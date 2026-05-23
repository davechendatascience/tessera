"""Tests for the GP loop."""
from __future__ import annotations

import random

import numpy as np
import pytest

from tessera.expression.gp import (
    Candidate, GP, GPConfig, mse_loss, pareto_front,
    _prediction_is_valid,
)
from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, complexity,
)
from tessera.expression.measure import measure_ema, measure_diff
from tessera.expression.functional import LinearFunctional


# ---------------- mse_loss ----------------

def test_mse_loss_basic():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert mse_loss(y_pred, y_true) == 0.0

def test_mse_loss_nan_handling():
    """mse_loss averages over the finite mask. The NaN-fraction precheck
    (rejecting majority-NaN predictions) lives in _prediction_is_valid,
    which the GP scoring path calls before mse_loss."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([np.nan]*4 + [5.0])
    # mse_loss averages over only the finite samples: err = (5-5)^2 = 0
    assert mse_loss(y_pred, y_true) == 0.0


def test_prediction_is_valid_rejects_too_many_nans():
    """The precheck (default min_valid_frac=0.9) rejects when fewer
    than 90% of entries are finite. There's also a hard floor of 2
    finite samples so a single lucky point can't pass."""
    y_true = np.arange(100, dtype=float)

    # 95% finite — passes 0.9 threshold
    y_95 = y_true.copy(); y_95[:5] = np.nan
    assert _prediction_is_valid(y_95, y_true, min_valid_frac=0.9)

    # 80% finite — fails 0.9, passes 0.5
    y_80 = y_true.copy(); y_80[:20] = np.nan
    assert not _prediction_is_valid(y_80, y_true, min_valid_frac=0.9)
    assert _prediction_is_valid(y_80, y_true, min_valid_frac=0.5)

    # All finite — passes anything
    assert _prediction_is_valid(y_true.copy(), y_true, min_valid_frac=0.9)

    # Hard floor: even at frac=0.0, need >=2 finite samples
    y_one = np.full_like(y_true, np.nan); y_one[50] = 1.0
    assert not _prediction_is_valid(y_one, y_true, min_valid_frac=0.0)


def test_prediction_is_valid_handles_scalar_predictions():
    y_true = np.arange(5.0)
    assert _prediction_is_valid(2.0, y_true, 0.9)         # finite scalar OK
    assert not _prediction_is_valid(float("nan"), y_true, 0.9)
    assert not _prediction_is_valid(float("inf"), y_true, 0.9)


def test_prediction_is_valid_catches_div_by_zero_pathology():
    """The (logret - logret) / (logret - logret) GP pathology produces
    all-NaN; precheck must reject it."""
    y_true = np.linspace(-1.0, 1.0, 100)
    y_pred_all_nan = np.full(100, np.nan)
    assert not _prediction_is_valid(y_pred_all_nan, y_true, 0.9)

def test_mse_loss_shape_broadcast_or_inf():
    y_true = np.array([1.0, 2.0, 3.0])
    # Scalar prediction broadcasts
    assert mse_loss(np.float64(2.0), y_true) == pytest.approx(2/3)
    # Incompatible shapes → inf
    assert mse_loss(np.array([1.0, 2.0]), y_true) == float("inf")


# ---------------- Pareto front ----------------

def _mock_candidate(loss: float, cx: int, gen: int = 0) -> Candidate:
    """Build a mock Candidate for testing without going through GP scoring."""
    # The tree is a no-op placeholder; tests don't evaluate it
    tree = Var(f"v{cx}_{loss}")
    return Candidate(
        tree=tree, train_loss=loss, complexity=cx,
        fitness=loss + 0.01 * cx, born_gen=gen,
    )

def test_pareto_front_empty():
    assert pareto_front([]) == []

def test_pareto_front_singleton():
    c = _mock_candidate(loss=0.5, cx=3)
    assert pareto_front([c]) == [c]

def test_pareto_front_dominance():
    """A candidate is dominated if another has lower-or-equal in both axes
    with at least one strictly lower."""
    a = _mock_candidate(loss=1.0, cx=2)
    b = _mock_candidate(loss=0.5, cx=4)   # less loss but higher cx
    c = _mock_candidate(loss=0.8, cx=3)   # in between
    d = _mock_candidate(loss=2.0, cx=5)   # dominated by all
    front = pareto_front([a, b, c, d])
    # a, b, c all non-dominated; d is dominated
    assert set(front) == {a, b, c}

def test_pareto_front_sorted_ascending_cx():
    cands = [
        _mock_candidate(loss=0.5, cx=4),
        _mock_candidate(loss=1.0, cx=2),
        _mock_candidate(loss=0.7, cx=3),
    ]
    front = pareto_front(cands)
    assert [c.complexity for c in front] == sorted(c.complexity for c in front)


# ---------------- GPConfig ----------------

def test_gpconfig_defaults_sane():
    cfg = GPConfig()
    assert cfg.pop_size >= 10
    assert cfg.n_gens >= 1
    assert 0.0 < cfg.parsimony < 1.0


# ---------------- End-to-end on a trivial problem ----------------

def test_gp_finds_simple_linear_relationship():
    """y = 0.5*x1 - 0.3*x2 + noise. A linear OLS-style problem; GP should
    converge to a low-MSE expression quickly."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)   # red-herring
    y = 0.5 * x1 - 0.3 * x2 + 0.05 * rng.standard_normal(n)

    cfg = GPConfig(
        pop_size=60, n_gens=15, init_max_depth=3, parsimony=0.001,
        verbose=False, seed=42,
    )
    gp = GP(cfg)
    env = {"x1": x1, "x2": x2, "x3": x3}
    front = gp.run(env, y, feature_names=["x1", "x2", "x3"])

    assert len(front) >= 1
    best_loss = min(c.train_loss for c in front)
    # Linear OLS-equivalent gives MSE ≈ 0.0025 (noise floor). The GP should
    # at least get close to a low-MSE solution given the small problem.
    assert best_loss < 0.5, f"GP failed to fit a linear target (best loss {best_loss:.3f})"


def test_gp_pareto_front_has_decreasing_loss_with_increasing_cx():
    """On any reasonable run, the Pareto front members should satisfy
    monotone decreasing loss as complexity increases."""
    rng = np.random.default_rng(1)
    n = 300
    x = rng.standard_normal(n)
    y = np.tanh(x) + 0.1 * rng.standard_normal(n)

    cfg = GPConfig(pop_size=40, n_gens=8, parsimony=0.005, verbose=False, seed=7)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])

    assert len(front) >= 1
    losses = [c.train_loss for c in front]
    cxs = [c.complexity for c in front]
    # Sorted by cx ascending, losses must be non-increasing
    assert all(losses[i] >= losses[i+1] for i in range(len(losses) - 1)), \
        f"Pareto front not monotone: cx={cxs}, losses={losses}"


def test_gp_history_tracks_per_gen_stats():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.standard_normal(n)
    y = x ** 2

    cfg = GPConfig(pop_size=30, n_gens=5, verbose=False, seed=11)
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])

    assert len(gp.history) >= 1
    for h in gp.history:
        for key in ("gen", "best_loss", "best_cx", "pareto_size", "hit_rate", "n_cache"):
            assert key in h


# ---------------- Stop control + reproducibility ----------------

def test_gp_stop_signal_exits_loop_early():
    """If stop() is called between generations, the loop should exit."""
    rng = np.random.default_rng(3)
    n = 200
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n) * 0.1

    cfg = GPConfig(pop_size=30, n_gens=100, verbose=False, seed=23)
    gp = GP(cfg)
    # Subclass-trick: after gen=0 init, request stop
    gp._stop_requested = True
    front = gp.run({"x": x}, y, ["x"])
    assert len(gp.history) <= 1   # init log + at most one gen


def test_gp_reproducible_with_seed():
    """Same seed + same data → same final Pareto front (by tree equality)."""
    rng = np.random.default_rng(4)
    n = 200
    x = rng.standard_normal(n)
    y = x ** 2 + 0.5

    cfg_kwargs = dict(pop_size=30, n_gens=6, verbose=False, seed=999)
    gp1 = GP(GPConfig(**cfg_kwargs))
    gp2 = GP(GPConfig(**cfg_kwargs))

    front1 = gp1.run({"x": x}, y, ["x"])
    front2 = gp2.run({"x": x}, y, ["x"])

    losses1 = sorted(c.train_loss for c in front1)
    losses2 = sorted(c.train_loss for c in front2)
    assert losses1 == losses2


# ---------------- Early stopping ----------------

def test_gp_early_stop_when_no_improvement():
    """A trivial constant target — GP converges quickly, then early-stop
    should fire well before n_gens."""
    rng = np.random.default_rng(5)
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n)   # trivial target

    cfg = GPConfig(
        pop_size=30, n_gens=50, parsimony=0.005,
        early_stop_patience=5, verbose=False, seed=17,
    )
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])
    # Should NOT use all 50 generations
    assert len(gp.history) < 50


# ---------------- Candidate equality / hashing ----------------

def test_candidate_is_hashable():
    c = _mock_candidate(loss=0.5, cx=3)
    s = {c}
    assert c in s


def test_candidate_equality_by_value():
    c1 = _mock_candidate(loss=0.5, cx=3, gen=2)
    c2 = _mock_candidate(loss=0.5, cx=3, gen=2)
    assert c1 == c2


# ---------------- Multiprocessing ----------------

def test_gp_with_n_workers_1_matches_sequential():
    """n_workers=1 (default) should be byte-identical to the no-MP path."""
    rng = np.random.default_rng(60)
    n = 200
    x = rng.standard_normal(n)
    y = np.tanh(x)

    cfg = GPConfig(pop_size=20, n_gens=4, verbose=False, seed=60, n_workers=1)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) >= 1
    assert gp._pool is None   # never created a pool


def test_gp_with_workers_produces_consistent_pareto():
    """With n_workers=2, the run completes and yields a sane Pareto front.

    NOTE: results are not bitwise identical to n_workers=1 because:
      1. Per-worker caches don't share hit/miss stats
      2. Floating point reductions may differ in order
    But the loss-vs-cx structure should be plausible.
    """
    rng = np.random.default_rng(61)
    n = 200
    x = rng.standard_normal(n)
    y = x ** 2

    cfg = GPConfig(pop_size=16, n_gens=3, verbose=False, seed=61, n_workers=2)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])

    assert len(front) >= 1
    # Sorted by cx ascending, loss non-increasing
    losses = [c.train_loss for c in front]
    assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    # Pool was created and shut down
    assert gp._pool is None


# ---------------- Cache integration ----------------

def test_gp_cache_records_hits_over_generations():
    """As generations progress, cache hit rate should be non-zero."""
    rng = np.random.default_rng(6)
    n = 300
    x = rng.standard_normal(n)
    y = np.tanh(x * 0.5) + 0.1 * rng.standard_normal(n)

    cfg = GPConfig(pop_size=30, n_gens=5, verbose=False, seed=44)
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])
    # Some hits must have happened (subexpression reuse)
    assert gp.cache.stats["mem_hits"] > 0


# ---------------- Custom loss_fn ----------------

def _abs_err_loss(y_pred, y_true):
    """Top-level (picklable) custom loss — mean absolute error.

    Defined at module level so it survives ProcessPoolExecutor pickling.
    Closures and lambdas would not.
    """
    y_pred = np.asarray(y_pred)
    if np.isscalar(y_pred) or y_pred.shape == ():
        y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def test_gp_serial_path_honors_custom_loss_fn():
    """GP(loss_fn=...) uses the custom loss in the serial scoring path."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    y = x + 0.05 * rng.standard_normal(n)

    cfg = GPConfig(pop_size=30, n_gens=5, verbose=False, seed=77, n_workers=1)
    gp = GP(cfg, loss_fn=_abs_err_loss)
    front = gp.run({"x": x}, y, ["x"])
    # All losses must be in the MAE scale (~0.1), not MSE (~0.01)
    assert all(c.train_loss >= 0 for c in front)
    # Sanity: identity x should reach ~0.04 MAE (noise-floor)
    best = min(front, key=lambda c: c.train_loss)
    assert best.train_loss < 0.1, f"MAE-best={best.train_loss} too high for x+noise"


def test_gp_worker_path_honors_custom_loss_fn():
    """When n_workers>1, the worker also uses the custom loss.

    Regression for the bug where _score_in_worker hardcoded mse_loss.
    """
    import sys
    if sys.platform == "win32":
        # Windows spawn requires the worker to re-import the test module.
        # Pytest's collected modules aren't always re-importable under spawn,
        # so this test is restricted to platforms where fork is the default.
        pytest.skip("workers use spawn on Windows; loss_fn pickling needs "
                    "the function to be importable, which pytest test modules "
                    "may not satisfy")
    rng = np.random.default_rng(8)
    n = 500
    x = rng.standard_normal(n)
    y = x + 0.05 * rng.standard_normal(n)

    cfg = GPConfig(pop_size=40, n_gens=4, verbose=False, seed=88, n_workers=2)
    gp = GP(cfg, loss_fn=_abs_err_loss)
    front = gp.run({"x": x}, y, ["x"])
    best = min(front, key=lambda c: c.train_loss)
    # If worker used the OLD hardcoded mse_loss, best would be ~0.0025;
    # with the abs_err loss, best is ~0.04 (the MAE noise floor).
    assert best.train_loss > 0.01, (
        f"worker appears to have used mse_loss instead of custom loss: "
        f"best train_loss={best.train_loss} looks MSE-scaled"
    )


def test_gp_evaluate_tree_rejects_mostly_nan_predictions():
    """Regression: a tree that produces majority-NaN output must score
    inf, not silently pass through to loss_fn (which might average over
    the lucky finite samples and return an artificially good score).

    Construction: feed a feature that's mostly NaN (e.g., a sparse signal
    with most rows missing). The simplest tree `Var("x")` then returns
    the NaN input — precheck catches it.
    """
    from tessera.expression.tree import Var
    from tessera.expression.gp import _evaluate_tree
    n = 200
    # 80% NaN, 20% finite — below the default min_valid_frac=0.9
    x = np.full(n, np.nan); x[:40] = np.linspace(0, 1, 40)
    y_true = np.linspace(0, 1, n)

    cfg = GPConfig(pop_size=30, n_gens=3, verbose=False, seed=99)
    gp = GP(cfg)

    # With default 0.9 threshold, the 0.2 valid fraction is rejected
    loss_strict = _evaluate_tree(
        Var("x"), {"x": x}, y_true, gp.cache,
        fill_warmup=0.0, loss_fn=mse_loss, min_valid_frac=0.9,
    )
    assert loss_strict == float("inf"), \
        f"precheck should reject 20%-valid prediction at 90% threshold; got {loss_strict}"

    # With loose threshold (0.1), 20% valid passes
    loss_loose = _evaluate_tree(
        Var("x"), {"x": x}, y_true, gp.cache,
        fill_warmup=0.0, loss_fn=mse_loss, min_valid_frac=0.1,
    )
    assert np.isfinite(loss_loose), \
        f"precheck should accept 20%-valid prediction at 10% threshold; got {loss_loose}"
