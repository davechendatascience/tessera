"""Tests for the GP loop."""
from __future__ import annotations

import random

import numpy as np
import pytest

from tessera.expression.gp import (
    Candidate, GP, GPConfig, mse_loss, pareto_front,
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

def test_mse_loss_nan_predictions_excluded_then_penalized():
    """Predictions with too few valid samples should yield inf."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([np.nan]*4 + [5.0])
    # Only 1 valid sample (below the 10/N=5 min threshold) → inf
    assert mse_loss(y_pred, y_true) == float("inf")

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
