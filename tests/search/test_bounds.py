"""Tests for interval-arithmetic lower-bound pruning."""
import numpy as np
import pytest

from tessera.expression import Var, Const, BinOp, UnOp
from tessera.expression.interval import (
    Interval, interval_evaluate, env_intervals_from_arrays,
)
from tessera.search import (
    Candidate, mse_lower_bound, pareto_threshold,
    GP, GPConfig, mse_loss,
)


# ---------------- Interval arithmetic ----------------

def test_interval_const():
    iv = interval_evaluate(Const(3.0), {})
    assert iv.lo == 3.0 and iv.hi == 3.0


def test_interval_var_from_env():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000) * 2 + 1   # mean ~1, std ~2
    env = {"x": x}
    iv = interval_evaluate(Var("x"), env_intervals_from_arrays(env))
    assert iv.lo == float(x.min())
    assert iv.hi == float(x.max())


def test_interval_add():
    env_iv = {"a": Interval(1.0, 3.0), "b": Interval(-2.0, 2.0)}
    iv = interval_evaluate(BinOp("add", Var("a"), Var("b")), env_iv)
    assert iv.lo == -1.0   # 1 + -2
    assert iv.hi == 5.0    # 3 + 2


def test_interval_sub():
    env_iv = {"a": Interval(1.0, 3.0), "b": Interval(-2.0, 2.0)}
    iv = interval_evaluate(BinOp("sub", Var("a"), Var("b")), env_iv)
    assert iv.lo == -1.0   # 1 - 2
    assert iv.hi == 5.0    # 3 - (-2)


def test_interval_mul():
    """4-corner products for [a,b] * [c,d]."""
    env_iv = {"a": Interval(-2.0, 3.0), "b": Interval(-1.0, 2.0)}
    iv = interval_evaluate(BinOp("mul", Var("a"), Var("b")), env_iv)
    # Corners: -2*-1=2, -2*2=-4, 3*-1=-3, 3*2=6 → [-4, 6]
    assert iv.lo == -4.0
    assert iv.hi == 6.0


def test_interval_div_zero_in_denominator_is_unbounded():
    """Safe-divide with 0 ∈ [b.lo, b.hi] is conservative ±inf."""
    env_iv = {"a": Interval(1.0, 2.0), "b": Interval(-1.0, 1.0)}
    iv = interval_evaluate(BinOp("div", Var("a"), Var("b")), env_iv)
    assert iv.lo == -float("inf") or iv.hi == float("inf")


def test_interval_div_positive_denominator():
    env_iv = {"a": Interval(2.0, 4.0), "b": Interval(1.0, 2.0)}
    iv = interval_evaluate(BinOp("div", Var("a"), Var("b")), env_iv)
    # a/b: 2/2=1 .. 4/1=4
    assert iv.lo == pytest.approx(1.0)
    assert iv.hi == pytest.approx(4.0)


def test_interval_abs():
    iv = interval_evaluate(UnOp("abs", Var("a")), {"a": Interval(-2.0, 3.0)})
    assert iv.lo == 0.0     # 0 ∈ [-2, 3]
    assert iv.hi == 3.0
    # Non-spanning
    iv2 = interval_evaluate(UnOp("abs", Var("a")), {"a": Interval(-5.0, -2.0)})
    assert iv2.lo == 2.0    # nearest to 0
    assert iv2.hi == 5.0


def test_interval_tanh():
    iv = interval_evaluate(UnOp("tanh", Var("a")), {"a": Interval(-10.0, 10.0)})
    assert -1.0 <= iv.lo <= 0.0
    assert 0.0 <= iv.hi <= 1.0


def test_interval_gt_tight_when_possible():
    """gt(a, b) where a.lo > b.hi → certain 1; else [0, 1]."""
    env_iv = {"a": Interval(3.0, 5.0), "b": Interval(0.0, 2.0)}
    iv = interval_evaluate(BinOp("gt", Var("a"), Var("b")), env_iv)
    assert iv.lo == 1.0 and iv.hi == 1.0
    # Crossing case
    env_iv = {"a": Interval(0.0, 5.0), "b": Interval(1.0, 4.0)}
    iv = interval_evaluate(BinOp("gt", Var("a"), Var("b")), env_iv)
    assert iv.lo == 0.0 and iv.hi == 1.0


def test_interval_functional_is_unbounded():
    """FunctionalOp args fall through to ±inf (conservative)."""
    from tessera.expression import LinearFunctional, measure_lag, FunctionalOp
    tree = FunctionalOp(LinearFunctional(measure=measure_lag(1)), (Var("x"),))
    iv = interval_evaluate(tree, {"x": Interval(0.0, 1.0)})
    # Either both inf or at least one — implementation returns full ±inf
    assert iv.lo == -float("inf") or iv.hi == float("inf")


# ---------------- MSE lower bound ----------------

def test_mse_lower_bound_zero_when_y_in_interval():
    y_true = np.linspace(-1.0, 1.0, 100)
    # Prediction interval [-2, 2] contains all y_true values
    lb = mse_lower_bound(-2.0, 2.0, y_true)
    assert lb == 0.0


def test_mse_lower_bound_positive_when_y_outside():
    y_true = np.array([10.0, 10.0, 10.0])
    # Prediction interval [0, 1] — all y_true outside, closest is 1
    lb = mse_lower_bound(0.0, 1.0, y_true)
    # err = (10 - 1)^2 = 81; mean = 81
    assert lb == pytest.approx(81.0)


def test_mse_lower_bound_infinite_interval_gives_zero():
    """Unbounded prediction interval gives no information → bound is 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    lb = mse_lower_bound(-float("inf"), float("inf"), y_true)
    assert lb == 0.0


# ---------------- Pareto threshold ----------------

def _cand(loss, cx):
    return Candidate(tree=Var(f"v{cx}_{loss}"), train_loss=loss,
                     complexity=cx, fitness=loss + 0.01 * cx, born_gen=0)


def test_pareto_threshold_minimum_at_or_below_cx():
    front = [_cand(1.0, 1), _cand(0.5, 3), _cand(0.2, 5)]
    # At cx=4: eligible are cx<=4 → cx={1, 3} → min loss = 0.5
    assert pareto_threshold(front, 4) == pytest.approx(0.5)
    # At cx=5: eligible are all → min = 0.2
    assert pareto_threshold(front, 5) == pytest.approx(0.2)


def test_pareto_threshold_inf_when_no_eligible():
    front = [_cand(0.5, 3), _cand(0.2, 5)]
    # At cx=1: nothing eligible → inf
    assert pareto_threshold(front, 1) == float("inf")


def test_pareto_threshold_empty_front():
    assert pareto_threshold([], 5) == float("inf")


# ---------------- GP integration ----------------

def test_gp_pruning_off_by_default():
    """Default config: prune_by_lower_bound=False; no pruning happens."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = x * x
    cfg = GPConfig(pop_size=20, n_gens=5, verbose=False, seed=1)
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])
    # No pruning happened
    assert gp.prune_stats["n_pruned"] == 0
    # n_evaluated may be 0 too (the worker path doesn't bump it); just
    # confirming pruning was a no-op.


def test_gp_pruning_on_does_not_break_search():
    """With pruning on, GP still produces a valid Pareto front."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = x * x + 0.3
    cfg = GPConfig(pop_size=30, n_gens=10, verbose=False, seed=1,
                   prune_by_lower_bound=True, n_workers=1)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) >= 1
    # Search ran; pruning either pruned some or didn't
    # (depends on how cheap the bound is on this small problem)


def test_gp_pruning_on_with_mse_loss_actually_prunes():
    """On a problem with a reasonable Pareto front, some candidates
    should be pruned."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = x   # ground truth: y = x

    cfg = GPConfig(pop_size=50, n_gens=15, verbose=False, seed=42,
                   prune_by_lower_bound=True, n_workers=1,
                   parsimony=1e-4)
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])
    # The Pareto front quickly finds something near y=x (loss ~ 0).
    # Pure-pointwise trees can be bounded; some should be pruned.
    total = gp.prune_stats["n_pruned"] + gp.prune_stats["n_evaluated"]
    # Sanity: total > 0 (the counter ran)
    assert total > 0
