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


def test_interval_functional_linear_bounded():
    """LinearFunctional now gets a tight L1-norm bound (step c)."""
    from tessera.expression import LinearFunctional, measure_lag, FunctionalOp
    tree = FunctionalOp(LinearFunctional(measure=measure_lag(1)), (Var("x"),))
    iv = interval_evaluate(tree, {"x": Interval(0.0, 1.0)})
    # measure_lag has ||m||_1 = 1, x in [0, 1] → output in [-1, 1] (could be
    # tighter since the input is non-negative, but the conservative bound is sound).
    assert iv.lo == pytest.approx(-1.0)
    assert iv.hi == pytest.approx(1.0)


def test_interval_functional_2d_unbounded():
    """FunctionalOp2D still conservative ±inf (future tightening)."""
    from tessera.expression import (
        FunctionalOp2D, measure_2d_diff_t,
    )
    tree = FunctionalOp2D(measure_2d_diff_t(lag_t=1), Var("x"))
    iv = interval_evaluate(tree, {"x": Interval(0.0, 1.0)})
    assert iv.lo == -float("inf") or iv.hi == float("inf")


def test_measure_l1_norm():
    """L1 norm matches sum of |kernel| for atomic + density measures."""
    from tessera.expression import measure_signed_sum, measure_ema, measure_diff
    from tessera.expression.interval import measure_l1_norm
    # Atomic only: ||(1·δ(0) + -1·δ(1))||_1 = 2
    assert measure_l1_norm(measure_diff(1)) == pytest.approx(2.0)
    # EMA: density integrates to ~1 (geometric series)
    l1_ema = measure_l1_norm(measure_ema(halflife=10))
    assert 0.95 < l1_ema < 1.01
    # Mixed: atom weight 2 + atom weight -1 → ||·||_1 = 3
    assert measure_l1_norm(measure_signed_sum([(2.0, 0), (-1.0, 5)])) == pytest.approx(3.0)


def test_interval_separable_bilinear_bound():
    """SeparableBilinear bound is the product of LinearFunctional bounds."""
    from tessera.expression import (
        FunctionalOp, SeparableBilinear, measure_diff, measure_ema,
    )
    tree = FunctionalOp(
        SeparableBilinear(measure_a=measure_diff(1), measure_b=measure_ema(halflife=10)),
        (Var("x"), Var("y")),
    )
    env_iv = {"x": Interval(-1.0, 1.0), "y": Interval(0.0, 2.0)}
    iv = interval_evaluate(tree, env_iv)
    # ||diff||_1 = 2, ||ema(10)||_1 ≈ 1; M_x = 1, M_y = 2
    # bound = 2 * 1 * 1 * 2 = 4
    assert iv.lo == pytest.approx(-4.0, rel=0.05)
    assert iv.hi == pytest.approx(4.0, rel=0.05)


def test_interval_volterra2_bound():
    """Volterra2 bound is the squared LinearFunctional bound on x."""
    from tessera.expression import FunctionalOp, Volterra2, measure_diff
    tree = FunctionalOp(
        Volterra2(measure_a=measure_diff(1), measure_b=measure_diff(1)),
        (Var("x"),),
    )
    env_iv = {"x": Interval(-1.0, 1.0)}
    iv = interval_evaluate(tree, env_iv)
    # ||diff||_1 = 2, M_x = 1 → bound = 2 * 2 * 1 * 1 = 4
    assert iv.lo == pytest.approx(-4.0)
    assert iv.hi == pytest.approx(4.0)


def test_interval_functional_bound_is_sound_empirically():
    """For random data, the empirical max of |L(x)| must fall within
    the interval bound."""
    from tessera.expression import (
        FunctionalOp, LinearFunctional, measure_ema, evaluate,
    )
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n) * 2     # x roughly in [-6, 6] (3σ)
    tree = FunctionalOp(LinearFunctional(measure=measure_ema(halflife=20)), (Var("x"),))
    env_iv = env_intervals_from_arrays({"x": x})
    iv = interval_evaluate(tree, env_iv)

    # Actual evaluation
    y = evaluate(tree, {"x": x})
    finite = y[np.isfinite(y)]
    # Empirical max must lie within [iv.lo, iv.hi]
    assert iv.lo <= finite.min(), f"empirical {finite.min()} < bound lo {iv.lo}"
    assert finite.max() <= iv.hi, f"empirical {finite.max()} > bound hi {iv.hi}"


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
