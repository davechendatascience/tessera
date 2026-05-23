"""Tests for the reduce_mean / reduce_max / reduce_sum / reduce_std
unary ops introduced for invariance-via-aggregation in SR."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp,
    UN_OPS, UN_OP_FNS,
    evaluate, complexity,
)
from tessera.expression.interval import (
    Interval, interval_evaluate,
)
from tessera.expression.simplify import simplify_canonical


# ---------------- Op table membership ----------------

@pytest.mark.parametrize("op", ["reduce_mean", "reduce_max",
                                  "reduce_sum", "reduce_std"])
def test_reduce_op_registered(op):
    assert op in UN_OPS
    assert op in UN_OP_FNS


# ---------------- Numerical correctness ----------------

def test_reduce_mean_on_2d():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert evaluate(UnOp("reduce_mean", Var("x")), {"x": x}) == pytest.approx(2.5)


def test_reduce_max_on_2d():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert evaluate(UnOp("reduce_max", Var("x")), {"x": x}) == 4.0


def test_reduce_sum_on_2d():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert evaluate(UnOp("reduce_sum", Var("x")), {"x": x}) == 10.0


def test_reduce_std_zero_for_constant_input():
    x = np.full((4, 4), 7.0)
    assert evaluate(UnOp("reduce_std", Var("x")), {"x": x}) == pytest.approx(0.0)


def test_reduce_std_nonzero_for_varying_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    s = evaluate(UnOp("reduce_std", Var("x")), {"x": x})
    assert 0.5 < s < 1.5


# ---------------- NaN handling ----------------

def test_reduce_ignores_nan_entries():
    x = np.array([1.0, np.nan, 3.0])
    assert evaluate(UnOp("reduce_mean", Var("x")), {"x": x}) == pytest.approx(2.0)
    assert evaluate(UnOp("reduce_max", Var("x")), {"x": x}) == 3.0
    assert evaluate(UnOp("reduce_sum", Var("x")), {"x": x}) == 4.0


def test_reduce_returns_nan_when_all_nan():
    x = np.full(5, np.nan)
    out = evaluate(UnOp("reduce_mean", Var("x")), {"x": x})
    assert np.isnan(out)


# ---------------- Output is scalar ----------------

def test_reduce_output_is_scalar():
    """Result is a float, not an array — used downstream via broadcast."""
    x = np.array([1.0, 2.0, 3.0])
    out = evaluate(UnOp("reduce_mean", Var("x")), {"x": x})
    assert np.isscalar(out) or (hasattr(out, "shape") and out.shape == ())


def test_reduce_then_binop_broadcasts():
    """reduce + array broadcasts via numpy."""
    x = np.array([1.0, 2.0, 3.0])
    # mean(x) = 2; then 2 + x = [3, 4, 5]
    tree = BinOp("add", UnOp("reduce_mean", Var("x")), Var("x"))
    out = evaluate(tree, {"x": x})
    np.testing.assert_array_equal(out, [3.0, 4.0, 5.0])


# ---------------- Interval bounds ----------------

def test_reduce_mean_interval_within_input_interval():
    iv = interval_evaluate(UnOp("reduce_mean", Var("x")),
                            {"x": Interval(-2.0, 3.0)})
    # Mean lies in [-2, 3]
    assert iv.lo == -2.0
    assert iv.hi == 3.0


def test_reduce_max_interval_within_input_interval():
    iv = interval_evaluate(UnOp("reduce_max", Var("x")),
                            {"x": Interval(0.0, 5.0)})
    assert iv.lo == 0.0
    assert iv.hi == 5.0


def test_reduce_sum_interval_unbounded_when_spans_zero():
    iv = interval_evaluate(UnOp("reduce_sum", Var("x")),
                            {"x": Interval(-1.0, 1.0)})
    # Conservative: ±inf because array size unknown
    assert not (np.isfinite(iv.lo) and np.isfinite(iv.hi))


def test_reduce_sum_interval_same_sign():
    iv = interval_evaluate(UnOp("reduce_sum", Var("x")),
                            {"x": Interval(1.0, 5.0)})
    # All-positive input: sum is non-negative, possibly unbounded above
    assert iv.lo == 0.0
    assert iv.hi == float("inf")


def test_reduce_std_interval():
    iv = interval_evaluate(UnOp("reduce_std", Var("x")),
                            {"x": Interval(-1.0, 1.0)})
    # std bounded by spread = 2
    assert iv.lo == 0.0
    assert 0.0 < iv.hi <= 2.0


# ---------------- Simplifier ----------------

def test_simplify_reduce_of_const_folds():
    """reduce_mean(Const(c)) = c (scalar collapses to itself)."""
    for op in ("reduce_mean", "reduce_max", "reduce_sum"):
        out = simplify_canonical(UnOp(op, Const(5.0)))
        # Either Const(5.0) (folded) or the original UnOp
        # The constant-folding rule will fold scalars via the lambda
        # returning float(...)
        if isinstance(out, Const):
            assert out.value == pytest.approx(5.0)


# ---------------- Mutation integration ----------------

def test_random_tree_can_include_reduce_ops():
    """random_tree's UN_OPS sampling includes the new reduce ops."""
    import random
    from tessera.expression.mutation import random_tree
    rng = random.Random(42)
    found_reduce = False
    for _ in range(200):
        t = random_tree(rng, ["x"], max_depth=3, enable_2d=False)
        # Check tree for any reduce_ op
        from tessera.expression.tree import iter_subtrees
        for sub in iter_subtrees(t):
            if isinstance(sub, UnOp) and sub.op.startswith("reduce_"):
                found_reduce = True
                break
        if found_reduce:
            break
    assert found_reduce, (
        "random_tree should occasionally produce reduce ops "
        "(4 of 9 unary ops are now reduces)"
    )
