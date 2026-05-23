"""Tests for the indicator primitives: gt, lt, ge, le, step."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp,
    BIN_OPS, UN_OPS, BIN_OP_FNS, UN_OP_FNS,
    simplify, evaluate, complexity,
)


# ---------------- Op tables include the new ops ----------------

def test_indicator_binops_in_table():
    for op in ("gt", "lt", "ge", "le"):
        assert op in BIN_OPS
        assert op in BIN_OP_FNS


def test_step_in_unop_table():
    assert "step" in UN_OPS
    assert "step" in UN_OP_FNS


# ---------------- Evaluation correctness ----------------

@pytest.fixture
def env():
    return {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "y": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
    }


def test_gt_element_wise(env):
    tree = BinOp("gt", Var("x"), Var("y"))
    out = evaluate(tree, env)
    np.testing.assert_array_equal(out, [0., 0., 0., 1., 1.])


def test_lt_element_wise(env):
    tree = BinOp("lt", Var("x"), Var("y"))
    out = evaluate(tree, env)
    np.testing.assert_array_equal(out, [1., 1., 0., 0., 0.])


def test_ge_equals_gt_or_eq(env):
    tree = BinOp("ge", Var("x"), Var("y"))
    out = evaluate(tree, env)
    np.testing.assert_array_equal(out, [0., 0., 1., 1., 1.])


def test_le_complement_of_gt(env):
    tree = BinOp("le", Var("x"), Var("y"))
    out = evaluate(tree, env)
    np.testing.assert_array_equal(out, [1., 1., 1., 0., 0.])


def test_step_unary(env):
    tree = UnOp("step", Var("x"))
    out = evaluate(tree, env)
    # x = [1, 2, 3, 4, 5]; all > 0
    np.testing.assert_array_equal(out, [1., 1., 1., 1., 1.])

    tree = UnOp("step", BinOp("sub", Var("x"), Const(3.0)))
    out = evaluate(tree, env)
    # x - 3 = [-2, -1, 0, 1, 2]; > 0 at indices 3 and 4
    np.testing.assert_array_equal(out, [0., 0., 0., 1., 1.])


def test_gt_lt_return_float64_not_bool():
    """Downstream ops (mul, sum, etc.) need float input — bool would
    silently break linear algebra."""
    out = BIN_OP_FNS["gt"](np.array([1.0, 2.0]), np.array([1.5, 1.5]))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0., 1.])


# ---------------- Pretty printing ----------------

def test_indicator_ops_print_infix():
    assert str(BinOp("gt", Var("x"), Var("y"))) == "(x > y)"
    assert str(BinOp("lt", Var("x"), Var("y"))) == "(x < y)"
    assert str(BinOp("ge", Var("x"), Var("y"))) == "(x >= y)"
    assert str(BinOp("le", Var("x"), Var("y"))) == "(x <= y)"
    assert str(UnOp("step", Var("x"))) == "step(x)"


# ---------------- Constant folding ----------------

def test_const_fold_gt_lt():
    assert simplify(BinOp("gt", Const(3.0), Const(2.0))) == Const(1.0)
    assert simplify(BinOp("gt", Const(1.0), Const(2.0))) == Const(0.0)
    assert simplify(BinOp("lt", Const(1.0), Const(2.0))) == Const(1.0)
    assert simplify(BinOp("lt", Const(3.0), Const(2.0))) == Const(0.0)


def test_const_fold_ge_le():
    # x == y case
    assert simplify(BinOp("ge", Const(2.0), Const(2.0))) == Const(1.0)
    assert simplify(BinOp("le", Const(2.0), Const(2.0))) == Const(1.0)


def test_const_fold_step():
    assert simplify(UnOp("step", Const(0.5))) == Const(1.0)
    assert simplify(UnOp("step", Const(-0.5))) == Const(0.0)
    assert simplify(UnOp("step", Const(0.0))) == Const(0.0)


# ---------------- Algebraic identities ----------------

def test_gt_lt_equal_args_fold_to_zero():
    """gt(x, x) and lt(x, x) are always false."""
    assert simplify(BinOp("gt", Var("x"), Var("x"))) == Const(0.0)
    assert simplify(BinOp("lt", Var("x"), Var("x"))) == Const(0.0)


def test_ge_le_equal_args_fold_to_one():
    """ge(x, x) and le(x, x) are always true."""
    assert simplify(BinOp("ge", Var("x"), Var("x"))) == Const(1.0)
    assert simplify(BinOp("le", Var("x"), Var("x"))) == Const(1.0)


def test_simplify_preserves_indicator_semantics():
    """Semantic equivalence between original and simplified indicator
    expressions on actual data."""
    rng = np.random.default_rng(0)
    n = 200
    env = {
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
    }
    cases = [
        BinOp("gt", Var("x"), Var("x")),                       # → 0
        BinOp("ge", Var("y"), Var("y")),                       # → 1
        BinOp("mul", BinOp("gt", Var("x"), Var("y")), Var("x")),
        BinOp("add", UnOp("step", Var("x")),
                     UnOp("step", UnOp("neg", Var("x")))),    # ~ 1 except at x=0
    ]
    for tree in cases:
        orig = np.asarray(evaluate(tree, env), dtype=np.float64)
        simp = np.asarray(evaluate(simplify(tree), env), dtype=np.float64)
        if orig.ndim == 0:
            orig = np.full(n, float(orig))
        if simp.ndim == 0:
            simp = np.full(n, float(simp))
        np.testing.assert_allclose(orig, simp, rtol=1e-9,
                                    err_msg=f"mismatch for {tree}")


# ---------------- Use in compound expressions ----------------

def test_indicator_gated_signal(env):
    """volume gated by `x > y` is a common SR pattern; verify it composes."""
    # signal = x * gt(x, y)   — keeps x where x > y, else 0
    tree = BinOp("mul", Var("x"), BinOp("gt", Var("x"), Var("y")))
    out = evaluate(tree, env)
    # x = [1,2,3,4,5]; y = 3 everywhere; gt(x, y) = [0,0,0,1,1]
    expected = np.array([0., 0., 0., 4., 5.])
    np.testing.assert_array_equal(out, expected)


def test_step_equivalent_to_gt_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    step_tree = UnOp("step", Var("x"))
    gt_tree = BinOp("gt", Var("x"), Const(0.0))
    np.testing.assert_array_equal(
        evaluate(step_tree, {"x": x}),
        evaluate(gt_tree, {"x": x}),
    )


def test_complexity_of_indicators_is_one_node():
    """Each indicator op is a single tree node — same as arithmetic."""
    assert complexity(BinOp("gt", Var("x"), Var("y"))) == 3
    assert complexity(UnOp("step", Var("x"))) == 2
