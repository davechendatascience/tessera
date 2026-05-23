"""Tests for tessera.expression.tree.simplify."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp, FunctionalOp,
    simplify, evaluate, complexity,
    LinearFunctional, measure_ema,
)


# ---------------- Algebraic identities ----------------

def test_x_minus_x_folds_to_zero():
    x = Var("x")
    tree = BinOp("sub", x, x)
    assert simplify(tree) == Const(0.0)
    # Nested in larger expression: (logvol - logrange) / (logret - logret) → 0
    tree = BinOp("div",
                 BinOp("sub", Var("logvol"), Var("logrange")),
                 BinOp("sub", Var("logret"), Var("logret")))
    assert simplify(tree) == Const(0.0)


def test_safe_div_zero_denominator():
    # X / 0 → 0  (matches safe-divide BIN_OP_FNS["div"])
    assert simplify(BinOp("div", Var("x"), Const(0.0))) == Const(0.0)
    # 0 / X → 0
    assert simplify(BinOp("div", Const(0.0), Var("x"))) == Const(0.0)


def test_identity_ops_with_const_zero():
    x = Var("x")
    assert simplify(BinOp("add", x, Const(0.0))) == x
    assert simplify(BinOp("add", Const(0.0), x)) == x
    assert simplify(BinOp("sub", x, Const(0.0))) == x
    # 0 - x → neg(x)
    assert simplify(BinOp("sub", Const(0.0), x)) == UnOp("neg", x)


def test_identity_ops_with_const_one():
    x = Var("x")
    assert simplify(BinOp("mul", x, Const(1.0))) == x
    assert simplify(BinOp("mul", Const(1.0), x)) == x
    assert simplify(BinOp("div", x, Const(1.0))) == x


def test_multiply_by_zero():
    x = Var("x")
    assert simplify(BinOp("mul", x, Const(0.0))) == Const(0.0)
    assert simplify(BinOp("mul", Const(0.0), x)) == Const(0.0)


def test_min_max_equal_args():
    x = Var("x")
    assert simplify(BinOp("min", x, x)) == x
    assert simplify(BinOp("max", x, x)) == x


def test_double_negation():
    x = Var("x")
    assert simplify(UnOp("neg", UnOp("neg", x))) == x


def test_abs_of_neg_drops_neg():
    x = Var("x")
    assert simplify(UnOp("abs", UnOp("neg", x))) == UnOp("abs", x)


# ---------------- Constant folding ----------------

def test_const_fold_binary():
    assert simplify(BinOp("add", Const(2.0), Const(3.0))) == Const(5.0)
    assert simplify(BinOp("sub", Const(5.0), Const(3.0))) == Const(2.0)
    assert simplify(BinOp("mul", Const(2.0), Const(3.0))) == Const(6.0)
    assert simplify(BinOp("div", Const(6.0), Const(2.0))) == Const(3.0)
    assert simplify(BinOp("min", Const(2.0), Const(5.0))) == Const(2.0)
    assert simplify(BinOp("max", Const(2.0), Const(5.0))) == Const(5.0)


def test_const_fold_unary():
    assert simplify(UnOp("neg", Const(3.0))) == Const(-3.0)
    assert simplify(UnOp("abs", Const(-3.0))) == Const(3.0)
    assert simplify(UnOp("sign", Const(-2.0))) == Const(-1.0)
    folded = simplify(UnOp("tanh", Const(0.0)))
    assert isinstance(folded, Const) and abs(folded.value) < 1e-12


def test_const_fold_nested():
    # (2 + 3) * (4 - 1) → 5 * 3 → 15
    inner = BinOp("mul",
                  BinOp("add", Const(2.0), Const(3.0)),
                  BinOp("sub", Const(4.0), Const(1.0)))
    assert simplify(inner) == Const(15.0)


# ---------------- Composite real-world case ----------------

def test_pnl_pathology_simplifies_to_abs_logret():
    """The cx=10 pathology from tessera_btc_1h_pnl.md:
       (((logvol - logrange) / (logret - logret)) + abs(logret))
    should simplify to abs(logret).
    """
    tree = BinOp(
        "add",
        BinOp("div",
              BinOp("sub", Var("logvol"), Var("logrange")),
              BinOp("sub", Var("logret"), Var("logret"))),
        UnOp("abs", Var("logret")),
    )
    assert complexity(tree) == 10
    s = simplify(tree)
    assert s == UnOp("abs", Var("logret"))
    assert complexity(s) == 2


# ---------------- Semantic equivalence ----------------

def test_simplification_preserves_evaluation():
    """For a battery of trees, simplified version must evaluate to the
    same numerical result as the original (within float tolerance)."""
    rng = np.random.default_rng(0)
    n = 1000
    env = {
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "z": rng.standard_normal(n),
    }
    cases = [
        BinOp("sub", Var("x"), Var("x")),  # → 0
        BinOp("div", BinOp("sub", Var("x"), Var("x")), Var("y")),  # → 0/y → 0 (safe)
        BinOp("add", BinOp("mul", Var("x"), Const(1.0)), Const(0.0)),  # → x
        BinOp("mul", Var("x"), BinOp("sub", Var("y"), Var("y"))),  # → 0
        UnOp("neg", UnOp("neg", Var("x"))),  # → x
        BinOp("div", Var("x"), Const(0.0)),  # → 0
        BinOp("add",
              BinOp("div",
                    BinOp("sub", Var("x"), Var("y")),
                    BinOp("sub", Var("z"), Var("z"))),
              UnOp("abs", Var("x"))),  # → abs(x)
        BinOp("mul", Const(2.0), BinOp("add", Const(3.0), Var("x"))),  # mixed
    ]
    def _to_array(v, length):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return np.full(length, float(a))
        return a

    for tree in cases:
        orig = _to_array(evaluate(tree, env), n)
        simp = _to_array(evaluate(simplify(tree), env), n)
        mask = np.isfinite(orig) & np.isfinite(simp)
        np.testing.assert_allclose(orig[mask], simp[mask], rtol=1e-9,
                                    err_msg=f"mismatch for {tree}")


# ---------------- Tree-with-FunctionalOp must recurse but not fold ----------------

def test_simplify_recurses_into_functional_args():
    """A FunctionalOp's child subtree should be simplified, but the
    FunctionalOp itself is preserved (no folding through measures)."""
    inner = BinOp("sub", Var("x"), Var("x"))   # → 0
    func = LinearFunctional(measure=measure_ema(halflife=24))
    tree = FunctionalOp(func, (inner,))
    simp = simplify(tree)
    assert isinstance(simp, FunctionalOp)
    assert simp.args == (Const(0.0),)
    assert simp.functional is func


# ---------------- Idempotence ----------------

def test_simplify_is_idempotent():
    """simplify(simplify(t)) == simplify(t) for any t."""
    rng = np.random.default_rng(1)
    trees = [
        BinOp("sub", Var("x"), Var("x")),
        BinOp("mul", Const(0.0), Var("y")),
        BinOp("add",
              BinOp("div", BinOp("sub", Var("x"), Var("x")), Var("y")),
              UnOp("abs", Var("x"))),
        UnOp("neg", UnOp("neg", UnOp("neg", Var("x")))),
    ]
    for t in trees:
        once = simplify(t)
        twice = simplify(once)
        assert once == twice, f"simplify not idempotent for {t}: {once} vs {twice}"
