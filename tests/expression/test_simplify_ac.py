"""Tests for AC normalisation (simplify_ac) and the canonical
composition simplify_canonical = simplify ∘ simplify_ac."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp,
    evaluate, complexity,
    simplify,
)
from tessera.expression.simplify import simplify_ac, simplify_canonical


# ---------------- Commutativity (AC normalisation rules) ----------------

def test_add_commutativity():
    """`a + b` and `b + a` produce the same canonical tree."""
    a, b = Var("a"), Var("b")
    t1 = BinOp("add", a, b)
    t2 = BinOp("add", b, a)
    assert simplify_ac(t1) == simplify_ac(t2)


def test_mul_commutativity():
    a, b = Var("a"), Var("b")
    assert simplify_ac(BinOp("mul", a, b)) == simplify_ac(BinOp("mul", b, a))


def test_min_max_commutativity():
    a, b = Var("a"), Var("b")
    assert simplify_ac(BinOp("min", a, b)) == simplify_ac(BinOp("min", b, a))
    assert simplify_ac(BinOp("max", a, b)) == simplify_ac(BinOp("max", b, a))


def test_sub_is_not_commutative():
    """sub is NOT in _AC_OPS — `a - b` and `b - a` are different forms."""
    a, b = Var("a"), Var("b")
    t1 = simplify_ac(BinOp("sub", a, b))
    t2 = simplify_ac(BinOp("sub", b, a))
    # The two normalised forms must remain distinct
    assert t1 != t2


def test_div_is_not_commutative():
    a, b = Var("a"), Var("b")
    t1 = simplify_ac(BinOp("div", a, b))
    t2 = simplify_ac(BinOp("div", b, a))
    assert t1 != t2


def test_gt_lt_not_commutative():
    """gt/lt/ge/le are NOT commutative — they return distinct functions
    when args swap."""
    a, b = Var("a"), Var("b")
    assert simplify_ac(BinOp("gt", a, b)) != simplify_ac(BinOp("gt", b, a))
    assert simplify_ac(BinOp("lt", a, b)) != simplify_ac(BinOp("lt", b, a))


# ---------------- Associativity (flatten + sort + rebuild) ----------------

def test_add_associativity_three_args():
    """(a + b) + c, a + (b + c), (c + b) + a all canonicalise to the same."""
    a, b, c = Var("a"), Var("b"), Var("c")
    t1 = BinOp("add", BinOp("add", a, b), c)
    t2 = BinOp("add", a, BinOp("add", b, c))
    t3 = BinOp("add", BinOp("add", c, b), a)
    assert simplify_ac(t1) == simplify_ac(t2) == simplify_ac(t3)


def test_mul_associativity_three_args():
    a, b, c = Var("a"), Var("b"), Var("c")
    t1 = BinOp("mul", BinOp("mul", a, b), c)
    t2 = BinOp("mul", a, BinOp("mul", b, c))
    assert simplify_ac(t1) == simplify_ac(t2)


def test_canonical_form_is_left_leaning():
    """After normalisation, the tree's shape is left-leaning."""
    a, b, c = Var("a"), Var("b"), Var("c")
    # Build a right-leaning tree
    t = BinOp("add", a, BinOp("add", b, c))
    normed = simplify_ac(t)
    # Top-level is always BinOp("add", ...)
    assert isinstance(normed, BinOp) and normed.op == "add"
    # Left child should itself be a BinOp (= left-leaning), not a Var
    assert isinstance(normed.a, BinOp), f"expected left-leaning; got {normed}"


def test_mixed_op_chain_only_flattens_same_op():
    """`(a + b) * c` does NOT flatten — different op chain."""
    a, b, c = Var("a"), Var("b"), Var("c")
    t = BinOp("mul", BinOp("add", a, b), c)
    normed = simplify_ac(t)
    # Outer mul is preserved (one mul, with children (a+b) and c)
    assert isinstance(normed, BinOp) and normed.op == "mul"


# ---------------- Recursion into non-AC nodes ----------------

def test_recurses_into_unop():
    """abs(b + a) → abs(a + b) (AC norm applied to UnOp's child)."""
    a, b = Var("a"), Var("b")
    t = UnOp("abs", BinOp("add", b, a))
    normed = simplify_ac(t)
    assert isinstance(normed, UnOp) and normed.op == "abs"
    assert normed.a == simplify_ac(BinOp("add", a, b))


def test_recurses_into_sub():
    a, b, c = Var("a"), Var("b"), Var("c")
    # (b + a) - c → (a + b) - c
    t = BinOp("sub", BinOp("add", b, a), c)
    normed = simplify_ac(t)
    assert isinstance(normed, BinOp) and normed.op == "sub"
    assert normed.a == simplify_ac(BinOp("add", a, b))


# ---------------- Semantic preservation ----------------

def test_simplify_ac_preserves_semantics():
    """For any tree, simplify_ac evaluates to the same numeric result."""
    rng = np.random.default_rng(0)
    n = 200
    env = {
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    }
    cases = [
        BinOp("add", Var("b"), Var("a")),
        BinOp("mul", Var("c"), BinOp("add", Var("b"), Var("a"))),
        BinOp("add", BinOp("add", Var("a"), Var("b")), Var("c")),
        BinOp("min", Var("b"), Var("a")),
        UnOp("abs", BinOp("mul", Var("b"), Var("a"))),
    ]
    for tree in cases:
        orig = np.asarray(evaluate(tree, env), dtype=np.float64)
        normed = np.asarray(evaluate(simplify_ac(tree), env), dtype=np.float64)
        if orig.ndim == 0:
            orig = np.full(n, float(orig))
        if normed.ndim == 0:
            normed = np.full(n, float(normed))
        np.testing.assert_allclose(orig, normed, rtol=1e-12,
                                    err_msg=f"mismatch for {tree}")


# ---------------- Idempotence ----------------

def test_simplify_ac_is_idempotent():
    a, b, c, d = Var("a"), Var("b"), Var("c"), Var("d")
    trees = [
        BinOp("add", BinOp("add", BinOp("add", d, c), b), a),
        BinOp("mul", BinOp("add", Var("b"), Var("a")),
                     BinOp("add", Var("d"), Var("c"))),
        UnOp("abs", BinOp("add", Var("b"), Var("a"))),
    ]
    for t in trees:
        once = simplify_ac(t)
        twice = simplify_ac(once)
        assert once == twice, f"not idempotent: {t}"


# ---------------- simplify_canonical = simplify ∘ simplify_ac ----------------

def test_canonical_clusters_then_folds_constants():
    """`2 + x + 3` should canonicalise THEN fold to `x + 5`."""
    t = BinOp("add", BinOp("add", Const(2.0), Var("x")), Const(3.0))
    canon = simplify_canonical(t)
    # Should be equivalent to x + 5
    # The exact form depends on sort key — Const(5) has cx=1, Var(x) has cx=1
    # so the sort tiebreaker (str) decides. Const(5)'s str is "5"; Var(x)'s
    # str is "x". '5' < 'x' so we expect Const(5) on the left.
    assert canon == BinOp("add", Const(5.0), Var("x"))


def test_canonical_preserves_semantics():
    """Numerical equivalence between original and canonical form."""
    rng = np.random.default_rng(1)
    n = 200
    env = {
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "z": rng.standard_normal(n),
    }
    cases = [
        BinOp("add", Const(2.0), BinOp("add", Var("x"), Const(3.0))),
        BinOp("mul", Var("y"), BinOp("add", Var("x"), Const(0.0))),  # x*y after fold
        BinOp("sub", Var("x"), Var("x")),                              # 0 after fold
        BinOp("add", Var("z"),
              BinOp("mul", Const(0.0), BinOp("add", Var("x"), Var("y")))),  # z
    ]
    for tree in cases:
        orig = np.asarray(evaluate(tree, env), dtype=np.float64)
        canon = np.asarray(evaluate(simplify_canonical(tree), env), dtype=np.float64)
        if orig.ndim == 0:
            orig = np.full(n, float(orig))
        if canon.ndim == 0:
            canon = np.full(n, float(canon))
        np.testing.assert_allclose(orig, canon, rtol=1e-12,
                                    err_msg=f"mismatch for {tree}")


def test_canonical_handles_a_plus_b_equals_b_plus_a_with_constants():
    """Tessera's complexity for `a + b` and `b + a` is now the same."""
    t1 = simplify_canonical(BinOp("add", Var("alpha"), Var("beta")))
    t2 = simplify_canonical(BinOp("add", Var("beta"), Var("alpha")))
    assert t1 == t2
    assert complexity(t1) == complexity(t2)


def test_canonical_is_idempotent():
    a, b, c, x = Var("a"), Var("b"), Var("c"), Var("x")
    trees = [
        BinOp("add", BinOp("add", c, b), a),
        BinOp("mul", BinOp("add", b, Const(2.0)), Const(3.0)),
        BinOp("add", Const(2.0), BinOp("add", x, Const(3.0))),
    ]
    for t in trees:
        once = simplify_canonical(t)
        twice = simplify_canonical(once)
        assert once == twice, (
            f"simplify_canonical not idempotent: {t}\n"
            f"  once:  {once}\n"
            f"  twice: {twice}"
        )


# ---------------- Complexity is non-increasing ----------------

def test_simplify_ac_does_not_grow_trees():
    """AC normalisation rearranges but never adds nodes."""
    a, b, c = Var("a"), Var("b"), Var("c")
    trees = [
        BinOp("add", BinOp("add", c, b), a),
        BinOp("mul", a, BinOp("mul", b, c)),
        UnOp("abs", BinOp("min", b, a)),
    ]
    for t in trees:
        normed = simplify_ac(t)
        assert complexity(normed) == complexity(t), (
            f"AC norm changed complexity: {complexity(t)} → {complexity(normed)}"
        )


def test_canonical_can_reduce_complexity():
    """Canonical (AC + rule-based) CAN reduce complexity, because
    constant folding kicks in after constants cluster."""
    t = BinOp("add", BinOp("add", Const(2.0), Var("x")), Const(3.0))
    assert complexity(t) == 5
    canon = simplify_canonical(t)
    assert complexity(canon) == 3   # `5 + x` is 3 nodes
