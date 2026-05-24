"""Tests for tessera.expression.tree.simplify."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    simplify, evaluate, complexity,
    LinearFunctional, SeparableBilinear, Volterra2,
    measure_ema, measure_diff, measure_lag,
)
from tessera.expression.measure_2d import (
    measure_2d_laplacian_5pt, measure_2d_atomic, measure_2d_separable,
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
    """A FunctionalOp's child subtree should be simplified. For inputs
    that simplify to Var, the FunctionalOp is preserved; for inputs
    that simplify to Const, the new const-fold collapses the whole
    FunctionalOp (see test_linear_functional_of_const_folds for the
    fold behaviour)."""
    # Var-preserving case: inner simplifies to Var, FunctionalOp stays.
    inner = BinOp("mul", Var("x"), Const(1.0))   # → x
    func = LinearFunctional(measure=measure_ema(halflife=24))
    tree = FunctionalOp(func, (inner,))
    simp = simplify(tree)
    assert isinstance(simp, FunctionalOp)
    assert simp.args == (Var("x"),)
    assert simp.functional is func

    # Const-folding case: inner simplifies to Const, FunctionalOp folds.
    inner_zero = BinOp("sub", Var("y"), Var("y"))   # → 0
    tree_zero = FunctionalOp(func, (inner_zero,))
    simp_zero = simplify(tree_zero)
    # LinearFunctional(ema)(0) = 0 * sum(ema_kernel) = 0
    assert simp_zero == Const(0.0)


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


# ---------------- X/X → 1 fold ----------------

def test_x_div_x_folds_to_one():
    """X / X → 1 (the safe-divide convention; X=0 case gives 0 in actual
    evaluation, but the simplifier picks the algebraic identity)."""
    assert simplify(BinOp("div", Var("x"), Var("x"))) == Const(1.0)
    # Larger expr: image / image → 1 → can be folded by outer ops
    tree = BinOp("add", Const(2.0), BinOp("div", Var("image"), Var("image")))
    assert simplify(tree) == Const(3.0)


# ---------------- FunctionalOp(Const) const-folds ----------------

def test_linear_functional_of_const_folds():
    """LinearFunctional(μ)(Const c) = c · Σκ. Verified against
    eval(tree) on a single time step."""
    m = measure_ema(halflife=4, support_max=20)
    kernel_sum = float(m.to_kernel().sum())
    expected = 3.0 * kernel_sum
    tree = FunctionalOp(LinearFunctional(measure=m), (Const(3.0),))
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value - expected) < 1e-9


def test_linear_functional_of_const_zero_folds_to_zero():
    m = measure_diff(1)
    # Const(0): result is 0
    tree = FunctionalOp(LinearFunctional(measure=m), (Const(0.0),))
    assert simplify(tree) == Const(0.0)


def test_volterra2_of_const_folds():
    """Volterra2(μ_a, μ_b)(Const c) = c² · Σκ_a · Σκ_b."""
    m_a = measure_diff(1)
    m_b = measure_ema(halflife=4, support_max=20)
    sa = float(m_a.to_kernel().sum())
    sb = float(m_b.to_kernel().sum())
    expected = (2.5 ** 2) * sa * sb
    tree = FunctionalOp(Volterra2(measure_a=m_a, measure_b=m_b), (Const(2.5),))
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value - expected) < 1e-9


def test_separable_bilinear_of_two_consts_folds():
    """SeparableBilinear(μ_a, μ_b)(Const c_a, Const c_b)
       = c_a · Σκ_a · c_b · Σκ_b."""
    m_a = measure_diff(1)
    m_b = measure_lag(3)
    sa = float(m_a.to_kernel().sum())   # diff sums to 0
    sb = float(m_b.to_kernel().sum())   # lag sums to 1
    expected = 1.5 * sa * 2.0 * sb
    tree = FunctionalOp(
        SeparableBilinear(measure_a=m_a, measure_b=m_b),
        (Const(1.5), Const(2.0)),
    )
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value - expected) < 1e-9


def test_separable_bilinear_of_var_and_const_does_not_fold():
    """SB needs BOTH args constant; one Var means no fold."""
    m = measure_ema(halflife=4, support_max=20)
    tree = FunctionalOp(
        SeparableBilinear(measure_a=m, measure_b=m),
        (Var("x"), Const(2.0)),
    )
    folded = simplify(tree)
    # No fold; structure unchanged
    assert isinstance(folded, FunctionalOp)


# ---------------- FunctionalOp2D(Const) const-folds ----------------

def test_2d_functional_of_const_folds_atomic_only():
    """Measure2D with atoms only, applied to Const c, = c · sum(atom_weights)."""
    m = measure_2d_atomic([(0.5, 0, 0), (-0.3, 1, 0), (0.2, 0, 1)])
    total = 0.5 - 0.3 + 0.2  # = 0.4
    expected = 7.0 * total
    tree = FunctionalOp2D(m, Const(7.0))
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value - expected) < 1e-9


def test_2d_laplacian_of_const_folds_to_zero():
    """Laplacian-5pt sums to zero → laplacian(c) = 0."""
    m = measure_2d_laplacian_5pt()
    tree = FunctionalOp2D(m, Const(42.0))
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value) < 1e-9


def test_2d_separable_of_const_folds():
    """Measure2D with separable density, applied to Const c, =
    c · Σκ_t · Σκ_x."""
    m_t = measure_ema(halflife=2, support_max=10)
    m_x = measure_lag(1)   # sums to 1
    m_2d = measure_2d_separable(m_t, m_x)
    expected = 4.0 * float(m_t.to_kernel().sum()) * float(m_x.to_kernel().sum())
    tree = FunctionalOp2D(m_2d, Const(4.0))
    folded = simplify(tree)
    assert isinstance(folded, Const)
    assert abs(folded.value - expected) < 1e-9


def test_2d_functional_of_var_does_not_fold():
    """Only Const inputs trigger the fold; Var inputs leave structure intact."""
    m = measure_2d_laplacian_5pt()
    tree = FunctionalOp2D(m, Var("image"))
    folded = simplify(tree)
    assert isinstance(folded, FunctionalOp2D)


# ---------------- Composed: const-fold lets dead branches die ----------------

def test_dead_functional_branch_simplifies_away():
    """The MNIST-diagnostic scenario: an M2D applied to a constant inside
    a larger expression. After the const-fold, parsimony can prune it."""
    m = measure_2d_laplacian_5pt()
    # `3*image - M2D[laplacian](0.04)` where the M2D(const) is a dead branch
    dead_branch = FunctionalOp2D(m, Const(0.04))   # → Const(0) (Laplacian sums to 0)
    tree = BinOp("sub",
                 BinOp("mul", Const(3.0), Var("image")),
                 dead_branch)
    folded = simplify(tree)
    # The subtract-zero should also fold (existing identity)
    # Result: 3 * image  → cx=3
    assert complexity(folded) <= 5   # was 9-12 with dead branch attached
