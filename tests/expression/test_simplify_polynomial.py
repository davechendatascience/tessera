"""Tests for tessera.expression.simplify.polynomial — additive
polynomial canonicalisation.

Covers:
  - Basic like-term folding: `2x + 3x → 5x`, `x*x + 2*x*x → 3*x*x`
  - Multi-variable monomials: `x*y + 2*x*y → 3*x*y`
  - Subtraction handling: `2x - x → x`
  - Constant folding: `5 + x + 3 → 8 + x`
  - Negative coefficients via UnOp(neg)
  - Coefficient cancellation: `x - x → 0`
  - Opaque terms preserved: `x + sin(x) + x → 2x + sin(x)`
  - Recursion into UnOp / BinOp non-additive
  - Idempotency
  - Monotone-in-complexity property
  - Evaluation equivalence (semantics preserved)
  - Polish-output realistic case
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, complexity, evaluate as eval_tree,
)
from tessera.expression.simplify import (
    simplify_polynomial, simplify_canonical, simplify_full,
)


# ----------------------------------------------------------------------
# 1. Basic like-term folding
# ----------------------------------------------------------------------

class TestLikeTermFolding:
    def test_two_x_plus_three_x(self):
        """2*x + 3*x → 5*x"""
        t = BinOp("add",
                  BinOp("mul", Const(2.0), Var("x")),
                  BinOp("mul", Const(3.0), Var("x")))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 30)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), 5.0 * x)
        # Complexity should be ≤ input.
        assert complexity(result) <= complexity(t)

    def test_xx_plus_two_xx(self):
        """x*x + 2*x*x → 3*x*x"""
        xx = BinOp("mul", Var("x"), Var("x"))
        t = BinOp("add", xx, BinOp("mul", Const(2.0), xx))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 30)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), 3.0 * x ** 2)

    def test_three_term_polynomial(self):
        """2*x + 3*x*x + x → 3*x + 3*x*x   (degrees sorted)"""
        t = BinOp("add",
                  BinOp("add",
                        BinOp("mul", Const(2.0), Var("x")),
                        BinOp("mul", Const(3.0),
                              BinOp("mul", Var("x"), Var("x")))),
                  Var("x"))
        result = simplify_polynomial(t)
        x = np.linspace(-1, 1, 40)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}), 3.0 * x + 3.0 * x ** 2,
            rtol=1e-9
        )


# ----------------------------------------------------------------------
# 2. Multi-variable monomials
# ----------------------------------------------------------------------

class TestMultiVariable:
    def test_xy_plus_two_xy(self):
        xy = BinOp("mul", Var("x"), Var("y"))
        t = BinOp("add", xy, BinOp("mul", Const(2.0), xy))
        result = simplify_polynomial(t)
        x = np.linspace(0.1, 2, 20)
        y = np.linspace(0.1, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x, "y": y}),
                                   3.0 * x * y)

    def test_xy_and_yx_same_monomial(self):
        """x*y and y*x should be recognised as the same monomial
        (commutative-mult); the matcher sorts exponents by var name."""
        t = BinOp("add",
                  BinOp("mul", Var("x"), Var("y")),
                  BinOp("mul", Var("y"), Var("x")))
        result = simplify_polynomial(t)
        # Should yield 2*x*y.
        x = np.linspace(0.1, 2, 20)
        y = np.linspace(0.1, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x, "y": y}),
                                   2.0 * x * y)
        # Tree should be substantially smaller than original.
        assert complexity(result) < complexity(t)


# ----------------------------------------------------------------------
# 3. Subtraction handling
# ----------------------------------------------------------------------

class TestSubtraction:
    def test_two_x_minus_x(self):
        """2*x - x → x"""
        t = BinOp("sub", BinOp("mul", Const(2.0), Var("x")), Var("x"))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), x)

    def test_x_minus_x_to_zero(self):
        t = BinOp("sub", Var("x"), Var("x"))
        result = simplify_polynomial(t)
        assert isinstance(result, Const)
        assert result.value == 0.0

    def test_constant_subtraction(self):
        """5 - 3 - x → 2 - x"""
        t = BinOp("sub",
                  BinOp("sub", Const(5.0), Const(3.0)),
                  Var("x"))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), 2.0 - x)


# ----------------------------------------------------------------------
# 4. Constants
# ----------------------------------------------------------------------

class TestConstants:
    def test_const_collection(self):
        """5 + x + 3 → 8 + x"""
        t = BinOp("add", BinOp("add", Const(5.0), Var("x")), Const(3.0))
        result = simplify_polynomial(t)
        x = np.linspace(-1, 1, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), 8.0 + x)

    def test_const_only(self):
        """3 + 5 + 2 → 10"""
        t = BinOp("add", BinOp("add", Const(3.0), Const(5.0)), Const(2.0))
        result = simplify_polynomial(t)
        assert isinstance(result, Const)
        assert result.value == pytest.approx(10.0)


# ----------------------------------------------------------------------
# 5. Negative coefficients
# ----------------------------------------------------------------------

class TestNegativeCoefficients:
    def test_negative_coef_via_neg_unop(self):
        """neg(2*x) + 3*x → x"""
        t = BinOp("add",
                  UnOp("neg", BinOp("mul", Const(2.0), Var("x"))),
                  BinOp("mul", Const(3.0), Var("x")))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), x)

    def test_negative_const_coef(self):
        """-3*x + 5*x → 2*x  (Const(-3.0))"""
        t = BinOp("add",
                  BinOp("mul", Const(-3.0), Var("x")),
                  BinOp("mul", Const(5.0), Var("x")))
        result = simplify_polynomial(t)
        x = np.linspace(-2, 2, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), 2.0 * x)

    def test_coef_minus_one_emits_neg(self):
        """5*x - 6*x should produce -x via UnOp(neg)"""
        t = BinOp("sub",
                  BinOp("mul", Const(5.0), Var("x")),
                  BinOp("mul", Const(6.0), Var("x")))
        result = simplify_polynomial(t)
        x = np.linspace(-1, 1, 20)
        np.testing.assert_allclose(eval_tree(result, {"x": x}), -x)


# ----------------------------------------------------------------------
# 6. Opaque terms
# ----------------------------------------------------------------------

class TestOpaqueTerms:
    def test_x_plus_sinx_plus_x(self):
        """x + sin(x) + x → 2*x + sin(x). sin is opaque (non-poly)."""
        t = BinOp("add",
                  BinOp("add", Var("x"), UnOp("sin", Var("x"))),
                  Var("x"))
        result = simplify_polynomial(t)
        x = np.linspace(-1, 1, 30)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}),
            2.0 * x + np.sin(x),
            rtol=1e-10,
        )
        # `x + x` (cx 3) becomes `2*x` (cx 3) — same complexity but
        # canonical structure. Monotone (≤) is the contract.
        assert complexity(result) <= complexity(t)

    def test_div_kept_opaque(self):
        """x + (1/x) + x — division can't fit the monomial pattern,
        kept opaque."""
        t = BinOp("add",
                  BinOp("add",
                        Var("x"),
                        BinOp("div", Const(1.0), Var("x"))),
                  Var("x"))
        result = simplify_polynomial(t)
        x = np.linspace(0.1, 2, 20)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}),
            2.0 * x + 1.0 / x,
            rtol=1e-10,
        )

    def test_negative_opaque_preserved(self):
        """x - sin(x) — opaque sin(x) subtracted, must keep sign."""
        t = BinOp("sub", Var("x"), UnOp("sin", Var("x")))
        result = simplify_polynomial(t)
        x = np.linspace(-1, 1, 20)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}),
            x - np.sin(x),
            rtol=1e-9,
        )


# ----------------------------------------------------------------------
# 7. Recursion (canonicalise INSIDE non-additive nodes)
# ----------------------------------------------------------------------

class TestRecursion:
    def test_canonicalise_inside_unop(self):
        """sin(x + x) → sin(2*x)"""
        t = UnOp("sin", BinOp("add", Var("x"), Var("x")))
        result = simplify_polynomial(t)
        assert isinstance(result, UnOp) and result.op == "sin"
        x = np.linspace(-1, 1, 20)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}),
            np.sin(2 * x),
            rtol=1e-9,
        )

    def test_canonicalise_inside_mul(self):
        """(2*x + 3*x) * y → 5*x*y (after polynomial-fold inside)"""
        inner = BinOp("add",
                      BinOp("mul", Const(2.0), Var("x")),
                      BinOp("mul", Const(3.0), Var("x")))
        t = BinOp("mul", inner, Var("y"))
        result = simplify_polynomial(t)
        x = np.linspace(0.1, 2, 20)
        y = np.linspace(0.1, 2, 20)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x, "y": y}),
            5.0 * x * y,
            rtol=1e-10,
        )

    def test_does_not_distribute_mul_over_add(self):
        """(x + 1) * y should NOT be expanded to x*y + y. We're a
        canonicaliser, not an expander."""
        t = BinOp("mul",
                  BinOp("add", Var("x"), Const(1.0)),
                  Var("y"))
        result = simplify_polynomial(t)
        # Output should still contain the product structure (top-level
        # is a mul, not an add).
        assert isinstance(result, BinOp) and result.op == "mul"


# ----------------------------------------------------------------------
# 8. Idempotency
# ----------------------------------------------------------------------

class TestIdempotency:
    @pytest.fixture
    def trees(self):
        return [
            # Basic polynomial
            BinOp("add",
                  BinOp("mul", Const(2.0), Var("x")),
                  BinOp("mul", Const(3.0), Var("x"))),
            # With multivariate
            BinOp("add",
                  BinOp("mul", Var("x"), Var("y")),
                  BinOp("mul", Const(2.0),
                        BinOp("mul", Var("x"), Var("y")))),
            # With opaque
            BinOp("add",
                  BinOp("add", Var("x"), UnOp("sin", Var("x"))),
                  Var("x")),
            # Negative coefs
            BinOp("sub",
                  BinOp("mul", Const(5.0), Var("x")),
                  BinOp("mul", Const(6.0), Var("x"))),
            # Nested mul, div as opaque
            BinOp("add",
                  BinOp("div", Const(1.0), Var("x")),
                  BinOp("div", Const(2.0), Var("x"))),
            # Constants only
            BinOp("add", BinOp("add", Const(3.0), Const(5.0)), Const(2.0)),
        ]

    def test_double_application_equals_single(self, trees):
        for t in trees:
            once = simplify_polynomial(t)
            twice = simplify_polynomial(once)
            assert str(once) == str(twice), (
                f"not idempotent on {t}: once={once} twice={twice}"
            )


# ----------------------------------------------------------------------
# 9. Monotone-in-complexity
# ----------------------------------------------------------------------

class TestMonotoneCx:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_polynomial_cx_non_increasing(self, seed):
        """Simplification must not increase complexity."""
        rng = np.random.default_rng(seed)
        # Build a random polynomial sum with several repeated monomials.
        terms = []
        for _ in range(8):
            coef = float(rng.uniform(-3, 3))
            deg = int(rng.integers(1, 4))
            mono = Var("x")
            for _ in range(deg - 1):
                mono = BinOp("mul", mono, Var("x"))
            terms.append(BinOp("mul", Const(coef), mono))
        # Left-fold into an add chain.
        t = terms[0]
        for term in terms[1:]:
            t = BinOp("add", t, term)
        result = simplify_polynomial(t)
        assert complexity(result) <= complexity(t)


# ----------------------------------------------------------------------
# 10. Semantics preservation on random samples
# ----------------------------------------------------------------------

class TestSemanticsPreservation:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_semantics_preserved_random(self, seed):
        rng = np.random.default_rng(seed)
        # Build a random polynomial with both monomial and opaque
        # subterms.
        terms = []
        for _ in range(5):
            coef = float(rng.uniform(-2, 2))
            mono = Var("x")
            deg = int(rng.integers(1, 3))
            for _ in range(deg - 1):
                mono = BinOp("mul", mono, Var("x"))
            terms.append(BinOp("mul", Const(coef), mono))
        # Add an opaque term
        terms.append(UnOp("sin", Var("x")))
        # Build add chain
        t = terms[0]
        for term in terms[1:]:
            t = BinOp("add", t, term)
        # Evaluate before and after
        x = np.linspace(-1.5, 1.5, 50)
        before = eval_tree(t, {"x": x})
        after = eval_tree(simplify_polynomial(t), {"x": x})
        np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-12)


# ----------------------------------------------------------------------
# 11. Realistic polish-output case
# ----------------------------------------------------------------------

class TestPolishOutputCase:
    def test_existing_xx_plus_polished_xx(self):
        """The §2.3 P3 motivating case: GP found existing `2*x*x`,
        polish appends `0.5*x + 0.3*x*x + ...`. After canonicalisation,
        the two x*x terms collapse to one."""
        existing = BinOp("mul", Const(2.0),
                         BinOp("mul", Var("x"), Var("x")))
        polish_addition = BinOp("add",
                                BinOp("mul", Const(0.5), Var("x")),
                                BinOp("mul", Const(0.3),
                                      BinOp("mul", Var("x"), Var("x"))))
        t = BinOp("add", existing, polish_addition)
        result = simplify_polynomial(t)
        # Should give: 0.5*x + 2.3*x*x
        x = np.linspace(-1, 1, 30)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x}),
            0.5 * x + 2.3 * x ** 2,
            rtol=1e-10,
        )
        # cx should drop substantially.
        assert complexity(result) <= complexity(t) - 4


# ----------------------------------------------------------------------
# 12. simplify_full pipeline
# ----------------------------------------------------------------------

class TestSimplifyFull:
    def test_full_pipeline_on_polish_case(self):
        """simplify_full = polynomial(canonical(.)) closes the loop."""
        t = BinOp("add",
                  BinOp("add",
                        BinOp("mul", Const(2.0),
                              BinOp("mul", Var("x"), Var("x"))),
                        BinOp("mul", Const(0.0), Var("y"))),  # × 0 fold
                  BinOp("mul", Const(0.3),
                        BinOp("mul", Var("x"), Var("x"))))
        result = simplify_full(t)
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        np.testing.assert_allclose(
            eval_tree(result, {"x": x, "y": y}),
            2.3 * x ** 2,
            rtol=1e-10,
        )

    def test_full_is_idempotent(self):
        t = BinOp("add",
                  BinOp("mul", Const(2.0), Var("x")),
                  BinOp("mul", Const(3.0), Var("x")))
        once = simplify_full(t)
        twice = simplify_full(once)
        assert str(once) == str(twice)
