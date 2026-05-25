"""Tests for tessera.expression.simplify.cas_fallback — CAS-based simplification."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, complexity, evaluate,
)
from tessera.expression.simplify.cas_fallback import (
    cas_simplify, simplify_front_with_cas,
    is_worth_cas_pass, get_backend,
    clear_cache, cache_size,
)


# Skip everything if no CAS backend available
pytestmark = pytest.mark.skipif(
    get_backend() is None,
    reason="No CAS backend (sympy or symengine) installed",
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear CAS cache between tests for clean isolation."""
    clear_cache()


# ----------------------------------------------------------------------
# 1. Predicates
# ----------------------------------------------------------------------

class TestIsWorthCASPass:
    def test_polynomial_only_no_cas(self):
        """Pure polynomial expressions don't benefit from CAS."""
        tree = BinOp("add", BinOp("mul", Const(2.0), Var("x")), Var("y"))
        assert not is_worth_cas_pass(tree)

    def test_trig_triggers_cas(self):
        tree = BinOp("add",
                     BinOp("mul", UnOp("sin", Var("x")), UnOp("sin", Var("x"))),
                     BinOp("mul", UnOp("cos", Var("x")), UnOp("cos", Var("x"))))
        assert is_worth_cas_pass(tree)

    def test_division_triggers_cas(self):
        tree = BinOp("div", Var("x"), Var("y"))
        assert is_worth_cas_pass(tree)

    def test_log_exp_triggers_cas(self):
        tree = UnOp("log", UnOp("exp", Var("x")))
        assert is_worth_cas_pass(tree)

    def test_comparison_op_disables_cas(self):
        """Trees containing gt/lt/ge/le should be skipped."""
        tree = BinOp("add",
                     BinOp("gt", Var("x"), Const(0.0)),
                     UnOp("sin", Var("x")))
        assert not is_worth_cas_pass(tree)

    def test_reduce_disables_cas(self):
        tree = BinOp("add", UnOp("reduce_max", Var("x")), Const(1.0))
        assert not is_worth_cas_pass(tree)


# ----------------------------------------------------------------------
# 2. Trig identity simplification
# ----------------------------------------------------------------------

class TestTrigIdentities:
    def test_sin_squared_plus_cos_squared(self):
        """sin(x)² + cos(x)² → 1"""
        sin_x = UnOp("sin", Var("x"))
        cos_x = UnOp("cos", Var("x"))
        tree = BinOp("add",
                     BinOp("mul", sin_x, sin_x),
                     BinOp("mul", cos_x, cos_x))
        simplified = cas_simplify(tree, verify_samples=10)
        # Should simplify to Const(1)
        x = np.linspace(-2, 2, 30)
        result = evaluate(simplified, {"x": x})
        np.testing.assert_allclose(result, np.ones_like(x), atol=1e-9)
        # And the simplified tree should be much smaller
        assert complexity(simplified) < complexity(tree)


# ----------------------------------------------------------------------
# 3. Polynomial / division cancellation
# ----------------------------------------------------------------------

class TestRationalCancellation:
    def test_division_collapse(self):
        """(2*x) / x → 2"""
        tree = BinOp("div",
                     BinOp("mul", Const(2.0), Var("x")),
                     Var("x"))
        simplified = cas_simplify(tree, verify_samples=10)
        # Should simplify to Const(2)
        x = np.linspace(1.0, 3.0, 20)  # avoid x=0
        result = evaluate(simplified, {"x": x})
        np.testing.assert_allclose(result, 2.0 * np.ones_like(x), atol=1e-6)


# ----------------------------------------------------------------------
# 4. log/exp identities
# ----------------------------------------------------------------------

class TestLogExpIdentities:
    def test_log_of_exp(self):
        """log(exp(x)) → x (approximate, given protected ops)"""
        tree = UnOp("log", UnOp("exp", Var("x")))
        simplified = cas_simplify(tree, verify_samples=10)
        # On safe range (positive x), should evaluate close to x
        x = np.linspace(0.5, 3.0, 20)
        orig_result = evaluate(tree, {"x": x})
        simp_result = evaluate(simplified, {"x": x})
        np.testing.assert_allclose(simp_result, orig_result, atol=1e-4)


# ----------------------------------------------------------------------
# 5. Returns original when simplification would be wrong / unhelpful
# ----------------------------------------------------------------------

class TestSafetyGuards:
    def test_no_op_for_polynomial(self):
        """For polynomial trees, the predicate says skip; original returned."""
        tree = BinOp("add", BinOp("mul", Const(2.0), Var("x")), Const(3.0))
        result = cas_simplify(tree)
        assert result is tree

    def test_does_not_increase_complexity(self):
        """If CAS expands to a bigger tree, original returned."""
        # x² + 2x + 1 = (x+1)² ; sympy expand would go the other way
        # Use a tree that would expand to something larger
        tree = UnOp("sin", BinOp("add", Var("x"), Const(1.0)))
        # The predicate says cas-worth (has sin); but sympy expand can't
        # simplify further. complexity should not increase.
        result = cas_simplify(tree)
        assert complexity(result) <= complexity(tree)

    def test_numerical_verification_blocks_divergence(self):
        """The verifier should reject simplifications that diverge from
        the original numerically."""
        # Just check: any simplification we accept must be numerically
        # equivalent. Run several trees and verify the property.
        trees = [
            BinOp("add",
                  BinOp("mul", UnOp("sin", Var("x")), UnOp("sin", Var("x"))),
                  BinOp("mul", UnOp("cos", Var("x")), UnOp("cos", Var("x")))),
            BinOp("div",
                  BinOp("mul", Const(3.0), Var("x")),
                  Var("x")),
            UnOp("log", UnOp("exp", Var("x"))),
        ]
        for tree in trees:
            simplified = cas_simplify(tree, verify_samples=20)
            # If simplified differs, must be numerically equivalent to original
            if simplified is not tree:
                # Eval both at random valid points
                rng = np.random.default_rng(123)
                x = rng.uniform(0.5, 3.0, 30)
                orig = evaluate(tree, {"x": x})
                simp = evaluate(simplified, {"x": x})
                assert np.allclose(orig, simp, atol=1e-4, rtol=1e-3)


# ----------------------------------------------------------------------
# 6. Caching
# ----------------------------------------------------------------------

class TestCaching:
    def test_cache_hit_returns_same_object(self):
        """Two calls on the same tree return identical results."""
        clear_cache()
        sin_x = UnOp("sin", Var("x"))
        cos_x = UnOp("cos", Var("x"))
        tree = BinOp("add",
                     BinOp("mul", sin_x, sin_x),
                     BinOp("mul", cos_x, cos_x))
        first = cas_simplify(tree, verify_samples=10)
        # Cache should now contain this entry
        assert cache_size() == 1
        # Second call
        second = cas_simplify(tree, verify_samples=10)
        # Same object identity (cache hit)
        assert first is second

    def test_clear_cache(self):
        sin_x = UnOp("sin", Var("x"))
        cos_x = UnOp("cos", Var("x"))
        tree = BinOp("add",
                     BinOp("mul", sin_x, sin_x),
                     BinOp("mul", cos_x, cos_x))
        cas_simplify(tree)
        assert cache_size() >= 1
        clear_cache()
        assert cache_size() == 0


# ----------------------------------------------------------------------
# 7. simplify_front_with_cas
# ----------------------------------------------------------------------

class TestSimplifyFront:
    def test_front_simplification(self):
        """Simplify a list of Candidate-like objects with .tree."""
        from tessera.search.base import Candidate
        sin_x = UnOp("sin", Var("x"))
        cos_x = UnOp("cos", Var("x"))
        identity_tree = BinOp("add",
                              BinOp("mul", sin_x, sin_x),
                              BinOp("mul", cos_x, cos_x))
        # Polynomial tree — should not change
        poly_tree = BinOp("add",
                          BinOp("mul", Const(2.0), Var("x")),
                          Const(3.0))
        front = [
            Candidate(tree=identity_tree, train_loss=0.5, complexity=complexity(identity_tree),
                      fitness=0.5, born_gen=0),
            Candidate(tree=poly_tree, train_loss=0.3, complexity=complexity(poly_tree),
                      fitness=0.3, born_gen=0),
        ]
        new_front = simplify_front_with_cas(front)
        assert len(new_front) == 2
        # Identity tree should have been simplified to lower cx
        assert new_front[0].complexity < front[0].complexity
        # Polynomial tree unchanged
        assert new_front[1].tree is poly_tree

    def test_no_op_without_backend(self):
        """When backend is None, returns front unchanged.
        (This test won't fire if backend IS available — skipped at top.)"""
        # Just check the basic shape — front length preserved
        from tessera.search.base import Candidate
        front = [
            Candidate(tree=Var("x"), train_loss=0.5, complexity=1,
                      fitness=0.5, born_gen=0)
        ]
        new_front = simplify_front_with_cas(front)
        assert len(new_front) == len(front)


# ----------------------------------------------------------------------
# 8. Backend detection
# ----------------------------------------------------------------------

class TestBackend:
    def test_backend_string(self):
        b = get_backend()
        assert b in ("sympy", "symengine")
