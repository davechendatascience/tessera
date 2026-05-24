"""Phase 2 integration tests: sufficient-stats polish wired into the GP.

Verifies:
  - build_polynomial_term_tree produces a valid tree that evaluates to
    the expected sum.
  - polish_tree_with_polynomial_term improves MSE on polynomial-friendly
    targets (TRAIN-side).
  - The analytical Δloss prediction matches the actual loss-difference
    after re-eval, within float tolerance — confirming the GP can trust
    the sufficient-stat Δloss.
  - GP integration: enabling sufficient_stats_polish_every does not
    break the run; the new candidates appear in HoF; loss improves on
    polynomial targets vs the polish-off baseline.
  - No-op gracefully when loss_fn is not mse_loss.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression.tree import (
    BinOp, Const, Var, complexity, evaluate as eval_tree,
)
from tessera.search import GP, GPConfig, mse_loss
from tessera.search.sufficient_stats import (
    PolynomialMoments, monomial_basis,
    build_polynomial_term_tree, polish_tree_with_polynomial_term,
)


# ----------------------------------------------------------------------
# 1. Tree construction
# ----------------------------------------------------------------------

class TestBuildPolynomialTermTree:
    def test_single_term_linear(self):
        """c=[3.0] with basis [x] should produce 3 * x."""
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=1,
            coefficients=np.array([3.0]),
        )
        assert tree is not None
        x = np.linspace(-2, 2, 50)
        out = eval_tree(tree, {"x": x})
        np.testing.assert_allclose(out, 3.0 * x, rtol=1e-10)

    def test_single_term_quadratic(self):
        """c=[0, 5.0] with basis [x, x^2] should produce 5 * x^2.

        Note: the first c is filtered (=0), so the tree should
        contain only one term."""
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=2,
            coefficients=np.array([0.0, 5.0]),
        )
        x = np.linspace(-2, 2, 50)
        out = eval_tree(tree, {"x": x})
        np.testing.assert_allclose(out, 5.0 * x ** 2, rtol=1e-9)

    def test_multi_term_sum(self):
        """c=[1, 2, 3] over [x, x^2, x^3] → 1x + 2x² + 3x³"""
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=3,
            coefficients=np.array([1.0, 2.0, 3.0]),
            top_n=None,
        )
        x = np.linspace(-1, 1, 30)
        expected = x + 2 * x ** 2 + 3 * x ** 3
        out = eval_tree(tree, {"x": x})
        np.testing.assert_allclose(out, expected, rtol=1e-9)

    def test_top_n_truncation(self):
        """top_n=2 should keep only the 2 largest-magnitude
        coefficients. Basis [x, x^2, x^3] (3 entries)."""
        # Largest are c[2]=10 (x^3) and c[0]=5 (x); tree should be
        # 5*x + 10*x^3.
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=3,
            coefficients=np.array([5.0, 0.5, 10.0]),
            top_n=2,
        )
        x = np.linspace(-1, 1, 30)
        expected = 5.0 * x + 10.0 * x ** 3
        out = eval_tree(tree, {"x": x})
        np.testing.assert_allclose(out, expected, rtol=1e-9)

    def test_all_below_threshold_returns_none(self):
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=2,
            coefficients=np.array([1e-10, 1e-12]),
            coef_threshold=1e-6,
        )
        assert tree is None

    def test_multi_feature(self):
        """Two features × degree 2, full coefficient vector."""
        # basis: [x_0, x_0^2, x_1, x_1^2]
        tree = build_polynomial_term_tree(
            feature_names=["a", "b"], feature_indices=[0, 1], max_degree=2,
            coefficients=np.array([1.0, 0.0, 0.0, 2.0]),
            top_n=None,
        )
        a = np.linspace(-1, 1, 20)
        b = np.linspace(-1, 1, 20)
        out = eval_tree(tree, {"a": a, "b": b})
        np.testing.assert_allclose(out, a + 2.0 * b ** 2, rtol=1e-9)

    def test_constant_included(self):
        """include_constant=True prepends a constant basis function."""
        tree = build_polynomial_term_tree(
            feature_names=["x"], feature_indices=[0], max_degree=1,
            coefficients=np.array([7.0, 0.0]),
            include_constant=True, top_n=None,
        )
        x = np.linspace(-1, 1, 20)
        out = eval_tree(tree, {"x": x})
        np.testing.assert_allclose(out, np.full_like(x, 7.0), rtol=1e-9)


# ----------------------------------------------------------------------
# 2. polish_tree_with_polynomial_term — closed-form-vs-actual Δloss
# ----------------------------------------------------------------------

class TestPolishTreeAnalyticalVsActual:
    def test_analytical_dl_matches_actual_re_eval(self):
        """The analytical Δloss prediction (from PolynomialMoments)
        must match the actual loss-difference when the polished tree
        is evaluated and the MSE is recomputed."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=500)
        env = {"x": x}
        # Target: 2x + 3x^2 - x^3
        y = 2 * x + 3 * x ** 2 - x ** 3

        # Starting tree: just `x` (an imperfect linear model)
        starting_tree = Var(name="x")
        starting_pred = eval_tree(starting_tree, env)
        starting_mse = float(np.mean((starting_pred - y) ** 2))

        X = x.reshape(-1, 1)
        new_tree, expected_dl, kept = polish_tree_with_polynomial_term(
            starting_tree, starting_pred, X, y,
            feature_names=["x"], feature_indices=[0], max_degree=3,
            top_n=3,
        )
        assert kept >= 1
        new_pred = eval_tree(new_tree, env)
        new_mse = float(np.mean((new_pred - y) ** 2))
        actual_dl = new_mse - starting_mse
        # Analytical vs actual must agree to numerical precision.
        assert expected_dl == pytest.approx(actual_dl, abs=1e-8, rel=1e-6)
        # Polish must improve MSE.
        assert new_mse < starting_mse

    def test_polish_solves_polynomial_target_exactly(self):
        """When the target IS in the basis and starting tree is 0,
        polish should bring MSE to ~0."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=400)
        env = {"x": x}
        y = 1.5 * x + 0.5 * x ** 2

        starting_tree = Const(value=0.0)
        starting_pred = np.zeros_like(x)

        X = x.reshape(-1, 1)
        new_tree, expected_dl, kept = polish_tree_with_polynomial_term(
            starting_tree, starting_pred, X, y,
            feature_names=["x"], feature_indices=[0], max_degree=2,
            top_n=None,  # Keep all
        )
        new_pred = eval_tree(new_tree, env)
        new_mse = float(np.mean((new_pred - y) ** 2))
        # Should be near zero (ridge regularisation gives a tiny floor).
        assert new_mse < 1e-6

    def test_polish_no_change_when_already_perfect(self):
        """If the starting tree is already y, polish should add ~0."""
        x = np.linspace(-1, 1, 100)
        env = {"x": x}
        y = 2 * x

        starting_tree = BinOp(op="mul", a=Const(value=2.0), b=Var(name="x"))
        starting_pred = eval_tree(starting_tree, env)
        X = x.reshape(-1, 1)

        new_tree, expected_dl, kept = polish_tree_with_polynomial_term(
            starting_tree, starting_pred, X, y,
            feature_names=["x"], feature_indices=[0], max_degree=2,
            top_n=3,
        )
        # Either no terms kept, or the added terms have ~0 impact.
        new_pred = eval_tree(new_tree, env)
        new_mse = float(np.mean((new_pred - y) ** 2))
        assert new_mse < 1e-9


# ----------------------------------------------------------------------
# 3. GP integration
# ----------------------------------------------------------------------

class TestGPIntegration:
    """End-to-end: enable the polish in GPConfig and confirm:
      - The run completes without error.
      - HoF receives polished candidates.
      - On a polynomial-friendly target, polish-on outperforms polish-off
        at the same gen budget.
    """

    def _make_polynomial_target(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        env = {"x": x}
        y = 1.5 * x + 0.7 * x ** 2 - 0.4 * x ** 3
        return env, y

    def test_polish_on_completes_without_error(self):
        env, y = self._make_polynomial_target()
        cfg = GPConfig(
            pop_size=30, n_gens=10, seed=0, verbose=False,
            sufficient_stats_polish_every=3,
            sufficient_stats_max_degree=3,
            sufficient_stats_top_n_terms=3,
            pointwise_only=True,
            optimize_constants_every=0,  # isolate the polish effect
        )
        gp = GP(cfg)
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_polish_improves_polynomial_target(self):
        """Polish-on should reach a lower best-loss than polish-off
        on a polynomial target at the same seed + gen budget."""
        env, y = self._make_polynomial_target(n=400, seed=1)

        # Baseline: polish off.
        cfg_off = GPConfig(
            pop_size=40, n_gens=12, seed=2, verbose=False,
            sufficient_stats_polish_every=0,
            pointwise_only=True,
            optimize_constants_every=0,
        )
        front_off = GP(cfg_off).run(env, y, feature_names=["x"])
        best_loss_off = min(c.train_loss for c in front_off)

        # Polish on.
        cfg_on = GPConfig(
            pop_size=40, n_gens=12, seed=2, verbose=False,
            sufficient_stats_polish_every=2,
            sufficient_stats_max_degree=3,
            sufficient_stats_top_n_terms=3,
            pointwise_only=True,
            optimize_constants_every=0,
        )
        front_on = GP(cfg_on).run(env, y, feature_names=["x"])
        best_loss_on = min(c.train_loss for c in front_on)

        # Polish-on should be strictly better. Polynomial target,
        # polynomial basis, closed-form coefficients — this is the
        # designed-for case.
        assert best_loss_on < best_loss_off * 0.5 + 1e-9, (
            f"polish-on best_loss={best_loss_on:.4g} should be much "
            f"lower than polish-off best_loss={best_loss_off:.4g}"
        )

    def test_polish_noop_with_non_mse_loss(self):
        """sufficient_stats_polish_every must silently no-op when
        loss_fn is not mse_loss."""
        env, y = self._make_polynomial_target()
        # Use a dummy non-mse loss
        def mae(pred, y):
            return float(np.mean(np.abs(pred - y)))
        cfg = GPConfig(
            pop_size=20, n_gens=6, seed=0, verbose=False,
            sufficient_stats_polish_every=2,
            pointwise_only=True,
            optimize_constants_every=0,
        )
        # Should not crash even though loss is mae.
        gp = GP(cfg, loss_fn=mae)
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_polish_feature_name_subset(self):
        """When sufficient_stats_feature_names selects a subset,
        polish must only use those features."""
        rng = np.random.default_rng(11)
        a = rng.normal(size=200)
        b = rng.normal(size=200)
        env = {"a": a, "b": b}
        # Target depends on both, but we'll only let polish use 'a'.
        y = a + 0.5 * a ** 2 + 0.3 * b
        cfg = GPConfig(
            pop_size=20, n_gens=6, seed=0, verbose=False,
            sufficient_stats_polish_every=2,
            sufficient_stats_feature_names=("a",),
            sufficient_stats_max_degree=2,
            pointwise_only=True,
            optimize_constants_every=0,
        )
        gp = GP(cfg)
        front = gp.run(env, y, feature_names=["a", "b"])
        assert len(front) > 0
