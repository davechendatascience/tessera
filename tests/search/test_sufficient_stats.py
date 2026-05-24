"""Tests for tessera.search.sufficient_stats — analytical Δloss machinery.

Covers:
  - Correctness of PolynomialMoments.delta_loss vs naive recomputation.
  - Closed-form optimal coefficients (and their Δloss-minimising property).
  - O(1)-in-N scaling — the whole point of Regime B.
  - monomial_basis helper.
  - Shape / input validation.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tessera.search.sufficient_stats import (
    PolynomialMoments,
    monomial_basis,
)


def _naive_mse(predictions: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((predictions - y) ** 2))


def _naive_delta_loss(
    X: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    basis,
    c: np.ndarray,
) -> float:
    """Reference implementation: actually compute δ(x) and remix the
    predictions, then take the loss diff. O(N · K) per query."""
    N = X.shape[0]
    delta = np.zeros(N, dtype=np.float64)
    for k, phi in enumerate(basis):
        delta += c[k] * phi(X)
    new_predictions = predictions + delta
    return _naive_mse(new_predictions, y) - _naive_mse(predictions, y)


# ----------------------------------------------------------------------
# 1. Correctness: delta_loss matches naive on synthetic data
# ----------------------------------------------------------------------

class TestDeltaLossCorrectness:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.N = 200
        self.X = rng.normal(size=(self.N, 3))
        self.y = (
            1.5 * self.X[:, 0] - 0.7 * self.X[:, 1] ** 2 + 0.1 * self.X[:, 2]
        )
        # An imperfect "current model" — just zeros, so residual = -y.
        self.predictions = np.zeros(self.N)
        self.basis = monomial_basis([0, 1, 2], max_degree=3)
        self.moments = PolynomialMoments.from_basis(
            self.X, self.y, self.predictions, self.basis
        )

    def test_matches_naive_zero_coefficients(self):
        c = np.zeros(len(self.basis))
        assert self.moments.delta_loss(c) == pytest.approx(0.0, abs=1e-12)

    def test_matches_naive_random_coefficients(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            c = rng.normal(scale=0.5, size=len(self.basis))
            sufficient = self.moments.delta_loss(c)
            naive = _naive_delta_loss(
                self.X, self.y, self.predictions, self.basis, c
            )
            assert sufficient == pytest.approx(naive, rel=1e-9, abs=1e-12)

    def test_matches_naive_nonzero_predictions(self):
        """Same correctness test but with a non-trivial current model."""
        rng = np.random.default_rng(11)
        predictions = rng.normal(scale=0.3, size=self.N)
        moments = PolynomialMoments.from_basis(
            self.X, self.y, predictions, self.basis
        )
        for _ in range(5):
            c = rng.normal(scale=0.5, size=len(self.basis))
            sufficient = moments.delta_loss(c)
            naive = _naive_delta_loss(self.X, self.y, predictions, self.basis, c)
            assert sufficient == pytest.approx(naive, rel=1e-9, abs=1e-12)


# ----------------------------------------------------------------------
# 2. Closed-form optimal coefficients
# ----------------------------------------------------------------------

class TestOptimalCoefficients:
    def test_optimal_delta_loss_is_non_positive(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(500, 2))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
        predictions = rng.normal(scale=0.1, size=500)
        basis = monomial_basis([0, 1], max_degree=2)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        opt = moments.optimal_delta_loss()
        assert opt <= 1e-12  # always ≤ 0 (Cauchy-Schwarz)

    def test_optimal_solves_polynomial_target(self):
        """If y IS a polynomial in the basis and predictions = 0, the
        optimal c should recover y exactly (up to ridge)."""
        rng = np.random.default_rng(5)
        N = 1000
        X = rng.normal(size=(N, 1))
        # y = 1*x + 2*x^2 + 0*x^3
        y = X[:, 0] + 2.0 * X[:, 0] ** 2
        predictions = np.zeros(N)
        basis = monomial_basis([0], max_degree=3)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        c_opt = moments.optimal_coefficients(ridge=1e-12)
        # Optimal c brings predictions to y, so residual after mutation ≈ 0;
        # Δloss ≈ -MSE(predictions, y) = -mean(y²).
        opt_delta = moments.optimal_delta_loss(ridge=1e-12)
        expected = -float(np.mean(y ** 2))
        assert opt_delta == pytest.approx(expected, rel=1e-6)
        # Coefficient vector should be [1, 2, 0] up to ridge bias.
        assert c_opt[0] == pytest.approx(1.0, abs=1e-3)
        assert c_opt[1] == pytest.approx(2.0, abs=1e-3)
        assert c_opt[2] == pytest.approx(0.0, abs=1e-3)

    def test_optimal_is_minimum_vs_random_perturbations(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(300, 2))
        y = 0.5 * X[:, 0] ** 2 - X[:, 1]
        predictions = rng.normal(scale=0.05, size=300)
        basis = monomial_basis([0, 1], max_degree=2)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        c_opt = moments.optimal_coefficients()
        opt_dl = moments.delta_loss(c_opt)
        for _ in range(20):
            c_perturbed = c_opt + rng.normal(scale=0.1, size=c_opt.shape)
            assert moments.delta_loss(c_perturbed) >= opt_dl - 1e-10


# ----------------------------------------------------------------------
# 3. Performance: O(1)-in-N scaling
# ----------------------------------------------------------------------

class TestScaling:
    @pytest.mark.parametrize("N", [1_000, 10_000, 100_000])
    def test_delta_loss_independent_of_N(self, N):
        """The delta_loss query itself must NOT scale with N — that's
        the whole point of Regime B. Only construction depends on N."""
        rng = np.random.default_rng(N)
        X = rng.normal(size=(N, 2))
        y = X[:, 0] + 0.5 * X[:, 1] ** 2
        predictions = np.zeros(N)
        basis = monomial_basis([0, 1], max_degree=3)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        c = rng.normal(size=len(basis))

        # Warm-up
        moments.delta_loss(c)

        # Time many queries
        t0 = time.perf_counter()
        for _ in range(1000):
            moments.delta_loss(c)
        elapsed = time.perf_counter() - t0
        # Should be MICROSECONDS per query regardless of N.
        per_query_us = elapsed * 1e6 / 1000
        # Generous bound: even at N=100k, K=6 (3 features × 2 degrees), per-
        # query should be well under 50us. Naive would be ~Nμs.
        assert per_query_us < 100, (
            f"N={N}: per-query {per_query_us:.1f}us — should be O(K²) "
            f"independent of N"
        )

    def test_speedup_vs_naive_at_large_N(self):
        """Acceptance criterion (a) from roadmap §2.3: ≥10× speedup vs
        naive Δloss at N=10000."""
        N = 10_000
        rng = np.random.default_rng(0)
        X = rng.normal(size=(N, 3))
        y = X[:, 0] + X[:, 1] ** 2 - X[:, 2]
        predictions = rng.normal(scale=0.1, size=N)
        basis = monomial_basis([0, 1, 2], max_degree=2)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        c = rng.normal(size=len(basis))

        # Warm-up both paths
        moments.delta_loss(c)
        _naive_delta_loss(X, y, predictions, basis, c)

        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            moments.delta_loss(c)
        t_sufficient = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _naive_delta_loss(X, y, predictions, basis, c)
        t_naive = time.perf_counter() - t0

        speedup = t_naive / t_sufficient
        assert speedup >= 10.0, (
            f"speedup at N={N} was {speedup:.1f}× — expected ≥ 10× per "
            f"roadmap §2.3 acceptance criterion"
        )


# ----------------------------------------------------------------------
# 4. monomial_basis helper
# ----------------------------------------------------------------------

class TestMonomialBasis:
    def test_basis_size(self):
        # 2 features × 3 degrees = 6 basis functions
        basis = monomial_basis([0, 1], max_degree=3)
        assert len(basis) == 6

    def test_basis_size_with_constant(self):
        basis = monomial_basis([0, 1], max_degree=3, include_constant=True)
        assert len(basis) == 7

    def test_basis_outputs(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        basis = monomial_basis([0, 2], max_degree=2, include_constant=True)
        # Expected: constant, X[:,0], X[:,0]^2, X[:,2], X[:,2]^2
        outputs = [phi(X) for phi in basis]
        np.testing.assert_array_equal(outputs[0], np.ones(50))
        np.testing.assert_array_equal(outputs[1], X[:, 0])
        np.testing.assert_array_equal(outputs[2], X[:, 0] ** 2)
        np.testing.assert_array_equal(outputs[3], X[:, 2])
        np.testing.assert_array_equal(outputs[4], X[:, 2] ** 2)

    def test_closure_capture(self):
        """Regression: the lambda-in-loop capture pitfall. Each basis
        function must remember its OWN (d, k), not the loop's last."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        basis = monomial_basis([0, 1, 2, 3], max_degree=2)
        # 4 features × 2 degrees = 8 functions, each unique
        outputs = np.stack([phi(X) for phi in basis])
        # No two rows should be identical (different (d, k) pairs)
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                assert not np.allclose(outputs[i], outputs[j]), (
                    f"basis {i} and {j} returned identical values — "
                    f"likely a loop-closure capture bug"
                )

    def test_max_degree_validation(self):
        with pytest.raises(ValueError, match="max_degree"):
            monomial_basis([0], max_degree=0)


# ----------------------------------------------------------------------
# 5. Input validation
# ----------------------------------------------------------------------

class TestInputValidation:
    def test_empty_basis_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        predictions = np.zeros(10)
        with pytest.raises(ValueError, match="basis must be non-empty"):
            PolynomialMoments.from_basis(X, y, predictions, [])

    def test_shape_mismatch_y_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(5)  # wrong
        predictions = np.zeros(10)
        basis = monomial_basis([0], max_degree=1)
        with pytest.raises(ValueError, match="size mismatch"):
            PolynomialMoments.from_basis(X, y, predictions, basis)

    def test_shape_mismatch_predictions_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        predictions = np.zeros(5)  # wrong
        basis = monomial_basis([0], max_degree=1)
        with pytest.raises(ValueError, match="size mismatch"):
            PolynomialMoments.from_basis(X, y, predictions, basis)

    def test_bad_basis_output_shape_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        predictions = np.zeros(10)
        bad_basis = [lambda X: np.zeros(5)]  # returns wrong size
        with pytest.raises(ValueError, match="basis function"):
            PolynomialMoments.from_basis(X, y, predictions, bad_basis)

    def test_delta_loss_wrong_c_size_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        predictions = np.zeros(10)
        basis = monomial_basis([0], max_degree=2)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        with pytest.raises(ValueError, match="coefficient shape mismatch"):
            moments.delta_loss(np.zeros(5))

    def test_1d_X_promoted_to_2d(self):
        """Convenience: a 1-D X (N,) should be auto-reshaped to (N, 1)."""
        X = np.linspace(-1, 1, 50)  # 1-D
        y = X ** 2
        predictions = np.zeros(50)
        basis = monomial_basis([0], max_degree=2)
        moments = PolynomialMoments.from_basis(X, y, predictions, basis)
        # Should successfully construct
        assert moments.N == 50
        assert moments.K == 2
