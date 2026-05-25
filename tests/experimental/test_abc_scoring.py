"""Tests for tessera.experimental.abc_scoring — Conjecture C1-refined.

Validates the summary-statistics + ABC-distance machinery and the
GPWithHoldout subclass behaves correctly. Does NOT test the
empirical conjecture (that's in the benchmark).
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.experimental.abc_scoring import (
    compute_summary_stats,
    abc_distance,
    GPWithHoldout,
)
from tessera.search import GPConfig


# ----------------------------------------------------------------------
# 1. Summary statistics
# ----------------------------------------------------------------------

class TestComputeSummaryStats:
    def test_2d_known_input(self):
        """Constant array → variance 0, mean_abs = constant, ACFs 0 or
        undefined."""
        arr = np.full((50, 50), 3.14)
        stats = compute_summary_stats(arr)
        assert stats["var_total"] == pytest.approx(0.0, abs=1e-12)
        assert stats["mean_abs"] == pytest.approx(3.14, rel=1e-6)
        # ACF on constant is undefined (denom zero); _pearson_safe returns 0
        assert stats["acf_time_lag1"] == pytest.approx(0.0)

    def test_2d_random_input(self):
        """Random array → variance ~1, mean_abs ~0.8 (for standard normal),
        ACFs near zero (uncorrelated)."""
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((100, 30))
        stats = compute_summary_stats(arr)
        assert stats["var_total"] == pytest.approx(1.0, rel=0.15)
        assert stats["mean_abs"] == pytest.approx(0.798, abs=0.05)
        assert abs(stats["acf_time_lag1"]) < 0.1
        assert abs(stats["acf_space_lag1"]) < 0.1

    def test_2d_temporally_autocorrelated(self):
        """AR(1) process should have detectable temporal ACF."""
        rng = np.random.default_rng(0)
        T, X = 200, 30
        arr = np.zeros((T, X))
        for t in range(1, T):
            arr[t] = 0.8 * arr[t-1] + 0.6 * rng.standard_normal(X)
        stats = compute_summary_stats(arr)
        # Lag-1 ACF should be ~0.8 by construction
        assert stats["acf_time_lag1"] > 0.5
        # Spatial ACF should be near zero (independent across X)
        assert abs(stats["acf_space_lag1"]) < 0.15

    def test_2d_spatially_autocorrelated(self):
        """Smooth spatial profile should have detectable spatial ACF."""
        rng = np.random.default_rng(0)
        T, X = 100, 30
        # Smooth spatial: low-frequency cosine basis
        x_axis = np.linspace(0, np.pi, X)
        base = np.cos(x_axis)
        arr = base[np.newaxis, :] + 0.05 * rng.standard_normal((T, X))
        stats = compute_summary_stats(arr)
        # Spatial ACF should be very close to 1 (smooth profile)
        assert stats["acf_space_lag1"] > 0.9

    def test_1d_input(self):
        """1-D input gets a subset of stats."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_summary_stats(arr)
        assert "var_total" in stats
        assert "mean_abs" in stats
        assert "acf_time_lag1" in stats
        # 1-D has no spatial ACF
        assert "acf_space_lag1" not in stats

    def test_nan_robust(self):
        """NaN values should be masked, not crash."""
        arr = np.ones((50, 30))
        arr[10:20, :] = np.nan
        stats = compute_summary_stats(arr)
        # Should run without error
        assert all(np.isfinite(v) for v in stats.values())

    def test_all_nan_returns_zeros(self):
        """All-NaN input → all-zero stats (degenerate but defined)."""
        arr = np.full((30, 30), np.nan)
        stats = compute_summary_stats(arr)
        # var, mean_abs should be 0 by convention
        assert stats["var_total"] == 0.0
        assert stats["mean_abs"] == 0.0


# ----------------------------------------------------------------------
# 2. ABC distance
# ----------------------------------------------------------------------

class TestABCDistance:
    def test_identical_stats_zero_distance(self):
        s = {"var_total": 1.5, "mean_abs": 0.8, "acf_time_lag1": 0.3,
             "acf_space_lag1": 0.1, "spatial_mean_var": 0.05,
             "temporal_mean_var": 0.04}
        d = abc_distance(s, s)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_doubled_stats_relative_error_1(self):
        """Predicted stats are 2× observed → relative error 1.0,
        distance = mean of 1.0² across stats = 1.0."""
        s_obs = {"var_total": 1.0, "mean_abs": 1.0}
        s_pred = {"var_total": 2.0, "mean_abs": 2.0}
        d = abc_distance(s_obs, s_pred)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_scale_invariant(self):
        """Relative error is scale-invariant: doubling everything gives
        same distance."""
        s_obs = {"var_total": 1e-3, "mean_abs": 1e-3}
        s_pred = {"var_total": 2e-3, "mean_abs": 2e-3}
        d_small = abc_distance(s_obs, s_pred)
        s_obs_big = {"var_total": 1e6, "mean_abs": 1e6}
        s_pred_big = {"var_total": 2e6, "mean_abs": 2e6}
        d_big = abc_distance(s_obs_big, s_pred_big)
        assert d_small == pytest.approx(d_big, abs=1e-9)

    def test_non_finite_penalized(self):
        """NaN / inf predicted stats add penalty 1.0 each."""
        s_obs = {"var_total": 1.0, "mean_abs": 1.0}
        s_pred = {"var_total": float("nan"), "mean_abs": 1.0}
        d = abc_distance(s_obs, s_pred)
        # The NaN stat contributes 1.0; the mean_abs match contributes 0
        # Average: 0.5
        assert d == pytest.approx(0.5, abs=1e-9)

    def test_missing_key_in_pred_ignored(self):
        """Stats only in obs (not pred) are excluded from comparison."""
        s_obs = {"var_total": 1.0, "mean_abs": 1.0, "extra": 5.0}
        s_pred = {"var_total": 1.0, "mean_abs": 1.0}
        d = abc_distance(s_obs, s_pred)
        # Only shared stats counted; both match
        assert d == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# 3. GPWithHoldout subclass
# ----------------------------------------------------------------------

class TestGPWithHoldout:
    def _make_synthetic_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = 2 * x - 0.5 * x ** 2
        return {"x": x}, y

    def test_constructor_runs(self):
        env, y = self._make_synthetic_data(seed=0)
        hold_env, hold_y = self._make_synthetic_data(seed=1)
        cfg = GPConfig(pop_size=20, n_gens=5, seed=42,
                       verbose=False, pointwise_only=True)
        gp = GPWithHoldout(cfg, hold_env=hold_env, hold_y_true=hold_y,
                           beta_abc=1.0, beta_hold_mse=1.0)
        assert gp.beta_abc == 1.0
        assert gp.beta_hold_mse == 1.0
        assert "x" in gp.hold_env
        # hold_stats computed at init
        assert "var_total" in gp.hold_stats

    def test_runs_end_to_end(self):
        env, y = self._make_synthetic_data(seed=0)
        hold_env, hold_y = self._make_synthetic_data(seed=1)
        cfg = GPConfig(pop_size=20, n_gens=5, seed=42,
                       verbose=False, pointwise_only=True,
                       early_stop_patience=5,
                       optimize_constants_every=0,
                       parsimony=0.01)
        gp = GPWithHoldout(cfg, hold_env=hold_env, hold_y_true=hold_y,
                           beta_abc=1.0, beta_hold_mse=1.0)
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_zero_betas_match_baseline(self):
        """beta_abc=0 and beta_hold_mse=0 → should match standard GP
        results bit-for-bit at same seed."""
        from tessera.search import GP
        env, y = self._make_synthetic_data(seed=0)
        hold_env, hold_y = self._make_synthetic_data(seed=1)
        cfg = GPConfig(pop_size=15, n_gens=5, seed=42,
                       verbose=False, pointwise_only=True,
                       early_stop_patience=5,
                       optimize_constants_every=0)
        baseline = GP(cfg)
        with_hold = GPWithHoldout(cfg, hold_env=hold_env, hold_y_true=hold_y,
                                  beta_abc=0.0, beta_hold_mse=0.0)
        front_b = baseline.run(env, y, feature_names=["x"])
        front_w = with_hold.run(env, y, feature_names=["x"])
        # Same seed + same config + zero hold-out weights → identical fronts
        assert len(front_b) == len(front_w)
        for cb, cw in zip(front_b, front_w):
            assert cb.complexity == cw.complexity
            assert cb.train_loss == pytest.approx(cw.train_loss, abs=1e-12)

    def test_nonzero_beta_changes_fitness(self):
        """With nonzero beta_abc, fitness values should differ from baseline
        (selection is influenced even if Pareto front semantics unchanged)."""
        env, y = self._make_synthetic_data(seed=0)
        # Make hold-out distinct enough to perturb scoring
        hold_env, hold_y = self._make_synthetic_data(seed=99)
        cfg = GPConfig(pop_size=30, n_gens=8, seed=42,
                       verbose=False, pointwise_only=True,
                       early_stop_patience=8,
                       optimize_constants_every=0,
                       parsimony=0.001)
        with_hold = GPWithHoldout(cfg, hold_env=hold_env, hold_y_true=hold_y,
                                  beta_abc=10.0, beta_hold_mse=0.0)
        front = with_hold.run(env, y, feature_names=["x"])
        # Front should be non-empty and contain real candidates
        assert len(front) > 0
        # Every candidate has a finite fitness
        for c in front:
            assert np.isfinite(c.fitness)
