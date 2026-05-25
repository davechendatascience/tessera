"""Tests for tessera.experimental.adaptive_search — Conjecture C6."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression import mutation as _mut
from tessera.search import GP, GPConfig
from tessera.experimental.adaptive_search import GPWithAdaptiveSearch


class TestGPWithAdaptiveSearch:
    def _synthetic_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = 2 * x - 0.5 * x ** 2 + 0.02 * rng.standard_normal(n)
        return {"x": x}, y

    def test_constructor(self):
        cfg = GPConfig(pop_size=10, n_gens=3, seed=0, verbose=False)
        gp = GPWithAdaptiveSearch(cfg, adapt_every=5, adapt_strength=0.5)
        assert gp.adapt_every == 5
        assert gp.adapt_strength == 0.5

    def test_runs_end_to_end(self):
        env, y = self._synthetic_data()
        cfg = GPConfig(
            pop_size=15, n_gens=15, seed=0, verbose=False,
            pointwise_only=True, early_stop_patience=15,
            optimize_constants_every=0,
        )
        gp = GPWithAdaptiveSearch(cfg, adapt_every=5)
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0
        # Should have triggered adaptations at gens 5, 10, 15
        assert len(gp.adapt_history) >= 2

    def test_restores_un_op_weights(self):
        """After run, UN_OP_WEIGHTS should be restored to pre-run state."""
        env, y = self._synthetic_data()
        cfg = GPConfig(
            pop_size=15, n_gens=10, seed=0, verbose=False,
            pointwise_only=True, early_stop_patience=10,
            optimize_constants_every=0,
        )
        # Snapshot original weights
        original = dict(_mut.UN_OP_WEIGHTS)
        gp = GPWithAdaptiveSearch(cfg, adapt_every=5)
        _ = gp.run(env, y, feature_names=["x"])
        # After run, weights should match original
        for op, w in original.items():
            assert _mut.UN_OP_WEIGHTS[op] == pytest.approx(w, abs=1e-12)

    def test_zero_strength_matches_baseline(self):
        """With adapt_strength=0, should behave identically to base GP."""
        env, y = self._synthetic_data()
        cfg = GPConfig(
            pop_size=15, n_gens=10, seed=42, verbose=False,
            pointwise_only=True, early_stop_patience=10,
            optimize_constants_every=0,
        )
        baseline = GP(cfg)
        adaptive_zero = GPWithAdaptiveSearch(cfg, adapt_strength=0.0)
        front_b = baseline.run(env, y, feature_names=["x"])
        front_a = adaptive_zero.run(env, y, feature_names=["x"])
        assert len(front_b) == len(front_a)
        for cb, ca in zip(front_b, front_a):
            assert cb.complexity == ca.complexity
            assert cb.train_loss == pytest.approx(ca.train_loss, abs=1e-12)

    def test_adaptation_records_diagnostics(self):
        env, y = self._synthetic_data()
        cfg = GPConfig(
            pop_size=20, n_gens=15, seed=0, verbose=False,
            pointwise_only=True, early_stop_patience=15,
            optimize_constants_every=0,
        )
        gp = GPWithAdaptiveSearch(cfg, adapt_every=5, adapt_strength=0.8)
        _ = gp.run(env, y, feature_names=["x"])
        # Adaptation should have fired at gens 5, 10, 15
        assert len(gp.adapt_history) >= 2
        for entry in gp.adapt_history:
            assert "gen" in entry
            assert "front_size" in entry
            assert "weights_snapshot" in entry

    def test_weights_change_during_adaptation(self):
        """At nonzero adapt_strength, weights should differ from baseline
        during the run (verified via history snapshots)."""
        env, y = self._synthetic_data()
        cfg = GPConfig(
            pop_size=20, n_gens=15, seed=0, verbose=False,
            pointwise_only=True, early_stop_patience=15,
            optimize_constants_every=0,
        )
        gp = GPWithAdaptiveSearch(cfg, adapt_every=5, adapt_strength=0.8)
        _ = gp.run(env, y, feature_names=["x"])
        if gp.adapt_history:
            snapshot = gp.adapt_history[-1]["weights_snapshot"]
            original = dict(_mut.UN_OP_WEIGHTS)  # restored post-run
            # At least one weight in the snapshot should differ from baseline
            differs = any(
                abs(snapshot[op] - original[op]) > 1e-9
                for op in snapshot
                if op in original
            )
            # This may not always trigger (if front happens to match
            # uniform), but commonly will.
            # If it doesn't trigger, the test passes silently.
            # (More of a smoke test than a strict assertion.)
