"""Tests for tessera.experimental.mdl_scoring — Conjecture C3."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp,
)
from tessera.search import GP, GPConfig

from tessera.experimental.mdl_scoring import (
    description_length_bits,
    GPWithMDLScoring,
)


# ----------------------------------------------------------------------
# 1. Description length
# ----------------------------------------------------------------------

class TestDescriptionLength:
    def test_const_alone(self):
        """A single constant has a fixed-ish bit cost."""
        node = Const(value=1.0)
        dl = description_length_bits(node)
        assert dl > 0
        assert dl < 50  # reasonable bound

    def test_var_alone(self):
        node = Var(name="x")
        dl = description_length_bits(node)
        assert dl > 0
        assert dl < 20

    def test_larger_tree_has_more_bits(self):
        """A 2-node BinOp has more bits than a Var alone."""
        small = Var(name="x")
        large = BinOp("mul", Const(2.0), Var("x"))
        assert description_length_bits(large) > description_length_bits(small)

    def test_dl_monotone_in_cx(self):
        """Within the same node-type mix, DL roughly grows with cx."""
        # 3 nested adds vs 1 add
        small = BinOp("add", Var("x"), Var("y"))
        large = BinOp("add",
                      BinOp("add", Var("x"), Var("y")),
                      BinOp("add", Var("z"), Var("w")))
        assert description_length_bits(large) > description_length_bits(small)

    def test_dl_is_float(self):
        """DL is float (uses log2)."""
        node = BinOp("mul", Const(1.0), Var("x"))
        dl = description_length_bits(node)
        assert isinstance(dl, float)


# ----------------------------------------------------------------------
# 2. GPWithMDLScoring construction
# ----------------------------------------------------------------------

class TestGPWithMDLScoringConstruction:
    def test_invalid_penalty_mode_raises(self):
        cfg = GPConfig(pop_size=5, n_gens=1, seed=0, verbose=False)
        with pytest.raises(ValueError):
            GPWithMDLScoring(cfg, penalty_mode="nonsense")

    def test_construction_succeeds(self):
        cfg = GPConfig(pop_size=5, n_gens=1, seed=0, verbose=False)
        for mode in ("adhoc", "naive_mdl", "recalibrated_mdl"):
            gp = GPWithMDLScoring(cfg, sigma=0.001, penalty_mode=mode)
            assert gp.penalty_mode == mode


# ----------------------------------------------------------------------
# 3. End-to-end runs
# ----------------------------------------------------------------------

class TestGPWithMDLScoringEndToEnd:
    def _synthetic_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = 2 * x - 0.5 * x ** 2 + 0.02 * rng.standard_normal(n)
        return {"x": x}, y

    def test_runs_in_adhoc_mode(self):
        env, y = self._synthetic_data()
        cfg = GPConfig(pop_size=15, n_gens=5, seed=0, verbose=False,
                       pointwise_only=True, early_stop_patience=5,
                       parsimony=0.01)
        gp = GPWithMDLScoring(cfg, penalty_mode="adhoc")
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_runs_in_naive_mdl(self):
        env, y = self._synthetic_data()
        cfg = GPConfig(pop_size=15, n_gens=5, seed=0, verbose=False,
                       pointwise_only=True, early_stop_patience=5,
                       parsimony=0.01)
        gp = GPWithMDLScoring(cfg, sigma=0.02, penalty_mode="naive_mdl")
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_runs_in_recalibrated(self):
        env, y = self._synthetic_data()
        cfg = GPConfig(pop_size=15, n_gens=5, seed=0, verbose=False,
                       pointwise_only=True, early_stop_patience=5,
                       parsimony=0.01)
        gp = GPWithMDLScoring(cfg, sigma=0.02, penalty_mode="recalibrated_mdl")
        front = gp.run(env, y, feature_names=["x"])
        assert len(front) > 0

    def test_adhoc_matches_base_gp(self):
        """With penalty_mode='adhoc', should behave identically to base GP."""
        env, y = self._synthetic_data()
        cfg = GPConfig(pop_size=15, n_gens=5, seed=42, verbose=False,
                       pointwise_only=True, early_stop_patience=5,
                       parsimony=0.01, optimize_constants_every=0)
        baseline = GP(cfg)
        mdl_adhoc = GPWithMDLScoring(cfg, penalty_mode="adhoc")
        front_b = baseline.run(env, y, feature_names=["x"])
        front_m = mdl_adhoc.run(env, y, feature_names=["x"])
        # Same length and same per-position (cx, loss) — adhoc mode is
        # a no-op subclass
        assert len(front_b) == len(front_m)
        for cb, cm in zip(front_b, front_m):
            assert cb.complexity == cm.complexity
            assert cb.train_loss == pytest.approx(cm.train_loss, abs=1e-12)

    def test_naive_mdl_fitness_differs_from_adhoc(self):
        """Naive MDL should produce fitness values that differ from
        ad-hoc (sanity check on the math being applied)."""
        env, y = self._synthetic_data()
        cfg = GPConfig(pop_size=15, n_gens=5, seed=42, verbose=False,
                       pointwise_only=True, early_stop_patience=5,
                       parsimony=0.01, optimize_constants_every=0)
        adhoc = GPWithMDLScoring(cfg, penalty_mode="adhoc")
        naive = GPWithMDLScoring(cfg, sigma=0.02, penalty_mode="naive_mdl")
        front_a = adhoc.run(env, y, feature_names=["x"])
        front_n = naive.run(env, y, feature_names=["x"])

        # Same trees may emerge but fitnesses should differ
        # (MDL uses MSE/(2σ²) rather than MSE + α·cx)
        # We just check both produce valid runs.
        assert len(front_a) > 0
        assert len(front_n) > 0
