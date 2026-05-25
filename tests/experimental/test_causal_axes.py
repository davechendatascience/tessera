"""Tests for tessera.experimental.causal_axes — Conjecture C4."""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression.measure_2d import (
    Atom2D, Measure2D,
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
)
from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp2D,
)
from tessera.search import GPConfig

from tessera.experimental.causal_axes import (
    is_pure_spatial_m2d,
    tree_violates_causal_spatial,
    count_violating_m2ds,
    GPWithCausalAxes,
)


# ----------------------------------------------------------------------
# 1. is_pure_spatial_m2d
# ----------------------------------------------------------------------

class TestIsPureSpatialM2D:
    def test_laplacian_5pt_is_pure_spatial(self):
        """The canonical 5-point Laplacian has 3 atoms all at lag_t=0,
        different lag_x — pure spatial."""
        lap = measure_2d_laplacian_5pt()
        assert is_pure_spatial_m2d(lap)

    def test_diff_t_is_not_pure_spatial(self):
        """A time-difference measure has atoms spanning lag_t — violates."""
        diff_t = measure_2d_diff_t(lag_t=1)
        assert not is_pure_spatial_m2d(diff_t)

    def test_diff_t_higher_lag_not_pure_spatial(self):
        diff_t = measure_2d_diff_t(lag_t=3)
        assert not is_pure_spatial_m2d(diff_t)

    def test_grad_x_is_pure_spatial(self):
        """A spatial gradient has atoms all at lag_t=0, different lag_x."""
        grad_x = measure_2d_grad_x()
        assert is_pure_spatial_m2d(grad_x)

    def test_manual_pure_spatial(self):
        """Construct a custom pure-spatial: 2 atoms at same lag_t."""
        m = Measure2D(atoms=(
            Atom2D(weight=0.5, lag_t=0, lag_x=-1),
            Atom2D(weight=-0.5, lag_t=0, lag_x=1),
        ))
        assert is_pure_spatial_m2d(m)

    def test_manual_temporal_only(self):
        m = Measure2D(atoms=(
            Atom2D(weight=1.0, lag_t=0, lag_x=0),
            Atom2D(weight=-1.0, lag_t=2, lag_x=0),
        ))
        assert not is_pure_spatial_m2d(m)

    def test_manual_mixed_temporal_and_spatial(self):
        """Atoms spanning both lag_t AND lag_x → still temporal (violates)."""
        m = Measure2D(atoms=(
            Atom2D(weight=1.0, lag_t=0, lag_x=0),
            Atom2D(weight=-1.0, lag_t=1, lag_x=1),
        ))
        assert not is_pure_spatial_m2d(m)

    def test_single_atom_is_pure_spatial(self):
        """One atom can't span anything; trivially pure-spatial."""
        m = Measure2D(atoms=(Atom2D(weight=1.0, lag_t=0, lag_x=0),))
        assert is_pure_spatial_m2d(m)

    def test_single_atom_nonzero_lag_t_pure_spatial(self):
        """Single atom with lag_t > 0 — still trivially pure-spatial
        (no spanning)."""
        m = Measure2D(atoms=(Atom2D(weight=1.0, lag_t=5, lag_x=0),))
        assert is_pure_spatial_m2d(m)


# ----------------------------------------------------------------------
# 2. tree_violates_causal_spatial
# ----------------------------------------------------------------------

class TestTreeViolatesCausalSpatial:
    def test_simple_pure_spatial_tree(self):
        """tree = M2D[Laplacian](U) — pure spatial, OK."""
        lap = measure_2d_laplacian_5pt()
        tree = FunctionalOp2D(lap, Var("U"))
        assert not tree_violates_causal_spatial(tree)

    def test_simple_temporal_tree(self):
        """tree = M2D[diff_t](U) — temporal, violates."""
        diff_t = measure_2d_diff_t(lag_t=1)
        tree = FunctionalOp2D(diff_t, Var("U"))
        assert tree_violates_causal_spatial(tree)

    def test_nested_pure_spatial_tree(self):
        """tree = (M2D[Laplacian](U) * 0.05) — nested but pure spatial."""
        lap = measure_2d_laplacian_5pt()
        inner = FunctionalOp2D(lap, Var("U"))
        tree = BinOp("mul", Const(0.05), inner)
        assert not tree_violates_causal_spatial(tree)

    def test_mixed_tree_with_one_violation(self):
        """tree contains both spatial and temporal M2Ds → violates."""
        lap = measure_2d_laplacian_5pt()
        diff_t = measure_2d_diff_t(lag_t=1)
        spatial = FunctionalOp2D(lap, Var("U"))
        temporal = FunctionalOp2D(diff_t, Var("U"))
        tree = BinOp("add", spatial, temporal)
        assert tree_violates_causal_spatial(tree)

    def test_tree_without_m2d(self):
        """tree has no Measure2D → trivially OK."""
        tree = BinOp("mul", Const(0.05), Var("U"))
        assert not tree_violates_causal_spatial(tree)

    def test_count_violating_m2ds(self):
        """count_violating_m2ds is exact count."""
        lap = measure_2d_laplacian_5pt()
        diff_t1 = measure_2d_diff_t(lag_t=1)
        diff_t2 = measure_2d_diff_t(lag_t=3)
        spatial = FunctionalOp2D(lap, Var("U"))
        temporal1 = FunctionalOp2D(diff_t1, Var("U"))
        temporal2 = FunctionalOp2D(diff_t2, Var("U"))
        tree = BinOp("add", spatial, BinOp("add", temporal1, temporal2))
        assert count_violating_m2ds(tree) == 2


# ----------------------------------------------------------------------
# 3. GPWithCausalAxes
# ----------------------------------------------------------------------

class TestGPWithCausalAxes:
    def _make_heat_eq_data(self, T=50, X=16, seed=0):
        """Minimal heat-equation trajectory for testing."""
        from tessera.expression.measure_2d import measure_2d_laplacian_5pt
        rng = np.random.default_rng(seed)
        U = np.zeros((T, X))
        xs = np.arange(X) - X / 2
        U[0] = 5.0 * np.exp(-(xs ** 2) / 8.0)
        for t in range(1, T):
            lap = np.zeros_like(U[t-1])
            lap[1:-1] = U[t-1, :-2] - 2.0 * U[t-1, 1:-1] + U[t-1, 2:]
            U[t] = U[t-1] + 0.05 * lap + 0.002 * rng.standard_normal(X)
        dt_U = np.zeros_like(U)
        dt_U[:-1] = U[1:] - U[:-1]
        return U, dt_U

    def test_constructor_runs(self):
        cfg = GPConfig(pop_size=20, n_gens=5, seed=0,
                       verbose=False, enable_2d=True)
        gp = GPWithCausalAxes(cfg, penalty=1e6)
        assert gp.causal_penalty == 1e6

    def test_runs_end_to_end_on_heat_eq(self):
        U, dt_U = self._make_heat_eq_data(T=50, X=16, seed=0)
        cfg = GPConfig(
            pop_size=20, n_gens=5, seed=0,
            verbose=False, enable_2d=True,
            early_stop_patience=5, fill_warmup=0.0,
            parsimony=0.0001,
        )
        gp = GPWithCausalAxes(cfg, penalty=1e6)
        front = gp.run({"U": U}, dt_U, feature_names=["U"])
        assert len(front) > 0

    def test_violating_tree_gets_penalty(self):
        """A tree containing diff_t M2D should score with penalty."""
        from tessera.expression.tree import FunctionalOp2D
        from tessera.expression.measure_2d import measure_2d_diff_t
        from tessera.search.scoring import _evaluate_tree
        from tessera.expression.cache import FunctionalCache

        U, dt_U = self._make_heat_eq_data(T=50, X=16, seed=0)
        cfg = GPConfig(
            pop_size=10, n_gens=2, seed=0,
            verbose=False, enable_2d=True,
            early_stop_patience=2, fill_warmup=0.0,
            parsimony=0.0,
        )

        # Construct a temporal-violating tree manually
        diff_t = measure_2d_diff_t(lag_t=1)
        violating_tree = FunctionalOp2D(diff_t, Var("U"))

        gp = GPWithCausalAxes(cfg, penalty=1e6)
        cand = gp._score(violating_tree, {"U": U}, dt_U, born_gen=0)

        # Fitness should be huge due to penalty
        assert cand.fitness > 1e5

    def test_zero_penalty_matches_baseline(self):
        """With penalty=0, fitness equals base GP."""
        from tessera.search import GP

        U, dt_U = self._make_heat_eq_data(T=50, X=16, seed=42)
        cfg = GPConfig(
            pop_size=15, n_gens=3, seed=42,
            verbose=False, enable_2d=True,
            early_stop_patience=3, fill_warmup=0.0,
            optimize_constants_every=0,
        )
        baseline = GP(cfg)
        with_causal = GPWithCausalAxes(cfg, penalty=0.0)
        front_b = baseline.run({"U": U}, dt_U, feature_names=["U"])
        front_c = with_causal.run({"U": U}, dt_U, feature_names=["U"])
        assert len(front_b) == len(front_c)
        for cb, cc in zip(front_b, front_c):
            assert cb.complexity == cc.complexity
            assert cb.train_loss == pytest.approx(cc.train_loss, abs=1e-12)
