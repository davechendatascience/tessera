"""Stage 2a tests: Tier A model-class discriminators.

Per Stage 2 design contract:
  - classify_model_class(traj) -> ModelClass
  - Three discriminators run cheaply; PDE/ODE/algebraic distinguished
    by permutation invariance + autocorrelation structure + stencil
    locality
  - All 11 canonical systems classified to their declared ModelClass
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.workbench import get_system, list_systems, ModelClass
from tessera.workbench.signatures import (
    classify_model_class,
    compute_permutation_invariance,
    compute_autocorrelation_structure,
    compute_stencil_locality,
)
from tessera.workbench.signatures.types import SignatureValue


# Canonical (system_id, dt, t_max) settings for the classification test.
# Burgers needs small dt for CFL stability; heat and the others use longer
# windows so signatures have enough samples.
CLASSIFY_FIXTURES = [
    ("harmonic_1d", {"t_max": 10.0, "dt": 0.01}),
    ("damped_harmonic_1d", {"t_max": 10.0, "dt": 0.01}),
    ("vdp", {"t_max": 10.0, "dt": 0.01}),
    ("lorenz63", {"t_max": 10.0, "dt": 0.01}),
    ("fhn", {"t_max": 10.0, "dt": 0.01}),
    ("heat_1d", {"t_max": 10.0, "dt": 0.1}),
    ("burgers_1d", {"t_max": 2.0, "dt": 0.005}),
    ("linear_pendulum", {"t_max": 10.0, "dt": 0.01}),
    ("nonlinear_pendulum", {"t_max": 10.0, "dt": 0.01}),
    ("kepler", {"t_max": 10.0, "dt": 0.01}),
]


class TestPermutationInvariance:
    def test_algebraic_is_high(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        sv = compute_permutation_invariance(traj)
        # Algebraic with iid inputs should be permutation-invariant
        # (within sampling noise) — score near 1
        assert isinstance(sv, SignatureValue)
        assert sv.value > 0.3, f"algebraic should be permutation-invariant; got {sv.value}"

    def test_ode_is_low(self):
        # Strong temporal correlation = low permutation invariance
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_permutation_invariance(traj)
        assert sv.value < 0.2, f"ODE should be order-dependent; got {sv.value}"

    def test_tiny_trajectory_returns_low_confidence(self):
        s = get_system("harmonic_1d")
        # Make a tiny trajectory — should report low confidence
        from tessera.workbench import Trajectory
        traj = Trajectory(
            t=np.arange(5).astype(float), state=np.zeros((5, 2)),
            observable=np.zeros((5, 2)), system_id="test",
            params={}, ic=np.zeros(2),
        )
        sv = compute_permutation_invariance(traj)
        assert sv.confidence == 0.0


class TestAutocorrelationStructure:
    def test_algebraic_has_zero_time_acf(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        sv = compute_autocorrelation_structure(traj)
        assert sv.value["time_acf_max"] < 0.2
        assert sv.value["space_acf_max"] is None  # not 2D

    def test_ode_has_high_time_acf(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_autocorrelation_structure(traj)
        assert sv.value["time_acf_max"] > 0.5

    def test_pde_has_both_time_and_space_acf(self):
        s = get_system("heat_1d")
        traj = s.generate(t_max=20.0, dt=0.5, seed=0)
        sv = compute_autocorrelation_structure(traj)
        assert sv.value["time_acf_max"] > 0.5
        assert sv.value["space_acf_max"] is not None
        assert sv.value["space_acf_max"] > 0.5


class TestStencilLocality:
    def test_heat_eq_has_finite_stencil(self):
        # Heat equation's 3-point Laplacian: u(t+1) - u(t) = alpha*(u_{x-1} - 2u_x + u_{x+1})
        # Linear stencil of half-width 1. Should be detected.
        s = get_system("heat_1d")
        traj = s.generate(t_max=20.0, dt=0.5, seed=0)
        sv = compute_stencil_locality(traj)
        assert sv.value is not None
        assert sv.value <= 2, f"heat eq should have small stencil; got {sv.value}"

    def test_ode_returns_none(self):
        s = get_system("lorenz63")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_stencil_locality(traj)
        # ODEs don't have spatial structure → stencil test returns None
        assert sv.value is None

    def test_nan_input_returns_safely(self):
        from tessera.workbench import Trajectory
        # Inject NaN
        state = np.full((50, 30), np.nan)
        traj = Trajectory(
            t=np.arange(50).astype(float), state=state, observable=state,
            system_id="test", params={}, ic=np.zeros(30),
        )
        sv = compute_stencil_locality(traj)
        assert sv.value is None
        assert sv.confidence == 0.0


class TestClassifyModelClass:
    """The headline test: classify_model_class must correctly identify
    the model class of every canonical system from data alone."""

    @pytest.mark.parametrize("sid,kwargs", CLASSIFY_FIXTURES)
    def test_canonical_systems_classify_correctly(self, sid, kwargs):
        s = get_system(sid)
        traj = s.generate(seed=0, **kwargs)
        inferred, diag = classify_model_class(traj)
        declared = s.model_class
        assert inferred == declared, (
            f"{sid}: declared={declared.value} but inferred={inferred.value}; "
            f"diagnostics = "
            f"perm={diag['permutation_invariance'].value:.3f}, "
            f"t_acf={diag['autocorrelation_structure'].value.get('time_acf_max', 0):.3f}, "
            f"s_acf={diag['autocorrelation_structure'].value.get('space_acf_max')}"
        )

    def test_algebraic_classifies_correctly(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        inferred, diag = classify_model_class(traj)
        assert inferred == ModelClass.ALGEBRAIC

    def test_diagnostics_complete(self):
        """Diagnostics dict must contain all 3 Tier A signature values."""
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=5.0, dt=0.01, seed=0)
        _, diag = classify_model_class(traj)
        assert "permutation_invariance" in diag
        assert "autocorrelation_structure" in diag
        assert "stencil_locality" in diag


class TestSignatureValueContract:
    """SignatureValue must carry value, confidence, n_samples_used."""

    def test_fields_populated(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=5.0, dt=0.01, seed=0)
        sv = compute_permutation_invariance(traj)
        assert hasattr(sv, "value")
        assert hasattr(sv, "confidence")
        assert hasattr(sv, "n_samples_used")
        assert 0.0 <= sv.confidence <= 1.0
        assert sv.n_samples_used == traj.state.shape[0]

    def test_is_reliable_threshold(self):
        sv = SignatureValue(value=1.0, confidence=0.8, n_samples_used=100)
        assert sv.is_reliable(0.5)
        assert sv.is_reliable(0.8)
        assert not sv.is_reliable(0.9)
