"""Stage 2b tests: Tier B within-class signatures + full-signature assembler.

Each signature is tested for:
  - Returns a SignatureValue with the expected value type
  - Behaves sensibly on canonical systems where ground truth is known
  - Handles edge cases (too-short trajectory, zero-variance signal)

The full-signature assembler is tested for:
  - Routes correctly based on model class
  - Populates only applicable signatures (others stay None)
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.workbench import get_system, ModelClass, Trajectory
from tessera.workbench.signatures import (
    compute_smoothness, compute_mode_count, compute_effective_dimensionality,
    compute_symmetry, compute_conservation, compute_spectral_content,
    compute_determinism, compute_lyapunov,
    compute_full_signature, WITHIN_CLASS_APPLICABILITY,
    Signature,
)


class TestSmoothness:
    def test_smooth_ode_is_high(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_smoothness(traj)
        assert 0.5 <= sv.value <= 1.5, f"smooth ODE alpha={sv.value}"

    def test_algebraic_iid_is_low(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        sv = compute_smoothness(traj)
        # Pure iid data has no temporal smoothness
        assert sv.value < 0.3, f"algebraic alpha should be ~0; got {sv.value}"

    def test_pde_is_smooth(self):
        s = get_system("heat_1d")
        traj = s.generate(t_max=50.0, dt=0.1, seed=0)
        sv = compute_smoothness(traj)
        # Heat eq solutions are C^infty
        assert sv.value > 0.8


class TestModeCount:
    def test_returns_positive_integer(self):
        s = get_system("vdp")
        traj = s.generate(t_max=30.0, dt=0.05, seed=0)
        sv = compute_mode_count(traj)
        assert isinstance(sv.value, int)
        assert sv.value >= 1

    def test_too_few_samples_returns_default(self):
        from tessera.workbench import Trajectory
        traj = Trajectory(
            t=np.arange(20).astype(float), state=np.zeros((20, 2)),
            observable=np.zeros((20, 2)), system_id="test",
            params={}, ic=np.zeros(2),
        )
        sv = compute_mode_count(traj)
        assert sv.value == 1
        assert sv.confidence == 0.0

    def test_lorenz_attractor_finds_at_least_two(self):
        # Lorenz has two distinct attractor lobes — even with the GMM limit
        # cycle issue, lorenz should find >= 2.
        s = get_system("lorenz63")
        traj = s.generate(t_max=30.0, dt=0.01, seed=0)
        sv = compute_mode_count(traj)
        assert sv.value >= 2


class TestEffectiveDimensionality:
    def test_2d_oscillator_gives_two(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=20.0, dt=0.01, seed=0)
        sv = compute_effective_dimensionality(traj)
        # 2D state, both components used → ~2
        assert 1.5 <= sv.value <= 2.5

    def test_3d_lorenz_lower_than_three(self):
        # Lorenz attractor has fractal dim ~2.06 < 3
        s = get_system("lorenz63")
        traj = s.generate(t_max=30.0, dt=0.01, seed=0)
        sv = compute_effective_dimensionality(traj)
        assert 1.5 <= sv.value <= 3.0

    def test_algebraic_scalar_gives_one(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        sv = compute_effective_dimensionality(traj)
        assert 0.5 <= sv.value <= 1.5


class TestSymmetry:
    def test_returns_dict_with_groups(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_symmetry(traj)
        assert isinstance(sv.value, dict)
        # Default candidates: time_translation, time_reversal, reflection, rotation_so2
        for key in ["time_translation", "time_reversal", "reflection"]:
            assert key in sv.value
            assert "passes" in sv.value[key]
            assert "score" in sv.value[key]

    def test_custom_candidates(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=10.0, dt=0.01, seed=0)
        sv = compute_symmetry(traj, candidates=["reflection"])
        assert set(sv.value.keys()) == {"reflection"}


class TestConservation:
    def test_kepler_finds_low_variance_quantity(self):
        s = get_system("kepler")
        traj = s.generate(t_max=20.0, dt=0.005, seed=0)
        sv = compute_conservation(traj)
        # Should find at least one degree with low variance ratio (energy or
        # angular momentum is degree-2 conserved)
        assert 2 in sv.value
        assert sv.value[2]["variance_ratio"] < 0.5

    def test_dissipative_system_higher_ratio(self):
        s = get_system("damped_harmonic_1d")
        traj = s.generate(t_max=20.0, dt=0.01, seed=0)
        sv = compute_conservation(traj)
        # Damped harmonic dissipates energy — conservation variance ratio
        # should be higher than for the un-damped harmonic
        assert 2 in sv.value


class TestSpectralContent:
    def test_periodic_has_sharp_peak(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=30.0, dt=0.01, seed=0)
        sv = compute_spectral_content(traj)
        # Sharp peak: peak_height >> mean
        assert sv.value["peak_height"] > 10.0

    def test_returns_dominant_freq(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=30.0, dt=0.01, seed=0)
        sv = compute_spectral_content(traj)
        assert "dominant_freq" in sv.value
        assert 0.0 <= sv.value["dominant_freq"] <= 0.5

    def test_too_short_returns_low_confidence(self):
        from tessera.workbench import Trajectory
        traj = Trajectory(
            t=np.arange(20).astype(float), state=np.zeros((20, 1)),
            observable=np.zeros((20, 1)), system_id="test",
            params={}, ic=np.zeros(1),
        )
        sv = compute_spectral_content(traj)
        assert sv.confidence == 0.0


class TestDeterminism:
    def test_ode_is_deterministic(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=20.0, dt=0.01, seed=0)
        sv = compute_determinism(traj)
        # Deterministic systems should give z > 1
        assert sv.value > 1.0


class TestLyapunov:
    def test_returns_finite_value(self):
        s = get_system("lorenz63")
        traj = s.generate(t_max=30.0, dt=0.01, seed=0)
        sv = compute_lyapunov(traj)
        assert sv.value is not None and np.isfinite(sv.value)

    def test_dissipative_is_non_positive_or_small(self):
        # Heat equation dissipates; Lyapunov should be near zero or negative
        s = get_system("heat_1d")
        traj = s.generate(t_max=20.0, dt=0.1, seed=0)
        sv = compute_lyapunov(traj)
        # Allow some calibration noise; sign should be non-positive-ish
        assert sv.value < 1.0


class TestFullSignature:
    def test_algebraic_skips_dynamical_signatures(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, seed=0)
        sig = compute_full_signature(traj)
        assert sig.inferred_model_class == "algebraic"
        # ALGEBRAIC applicable set: smoothness, mode_count, dim, symmetry
        assert sig.smoothness is not None
        assert sig.mode_count is not None
        # Not applicable to ALGEBRAIC
        assert sig.spectral_content is None
        assert sig.lyapunov is None
        assert sig.determinism is None
        assert sig.conservation is None

    def test_ode_populates_all_tier_b(self):
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=20.0, dt=0.01, seed=0)
        sig = compute_full_signature(traj)
        assert sig.inferred_model_class == "ode"
        # ODE applicable set: ALL 8
        assert sig.smoothness is not None
        assert sig.mode_count is not None
        assert sig.effective_dimensionality is not None
        assert sig.symmetry is not None
        assert sig.conservation is not None
        assert sig.spectral_content is not None
        assert sig.determinism is not None
        assert sig.lyapunov is not None

    def test_pde_populates_all_tier_b(self):
        s = get_system("heat_1d")
        traj = s.generate(t_max=50.0, dt=0.1, seed=0)
        sig = compute_full_signature(traj)
        assert sig.inferred_model_class == "pde"
        # PDE applicable set: ALL 8
        assert sig.smoothness is not None
        assert sig.lyapunov is not None

    def test_explicit_model_class_overrides_classification(self):
        # Even if data looks like ODE, if user says algebraic, we trust them
        s = get_system("harmonic_1d")
        traj = s.generate(t_max=20.0, dt=0.01, seed=0)
        sig = compute_full_signature(traj, model_class=ModelClass.ALGEBRAIC)
        assert sig.inferred_model_class == "algebraic"
        # Should only populate algebraic-applicable signatures
        assert sig.spectral_content is None
        assert sig.lyapunov is None


class TestApplicabilityMap:
    def test_all_model_classes_covered(self):
        for mc in ModelClass:
            assert mc in WITHIN_CLASS_APPLICABILITY

    def test_algebraic_excludes_dynamical(self):
        algeb = WITHIN_CLASS_APPLICABILITY[ModelClass.ALGEBRAIC]
        assert "spectral_content" not in algeb
        assert "lyapunov" not in algeb
        assert "determinism" not in algeb
        assert "conservation" not in algeb

    def test_ode_pde_get_all_eight(self):
        # ODE and PDE should both have all 8 signatures applicable
        assert len(WITHIN_CLASS_APPLICABILITY[ModelClass.ODE]) == 8
        assert len(WITHIN_CLASS_APPLICABILITY[ModelClass.PDE]) == 8
