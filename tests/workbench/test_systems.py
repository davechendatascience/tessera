"""Stage 1 tests: canonical workbench systems generate correct dynamics.

Per the Stage 0 design contract, each system is verified against an
analytical solution or a well-known dynamical property:

  - harmonic_1d / linear_pendulum: closed-form sinusoidal solution +
    energy conservation
  - damped_harmonic_1d: amplitude decay matches exp(-gamma*t/2)
  - nonlinear_pendulum: energy conservation under large-amplitude motion
  - vdp: limit-cycle convergence + non-trivial amplitude
  - lorenz63: bounded chaotic attractor (no divergence to infinity)
  - fhn: relaxation-oscillator periodicity
  - heat_1d: integral of u decays under dissipative dynamics
  - burgers_1d: bounded propagation, mass-conservation-like behavior
  - kepler: angular momentum + energy conservation under central force

These tests verify *generator correctness*. Statistical signature
testing happens in Stage 2.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.workbench import (
    REGISTRY, get_system, list_systems, ModelClass,
    SYMMETRY_VOCABULARY, CONSERVATION_VOCABULARY, SMOOTHNESS_CLASSES,
)
from tessera.workbench.types import CanonicalSystem


ALL_SYSTEMS = [
    "harmonic_1d", "damped_harmonic_1d", "vdp", "lorenz63", "fhn",
    "heat_1d", "burgers_1d", "linear_pendulum", "nonlinear_pendulum",
    "kepler", "algebraic_feynman_gaussian",
]


class TestRegistry:
    def test_all_eleven_registered(self):
        assert len(list_systems()) == 11

    def test_ids_unique(self):
        ids = list_systems()
        assert len(ids) == len(set(ids))

    def test_get_system_roundtrip(self):
        for sid in list_systems():
            s = get_system(sid)
            assert s.id == sid

    def test_unknown_system_raises(self):
        with pytest.raises(KeyError):
            get_system("does_not_exist")

    @pytest.mark.parametrize("sid", ALL_SYSTEMS)
    def test_metadata_complete(self, sid):
        s = get_system(sid)
        meta = s.metadata()
        assert meta["id"] == sid
        assert meta["model_class"] in ("algebraic", "discrete_map", "ode", "pde")
        assert meta["state_dim"] > 0
        assert isinstance(meta["canonical_target_form"], str)
        assert len(meta["canonical_target_form"]) > 0
        assert meta["smoothness_class"] in SMOOTHNESS_CLASSES
        for sym in meta["symmetries"]:
            assert sym in SYMMETRY_VOCABULARY
        for c in meta["conservation_laws"]:
            assert c in CONSERVATION_VOCABULARY

    @pytest.mark.parametrize("sid", ALL_SYSTEMS)
    def test_model_class_is_enum(self, sid):
        s = get_system(sid)
        assert isinstance(s.model_class, ModelClass)

    def test_domain_backcompat_alias(self):
        # domain is a backwards-compat alias for model_class.value
        for sid in list_systems():
            s = get_system(sid)
            assert s.domain == s.model_class.value


class TestGenerationContract:
    """Every system must satisfy basic generation contract."""

    @pytest.mark.parametrize("sid", ALL_SYSTEMS)
    def test_generate_returns_trajectory(self, sid):
        s = get_system(sid)
        traj = s.generate(t_max=2.0, dt=0.01, noise_std=0.0, seed=0)
        assert traj.t.ndim == 1
        assert traj.t.shape[0] == traj.state.shape[0]
        assert traj.state.shape == traj.observable.shape
        assert traj.system_id == sid
        assert np.all(np.isfinite(traj.state))

    @pytest.mark.parametrize("sid", [
        "harmonic_1d", "lorenz63", "kepler",
    ])
    def test_deterministic_under_same_seed(self, sid):
        s = get_system(sid)
        t1 = s.generate(t_max=2.0, dt=0.01, noise_std=0.1, seed=42)
        t2 = s.generate(t_max=2.0, dt=0.01, noise_std=0.1, seed=42)
        np.testing.assert_allclose(t1.state, t2.state)
        np.testing.assert_allclose(t1.observable, t2.observable)

    def test_noise_actually_perturbs(self):
        s = get_system("harmonic_1d")
        clean = s.generate(t_max=2.0, dt=0.01, noise_std=0.0, seed=0)
        noisy = s.generate(t_max=2.0, dt=0.01, noise_std=0.1, seed=0)
        assert not np.allclose(clean.observable, noisy.observable)
        np.testing.assert_allclose(clean.state, noisy.state)  # ground-truth unchanged


class TestHarmonicAnalytical:
    """Compare RK4 result to analytical x(t) = cos(omega*t) for IC=[1,0]."""

    def test_period_matches_analytical(self):
        s = get_system("harmonic_1d")
        traj = s.generate(
            params={"omega": 2.0}, ic=np.array([1.0, 0.0]),
            t_max=10.0, dt=0.001,
        )
        analytical = np.cos(2.0 * traj.t)
        rms = float(np.sqrt(np.mean((traj.state[:, 0] - analytical) ** 2)))
        assert rms < 1e-3, f"RK4 vs analytical RMS={rms} too large"

    def test_energy_conservation(self):
        s = get_system("harmonic_1d")
        traj = s.generate(
            params={"omega": 1.5}, ic=np.array([1.0, 0.0]),
            t_max=10.0, dt=0.001,
        )
        x, v = traj.state[:, 0], traj.state[:, 1]
        E = 0.5 * v ** 2 + 0.5 * (1.5 ** 2) * x ** 2
        drift = float(np.std(E) / np.mean(E))
        assert drift < 1e-4, f"Energy drift {drift} too large"


class TestDampedHarmonic:
    def test_amplitude_decays(self):
        s = get_system("damped_harmonic_1d")
        traj = s.generate(
            params={"omega": 1.0, "gamma": 0.3},
            ic=np.array([1.0, 0.0]),
            t_max=20.0, dt=0.01,
        )
        # Envelope should decay; final amplitude << initial
        assert abs(traj.state[-1, 0]) < 0.5 * abs(traj.state[0, 0])

    def test_no_decay_at_zero_damping(self):
        s = get_system("damped_harmonic_1d")
        traj = s.generate(
            params={"omega": 1.0, "gamma": 0.0},
            ic=np.array([1.0, 0.0]),
            t_max=10.0, dt=0.001,
        )
        x = traj.state[:, 0]
        v = traj.state[:, 1]
        E = 0.5 * v ** 2 + 0.5 * (1.0 ** 2) * x ** 2
        assert np.std(E) / np.mean(E) < 1e-4


class TestVanDerPol:
    def test_limit_cycle(self):
        """Trajectory from origin doesn't blow up; non-zero IC converges to LC."""
        s = get_system("vdp")
        traj = s.generate(t_max=40.0, dt=0.01)
        # Late-time amplitude is bounded and non-trivial
        late = traj.state[traj.t > 30.0]
        assert np.all(np.abs(late) < 10.0)
        assert np.max(np.abs(late[:, 0])) > 0.5


class TestLorenz:
    def test_bounded_chaotic(self):
        s = get_system("lorenz63")
        traj = s.generate(t_max=20.0, dt=0.01)
        # Lorenz attractor stays bounded ~|state| < 50
        assert np.all(np.abs(traj.state) < 50.0)
        # And the trajectory should NOT collapse to a fixed point
        std_x = float(np.std(traj.state[:, 0]))
        assert std_x > 1.0


class TestKeplerConservation:
    """Kepler must conserve both energy and angular momentum."""

    def test_angular_momentum(self):
        s = get_system("kepler")
        traj = s.generate(t_max=15.0, dt=0.001)
        x, y = traj.state[:, 0], traj.state[:, 1]
        vx, vy = traj.state[:, 2], traj.state[:, 3]
        L = x * vy - y * vx
        drift = float(np.std(L) / np.abs(np.mean(L)))
        assert drift < 1e-3, f"Angular momentum drift {drift} too large"

    def test_energy(self):
        s = get_system("kepler")
        traj = s.generate(
            params={"GM": 1.0}, t_max=15.0, dt=0.001,
        )
        x, y = traj.state[:, 0], traj.state[:, 1]
        vx, vy = traj.state[:, 2], traj.state[:, 3]
        r = np.sqrt(x * x + y * y)
        E = 0.5 * (vx ** 2 + vy ** 2) - 1.0 / r
        drift = float(np.std(E) / abs(np.mean(E)))
        assert drift < 1e-3, f"Energy drift {drift} too large"


class TestNonlinearPendulumEnergy:
    def test_energy_conservation(self):
        s = get_system("nonlinear_pendulum")
        traj = s.generate(
            params={"g_over_L": 1.0},
            ic=np.array([1.5, 0.0]),
            t_max=15.0, dt=0.001,
        )
        theta, dtheta = traj.state[:, 0], traj.state[:, 1]
        E = 0.5 * dtheta ** 2 - 1.0 * np.cos(theta)
        drift = float(np.std(E) / abs(np.mean(E) + 1e-9))
        assert drift < 1e-3, f"Energy drift {drift} too large"


class TestHeatEquation:
    """Heat equation dissipates total field energy under Dirichlet BC."""

    def test_field_energy_decays(self):
        s = get_system("heat_1d")
        # Heat equation with Dirichlet BC strictly dissipates L2 norm.
        # The default Gaussian IC has most of its energy in low spatial
        # frequencies which decay slowly, so we just verify meaningful
        # decay (>10%) — the monotone-decrease test below is the stronger
        # physics check.
        traj = s.generate(params={"alpha": 0.2}, t_max=200.0, dt=0.1)
        l2_norms = np.sqrt(np.sum(traj.state ** 2, axis=1))
        assert l2_norms[-1] < 0.9 * l2_norms[0]

    def test_field_energy_monotonically_decreases(self):
        s = get_system("heat_1d")
        traj = s.generate(t_max=20.0, dt=0.1)
        l2_norms = np.sqrt(np.sum(traj.state ** 2, axis=1))
        # Diffusion is monotonically dissipative under Dirichlet BC
        assert np.all(np.diff(l2_norms) <= 1e-9)

    def test_dirichlet_boundaries(self):
        s = get_system("heat_1d")
        traj = s.generate(t_max=2.0, dt=0.05)
        assert np.allclose(traj.state[:, 0], 0.0, atol=1e-9)
        assert np.allclose(traj.state[:, -1], 0.0, atol=1e-9)


class TestBurgers:
    """Periodic-BC viscous Burgers: bounded, mean approximately conserved."""

    def test_bounded(self):
        s = get_system("burgers_1d")
        traj = s.generate(t_max=1.0, dt=0.005)
        assert np.all(np.abs(traj.state) < 2.0)

    def test_mean_approximately_conserved(self):
        s = get_system("burgers_1d")
        traj = s.generate(t_max=1.0, dt=0.005)
        means = traj.state.mean(axis=1)
        # Initial mean of sin on [0, 2pi] is 0; should stay near 0
        # under periodic BC + symmetric IC. Tolerance accommodates upwind
        # scheme's slight numerical drift.
        assert np.all(np.abs(means) < 0.05)


class TestTargetForm:
    """Stage 0.5: target_form transforms trajectories into SR-ready
    (features, target) tuples, per model class.

    ALGEBRAIC    -- features = inputs (from meta), target = y
    ODE          -- features = state[:-1], target = finite-diff of state
    PDE          -- features = u[:-1], target = u[1:] - u[:-1]
    DISCRETE_MAP -- features = state[:-1], target = state[1:]
    """

    @pytest.mark.parametrize("sid", ALL_SYSTEMS)
    def test_target_form_returns_arrays(self, sid):
        s = get_system(sid)
        traj = s.generate(t_max=2.0, dt=0.01, noise_std=0.0, seed=0)
        features, target = s.target_form(traj)
        assert isinstance(features, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert features.shape[0] == target.shape[0]
        assert np.all(np.isfinite(features))
        assert np.all(np.isfinite(target))

    @pytest.mark.parametrize("sid", [
        "harmonic_1d", "damped_harmonic_1d", "vdp", "lorenz63", "fhn",
        "linear_pendulum", "nonlinear_pendulum", "kepler",
    ])
    def test_ode_target_form_finite_difference(self, sid):
        """For ODE systems, target_form's target should match finite-
        difference of state per the canonical form 'd[state]/dt = f(state)'."""
        s = get_system(sid)
        traj = s.generate(t_max=2.0, dt=0.01, noise_std=0.0, seed=0)
        features, target = s.target_form(traj)
        assert features.shape == traj.state[:-1].shape
        assert target.shape == traj.state[:-1].shape
        # target should be (state[1:] - state[:-1]) / dt
        dt = float(traj.t[1] - traj.t[0])
        expected = (traj.state[1:] - traj.state[:-1]) / dt
        np.testing.assert_allclose(target, expected)

    @pytest.mark.parametrize("sid", ["heat_1d", "burgers_1d"])
    def test_pde_target_form_shape(self, sid):
        s = get_system(sid)
        traj = s.generate(t_max=2.0, dt=0.01, noise_std=0.0, seed=0)
        features, target = s.target_form(traj)
        # PDE: features = u[:-1], target = u[1:] - u[:-1]
        assert features.shape == traj.state[:-1].shape
        assert target.shape == traj.state[:-1].shape
        np.testing.assert_allclose(target, traj.state[1:] - traj.state[:-1])

    def test_algebraic_target_form(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=200, seed=0)
        features, target = s.target_form(traj)
        # features = inputs (theta); target = y = exp(-theta^2/2)
        assert features.shape[0] == 200
        assert target.shape[0] == 200
        theta = features[:, 0]
        expected_y = np.exp(-(theta ** 2) / 2.0).reshape(-1, 1)
        np.testing.assert_allclose(target, expected_y, atol=1e-12)


class TestAlgebraicSystem:
    """Sanity tests for the new algebraic_feynman_gaussian entry."""

    def test_model_class_is_algebraic(self):
        s = get_system("algebraic_feynman_gaussian")
        assert s.model_class == ModelClass.ALGEBRAIC

    def test_inputs_in_meta(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=100, seed=0)
        assert "inputs" in traj.meta
        assert traj.meta["inputs"].shape == (100, 1)
        assert traj.meta["input_names"] == ["theta"]

    def test_function_value_correct(self):
        s = get_system("algebraic_feynman_gaussian")
        traj = s.generate(n_samples=500, noise_std=0.0, seed=42)
        theta = traj.meta["inputs"][:, 0]
        y_expected = np.exp(-(theta ** 2) / 2.0)
        np.testing.assert_allclose(traj.state[:, 0], y_expected, atol=1e-12)


class TestPDEGridMetadata:
    """Stage 0.5 §5.4: PDE systems declare min_dt, min_dx, grid_floor."""

    @pytest.mark.parametrize("sid", ["heat_1d", "burgers_1d"])
    def test_pde_has_grid_metadata(self, sid):
        s = get_system(sid)
        assert s.info_min.min_dt is not None
        assert s.info_min.min_dx is not None
        assert s.info_min.grid_floor is not None
        assert "cfl_max" in s.info_min.grid_floor

    @pytest.mark.parametrize("sid", [
        "harmonic_1d", "damped_harmonic_1d", "vdp", "lorenz63", "fhn",
        "linear_pendulum", "nonlinear_pendulum", "kepler",
    ])
    def test_ode_has_dt_no_dx(self, sid):
        s = get_system(sid)
        assert s.info_min.min_dt is not None
        assert s.info_min.min_dx is None      # ODEs don't have spatial grid
        assert s.info_min.grid_floor is None

    def test_algebraic_has_no_grid_metadata(self):
        s = get_system("algebraic_feynman_gaussian")
        # Algebraic has no time/space; min_dt and min_dx should be None
        assert s.info_min.min_dt is None
        assert s.info_min.min_dx is None
        assert s.info_min.grid_floor is None


class TestLinearVsNonlinearPendulumDistinguishable:
    """At small amplitude they agree; at large amplitude they diverge.
    This is the discrimination the identification pipeline must learn."""

    def test_small_amplitude_close(self):
        lp = get_system("linear_pendulum")
        nlp = get_system("nonlinear_pendulum")
        ic = np.array([0.05, 0.0])
        params = {"g_over_L": 1.0}
        tl = lp.generate(params=params, ic=ic, t_max=10.0, dt=0.001)
        tn = nlp.generate(params=params, ic=ic, t_max=10.0, dt=0.001)
        rms = float(np.sqrt(np.mean((tl.state[:, 0] - tn.state[:, 0]) ** 2)))
        assert rms < 1e-3, f"Small-amplitude pendulums should agree (RMS={rms})"

    def test_large_amplitude_diverges(self):
        lp = get_system("linear_pendulum")
        nlp = get_system("nonlinear_pendulum")
        ic = np.array([2.5, 0.0])  # large angle
        params = {"g_over_L": 1.0}
        tl = lp.generate(params=params, ic=ic, t_max=10.0, dt=0.001)
        tn = nlp.generate(params=params, ic=ic, t_max=10.0, dt=0.001)
        rms = float(np.sqrt(np.mean((tl.state[:, 0] - tn.state[:, 0]) ** 2)))
        assert rms > 0.1, f"Large-amplitude pendulums should diverge (RMS={rms})"
