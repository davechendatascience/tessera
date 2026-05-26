"""The 10 canonical systems for the methodology workbench.

Each system implements `CanonicalSystem`. The registry at the bottom of
this module is the source of truth — `get_system(id)` and `list_systems()`
provide the public API.

Per Section 4 of the Stage 0 design note, this list spans ODE/PDE,
conservative/dissipative, single-mode/multi-mode, polynomial/trig,
single-equation/coupled — the discriminations the identification
pipeline must learn to make.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .integrators import (
    rk4_integrate, forward_euler_pde_1d, apply_observation_noise,
)
from .types import (
    CanonicalSystem, InformationRequirements, Trajectory,
    validate_symmetries, validate_conservation, validate_smoothness,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _resolve_params(self, params):
    out = self.default_params()
    if params is not None:
        out.update(params)
    return out


def _resolve_ic(self, ic):
    return self.default_ic() if ic is None else np.asarray(ic, dtype=np.float64)


# ----------------------------------------------------------------------
# 1. Harmonic oscillator
# ----------------------------------------------------------------------

class HarmonicOscillator1D(CanonicalSystem):
    id = "harmonic_1d"
    domain = "ode"
    dynamics_doc = "ddot{x} = -omega^2 * x  (state: [x, dx/dt])"
    state_dim = 2
    observable_dim = 2
    parameters = {"omega": (0.5, 5.0)}
    symmetries = validate_symmetries(["time_translation", "time_reversal", "reflection"])
    conservation_laws = validate_conservation(["energy"])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1
    info_min = InformationRequirements(
        min_samples=50, min_trajectories=1, noise_max=0.1,
        identifiability_proof=None,
    )

    def default_params(self):
        return {"omega": 1.0}

    def default_ic(self):
        return np.array([1.0, 0.0], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=20.0, dt=0.01,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        omega = p["omega"]

        def rhs(y):
            return np.array([y[1], -(omega ** 2) * y[0]])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 2. Damped harmonic oscillator
# ----------------------------------------------------------------------

class DampedHarmonicOscillator1D(CanonicalSystem):
    id = "damped_harmonic_1d"
    domain = "ode"
    dynamics_doc = "ddot{x} = -omega^2 * x - gamma * dx/dt  (state: [x, dx/dt])"
    state_dim = 2
    observable_dim = 2
    parameters = {"omega": (0.5, 5.0), "gamma": (0.0, 1.0)}
    symmetries = validate_symmetries(["time_translation", "reflection"])
    conservation_laws = validate_conservation([])  # dissipation breaks energy conservation
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1
    info_min = InformationRequirements(
        min_samples=100, min_trajectories=1, noise_max=0.05,
    )

    def default_params(self):
        return {"omega": 1.0, "gamma": 0.2}

    def default_ic(self):
        return np.array([1.0, 0.0], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=20.0, dt=0.01,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        omega = p["omega"]
        gamma = p["gamma"]

        def rhs(y):
            return np.array([y[1], -(omega ** 2) * y[0] - gamma * y[1]])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 3. Van der Pol oscillator
# ----------------------------------------------------------------------

class VanDerPol(CanonicalSystem):
    id = "vdp"
    domain = "ode"
    dynamics_doc = "ddot{x} = mu * (1 - x^2) * dx/dt - x  (state: [x, dx/dt])"
    state_dim = 2
    observable_dim = 2
    parameters = {"mu": (0.1, 5.0)}
    symmetries = validate_symmetries(["time_translation", "reflection"])
    conservation_laws = validate_conservation([])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1  # single stable limit cycle (post-transient)
    info_min = InformationRequirements(
        min_samples=200, min_trajectories=1, noise_max=0.05,
        excitation_requirements=["non-equilibrium IC required (origin is unstable)"],
    )

    def default_params(self):
        return {"mu": 1.0}

    def default_ic(self):
        return np.array([2.0, 0.0], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=30.0, dt=0.01,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        mu = p["mu"]

        def rhs(y):
            x, v = y[0], y[1]
            return np.array([v, mu * (1.0 - x * x) * v - x])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 4. Lorenz-63
# ----------------------------------------------------------------------

class Lorenz63(CanonicalSystem):
    id = "lorenz63"
    domain = "ode"
    dynamics_doc = (
        "dx/dt = sigma * (y - x); dy/dt = x * (rho - z) - y; "
        "dz/dt = x * y - beta * z  (state: [x, y, z])"
    )
    state_dim = 3
    observable_dim = 3
    parameters = {"sigma": (5.0, 15.0), "rho": (20.0, 40.0), "beta": (1.0, 4.0)}
    symmetries = validate_symmetries(["time_translation"])
    conservation_laws = validate_conservation([])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1  # single chaotic attractor (lobes are not separate modes)
    info_min = InformationRequirements(
        min_samples=2000, min_trajectories=1, noise_max=0.02,
        excitation_requirements=["transient skip needed to reach attractor"],
    )

    def default_params(self):
        return {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

    def default_ic(self):
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=30.0, dt=0.01,
                 noise_std=0.0, seed=None, transient_skip: float = 2.0,
                 **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        sigma, rho, beta = p["sigma"], p["rho"], p["beta"]

        def rhs(y):
            x, yy, z = y[0], y[1], y[2]
            return np.array([sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z])

        t_full, states_full = rk4_integrate(rhs, y0, t_max + transient_skip, dt)
        n_skip = int(round(transient_skip / dt))
        t = t_full[n_skip:] - t_full[n_skip]
        states = states_full[n_skip:]
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
            meta={"transient_skip": transient_skip},
        )


# ----------------------------------------------------------------------
# 5. FitzHugh-Nagumo
# ----------------------------------------------------------------------

class FitzHughNagumo(CanonicalSystem):
    id = "fhn"
    domain = "ode"
    dynamics_doc = (
        "dv/dt = v - v^3/3 - w + I; "
        "dw/dt = epsilon * (v + a - b * w)  (state: [v, w])"
    )
    state_dim = 2
    observable_dim = 2
    parameters = {
        "epsilon": (0.05, 0.2), "a": (0.5, 1.0),
        "b": (0.5, 1.0), "I": (0.0, 1.0),
    }
    symmetries = validate_symmetries(["time_translation"])
    conservation_laws = validate_conservation([])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1  # excitable or oscillatory depending on parameters
    info_min = InformationRequirements(
        min_samples=300, min_trajectories=2, noise_max=0.03,
        excitation_requirements=[
            "multiple IC needed to distinguish excitable vs oscillatory regimes",
        ],
    )

    def default_params(self):
        # I=0.5 puts us in oscillatory regime (limit cycle)
        return {"epsilon": 0.08, "a": 0.7, "b": 0.8, "I": 0.5}

    def default_ic(self):
        return np.array([0.0, 0.0], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=200.0, dt=0.05,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        eps, a, b, I = p["epsilon"], p["a"], p["b"], p["I"]

        def rhs(y):
            v, w = y[0], y[1]
            return np.array([v - (v ** 3) / 3.0 - w + I, eps * (v + a - b * w)])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 6. 1D Heat equation
# ----------------------------------------------------------------------

class Heat1D(CanonicalSystem):
    id = "heat_1d"
    domain = "pde"
    dynamics_doc = "du/dt = alpha * d^2u/dx^2 with Dirichlet BC u(boundary) = 0"
    state_dim = 64       # default grid size; configurable via generate(n_x=...)
    observable_dim = 64
    parameters = {"alpha": (0.01, 0.2)}
    symmetries = validate_symmetries(["time_translation"])
    conservation_laws = validate_conservation([])  # dissipative
    smoothness_class = validate_smoothness("c_infty")
    mode_count = 1
    info_min = InformationRequirements(
        min_samples=100, min_trajectories=3, noise_max=0.02,
        excitation_requirements=[
            "multi-trajectory (varying IC) required for mechanism vs aggregate disambiguation",
        ],
    )

    def default_params(self):
        return {"alpha": 0.05}

    def default_ic(self, n_x: int = 64, amplitude: float = 10.0):
        xs = np.arange(n_x, dtype=np.float64) - n_x / 2
        u0 = amplitude * np.exp(-(xs ** 2) / (2 * 5.0 ** 2))
        u0[0] = 0.0
        u0[-1] = 0.0
        return u0

    def generate(self, *, params=None, ic=None, t_max=20.0, dt=0.1,
                 noise_std=0.0, seed=None, n_x: int = 64,
                 **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        alpha = p["alpha"]
        u0 = self.default_ic(n_x=n_x) if ic is None else np.asarray(ic, dtype=np.float64)
        n_x_actual = u0.shape[0]

        def rhs(u):
            lap = np.zeros_like(u)
            lap[1:-1] = u[:-2] - 2.0 * u[1:-1] + u[2:]
            return alpha * lap

        t, fields = forward_euler_pde_1d(rhs, u0, t_max, dt)
        obs = apply_observation_noise(fields, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=fields, observable=obs, system_id=self.id,
            params=p, ic=u0, noise_std=noise_std, seed=seed,
            meta={"n_x": n_x_actual},
        )


# ----------------------------------------------------------------------
# 7. 1D Burgers' equation (viscous)
# ----------------------------------------------------------------------

class Burgers1D(CanonicalSystem):
    id = "burgers_1d"
    domain = "pde"
    dynamics_doc = (
        "du/dt = -u * du/dx + nu * d^2u/dx^2 (viscous Burgers; "
        "upwind for advection, centered for diffusion, periodic BC)"
    )
    state_dim = 128
    observable_dim = 128
    parameters = {"nu": (0.005, 0.1)}
    symmetries = validate_symmetries(["space_translation", "time_translation"])
    conservation_laws = validate_conservation([])  # viscous dissipation
    smoothness_class = validate_smoothness("c_2")  # shock formation softened by viscosity
    mode_count = 1
    info_min = InformationRequirements(
        min_samples=200, min_trajectories=3, noise_max=0.01,
        excitation_requirements=["shock-forming IC (e.g., sinusoidal initial profile)"],
    )

    def default_params(self):
        return {"nu": 0.02}

    def default_ic(self, n_x: int = 128):
        xs = np.linspace(0.0, 2.0 * np.pi, n_x, endpoint=False)
        return np.sin(xs)

    def generate(self, *, params=None, ic=None, t_max=2.0, dt=0.005,
                 noise_std=0.0, seed=None, n_x: int = 128,
                 **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        nu = p["nu"]
        u0 = self.default_ic(n_x=n_x) if ic is None else np.asarray(ic, dtype=np.float64)
        n_x_actual = u0.shape[0]
        L = 2.0 * np.pi
        dx = L / n_x_actual

        def rhs(u):
            up = np.roll(u, -1)
            um = np.roll(u, 1)
            adv = np.where(u >= 0, u * (u - um) / dx, u * (up - u) / dx)
            diff = (up - 2.0 * u + um) / (dx ** 2)
            return -adv + nu * diff

        t, fields = forward_euler_pde_1d(rhs, u0, t_max, dt)
        obs = apply_observation_noise(fields, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=fields, observable=obs, system_id=self.id,
            params=p, ic=u0, noise_std=noise_std, seed=seed,
            meta={"n_x": n_x_actual, "L": L, "dx": dx},
        )


# ----------------------------------------------------------------------
# 8. Linear pendulum (small-angle)
# ----------------------------------------------------------------------

class LinearPendulum(CanonicalSystem):
    id = "linear_pendulum"
    domain = "ode"
    dynamics_doc = "ddot{theta} = -(g/L) * theta  (small-angle approx; state: [theta, dtheta/dt])"
    state_dim = 2
    observable_dim = 2
    parameters = {"g_over_L": (1.0, 10.0)}
    symmetries = validate_symmetries(["time_translation", "time_reversal", "reflection"])
    conservation_laws = validate_conservation(["energy"])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1
    info_min = InformationRequirements(
        min_samples=50, min_trajectories=1, noise_max=0.1,
        identifiability_proof="trivially identifiable; period gives g/L exactly",
    )

    def default_params(self):
        return {"g_over_L": 9.81}

    def default_ic(self):
        return np.array([0.1, 0.0], dtype=np.float64)  # small initial angle

    def generate(self, *, params=None, ic=None, t_max=10.0, dt=0.01,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        gL = p["g_over_L"]

        def rhs(y):
            return np.array([y[1], -gL * y[0]])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 9. Nonlinear pendulum (full sin)
# ----------------------------------------------------------------------

class NonlinearPendulum(CanonicalSystem):
    id = "nonlinear_pendulum"
    domain = "ode"
    dynamics_doc = "ddot{theta} = -(g/L) * sin(theta)  (full pendulum; state: [theta, dtheta/dt])"
    state_dim = 2
    observable_dim = 2
    parameters = {"g_over_L": (1.0, 10.0)}
    symmetries = validate_symmetries(["time_translation", "time_reversal", "reflection"])
    conservation_laws = validate_conservation(["energy"])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 2  # libration (bound) vs rotation (unbound) — separated by separatrix
    info_min = InformationRequirements(
        min_samples=200, min_trajectories=3, noise_max=0.05,
        excitation_requirements=[
            "multiple amplitudes required to distinguish from linear pendulum",
            "large-amplitude IC needed to see trig nonlinearity",
        ],
    )

    def default_params(self):
        return {"g_over_L": 9.81}

    def default_ic(self):
        return np.array([1.5, 0.0], dtype=np.float64)  # large-amplitude libration

    def generate(self, *, params=None, ic=None, t_max=10.0, dt=0.01,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        gL = p["g_over_L"]

        def rhs(y):
            return np.array([y[1], -gL * np.sin(y[0])])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# 10. Kepler (2D inverse-square central force)
# ----------------------------------------------------------------------

class Kepler2D(CanonicalSystem):
    id = "kepler"
    domain = "ode"
    dynamics_doc = (
        "2D Kepler problem: r''(t) = -GM * r / |r|^3  "
        "(state: [x, y, dx/dt, dy/dt])"
    )
    state_dim = 4
    observable_dim = 4
    parameters = {"GM": (0.5, 2.0)}
    symmetries = validate_symmetries(
        ["time_translation", "time_reversal", "rotation_so2"]
    )
    conservation_laws = validate_conservation(["energy", "angular_momentum"])
    smoothness_class = validate_smoothness("analytic")
    mode_count = 1  # bound elliptic orbit when E < 0 (single closed orbit)
    info_min = InformationRequirements(
        min_samples=500, min_trajectories=1, noise_max=0.01,
        excitation_requirements=["bound IC (E < 0) for closed orbit"],
        identifiability_proof="orbital elements + Kepler's third law",
    )

    def default_params(self):
        return {"GM": 1.0}

    def default_ic(self):
        # Bound elliptic orbit: start at (1, 0) with velocity (0, 0.8)
        # gives an ellipse around the origin with E < 0
        return np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float64)

    def generate(self, *, params=None, ic=None, t_max=20.0, dt=0.005,
                 noise_std=0.0, seed=None, **kwargs) -> Trajectory:
        p = _resolve_params(self, params)
        y0 = _resolve_ic(self, ic)
        GM = p["GM"]

        def rhs(y):
            x, yy, vx, vy = y[0], y[1], y[2], y[3]
            r2 = x * x + yy * yy
            r3 = r2 * np.sqrt(r2)
            return np.array([vx, vy, -GM * x / r3, -GM * yy / r3])

        t, states = rk4_integrate(rhs, y0, t_max, dt)
        obs = apply_observation_noise(states, noise_std, _rng(seed))
        return Trajectory(
            t=t, state=states, observable=obs, system_id=self.id,
            params=p, ic=y0, noise_std=noise_std, seed=seed,
        )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

REGISTRY: dict[str, CanonicalSystem] = {}


def register(system: CanonicalSystem) -> None:
    """Add a canonical system to the registry.

    System IDs must be unique. Re-registering is an error (intentional —
    catches accidental redefinition).
    """
    if system.id in REGISTRY:
        raise ValueError(f"System {system.id!r} already registered")
    REGISTRY[system.id] = system


def get_system(system_id: str) -> CanonicalSystem:
    """Look up a registered canonical system by id."""
    if system_id not in REGISTRY:
        raise KeyError(
            f"Unknown system {system_id!r}. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[system_id]


def list_systems() -> list[str]:
    """Return the IDs of all registered systems, in registration order."""
    return list(REGISTRY.keys())


# Register all 10 canonical systems.
for _cls in [
    HarmonicOscillator1D,
    DampedHarmonicOscillator1D,
    VanDerPol,
    Lorenz63,
    FitzHughNagumo,
    Heat1D,
    Burgers1D,
    LinearPendulum,
    NonlinearPendulum,
    Kepler2D,
]:
    register(_cls())
