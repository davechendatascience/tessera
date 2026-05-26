"""Minimal integrators for workbench generators.

Pure-numpy RK4 for ODEs and forward-Euler for 1D PDEs. We avoid scipy
to keep the workbench dependency-light and integrators deterministic
in pure numpy (no fortran-call nondeterminism across platforms).

For chaotic systems (e.g., Lorenz), RK4 at dt=0.01 is sufficient for
the methodology workbench's purposes — we don't need finite-time
attractor exactness, only consistent trajectories for SR fitting.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical RK4 for autonomous ODEs.

    Parameters
    ----------
    rhs : callable(y) -> dy/dt
    y0 : initial state, shape (state_dim,)
    t_max, dt : integration window and step

    Returns
    -------
    t : shape (n_steps + 1,)
    states : shape (n_steps + 1, state_dim)
    """
    n_steps = int(round(t_max / dt))
    state_dim = y0.shape[0]
    states = np.zeros((n_steps + 1, state_dim), dtype=np.float64)
    states[0] = y0
    y = y0.copy()
    for i in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[i + 1] = y
    t = np.arange(n_steps + 1) * dt
    return t, states


def forward_euler_pde_1d(
    rhs: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    t_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler time stepping for 1D PDEs on a fixed spatial grid.

    Parameters
    ----------
    rhs : callable(u) -> du/dt, where u is the current field shape (n_x,)
    u0 : initial field, shape (n_x,)
    t_max, dt : integration window and step

    Returns
    -------
    t : shape (n_steps + 1,)
    fields : shape (n_steps + 1, n_x)
    """
    n_steps = int(round(t_max / dt))
    n_x = u0.shape[0]
    fields = np.zeros((n_steps + 1, n_x), dtype=np.float64)
    fields[0] = u0
    u = u0.copy()
    for i in range(n_steps):
        u = u + dt * rhs(u)
        fields[i + 1] = u
    t = np.arange(n_steps + 1) * dt
    return t, fields


def apply_observation_noise(
    state: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply iid Gaussian observation noise to a state array.

    Returns a copy; does not mutate input.
    """
    if noise_std <= 0.0:
        return state.copy()
    return state + noise_std * rng.standard_normal(state.shape)
