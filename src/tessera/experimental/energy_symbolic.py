"""Energy-based symbolic learning primitives (Conjecture E1).

Provenance: `docs/research/energy_based_symbolic_learning.md`.

Status: **UNTESTED** at module-add time. Gradient-free, GPU-oriented
(JAX). Implements the building blocks for the E1 experiments:

  - Synthetic Ising systems: random sparse couplings J + fields h,
    batch energy evaluation, and Gibbs sampling.
  - A GPU energy minimizer: ballistic Simulated Bifurcation (bSB,
    Goto/Tatsumura/Dixon 2019) — solves Ising by integrating a CLASSICAL
    HAMILTONIAN system (coupled nonlinear oscillators). No gradients, no
    quantum hardware; pure ODE integration, jittable + vmappable over
    replicas. This is the gradient-free optimizer the strategy note
    centres on.
  - Hamiltonian discovery, energy-labeled (E1a): recover H(s) from
    (s, energy) pairs. Because H = Σ Jᵢⱼ sᵢsⱼ + Σ hᵢ sᵢ is a degree-2
    additive polynomial, this reuses the C8 detector
    (tessera.experimental.additive_polynomial).

E1b (inverse Ising from samples only, pseudo-likelihood) is scaffolded
separately once E1a + the minimizer are validated.

Graduation criterion (E1): tessera recovers a sparse Ising
Hamiltonian's structure gradient-free, with coupling recovery clearly
above a non-energy baseline; AND the bSB minimizer finds ground states
of small instances (brute-force-checkable) reliably.

Removal criterion: bSB doesn't reliably minimize small Ising
instances, OR energy-labeled discovery (≈C8) fails.

Initial commit: 2026-05-29
Last evaluation: never

Note on JAX: every function here works on numpy too (uses
`tessera.backend.array_module` style where needed), but the minimizer
and sampler are written to be `jax.jit`/`vmap`-friendly so they run on
GPU. They accept an explicit `xp` (numpy or jax.numpy) where it matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------
# Synthetic Ising systems
# ---------------------------------------------------------------------

@dataclass
class IsingSystem:
    """A pairwise Ising Hamiltonian H(s) = -Σ_{i<j} J_ij s_i s_j - Σ_i h_i s_i.

    J is symmetric with zero diagonal (n, n); h is (n,). Spins s ∈ {-1,+1}.
    Sign convention: H is the energy to MINIMIZE (lower = more probable).
    """
    J: np.ndarray   # (n, n) symmetric, zero diagonal
    h: np.ndarray   # (n,)

    @property
    def n(self) -> int:
        return self.h.shape[0]


def random_ising(n: int, density: float = 0.3, seed: int = 0,
                 coupling_scale: float = 1.0,
                 field_scale: float = 0.5) -> IsingSystem:
    """Random sparse symmetric Ising system.

    `density` = fraction of off-diagonal pairs with a nonzero coupling.
    Couplings ~ N(0, coupling_scale); fields ~ N(0, field_scale).
    """
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.normal(0.0, coupling_scale)
                J[i, j] = w
                J[j, i] = w
    h = rng.normal(0.0, field_scale, size=n)
    return IsingSystem(J=J, h=h)


def ising_energy(s: np.ndarray, sys: IsingSystem) -> np.ndarray:
    """Energy of spin configs. `s` is (n,) or (batch, n) in {-1,+1}.

    H(s) = -0.5 sᵀJs - hᵀs  (the 0.5 because J is doubly-counted
    symmetric; equals -Σ_{i<j} J_ij s_i s_j - Σ_i h_i s_i).
    Returns scalar or (batch,).
    """
    s = np.asarray(s, dtype=np.float64)
    single = (s.ndim == 1)
    if single:
        s = s[None]
    quad = -0.5 * np.einsum("bi,ij,bj->b", s, sys.J, s)
    lin = -s @ sys.h
    e = quad + lin
    return e[0] if single else e


def sample_ising_gibbs(sys: IsingSystem, beta: float = 1.0,
                       n_samples: int = 500, n_sweeps: int = 50,
                       seed: int = 0) -> np.ndarray:
    """Gibbs-sample spin configs from p(s) ∝ exp(-beta·H(s)).

    Returns (n_samples, n) array in {-1,+1}. Numpy reference
    implementation (vectorized over samples; sequential over spins).
    Used to generate data for the E1b inverse problem.
    """
    n = sys.n
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_samples, n))
    for _ in range(n_sweeps):
        for i in range(n):
            # local field on spin i: h_i + Σ_j J_ij s_j
            local = sys.h[i] + s @ sys.J[i]           # (n_samples,)
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local))
            u = rng.random(n_samples)
            s[:, i] = np.where(u < p_up, 1.0, -1.0)
    return s


def brute_force_ground_state(sys: IsingSystem) -> tuple[np.ndarray, float]:
    """Exact ground state by enumeration. Only for small n (<= ~20)."""
    n = sys.n
    if n > 22:
        raise ValueError("brute force only for n <= 22")
    best_s, best_e = None, np.inf
    for code in range(1 << n):
        s = np.array([1.0 if (code >> b) & 1 else -1.0 for b in range(n)])
        e = float(ising_energy(s, sys))
        if e < best_e:
            best_e, best_s = e, s
    return best_s, best_e


# ---------------------------------------------------------------------
# Ballistic Simulated Bifurcation — GPU energy minimizer (gradient-free)
# ---------------------------------------------------------------------

def simulated_bifurcation(
    sys: IsingSystem,
    n_replicas: int = 64,
    n_steps: int = 400,
    dt: float = 0.5,
    c0: Optional[float] = None,
    seed: int = 0,
    use_jax: bool = False,
):
    """Ballistic Simulated Bifurcation (Goto et al. 2019).

    Minimizes the Ising energy by integrating a classical Hamiltonian
    system of coupled nonlinear oscillators. Gradient-free; the only
    "signal" is the local field Σ_j J_ij x_j (the same pairwise credit
    structure the energy encodes).

    Runs `n_replicas` independent trajectories from random initial
    conditions (vmapped on GPU when use_jax=True); returns the best
    spin config found and its energy.

    Returns (best_spins (n,), best_energy float).
    """
    n = sys.n
    if c0 is None:
        # Goto's recommended scale: c0 = 0.5 / (sqrt(n) * std(J))
        offdiag = sys.J[np.triu_indices(n, k=1)]
        j_std = float(np.std(offdiag)) if offdiag.size and np.std(offdiag) > 0 else 1.0
        c0 = 0.5 / (np.sqrt(n) * j_std)

    if use_jax:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            use_jax = False

    if use_jax:
        import jax
        import jax.numpy as jnp
        J = jnp.asarray(sys.J); h = jnp.asarray(sys.h)
        key = jax.random.PRNGKey(seed)
        x0 = 0.1 * jax.random.normal(key, (n_replicas, n))
        y0 = 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (n_replicas, n))
        a_sched = jnp.linspace(0.0, 1.0, n_steps)

        def step(carry, a):
            x, y = carry
            # dy = (-(1 - a) x + c0 (Jx + h)) dt ; dx = a0 y dt  (a0=1)
            field = x @ J + h                      # (R, n)
            y = y + dt * (-(1.0 - a) * x + c0 * field)
            x = x + dt * y
            # inelastic walls at ±1 (ballistic SB)
            over = jnp.abs(x) > 1.0
            x = jnp.clip(x, -1.0, 1.0)
            y = jnp.where(over, 0.0, y)
            return (x, y), None

        (xf, _), _ = jax.lax.scan(step, (x0, y0), a_sched)
        spins = jnp.sign(xf)
        spins = jnp.where(spins == 0, 1.0, spins)
        # energies
        quad = -0.5 * jnp.einsum("bi,ij,bj->b", spins, J, spins)
        lin = -spins @ h
        energies = np.asarray(quad + lin)
        spins = np.asarray(spins)
    else:
        rng = np.random.default_rng(seed)
        x = 0.1 * rng.standard_normal((n_replicas, n))
        y = 0.1 * rng.standard_normal((n_replicas, n))
        J = sys.J; h = sys.h
        for k in range(n_steps):
            a = k / max(n_steps - 1, 1)
            field = x @ J + h
            y = y + dt * (-(1.0 - a) * x + c0 * field)
            x = x + dt * y
            over = np.abs(x) > 1.0
            x = np.clip(x, -1.0, 1.0)
            y[over] = 0.0
        spins = np.sign(x)
        spins[spins == 0] = 1.0
        energies = ising_energy(spins, sys)

    best = int(np.argmin(energies))
    return spins[best], float(energies[best])


# ---------------------------------------------------------------------
# E1a — Hamiltonian discovery from (s, energy) pairs  (reuses C8)
# ---------------------------------------------------------------------

def discover_hamiltonian_energy_labeled(
    s: np.ndarray, energies: np.ndarray,
    *, r2_threshold: float = 0.99, top_n: int = 64,
):
    """Recover H(s) from (config, energy) pairs via the C8 additive-
    polynomial detector. H is a degree-2 additive polynomial in the
    spins (pairwise products + linear), so C8 is the right tool.

    `s`: (N, n) configs in {-1,+1}. `energies`: (N,).
    Returns the AdditivePolynomialFit (or None), plus the recovered
    coupling matrix estimate for comparison to ground truth.
    """
    from tessera.experimental.additive_polynomial import detect_additive_polynomial

    n = s.shape[1]
    env = {f"s{i}": s[:, i].astype(np.float64) for i in range(n)}
    fit = detect_additive_polynomial(
        env, np.asarray(energies, dtype=np.float64),
        max_degree=2, r2_threshold=r2_threshold, top_n=top_n,
    )
    return fit


def fit_to_coupling_matrix(fit, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (J_est, h_est) from an AdditivePolynomialFit over s0..s{n-1}.

    Linear term coeff on s_i → -h_i (sign per the H convention here);
    bilinear coeff on s_i·s_j → -J_ij. Returns (J_est, h_est).
    """
    J_est = np.zeros((n, n)); h_est = np.zeros(n)
    if fit is None:
        return J_est, h_est
    for coef, exps in zip(fit.coefficients, fit.monomial_exponents):
        vars_in = sorted(exps.keys())
        # exps maps name->power; names are 's{i}'
        idxs = []
        for name, p in exps.items():
            i = int(name[1:])
            idxs.extend([i] * p)
        if len(idxs) == 1:
            h_est[idxs[0]] = -coef               # linear → field
        elif len(idxs) == 2 and idxs[0] != idxs[1]:
            i, j = idxs
            J_est[i, j] = -coef                  # bilinear → coupling
            J_est[j, i] = -coef
        # s_i^2 = 1 for ±1 spins → constant; ignore
    return J_est, h_est


__all__ = [
    "IsingSystem",
    "random_ising",
    "ising_energy",
    "sample_ising_gibbs",
    "brute_force_ground_state",
    "simulated_bifurcation",
    "discover_hamiltonian_energy_labeled",
    "fit_to_coupling_matrix",
]
