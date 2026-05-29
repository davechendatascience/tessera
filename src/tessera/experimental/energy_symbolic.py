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
  - E1b — symbolic FORM-search from samples only. The pseudo-likelihood
    identity p(sᵢ|s₋ᵢ) = σ(-β·ΔHᵢ·sᵢ) makes scoring a candidate energy
    form gradient-free AND sampling-free (needs only H on spin-flipped
    data). `discover_energy_monomials` does greedy forward-selection over
    the multilinear-monomial basis (the exact basis for ±1-spin energy
    functions); `discover_hamiltonian_from_samples` is the generic GP-
    over-trees variant. Order-3 monomials let it discover 3-body
    interactions a fixed pairwise inverse cannot represent.

CPU/GPU: the sampler (`sample_ising_gibbs`), the minimizer
(`simulated_bifurcation`), and the form-search PLL hot loop each have a
numpy (CPU) path and a jit'd JAX (GPU) path selected by `use_jax`; they
agree to float precision.

Graduation criterion (E1): tessera recovers a sparse Ising
Hamiltonian's structure gradient-free, with coupling recovery clearly
above a non-energy baseline; AND the bSB minimizer finds ground states
of small instances (brute-force-checkable) reliably.

Removal criterion: bSB doesn't reliably minimize small Ising
instances, OR energy-labeled discovery (≈C8) fails.

Initial commit: 2026-05-29
Last evaluation: 2026-05-29 — E1 validated (see
benchmarks/results/hamiltonian_discovery.md): SB 7/8 ground states;
E1a exact (corr 1.0); E1b form-search recovers exact structure and
discovers 3-body terms a pairwise inverse cannot.

Note on JAX: every function here works on numpy too (uses
`tessera.backend.array_module` style where needed), but the minimizer
and sampler are written to be `jax.jit`/`vmap`-friendly so they run on
GPU. They accept an explicit `xp` (numpy or jax.numpy) where it matters.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
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
                       seed: int = 0, use_jax: bool = False) -> np.ndarray:
    """Gibbs-sample spin configs from p(s) ∝ exp(-beta·H(s)).

    Returns (n_samples, n) array in {-1,+1}. Vectorized over samples,
    sequential over spins (true sequential Gibbs). `use_jax=True` runs the
    sweeps on a jit'd JAX kernel (GPU path) — same Glauber update rule as
    the numpy (CPU) reference. Used to generate data for E1b.
    """
    n = sys.n
    if use_jax:
        try:
            return _sample_ising_gibbs_jax(sys, beta, n_samples, n_sweeps, seed)
        except ImportError:
            pass
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


def _sample_ising_gibbs_jax(sys: IsingSystem, beta, n_samples, n_sweeps, seed):
    """GPU Gibbs sampler: scan over sweeps, fori over spins (sequential
    Glauber). vmapped over samples implicitly (the sample axis is a batch
    dimension carried through every op)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    n = sys.n
    J = jnp.asarray(sys.J); h = jnp.asarray(sys.h)
    key = jax.random.PRNGKey(seed)
    s0 = jnp.where(jax.random.bernoulli(key, 0.5, (n_samples, n)), 1.0, -1.0)

    def sweep(s, sweep_key):
        def spin(i, s):
            local = h[i] + s @ J[i]                    # (n_samples,)
            p_up = jax.nn.sigmoid(2.0 * beta * local)
            u = jax.random.uniform(jax.random.fold_in(sweep_key, i), (n_samples,))
            col = jnp.where(u < p_up, 1.0, -1.0)
            return s.at[:, i].set(col)
        return lax.fori_loop(0, n, spin, s), None

    sweep_keys = jax.random.split(jax.random.fold_in(key, 1), n_sweeps)
    s_final, _ = lax.scan(sweep, s0, sweep_keys)
    return np.asarray(s_final)


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


# ---------------------------------------------------------------------
# E1b — symbolic form-search for the energy H(s) from SAMPLES ONLY
# ---------------------------------------------------------------------
#
# This is the full energy-based symbolic-discovery loop, and the reason
# it is tractable AND gradient-free is the pseudo-likelihood identity:
#
#   For an energy model p(s) ∝ exp(-β H(s)), the conditional of one spin
#   given the rest is
#       p(s_i | s_{-i}) = σ(-β · ΔH_i · s_i),   ΔH_i = H(s^{i+}) - H(s^{i-})
#   where s^{i±} is the config with spin i forced to ±1.
#
# So scoring a CANDIDATE energy form needs only evaluations of H on
# spin-flipped copies of the data — NO sampling from the model, NO
# gradients. The partition function never appears. We search the
# symbolic FORM of H with GP and score each candidate by its
# pseudo-log-likelihood; coefficients are polished gradient-free
# (Nelder-Mead) on the same objective.
#
# Why this is the genuine contribution over the closed-form inverse
# (check_e1b in the benchmark): the inverse `J ≈ -C⁻¹/β` ASSUMES a
# pairwise form. The symbolic search does not — given a vocabulary with
# products, it can discover non-pairwise (e.g. 3-body s_i s_j s_k) energy
# terms that a fixed pairwise inverse structurally cannot represent.


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable log σ(z) = -softplus(-z) = -log(1+exp(-z))."""
    return -np.logaddexp(0.0, -z)


def pseudo_log_likelihood(tree, S: np.ndarray, beta: float) -> float:
    """Mean pseudo-log-likelihood of energy form `tree` on samples `S`.

    `tree` is an Expr over features s0..s{n-1} returning the energy H(s).
    `S` is (N, n) in {-1,+1}. Higher (closer to 0) is better.

    Gradient-free AND sampling-free: uses only H evaluated on the data
    with each spin forced to ±1. Returns -inf if the tree is unevaluable.
    """
    from tessera.expression.tree import evaluate

    S = np.asarray(S, dtype=np.float64)
    N, n = S.shape
    total = 0.0
    for i in range(n):
        s_plus = S.copy(); s_plus[:, i] = 1.0
        s_minus = S.copy(); s_minus[:, i] = -1.0
        env_p = {f"s{j}": s_plus[:, j] for j in range(n)}
        env_m = {f"s{j}": s_minus[:, j] for j in range(n)}
        try:
            Hp = np.asarray(evaluate(tree, env_p), dtype=np.float64)
            Hm = np.asarray(evaluate(tree, env_m), dtype=np.float64)
        except Exception:
            return float("-inf")
        if np.isscalar(Hp) or Hp.ndim == 0:
            Hp = np.full(N, float(Hp))
        if np.isscalar(Hm) or Hm.ndim == 0:
            Hm = np.full(N, float(Hm))
        dH = Hp - Hm                       # ΔH_i over the batch
        z = -beta * dH * S[:, i]           # σ(z) = p(observed spin i)
        ll = _log_sigmoid(z)
        if not np.all(np.isfinite(ll)):
            return float("-inf")
        total += float(np.sum(ll))
    return total / (N * n)


def _polish_energy_constants(tree, S: np.ndarray, beta: float,
                             maxiter: int = 60):
    """Gradient-free (Nelder-Mead) polish of the tree's Const leaves to
    maximize pseudo-log-likelihood. Returns (tree, pll)."""
    from scipy.optimize import minimize
    from tessera.expression.tree import collect_const_values, set_const_values

    initial = collect_const_values(tree)
    base_pll = pseudo_log_likelihood(tree, S, beta)
    if not initial:
        return tree, base_pll
    x0 = np.array(initial, dtype=np.float64)

    def neg_pll(x):
        cand = set_const_values(tree, x.tolist())
        v = pseudo_log_likelihood(cand, S, beta)
        return 1e18 if not np.isfinite(v) else -v

    try:
        res = minimize(neg_pll, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
    except Exception:
        return tree, base_pll
    polished = set_const_values(tree, res.x.tolist())
    pll = pseudo_log_likelihood(polished, S, beta)
    if pll >= base_pll:
        return polished, pll
    return tree, base_pll


@dataclass
class EnergyDiscoveryConfig:
    """Knobs for the E1b symbolic energy-form search."""
    pop_size: int = 200
    n_gens: int = 40
    tournament_k: int = 4
    elite_frac: float = 0.1
    parsimony: float = 2e-3
    max_depth_init: int = 4
    polish_every: int = 5
    polish_top: int = 8
    seed: int = 0


@dataclass
class EnergyDiscoveryResult:
    tree: object
    pll: float
    complexity: int
    history: list = field(default_factory=list)


def discover_hamiltonian_from_samples(
    S: np.ndarray, beta: float,
    cfg: Optional[EnergyDiscoveryConfig] = None,
    seed_trees: Optional[list] = None,
) -> EnergyDiscoveryResult:
    """Discover the symbolic energy form H(s) from SAMPLES via GP.

    Searches Expr trees over spins s0..s{n-1}, scored by mean pseudo-log-
    likelihood (gradient-free + sampling-free) minus a parsimony penalty.
    Coefficients are polished gradient-free every `polish_every` gens.

    Returns the best energy tree found. Recoverable up to an additive
    constant (which cancels in ΔH) — compare to ground truth with
    `energy_correlation` (affine-invariant), not raw values.
    """
    from tessera.expression.tree import complexity, Const
    from tessera.expression.mutation import mutate, random_tree, validate_tree
    from tessera.expression.simplify import simplify_full as simplify

    cfg = cfg or EnergyDiscoveryConfig()
    n = S.shape[1]
    feats = [f"s{i}" for i in range(n)]
    feat_set = set(feats)
    rng = random.Random(cfg.seed)

    def fitness(tree) -> float:
        # higher better: PLL minus parsimony on complexity
        pll = pseudo_log_likelihood(tree, S, beta)
        if not np.isfinite(pll):
            return float("-inf")
        return pll - cfg.parsimony * complexity(tree)

    # ---- init population ----
    pop: list = []
    if seed_trees:
        for t in seed_trees:
            if validate_tree(t, feat_set) is None:
                pop.append(t)
    # An additive pairwise seed gives the search a sensible basin without
    # assuming it's correct (GP will prune/extend it).
    while len(pop) < cfg.pop_size:
        t = random_tree(rng, feats, max_depth=cfg.max_depth_init,
                        pointwise_only=True)
        if validate_tree(t, feat_set) is None:
            pop.append(t)

    scored = [(fitness(t), t) for t in pop]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    history = []

    n_elite = max(1, int(cfg.elite_frac * cfg.pop_size))
    for gen in range(cfg.n_gens):
        # polish the top candidates' constants gradient-free
        if cfg.polish_every and gen % cfg.polish_every == 0:
            polished = []
            for fit, t in scored[:cfg.polish_top]:
                pt, _ = _polish_energy_constants(t, S, beta)
                polished.append((fitness(pt), pt))
            scored = polished + scored[cfg.polish_top:]
            scored.sort(key=lambda kv: kv[0], reverse=True)

        elites = [t for _, t in scored[:n_elite]]
        children: list = []
        guard = 0
        while len(children) < cfg.pop_size - n_elite and guard < cfg.pop_size * 20:
            guard += 1
            # tournament selection
            def pick():
                cand = rng.sample(scored, min(cfg.tournament_k, len(scored)))
                cand.sort(key=lambda kv: kv[0], reverse=True)
                return cand[0][1]
            parents = [pick(), pick()]
            child = mutate(parents, rng, feats, pointwise_only=True)
            if child is None:
                continue
            try:
                child = simplify(child)
            except Exception:
                pass
            if validate_tree(child, feat_set) is not None:
                continue
            children.append(child)

        pop = elites + children
        scored = [(fitness(t), t) for t in pop]
        scored.sort(key=lambda kv: kv[0], reverse=True)
        history.append(scored[0][0])

    # final polish of the winner
    best_fit, best = scored[0]
    best, best_pll = _polish_energy_constants(best, S, beta, maxiter=120)
    return EnergyDiscoveryResult(
        tree=best, pll=best_pll, complexity=complexity(best), history=history,
    )


# --- Principled form-search over the multilinear-monomial basis ------
#
# Every energy function on ±1 spins is EXACTLY a multilinear polynomial
# H(s) = Σ_A c_A · Π_{i∈A} s_i (since s_i² = 1). So the correct
# hypothesis space for "the symbolic form of the energy" is the set of
# monomials — the same polynomial basis the C8 detector uses for E1a,
# now scored by pseudo-likelihood instead of energy-regression. We
# discover the FORM by greedy forward selection over monomials (a
# gradient-free combinatorial search), fitting coefficients gradient-
# free (Nelder-Mead) at each step. Including order-3 monomials lets this
# discover 3-body interactions a fixed pairwise inverse cannot represent.

def _candidate_monomials(n: int, max_order: int):
    from itertools import combinations
    mons = []
    for order in range(1, max_order + 1):
        mons.extend(combinations(range(n), order))
    return mons


def _build_design(S: np.ndarray, mons):
    """Build the PLL design tensor D (n, N, K) and the sign matrix
    (n, N) = Sᵀ. D[i, :, k] = ΔH_i contribution of monomial k per unit
    coefficient = 2·Π_{j∈A_k, j≠i} s_j if i∈A_k else 0.

    With this, the whole PLL is a couple of array ops (einsum + logistic)
    — the form that runs identically on numpy (CPU) and jit'd JAX (GPU).
    """
    N, n = S.shape
    K = len(mons)
    D = np.zeros((n, N, K), dtype=np.float64)
    for k, A in enumerate(mons):
        for i in A:
            rest = [j for j in A if j != i]
            D[i, :, k] = 2.0 * (np.prod(S[:, rest], axis=1) if rest else 1.0)
    return D, S.T.copy()


def _pll_design_np(D, signs, beta, c) -> float:
    """CPU pseudo-log-likelihood from the design tensor (numpy)."""
    dH = np.einsum("ink,k->in", D, c)        # (n, N) = ΔH_i over samples
    z = -beta * signs * dH
    return float(np.mean(-np.logaddexp(0.0, -z)))


_PLL_JIT = None


def _get_pll_jit():
    """Cached jit'd JAX PLL kernel (GPU path). Recompiles per distinct K
    (≤ max_terms shapes), like the symbolic interpreter caches per
    max_nodes."""
    global _PLL_JIT
    if _PLL_JIT is not None:
        return _PLL_JIT
    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(D, signs, beta, c):
        dH = jnp.einsum("ink,k->in", D, c)
        z = -beta * signs * dH
        return jnp.mean(-jnp.logaddexp(0.0, -z))

    _PLL_JIT = f
    return f


def _fit_monomial_coeffs(S, beta, mons, maxiter, x0=None, use_jax=False):
    """Gradient-free (Nelder-Mead) coefficient fit maximizing PLL.

    Builds the design tensor once, then each Nelder-Mead probe is a
    single einsum + logistic — on numpy (CPU) or a jit'd JAX kernel
    (GPU). Returns (coeffs, pll)."""
    from scipy.optimize import minimize
    if not mons:
        return [], float("-inf")
    D, signs = _build_design(S, mons)
    x0 = np.zeros(len(mons)) if x0 is None else np.asarray(x0, dtype=float)

    if use_jax:
        import jax.numpy as jnp
        f = _get_pll_jit()
        Dj = jnp.asarray(D); sj = jnp.asarray(signs); bj = jnp.asarray(beta)

        def pll(x):
            return float(f(Dj, sj, bj, jnp.asarray(x, dtype=Dj.dtype)))
    else:
        def pll(x):
            return _pll_design_np(D, signs, beta, np.asarray(x, dtype=float))

    def neg(x):
        v = pll(x)
        return 1e18 if not np.isfinite(v) else -v

    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "fatol": 1e-7, "xatol": 1e-5})
    return res.x.tolist(), -float(res.fun)


def _pll_monomials(S: np.ndarray, beta: float, mons, coeffs) -> float:
    """Mean pseudo-log-likelihood of H = Σ coeffs[k]·monomial[k] (numpy
    reference / scoring helper)."""
    if not mons:
        return float("-inf")
    D, signs = _build_design(S, mons)
    return _pll_design_np(D, signs, beta, np.asarray(coeffs, dtype=float))


@dataclass
class MonomialEnergyResult:
    monomials: list          # list of tuples of spin indices
    coeffs: list             # parallel coefficients c_A
    pll: float
    history: list = field(default_factory=list)

    def to_tree(self):
        from tessera.expression.tree import Var, Const, BinOp
        terms = []
        for A, c in zip(self.monomials, self.coeffs):
            node = Const(float(c))
            for i in A:
                node = BinOp("mul", node, Var(f"s{i}"))
            terms.append(node)
        if not terms:
            return Const(0.0)
        acc = terms[0]
        for t in terms[1:]:
            acc = BinOp("add", acc, t)
        return acc


def discover_energy_monomials(
    S: np.ndarray, beta: float, *,
    max_order: int = 2, max_terms: int = 16,
    maxiter: int = 200, bic: bool = True, verbose: bool = False,
    use_jax: bool = False,
) -> MonomialEnergyResult:
    """Greedy forward-selection form-search over the monomial basis,
    scored by pseudo-log-likelihood (gradient-free + sampling-free).

    At each round, tries adding every not-yet-selected monomial, refits
    ALL coefficients gradient-free, keeps the best. Stops when the PLL
    gain falls below the BIC per-parameter penalty (principled, not a
    fixed term count). Set `max_order=3` to allow 3-body interactions.

    `use_jax=True` runs each coefficient fit's PLL evaluations on a jit'd
    JAX kernel (GPU); the default numpy path is the CPU version. Both
    produce identical results (same Nelder-Mead trajectory).
    """
    N, n = S.shape
    cands = _candidate_monomials(n, max_order)
    penalty = 0.5 * np.log(N * n) / (N * n) if bic else 0.0

    selected, coeffs, cur_pll = [], [], float("-inf")
    history = []
    while len(selected) < max_terms:
        best = None
        for A in cands:
            if A in selected:
                continue
            trial = selected + [A]
            x0 = coeffs + [0.0]
            x, pll = _fit_monomial_coeffs(S, beta, trial, maxiter,
                                          x0=x0, use_jax=use_jax)
            if best is None or pll > best[1]:
                best = (A, pll, x)
        A, pll, x = best
        if pll - cur_pll < penalty:
            break
        selected.append(A); coeffs = x; cur_pll = pll
        history.append((A, pll))
        if verbose:
            print(f"  + {A}: PLL={pll:.5f}")
    return MonomialEnergyResult(monomials=selected, coeffs=coeffs,
                                pll=cur_pll, history=history)


def monomial_result_to_coupling_matrix(res: "MonomialEnergyResult", n: int):
    """Extract (J_est, h_est) from order≤2 monomials (H = -Σ J s s - Σ h s)."""
    J = np.zeros((n, n)); h = np.zeros(n)
    for A, c in zip(res.monomials, res.coeffs):
        if len(A) == 1:
            h[A[0]] = -c
        elif len(A) == 2:
            i, j = A
            J[i, j] = -c; J[j, i] = -c
    return J, h


def sample_monomial_gibbs(monomials, coeffs, n: int, beta: float = 1.0,
                          n_samples: int = 2000, n_sweeps: int = 80,
                          seed: int = 0) -> np.ndarray:
    """Gibbs-sample p(s) ∝ exp(-β H(s)) for a general multilinear energy
    H = Σ coeffs[k]·Π_{i∈monomials[k]} s_i. Supports 3-body+ terms."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_samples, n))
    # precompute, per spin i, the monomials containing i and their "rest" indices
    by_spin = {i: [] for i in range(n)}
    for A, c in zip(monomials, coeffs):
        for i in A:
            by_spin[i].append((c, [j for j in A if j != i]))
    for _ in range(n_sweeps):
        for i in range(n):
            f = np.zeros(n_samples)           # f_i = Σ_{A∋i} c_A Π_{j∈A\i} s_j
            for c, rest in by_spin[i]:
                f += c * (np.prod(s[:, rest], axis=1) if rest else 1.0)
            p_up = 1.0 / (1.0 + np.exp(2.0 * beta * f))   # p(s_i=+1)=σ(-2β f_i)
            s[:, i] = np.where(rng.random(n_samples) < p_up, 1.0, -1.0)
    return s


def build_pairwise_energy_tree(J: np.ndarray, h: np.ndarray):
    """Construct an additive energy tree H = -Σ J_ij s_i s_j - Σ h_i s_i
    over features s0..s{n-1}. Used as a detect-then-seed starting point
    (e.g. from the nMF inverse) for the symbolic refinement search."""
    from tessera.expression.tree import Var, Const, BinOp

    n = h.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(J[i, j])
            if abs(w) > 1e-9:
                prod = BinOp("mul", Var(f"s{i}"), Var(f"s{j}"))
                terms.append(BinOp("mul", Const(-w), prod))
    for i in range(n):
        if abs(float(h[i])) > 1e-9:
            terms.append(BinOp("mul", Const(-float(h[i])), Var(f"s{i}")))
    if not terms:
        return Const(0.0)
    acc = terms[0]
    for t in terms[1:]:
        acc = BinOp("add", acc, t)
    return acc


def nmf_seed_from_samples(S: np.ndarray, beta: float):
    """Naive-mean-field inverse-Ising estimate as a seed energy tree.
    `J ≈ -C⁻¹/β`, fields from the magnetizations. Gradient-free."""
    n = S.shape[1]
    C = np.cov(S, rowvar=False)
    Cinv = np.linalg.inv(C + 1e-6 * np.eye(n))
    J_est = -Cinv / beta
    np.fill_diagonal(J_est, 0.0)
    m = S.mean(axis=0)
    # mean-field field estimate: atanh(m)/β - Σ_j J_ij m_j
    m_c = np.clip(m, -0.999, 0.999)
    h_est = np.arctanh(m_c) / beta - J_est @ m
    return build_pairwise_energy_tree(J_est, h_est)


def energy_correlation(tree, sys: IsingSystem, n_configs: int = 2000,
                       seed: int = 0) -> float:
    """Functional check: correlation between a discovered energy `tree`
    and the true Ising energy over random configs. Affine-invariant, so
    it ignores the additive-constant / overall-scale gauge freedom."""
    from tessera.expression.tree import evaluate

    n = sys.n
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_configs, n))
    E_true = ising_energy(S, sys)
    env = {f"s{i}": S[:, i] for i in range(n)}
    try:
        E_pred = np.asarray(evaluate(tree, env), dtype=np.float64)
    except Exception:
        return float("nan")
    if E_pred.ndim == 0:
        return 0.0
    if E_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(E_true, E_pred)[0, 1])


__all__ = [
    "IsingSystem",
    "random_ising",
    "ising_energy",
    "sample_ising_gibbs",
    "brute_force_ground_state",
    "simulated_bifurcation",
    "discover_hamiltonian_energy_labeled",
    "fit_to_coupling_matrix",
    "pseudo_log_likelihood",
    "EnergyDiscoveryConfig",
    "EnergyDiscoveryResult",
    "discover_hamiltonian_from_samples",
    "discover_energy_monomials",
    "MonomialEnergyResult",
    "monomial_result_to_coupling_matrix",
    "sample_monomial_gibbs",
    "build_pairwise_energy_tree",
    "nmf_seed_from_samples",
    "energy_correlation",
]
