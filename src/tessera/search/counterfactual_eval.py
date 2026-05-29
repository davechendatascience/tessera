"""Counterfactual evaluation as a post-hoc selector for Pareto-front candidates.

Lifecycle status
----------------
**SHIPPED 2026-05-29** (graduated from `tessera.experimental.counterfactual_eval`).

History: originally C5 from `docs/research/process_discovery_sr.md` §6.5.
Theoretical pre-analysis: `docs/shipped/c5_counterfactual_eval_analysis.md`
(moved to shipped/ when validation completed).

Validation evidence
-------------------

Heat equation benchmark (`benchmarks/results/heat_equation_counterfactual_mvp_c5.md`):
- 2/2 mechanism candidates correctly identified when present
- 0/3 false positives when no mechanism candidate exists
- Counterfactual ratio score cleanly separates mechanism (median ≤ 1.5)
  from non-mechanism (median > 1.7)
- 1/5 seeds: counterfactual ranking produced a Pareto-strict improvement
  over train-loss selection (lower cx at same accuracy)

Feynman cross-benchmark (`benchmarks/results/feynman_counterfactual_validation.md`):
- CF ranking and train-loss selection AGREED in 24/24 (target, seed) pairs.
- Feynman has no natural-overfit failure mode for CF to discriminate against.
- Conditional verdict: CF helps when the benchmark has Class B candidates
  in the Pareto front; CF is neutral otherwise.

Deployment recommendation
-------------------------

Ship as explicit-opt-in selection tool, NOT as default. Use case:
when the user has reason to suspect natural-overfit candidates in the
Pareto front (i.e., the benchmark/data class is known to admit them).

The C5 architectural insight
----------------------------

The cross-experiment pattern in `tessera.experimental` is striking:
- C1 ABC scoring (scoring-layer): falsified
- C3 MDL scoring (scoring-layer): falsified
- C4 causal axes (search-layer): partial
- C6 adaptive search (search-layer): null
- **C5 counterfactual eval (SELECTION-LAYER): VALIDATED**

The selection layer is where post-hoc evidence about a candidate's
generalization profile lives. Scoring/search-layer interventions try
to BIAS the GP toward better candidates during the run; selection-
layer interventions evaluate ALREADY-DISCOVERED candidates by criteria
the GP couldn't see. The latter pattern won where the former didn't.

This module's design generalizes: any post-hoc per-candidate
evaluation that can produce a scalar score per candidate fits this
framework. The heat-equation counterfactual generator is one
instantiation; future domain-specific generators (weather, CAMELS,
tokamak) would follow the same `Counterfactual` interface.

API
---

Two layers:

    Generic (domain-independent):
        score_counterfactual(tree, counterfactuals) -> dict
            Evaluate a tree on each counterfactual; return MSE +
            ratio-vs-oracle per CF + aggregate (mean/median/max).
        rank_front_by_counterfactual(front, counterfactuals, score_key)
            Rank Pareto-front candidates by a chosen score key
            (default 'median_ratio'). Lowest is best.

    Heat-equation instance:
        HeatEqCounterfactual
            Dataclass: (name, U, dt_U, alpha, oracle_mse).
        generate_heat_eq_counterfactuals(T, X, alpha_base, noise_std_base)
            Produces 5 CFs: 2 IC perturbations, 1 doubled-α
            interventional, 1 10× noise perturbation, 1 geometric
            (smaller X). Each CF carries its own oracle_mse for
            ratio computation.

Adding a domain
---------------

For a new domain (e.g., weather PDE, CAMELS streamflow), implement a
factory function returning a list of dataclass instances that match
the duck-typed interface used by `score_counterfactual` — i.e., each
CF must have at least `name`, the data to evaluate the tree on (in
whatever form `eval_tree` expects), and `oracle_mse > 0` for the
ratio. The generic ranking will then work on the resulting
counterfactual list.

For now, only the heat-equation instance ships. The generic API is
the contract; new instances are domain-by-domain follow-ups.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tessera.expression.measure_2d import measure_2d_laplacian_5pt
from tessera.expression.tree import evaluate as eval_tree


# ---------------------------------------------------------------------
# Heat-equation counterfactual instance
# ---------------------------------------------------------------------

@dataclass
class HeatEqCounterfactual:
    """A perturbed heat-equation trajectory + its target + a name."""
    name: str
    U: np.ndarray              # shape (T, X)
    dt_U: np.ndarray           # shape (T, X)
    alpha: float               # diffusion coefficient used
    oracle_mse: float          # oracle MSE on this CF (for normalization)


def _simulate_heat_simple(T, X, alpha, noise_std, ic_seed, sim_seed,
                          amplitude=10.0):
    """Heat equation simulator — same as benchmarks but inlined here
    to avoid cross-package imports."""
    rng_ic = np.random.default_rng(ic_seed)
    rng_sim = np.random.default_rng(sim_seed)
    U = np.zeros((T, X), dtype=np.float64)
    xs = np.arange(X) - X / 2

    n_bumps = int(rng_ic.integers(2, 4))
    for _ in range(n_bumps):
        center = rng_ic.uniform(-X / 4, X / 4)
        width = rng_ic.uniform(3.0, 7.0)
        amp = amplitude * rng_ic.uniform(0.4, 1.0)
        U[0] += amp * np.exp(-((xs - center) ** 2) / (2 * width ** 2))

    for t in range(1, T):
        prev = U[t - 1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        U[t] = prev + alpha * lap + noise_std * rng_sim.standard_normal(X)
    return U


def _compute_oracle_mse(U, dt_U, alpha):
    lap = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    return float(np.mean((alpha * lap[interior] - dt_U[interior]) ** 2))


def generate_heat_eq_counterfactuals(
    T: int = 200,
    X: int = 32,
    alpha_base: float = 0.05,
    noise_std_base: float = 0.002,
) -> list[HeatEqCounterfactual]:
    """Generate a set of 5 counterfactual perturbations.

    The CFs probe different mechanism dimensions:
    - cf_ic_a, cf_ic_b: different initial conditions, same α
    - cf_alpha_2x: doubled diffusion coefficient (interventional)
    - cf_noise_10x: 10× higher noise level (regularity perturbation)
    - cf_smaller_x: smaller spatial grid (geometric perturbation)
    """
    cfs: list[HeatEqCounterfactual] = []

    # CF 1: different IC, same α
    U = _simulate_heat_simple(T=T, X=X, alpha=alpha_base,
                              noise_std=noise_std_base,
                              ic_seed=500, sim_seed=10)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    cfs.append(HeatEqCounterfactual(
        name="cf_ic_a",
        U=U, dt_U=dt_U, alpha=alpha_base,
        oracle_mse=_compute_oracle_mse(U, dt_U, alpha_base),
    ))

    # CF 2: another different IC
    U = _simulate_heat_simple(T=T, X=X, alpha=alpha_base,
                              noise_std=noise_std_base,
                              ic_seed=600, sim_seed=11)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    cfs.append(HeatEqCounterfactual(
        name="cf_ic_b",
        U=U, dt_U=dt_U, alpha=alpha_base,
        oracle_mse=_compute_oracle_mse(U, dt_U, alpha_base),
    ))

    # CF 3: doubled α — interventional
    alpha_2x = alpha_base * 2.0
    U = _simulate_heat_simple(T=T, X=X, alpha=alpha_2x,
                              noise_std=noise_std_base,
                              ic_seed=700, sim_seed=12)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    cfs.append(HeatEqCounterfactual(
        name="cf_alpha_2x",
        U=U, dt_U=dt_U, alpha=alpha_2x,
        oracle_mse=_compute_oracle_mse(U, dt_U, alpha_2x),
    ))

    # CF 4: 10× higher noise
    U = _simulate_heat_simple(T=T, X=X, alpha=alpha_base,
                              noise_std=noise_std_base * 10.0,
                              ic_seed=800, sim_seed=13)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    cfs.append(HeatEqCounterfactual(
        name="cf_noise_10x",
        U=U, dt_U=dt_U, alpha=alpha_base,
        oracle_mse=_compute_oracle_mse(U, dt_U, alpha_base),
    ))

    # CF 5: smaller X — geometric perturbation
    U = _simulate_heat_simple(T=T, X=X // 2, alpha=alpha_base,
                              noise_std=noise_std_base,
                              ic_seed=900, sim_seed=14)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    cfs.append(HeatEqCounterfactual(
        name="cf_smaller_x",
        U=U, dt_U=dt_U, alpha=alpha_base,
        oracle_mse=_compute_oracle_mse(U, dt_U, alpha_base),
    ))

    return cfs


# ---------------------------------------------------------------------
# Generic scoring + ranking
# ---------------------------------------------------------------------

def _evaluate_tree_on(tree, U, dt_U) -> float:
    """Evaluate a tree on (U, dt_U). Returns MSE on interior; inf on fail."""
    try:
        pred = eval_tree(tree, {"U": U}, fill_warmup=0.0)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != dt_U.shape or not np.isfinite(pred).all():
            return float("inf")
        interior = (slice(1, -1), slice(1, -1))
        return float(np.mean((pred[interior] - dt_U[interior]) ** 2))
    except Exception:
        return float("inf")


def score_counterfactual(
    tree, counterfactuals: list[HeatEqCounterfactual],
) -> dict:
    """Score a tree on a set of counterfactuals.

    Returns a dict with:
        per_cf : dict[cf_name -> dict(mse, ratio_vs_oracle)]
        n_finite : int — how many CFs gave finite MSE
        mean_ratio : float — mean of finite (mse / oracle_mse)
        max_ratio : float — max of finite ratios (worst CF)
        median_ratio : float
    """
    per_cf = {}
    ratios = []
    for cf in counterfactuals:
        mse = _evaluate_tree_on(tree, cf.U, cf.dt_U)
        if math.isfinite(mse) and cf.oracle_mse > 0:
            ratio = mse / cf.oracle_mse
            ratios.append(ratio)
        else:
            ratio = float("inf")
        per_cf[cf.name] = {"mse": mse, "ratio_vs_oracle": ratio}

    if ratios:
        finite_ratios = [r for r in ratios if math.isfinite(r)]
        mean_r = float(np.mean(finite_ratios)) if finite_ratios else float("inf")
        max_r = float(np.max(finite_ratios)) if finite_ratios else float("inf")
        median_r = float(np.median(finite_ratios)) if finite_ratios else float("inf")
    else:
        mean_r = max_r = median_r = float("inf")

    return {
        "per_cf": per_cf,
        "n_finite": sum(1 for r in ratios if math.isfinite(r)),
        "mean_ratio": mean_r,
        "max_ratio": max_r,
        "median_ratio": median_r,
    }


def rank_front_by_counterfactual(
    front: list,
    counterfactuals: list[HeatEqCounterfactual],
    score_key: str = "median_ratio",
) -> list[tuple]:
    """Rank Pareto-front candidates by counterfactual performance.

    Returns list of (cand, score_dict) sorted by score_key (lowest=best).
    Candidates with no finite CF go last.
    """
    scored = []
    for cand in front:
        s = score_counterfactual(cand.tree, counterfactuals)
        scored.append((cand, s))

    def sort_key(item):
        s = item[1]
        v = s.get(score_key, float("inf"))
        if not math.isfinite(v):
            return (1, float("inf"))
        return (0, v)

    scored.sort(key=sort_key)
    return scored


__all__ = [
    "HeatEqCounterfactual",
    "generate_heat_eq_counterfactuals",
    "score_counterfactual",
    "rank_front_by_counterfactual",
]
