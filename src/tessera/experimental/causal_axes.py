"""Causal direction priors via axis types (Conjecture C4).

Provenance: C4 from `docs/research/process_discovery_sr.md` §6.4.

Status: **PARTIAL VALIDATION** on heat equation (2026-05-26).
        See `benchmarks/results/heat_equation_causal_axes_mvp_c4.md`.
        Eliminates Class A-temporal (the temporal-derivative tautology
        shortcut) as designed, AND doesn't lose the right answer
        (Class C rate unchanged at 1/5). BUT does NOT boost Class C
        discovery — the GP falls back to A-spatial (2-atom first
        derivative) or degenerate (predict-zero). Necessary but not
        sufficient for mechanism discovery. Module preserved; possible
        future combinations with other interventions noted in the
        empirical report.

Conjecture C4
-------------
> "Causal direction priors at the tree level (extending axis types)
> reduce effective search space without losing the right answer."

The specific operationalization here: for dynamical-system targets
where the target is a TEMPORAL DERIVATIVE of the state variable
(e.g., dt_U for heat equation), forbid Measure2D operators whose
atoms span multiple time offsets. This eliminates trees that compute
the target via a temporal-derivative shortcut (Class A tautology in
the heat equation taxonomy):

  M2D[1·(0,0) + -1·(1,0)](U) = U[t,x] - U[t+1,x] ≈ -dt_U[t,x]

Trees of this shape FIT THE DATA WELL but offer no mechanism — they
restate the target via time differencing. The causal-axes constraint
forces the GP to express dt_U in terms of SPATIAL operators on U
(the actual physical mechanism: dt_U = α · ∇²U).

Generalization
--------------
The narrow rule "all Measure2D atoms must share lag_t" is the
strictest possible version. More general formulations might allow:
- Trees where lag_t spans bounded values (e.g., lag_t ∈ {0, 1} OK)
- Trees that mix temporal and spatial atoms IF the temporal atoms
  represent integration / smoothing (not differencing)

For this initial experiment we take the strict form. If C4 validates
with strict, we can explore looser variants.

Graduation criterion
--------------------
On heat equation discovery with single-trajectory training (no multi-
trajectory crutch, no reduce_* downweight as the only fix), the
causal-axes constraint raises Class C discovery rate strictly above
baseline (which sits at ~1/5 from the C1 experiment baseline).

Removal criterion
-----------------
The constraint either:
  - Eliminates Class C alongside Class A (too strict; nothing works)
  - Fails to raise Class C rate above baseline (constraint doesn't
    redirect search toward mechanism)

Initial commit: 2026-05-26
Last evaluation: 2026-05-26 (partial validation)

What's in this module
---------------------
    is_pure_spatial_m2d(measure_2d) -> bool
        True iff all atoms in the Measure2D have the same lag_t and
        sep_t is None (no temporal density).

    tree_violates_causal_spatial(node) -> bool
        True iff any Measure2D in the tree is NOT pure-spatial.

    GPWithCausalAxes(GP)
        GP subclass that applies a heavy fitness penalty to any
        candidate violating causal-spatial constraint. The penalty
        is large enough that violations consistently lose tournaments.

Why penalty rather than hard rejection
--------------------------------------
A hard rejection (regenerate or reject) interrupts the GP loop's
control flow. A penalty preserves the standard scoring path and lets
the GP "see" violators (which may have intermediate structures worth
mutating from) but never selects them as parents.

The penalty size matters: too small and violators leak through; too
large and selection becomes deterministic (Pareto-front degeneracy).
Default penalty = 1e6 was chosen so violators always lose tournaments
against any feasible candidate, but the Pareto front still uses
(cx, train_loss) without modification.

Connection to existing tessera
------------------------------
This is the first experimental use of tessera's axis system (in
spirit, though it doesn't currently use `tessera.expression.axes`
directly — that module enforces type/dimensional compatibility, not
temporal causality). If C4 validates, the axis module is the natural
home for the production version.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tessera.search.base import Candidate
from tessera.search.gp import GP, GPConfig
from tessera.expression.tree import (
    FunctionalOp2D, iter_subtrees,
)


# ---------------------------------------------------------------------
# Causal-axes constraint check
# ---------------------------------------------------------------------

def is_pure_spatial_m2d(measure_2d) -> bool:
    """Return True iff the Measure2D is pure-spatial.

    Pure-spatial means:
      - All atoms have the SAME lag_t (all at the same time slice)
      - sep_t is None (no temporal-density smoothing/integration)

    Examples (heat-eq notation):
      M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)]  → pure spatial  (Laplacian)
      M2D[1·(0,0) + -1·(1,0)]              → NOT pure spatial (temporal diff)
      M2D[1·(0,0) + -1·(0,1)]              → pure spatial (1st spatial diff)
    """
    if not hasattr(measure_2d, "atoms"):
        return False
    # No atoms = trivial; treat as pure spatial (nothing to violate)
    if not measure_2d.atoms:
        # If there's only a density part, check sep_t
        return getattr(measure_2d, "sep_t", None) is None
    lag_ts = {int(a.lag_t) for a in measure_2d.atoms}
    if len(lag_ts) > 1:
        return False
    # Check density part too
    if getattr(measure_2d, "sep_t", None) is not None:
        return False
    return True


def tree_violates_causal_spatial(node) -> bool:
    """Return True iff any Measure2D in the tree is NOT pure-spatial."""
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            if not is_pure_spatial_m2d(sub.measure_2d):
                return True
    return False


def count_violating_m2ds(node) -> int:
    """Count the number of Measure2D nodes in the tree that violate
    the causal-spatial constraint. Useful for diagnostics."""
    n = 0
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            if not is_pure_spatial_m2d(sub.measure_2d):
                n += 1
    return n


# ---------------------------------------------------------------------
# GP subclass with causal-axes penalty
# ---------------------------------------------------------------------

class GPWithCausalAxes(GP):
    """GP that penalizes trees violating the pure-spatial Measure2D
    constraint.

    Use case: dynamical-system targets where the target variable is
    a temporal derivative of the input field (heat equation, wave
    equation, advection-diffusion). The constraint forbids the
    obvious temporal-derivative shortcut, forcing the GP to express
    the target via spatial operators on the input.

    Parameters
    ----------
    cfg : GPConfig
    penalty : float
        Fitness penalty added per violating Measure2D in the tree.
        Default 1e6 — large enough that any feasible candidate beats
        any violator in tournaments.
    loss_fn : callable | None
        Forwarded to GP base.
    """

    def __init__(
        self,
        cfg: GPConfig,
        *,
        penalty: float = 1e6,
        loss_fn: Any = None,
    ) -> None:
        super().__init__(cfg, loss_fn=loss_fn)
        self.causal_penalty = float(penalty)

    def _score(self, tree, env, y_true, born_gen):
        cand = super()._score(tree, env, y_true, born_gen)
        n_violators = count_violating_m2ds(cand.tree)
        if n_violators == 0:
            return cand
        # Hard rejection: violators get inf in BOTH train_loss and fitness.
        # train_loss excludes from Pareto front (which uses (cx, train_loss));
        # fitness excludes from tournament selection.
        # An earlier version only penalized fitness, but that left violators
        # appearing in the final Pareto front output despite never being
        # selected for breeding — confusing the diagnostic.
        return Candidate(
            tree=cand.tree,
            train_loss=float("inf"),
            complexity=cand.complexity,
            fitness=float("inf"),
            born_gen=cand.born_gen,
        )


__all__ = [
    "is_pure_spatial_m2d",
    "tree_violates_causal_spatial",
    "count_violating_m2ds",
    "GPWithCausalAxes",
]
