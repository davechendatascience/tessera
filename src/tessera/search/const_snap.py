"""Constant snapping — round fitted constants to canonical physical values.

Step 2 of the Feynman improvement plan (see
`docs/research/from_data_to_mechanism.md` §6 for the methodological
discipline; see this session's plan for the full step ordering).

Motivation
----------
After scipy's `optimize_constants` polish, fitted scalars are usually
close-but-not-exact to known physical constants:

    Coulomb 1/(4π·ε)        →  factor 0.0796 ≈ 1/(4π)
    Stokes-Einstein 1/(6π)  →  factor 0.0531 ≈ 1/(6π)
    Spring potential ½kx²   →  factor 0.5    ≈ 1/2
    Gaussian exp(-x²/2)     →  factor 0.5    ≈ 1/2
    Ideal gas nkT/V         →  factor 1.0    ≈ 1

The const-opt step can polish to 4-6 digits but adding the equivalent
of "snap to nearest physical constant if very close" makes the report
read 1/(4π) instead of 0.07958... — and gives a free loss bump when
the snapped value is, in fact, the truth.

Independent justification (§6.1 discipline)
-------------------------------------------
Physical constants ARE often integer ratios of π / e / √-integers. This
is a property of *physics*, not a property of the Feynman dataset. The
snap library is hand-curated from canonical forms that appear in any
introductory physics curriculum; it doesn't reference the benchmark.

Safety
------
Snap is gated on loss preservation: a snap is only accepted if the
snapped tree's loss is within `accept_tol` (default 1e-6 absolute or
0.1% relative) of the polished tree's loss. Strictly-worse snaps are
rejected. This makes the operation safe-by-construction — it never
moves the Pareto front backwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import (
    Node, collect_const_values, set_const_values,
)
from .scoring import _evaluate_tree


# ---------------- Candidate library ----------------

def _build_snap_candidates() -> list[tuple[float, str]]:
    """Return list of (value, name) snap candidates.

    Hand-curated for physics constants commonly fitted in Feynman-style
    targets. Small denominators (≤ 10), small integer numerators (≤ 10),
    π/e/√ in canonical positions. ~80 entries total.
    """
    pi = math.pi
    e = math.e

    cands: list[tuple[float, str]] = []

    # Integers
    for k in range(-10, 11):
        cands.append((float(k), f"{k}"))

    # Half-integers and small rationals
    small_rationals = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 10),
        (2, 3), (3, 4), (3, 2), (5, 2), (5, 4), (7, 2),
    ]
    for num, den in small_rationals:
        cands.append((num / den, f"{num}/{den}"))
        cands.append((-num / den, f"-{num}/{den}"))

    # π forms (most common in physics)
    pi_factors = [
        (1.0, 1, "pi"), (1.0, 2, "pi^2"), (1.0, 3, "pi^3"),
        (2.0, 1, "2*pi"), (4.0, 1, "4*pi"), (6.0, 1, "6*pi"), (8.0, 1, "8*pi"),
    ]
    for coef, power, name in pi_factors:
        val = coef * (pi ** power)
        cands.append((val, name))
        cands.append((1.0 / val, f"1/({name})"))

    # π fractions in denominator: 1/(k·π^n) for small k, n
    for k_num, k_den in [(1, 2), (1, 3), (1, 4), (1, 6), (1, 8),
                         (1, 12), (1, 16), (1, 24)]:
        v = k_num / (k_den * pi)
        cands.append((v, f"{k_num}/({k_den}*pi)"))

    # π/k and k/π
    for k in [2, 3, 4, 6, 8]:
        cands.append((pi / k, f"pi/{k}"))
        cands.append((k / pi, f"{k}/pi"))

    # e (Euler)
    cands.append((e, "e"))
    cands.append((1.0 / e, "1/e"))
    cands.append((e * e, "e^2"))

    # √ forms
    for k in [2, 3, 5, 6, 10]:
        v = math.sqrt(k)
        cands.append((v, f"sqrt({k})"))
        cands.append((1.0 / v, f"1/sqrt({k})"))

    # 1/sqrt(2π) — Gaussian normalisation
    cands.append((1.0 / math.sqrt(2 * pi), "1/sqrt(2*pi)"))
    cands.append((math.sqrt(2 * pi), "sqrt(2*pi)"))

    return cands


SNAP_CANDIDATES: list[tuple[float, str]] = _build_snap_candidates()
"""Module-level list of (value, name) snap candidates. ~140 entries.
The name field is currently unused but available for downstream tools
that want to report which canonical form the constant was snapped to.
"""


# ---------------- Matching ----------------

def _find_best_match(
    c: float, rel_tol: float, abs_tol: float = 1e-8,
) -> tuple[float, str] | None:
    """Find the closest snap candidate to `c` within `rel_tol`.

    Returns (snap_value, name) if a match exists, else None. Best match
    is the one with smallest relative error.

    rel_tol logic: |c - snap| / max(|c|, abs_tol) < rel_tol.
    The abs_tol denominator floor handles c ≈ 0 cases (without it,
    division-by-zero turns every tiny c into a candidate for snap-to-0).
    """
    denom = max(abs(c), abs_tol)
    best: tuple[float, float, str] | None = None  # (rel_err, value, name)
    for value, name in SNAP_CANDIDATES:
        rel_err = abs(c - value) / denom
        if rel_err < rel_tol:
            if best is None or rel_err < best[0]:
                best = (rel_err, value, name)
    if best is None:
        return None
    return (best[1], best[2])


# ---------------- Snap pass ----------------

@dataclass
class SnapResult:
    """Diagnostic record of one snap-pass invocation."""
    n_consts: int
    n_matched: int
    accepted: bool
    delta_loss: float
    snapped_indices: tuple[int, ...]
    snapped_names: tuple[str, ...]


def snap_constants(
    tree: Node,
    env: dict[str, np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    cache: FunctionalCache,
    current_loss: float,
    *,
    fill_warmup: float = 0.0,
    min_valid_frac: float = 0.9,
    rel_tol: float = 0.005,
    accept_tol_abs: float = 1e-9,
    accept_tol_rel: float = 1e-3,
) -> tuple[Node, float, SnapResult]:
    """Try snapping each Const to a canonical-physical-value candidate.

    Protocol
    --------
    1. Collect all Const leaves in pre-order.
    2. For each value c, find the closest SNAP_CANDIDATES entry within
       `rel_tol` relative error. If none, leave c unchanged.
    3. Build a candidate tree with ALL matched consts replaced at once.
    4. Evaluate the candidate tree.
    5. Accept iff the candidate's loss is no worse than `current_loss`
       within (accept_tol_abs, accept_tol_rel). Otherwise reject and
       return the input tree.

    Parameters
    ----------
    tree, env, y_true, loss_fn, cache : as in `optimize_constants`.
    current_loss : float
        The polished tree's loss BEFORE snapping. Used as the floor
        for accept/reject.
    rel_tol : float
        Maximum relative distance from a candidate for a const to be
        considered a snap target. Default 0.005 = 0.5%.
    accept_tol_abs, accept_tol_rel : float
        Accept the snap if (snapped_loss - current_loss) ≤
        max(accept_tol_abs, accept_tol_rel * |current_loss|). Allows
        snap to be accepted even when it slightly degrades loss
        (canonical form preferred over a marginally better random float).

    Returns
    -------
    (snapped_tree, snapped_loss, SnapResult). If no snap was accepted,
    snapped_tree == tree and snapped_loss == current_loss.
    """
    consts = collect_const_values(tree)
    n_consts = len(consts)
    if not consts:
        return tree, current_loss, SnapResult(
            n_consts=0, n_matched=0, accepted=False,
            delta_loss=0.0, snapped_indices=(), snapped_names=(),
        )

    # For each const, find best snap (if any).
    snapped_indices: list[int] = []
    snapped_names: list[str] = []
    new_values = list(consts)
    for i, c in enumerate(consts):
        match = _find_best_match(c, rel_tol=rel_tol)
        if match is not None:
            snap_value, name = match
            new_values[i] = snap_value
            snapped_indices.append(i)
            snapped_names.append(name)

    n_matched = len(snapped_indices)
    if n_matched == 0:
        return tree, current_loss, SnapResult(
            n_consts=n_consts, n_matched=0, accepted=False,
            delta_loss=0.0, snapped_indices=(), snapped_names=(),
        )

    # Build candidate tree and evaluate.
    candidate_tree = set_const_values(tree, new_values)
    candidate_loss = _evaluate_tree(
        candidate_tree, env, y_true, cache,
        fill_warmup=fill_warmup,
        loss_fn=loss_fn,
        min_valid_frac=min_valid_frac,
    )

    if not np.isfinite(candidate_loss):
        return tree, current_loss, SnapResult(
            n_consts=n_consts, n_matched=n_matched, accepted=False,
            delta_loss=float("inf"),
            snapped_indices=tuple(snapped_indices),
            snapped_names=tuple(snapped_names),
        )

    delta = candidate_loss - current_loss
    tol = max(accept_tol_abs, accept_tol_rel * abs(current_loss))
    accepted = delta <= tol

    if accepted:
        return candidate_tree, float(candidate_loss), SnapResult(
            n_consts=n_consts, n_matched=n_matched, accepted=True,
            delta_loss=float(delta),
            snapped_indices=tuple(snapped_indices),
            snapped_names=tuple(snapped_names),
        )
    return tree, current_loss, SnapResult(
        n_consts=n_consts, n_matched=n_matched, accepted=False,
        delta_loss=float(delta),
        snapped_indices=tuple(snapped_indices),
        snapped_names=tuple(snapped_names),
    )


__all__ = [
    "SNAP_CANDIDATES",
    "SnapResult",
    "snap_constants",
]
