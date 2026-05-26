"""Tier B dispatcher — runs the within-class signatures appropriate
to a given model class, and aggregates them into a Signature.

Per Stage 0/0.5/2.5 design, signatures are not all applicable to every
class:
  - Spectral content meaningless for ALGEBRAIC (no time series)
  - Lyapunov meaningless without dynamics (ALGEBRAIC, often
    DISCRETE_MAP without enough history)
  - Conservation laws require state to be conserved-or-not (so it's
    meaningful for ODE/PDE but trivial for ALGEBRAIC)

This dispatcher selects the right subset per class and assembles the
results into a Signature dataclass for use in identification (Stage 5)
and scoring (Stage 6).
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..types import ModelClass, Trajectory
from .types import Signature, SignatureValue
from .conservation import compute_conservation
from .determinism import compute_determinism
from .dimensionality import compute_effective_dimensionality
from .lyapunov import compute_lyapunov
from .model_class import (
    classify_model_class,
    compute_autocorrelation_structure,
    compute_permutation_invariance,
    compute_stencil_locality,
)
from .modes import compute_mode_count
from .smoothness import compute_smoothness
from .spectral import compute_spectral_content
from .symmetry import compute_symmetry


# Which within-class signatures apply to which ModelClass.
# Algebraic gets smoothness + modes + dim + symmetry (no time-dependent ones).
# Dynamical (ODE/PDE/DISCRETE_MAP) get all.
WITHIN_CLASS_APPLICABILITY = {
    ModelClass.ALGEBRAIC: {
        "smoothness", "mode_count", "effective_dimensionality", "symmetry",
    },
    ModelClass.DISCRETE_MAP: {
        "smoothness", "mode_count", "effective_dimensionality", "symmetry",
        "conservation", "spectral_content", "determinism",
    },
    ModelClass.ODE: {
        "smoothness", "mode_count", "effective_dimensionality", "symmetry",
        "conservation", "spectral_content", "determinism", "lyapunov",
    },
    ModelClass.PDE: {
        "smoothness", "mode_count", "effective_dimensionality", "symmetry",
        "conservation", "spectral_content", "determinism", "lyapunov",
    },
}


def compute_within_class_signature(
    traj: Trajectory,
    model_class: ModelClass,
    *,
    symmetry_candidates: Optional[Iterable[str]] = None,
) -> dict[str, SignatureValue]:
    """Run the within-class signatures applicable to `model_class`.

    Returns a flat dict keyed by signature name; consumer wraps it into
    a Signature dataclass if desired.
    """
    applicable = WITHIN_CLASS_APPLICABILITY.get(model_class, set())
    out: dict[str, SignatureValue] = {}

    if "smoothness" in applicable:
        out["smoothness"] = compute_smoothness(traj)
    if "mode_count" in applicable:
        out["mode_count"] = compute_mode_count(traj)
    if "effective_dimensionality" in applicable:
        out["effective_dimensionality"] = compute_effective_dimensionality(traj)
    if "symmetry" in applicable:
        out["symmetry"] = compute_symmetry(traj, candidates=symmetry_candidates)
    if "conservation" in applicable:
        out["conservation"] = compute_conservation(traj)
    if "spectral_content" in applicable:
        out["spectral_content"] = compute_spectral_content(traj)
    if "determinism" in applicable:
        out["determinism"] = compute_determinism(traj)
    if "lyapunov" in applicable:
        out["lyapunov"] = compute_lyapunov(traj)

    return out


def compute_full_signature(
    traj: Trajectory,
    *,
    model_class: Optional[ModelClass] = None,
    symmetry_candidates: Optional[Iterable[str]] = None,
) -> Signature:
    """Run Tier A + applicable Tier B; assemble into a Signature.

    If `model_class` is provided, skip the Tier A classify step and go
    directly to within-class extraction. If None, classify first.
    """
    # Tier A (always run — they're cheap and we want diagnostics)
    perm = compute_permutation_invariance(traj)
    acf = compute_autocorrelation_structure(traj)
    stencil = compute_stencil_locality(traj)

    if model_class is None:
        inferred, _ = classify_model_class(traj)
    else:
        inferred = model_class

    tier_b = compute_within_class_signature(
        traj, inferred, symmetry_candidates=symmetry_candidates,
    )

    return Signature(
        inferred_model_class=inferred.value,
        permutation_invariance=perm,
        autocorrelation_structure=acf,
        stencil_locality=stencil,
        smoothness=tier_b.get("smoothness"),
        mode_count=tier_b.get("mode_count"),
        effective_dimensionality=tier_b.get("effective_dimensionality"),
        symmetry=tier_b.get("symmetry"),
        conservation=tier_b.get("conservation"),
        spectral_content=tier_b.get("spectral_content"),
        determinism=tier_b.get("determinism"),
        lyapunov=tier_b.get("lyapunov"),
    )
