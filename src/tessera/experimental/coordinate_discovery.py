"""Coordinate-discovery prepass (Conjecture C7).

Provenance: C7 from discussion 2026-05-29 — extends the detect-then-seed
architecture with explicit target-space coordinate discovery before
running the existing power-law / exp-wrapper detectors.

Theoretical pre-analysis: `docs/research/c7_coordinate_discovery.md`
(to be written alongside this module per the experimental discipline).

Status: **UNTESTED** at module-add time. Initial A/B target: Feynman
        benchmark (same as decompose v2). Honest expectation: neutral
        or modest improvement on Feynman because the existing power-law
        detector handles fractional exponents already (sqrt forms via
        exponent=0.5) and exp-wrapper handles Gaussian forms. C7 would
        primarily help on benchmarks where the natural target coordinate
        differs significantly from the raw observable — likely more
        impactful on real-data benchmarks (weather, CAMELS) than on
        Feynman.

The C7 conjecture
-----------------

A meaningful fraction of physical relationships look like power-law
products in a TRANSFORMED target space, where the transform is one
of a small physics-motivated library {identity, log, sqrt, square,
inverse}. By trying each transform on the target before applying
power-law detection, we can:

  1. Discover the right coordinate system AND the right power-law
     simultaneously (the Champion et al. 2019 PNAS framing for SR).
  2. Produce simpler seed trees with integer exponents where the
     baseline detector would have used fractional exponents.
  3. Catch forms that aren't power-law in raw or log space but ARE
     in some other coordinate.

The architectural pattern is the same as the existing
`power_law_seed` orchestrator: detect → seed → inject → let selection
decide. The coordinate library expands what's detectable.

Graduation criterion
--------------------
On the Feynman 30-equation benchmark: at least +1 exact transition
that the existing power-law + exp-wrapper detectors don't already
produce, with 0 regressions on currently-exact equations.

OR

On the weather PDE or CAMELS benchmark: produces a seed tree that
escapes the persistence trap (CAMELS) or beats the diffusion oracle
(weather) where the baseline detectors stay silent.

Removal criterion
-----------------
On Feynman: produces 0 new exact transitions AND no improvement on
seed tree complexity (no integer-exponent simplification).

AND

On real-data benchmarks: produces no improvement over the existing
prepass.

Initial commit: 2026-05-29
Last evaluation: never

What this module provides
-------------------------

    TARGET_TRANSFORMS : dict[str, callable]
        Library of target-space transforms. Each maps y -> y_transformed.
        Members: identity, log_abs, sqrt_abs, square, inverse.

    INVERSE_TRANSFORMS : dict[str, callable]
        Their inverses, used to wrap the discovered power-law seed.

    detect_coord_discovery_seed(env, y, r2_threshold, ...) -> (tree, fit, transform_name)
        Try each transform; run power-law detection on transformed y;
        return the seed tree wrapping the best-fitting transform's inverse.

Design notes
------------

Two failure modes we explicitly accept:

1. **Identity-dominates**: if power-law on raw y has the highest R²,
   coord-discovery degenerates to the existing power_law_seed. Then
   the experimental module is exactly equivalent to production — fine.

2. **Spurious transform-hit**: if some transform gives R²=0.999 but
   for a degenerate reason (y nearly constant, transformed to nearly
   linear), the seed will be misleading. We guard with:
     - Minimum positive-fraction check on the transformed target
     - R² margin requirement (transform R² must beat identity R² by
       >= margin to be accepted)

We do NOT try transforms applied to the FEATURES (x_i → log|x_i|
etc.) because the power-law detector already does that internally
(its design IS log-log space). The novelty is only in target-space
transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, Node,
)
from tessera.search.decompose import (
    PowerLawFit, detect_power_law, build_power_law_tree,
)


# ---------------- Transform library ----------------

def _identity(y: np.ndarray) -> np.ndarray:
    return y.copy()


def _log_abs(y: np.ndarray) -> np.ndarray:
    # log|y|; matches the exp-wrapper detector's transform
    return np.log(np.maximum(np.abs(y), 1e-300))


def _sqrt_abs(y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.abs(y), 0.0))


def _square(y: np.ndarray) -> np.ndarray:
    return y * y


def _inverse(y: np.ndarray) -> np.ndarray:
    return 1.0 / np.where(np.abs(y) > 1e-12, y, 1e-12)


TARGET_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity":  _identity,
    "log_abs":   _log_abs,
    "sqrt_abs":  _sqrt_abs,
    "square":    _square,
    "inverse":   _inverse,
}


# ---------------- Inverse transforms (tree-level) ----------------

def _wrap_identity(tree: Node) -> Node:
    return tree


def _wrap_exp(tree: Node) -> Node:
    """y_transformed = log|y|  →  y = exp(y_transformed)."""
    return UnOp("exp", tree)


def _wrap_square(tree: Node) -> Node:
    """y_transformed = sqrt|y|  →  y = (y_transformed)^2."""
    return BinOp("mul", tree, tree)


def _wrap_sqrt(tree: Node) -> Node:
    """y_transformed = y^2  →  y = sqrt(y_transformed)."""
    return UnOp("sqrt", tree)


def _wrap_inverse(tree: Node) -> Node:
    """y_transformed = 1/y  →  y = 1/y_transformed."""
    return BinOp("div", Const(1.0), tree)


INVERSE_TRANSFORMS: dict[str, Callable[[Node], Node]] = {
    "identity":  _wrap_identity,
    "log_abs":   _wrap_exp,
    "sqrt_abs":  _wrap_square,
    "square":    _wrap_sqrt,
    "inverse":   _wrap_inverse,
}


# ---------------- Result type ----------------

@dataclass
class CoordDiscoveryResult:
    """Result of a coordinate-discovery search."""
    transform_name: str
    inner_fit: PowerLawFit
    seed_tree: Node
    all_r2: dict[str, float]   # R² achieved per transform (None for skipped)

    def __str__(self) -> str:
        return (f"CoordDiscovery({self.transform_name}, inner={self.inner_fit}, "
                f"all_r2={self.all_r2})")


# ---------------- Detector ----------------

def detect_coord_discovery_seed(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    r2_threshold: float = 0.99,
    margin_over_identity: float = 0.0,
    skip_identity: bool = False,
    round_exponents: bool = True,
) -> Optional[CoordDiscoveryResult]:
    """Try each target transform; return the best power-law fit.

    Protocol:
      1. For each transform φ in TARGET_TRANSFORMS:
         - Apply: y_t = φ(y)
         - Run power-law detection on (env, y_t) with the threshold
         - Record R² (or None if the transform's preconditions failed)
      2. Pick the transform with the highest R² ≥ r2_threshold.
      3. If `margin_over_identity > 0`, require the chosen transform's
         R² to exceed identity's by at least that margin (avoids
         degenerate "transform marginally better than identity" picks).
      4. Build the seed tree: inverse_transform(power_law_seed).

    Parameters
    ----------
    skip_identity : bool
        If True, skip identity transform — useful when this module is
        used IN ADDITION to the production `power_law_seed` (which
        already handles identity); a True value here means "only seed
        from a non-trivial transform; defer identity case to the
        production path."

    Returns
    -------
    CoordDiscoveryResult or None.
    """
    all_r2: dict[str, float] = {}
    fits: dict[str, PowerLawFit] = {}

    for name, transform in TARGET_TRANSFORMS.items():
        if skip_identity and name == "identity":
            continue
        try:
            y_t = transform(y)
        except Exception:
            all_r2[name] = float("nan")
            continue
        if not np.all(np.isfinite(y_t)):
            # Try to clip to finite values for the regression. If too
            # many non-finite, skip.
            valid = np.isfinite(y_t)
            if valid.mean() < 0.95:
                all_r2[name] = float("nan")
                continue

        # Use a loose threshold for the per-transform regression to
        # learn its R², then apply the strict threshold afterward.
        fit = detect_power_law(env, y_t, r2_threshold=0.0)
        if fit is None:
            all_r2[name] = float("nan")
            continue
        all_r2[name] = fit.r2
        fits[name] = fit

    # Apply threshold + margin
    identity_r2 = all_r2.get("identity", float("nan"))
    best_name: Optional[str] = None
    best_r2 = -np.inf
    for name, r2 in all_r2.items():
        if not np.isfinite(r2) or r2 < r2_threshold:
            continue
        if (np.isfinite(identity_r2)
                and name != "identity"
                and r2 < identity_r2 + margin_over_identity):
            continue
        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    if best_name is None:
        return None

    inner_fit = fits[best_name]
    # Build the inner power-law tree, then wrap the inverse transform.
    inner_tree = build_power_law_tree(inner_fit, round_exponents=round_exponents)
    if inner_tree is None:
        return None
    seed_tree = INVERSE_TRANSFORMS[best_name](inner_tree)

    return CoordDiscoveryResult(
        transform_name=best_name,
        inner_fit=inner_fit,
        seed_tree=seed_tree,
        all_r2=all_r2,
    )


__all__ = [
    "TARGET_TRANSFORMS",
    "INVERSE_TRANSFORMS",
    "CoordDiscoveryResult",
    "detect_coord_discovery_seed",
]
