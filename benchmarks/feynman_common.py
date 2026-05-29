"""Shared helpers for Feynman A/B benchmark runners.

Background
----------
Each Feynman A/B runner (constopt, snap, decompose, decompose_v2, c7)
duplicated an identical `classify_verdict(rel)` function — the same
three-tier (exact / partial / failed) classifier mapping the candidate
relative-loss to a verdict string. This module centralizes that.

Per the integration note (2026-05-29): the helpers here are the
*Feynman-specific* verdict logic. The Weather and CAMELS verdict
classifiers are deliberately NOT consolidated here because their
domain conditions are different — Weather uses train_rel relative
to oracle ceiling; CAMELS uses two-anchor (persistence + engineering).
Forcing those into a common framework would muddle the semantic
distinctions.

What's shared
-------------
- `classify_feynman_verdict(rel)` — the three-tier classifier
- `EXACT_THRESHOLD`, `PARTIAL_THRESHOLD` — thresholds as named constants

Usage
-----
    from benchmarks.feynman_common import classify_feynman_verdict
    v = classify_feynman_verdict(0.005)   # → "exact"
"""
from __future__ import annotations

import numpy as np


EXACT_THRESHOLD: float = 0.01
"""rel < this → exact verdict. Same convention as AI Feynman paper's
'NSE_inf' (the noise-free recovery test): the discovered tree
matches the analytical form to within numerical const-opt precision.
"""

PARTIAL_THRESHOLD: float = 0.20
"""EXACT_THRESHOLD ≤ rel < this → partial verdict. Used historically
in tessera Feynman reports; meaningfully better than zero-prediction
baseline (rel ≈ 1.0) but not at the analytical-form recovery level.
"""


def classify_feynman_verdict(rel: float) -> str:
    """Classify a candidate's relative loss as exact / partial / failed.

    rel = train_loss / var(y) is the standard SR-on-Feynman metric.
    NaN/inf → failed.
    """
    if not np.isfinite(rel):
        return "failed"
    if rel < EXACT_THRESHOLD:
        return "exact"
    if rel < PARTIAL_THRESHOLD:
        return "partial"
    return "failed"


__all__ = [
    "EXACT_THRESHOLD",
    "PARTIAL_THRESHOLD",
    "classify_feynman_verdict",
]
