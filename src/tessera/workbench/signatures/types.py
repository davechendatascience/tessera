"""Type definitions for the workbench signatures module.

Per Stage 2.5 design contract (§7), signatures must support both:
  - Identification reading: signature value as matching key
  - Scoring reading: signature deviation as Pareto penalty

The SignatureValue carrier holds the measurement plus confidence and
sample-size metadata so the consuming layer (Stage 5 matching or
Stage 6 scoring) can downweight low-confidence signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SignatureValue:
    """A single signature measurement with confidence + provenance.

    `value` is the measurement (scalar or structured dict for grouped
    tests like symmetry). `confidence` is a measure of reliability in
    [0, 1]; 1 = fully reliable, 0 = no information. `n_samples_used`
    records the trajectory length used (for downstream noise floor
    estimation).
    """
    value: Any
    confidence: float = 1.0
    n_samples_used: int = 0
    notes: str = ""

    def is_reliable(self, threshold: float = 0.5) -> bool:
        return self.confidence >= threshold


@dataclass(frozen=True)
class Signature:
    """Full signature of a trajectory — both Tier A and Tier B results.

    Tier A fields are populated by the model-class classifier and are
    used for routing to within-class extractors. Tier B fields are
    populated only for the classified model class (others left None).

    Per Stage 0.5 §6 + Stage 2.5 §5, a Signature is the matching key
    in identification and the deviation source in scoring.
    """
    # Tier A — model class discriminators
    inferred_model_class: Optional[str] = None
    """str value of ModelClass (algebraic/discrete_map/ode/pde) — or None
    if classification failed (e.g., insufficient data)."""
    permutation_invariance: Optional[SignatureValue] = None
    autocorrelation_structure: Optional[SignatureValue] = None
    stencil_locality: Optional[SignatureValue] = None

    # Tier B — within-class signatures (populated per class)
    smoothness: Optional[SignatureValue] = None
    mode_count: Optional[SignatureValue] = None
    effective_dimensionality: Optional[SignatureValue] = None
    symmetry: Optional[SignatureValue] = None
    conservation: Optional[SignatureValue] = None
    spectral_content: Optional[SignatureValue] = None
    determinism: Optional[SignatureValue] = None
    lyapunov: Optional[SignatureValue] = None
