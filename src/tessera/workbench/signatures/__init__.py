"""tessera.workbench.signatures — data-derivable system properties.

Stage 2 deliverable per the design contracts in:
  - docs/research/methodology_workbench_and_library.md §6
  - docs/research/model_class_taxonomy.md §6 (Stage 5 pipeline)
  - docs/research/per_class_loss_and_multi_objective.md §5

Two-tier architecture
---------------------

**Tier A — model class discriminators** (cheap; always run first):
  - permutation_invariance: shuffle test (algebraic-vs-dynamical)
  - autocorrelation_structure: time/space ACF (algebraic / ODE / PDE)
  - stencil_locality: PDE-only test for finite stencil

  classify_model_class(traj) -> ModelClass

**Tier B — within-class signatures** (run after model class is known):
  - smoothness: Hölder regularity exponent
  - mode_count: number of attractors / regimes
  - effective_dimensionality: state-space manifold dimension
  - symmetry: pass/fail per candidate group
  - conservation: best low-order conserved scalar
  - spectral_content: power spectrum summary
  - determinism: k-NN forecast vs surrogate null
  - lyapunov: max Lyapunov exponent estimate

  compute_signature(traj, model_class) -> Signature

Bidirectional reading
---------------------

Per Stage 2.5 note, signatures serve two purposes:
  1. IDENTIFICATION (Stage 5): signature measured on unknown data →
     distance to library anchors
  2. SCORING (Stage 6): signature DEVIATION between declared system
     and candidate-fit's simulated trajectory → multi-objective
     Pareto penalty

Both readings use the same Signature dataclass. Stage 2 ships the
extractors; Stage 5 and Stage 6 consume them.
"""
from __future__ import annotations

from .types import Signature, SignatureValue
from .model_class import (
    classify_model_class,
    compute_permutation_invariance,
    compute_autocorrelation_structure,
    compute_stencil_locality,
)

__all__ = [
    "Signature",
    "SignatureValue",
    "classify_model_class",
    "compute_permutation_invariance",
    "compute_autocorrelation_structure",
    "compute_stencil_locality",
]
