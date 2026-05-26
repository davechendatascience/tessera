"""tessera.workbench — controlled simulators for methodology evaluation.

This subpackage implements the workbench described in
`docs/research/methodology_workbench_and_library.md` (Stage 0 design note).

Purpose
-------
The workbench is a curated set of canonical dynamical systems with
declared ground-truth metadata (dimensions, symmetries, conservation
laws, smoothness class, mode count). It is the foundation layer of the
library-learning + system-identification framework:

    Stage 1 (this module): canonical systems with generators + metadata
    Stage 2: signature extractors callable from data alone
    Stage 3: information-sufficiency calibration (sample-complexity per system)
    Stage 4: library construction — fit each canonical, store as anchor
    Stage 5: identification pipeline — data in → identified system out
    Stage 6: multi-objective Pareto scoring
    Stage 7: partial-information curriculum
    Stage 8: methodology paper deliverable

What this module is NOT
-----------------------

- Not a benchmark target. The systems here are *methodology test cases*.
- Not a replacement for existing tessera.experimental conjecture modules
  (those still validate scoring-layer interventions; the workbench
  evaluates *which* interventions help on *which* canonical class).
- Not a competitor to PySR / SINDy / AI-Feynman. The workbench is what
  those tools lack — a controlled environment to measure methodology
  contributions independently of any specific solver.

Usage
-----

    from tessera.workbench import get_system, list_systems

    list_systems()
    # ['harmonic_1d', 'damped_harmonic_1d', 'vdp', 'lorenz63',
    #  'fhn', 'heat_1d', 'burgers_1d', 'linear_pendulum',
    #  'nonlinear_pendulum', 'kepler']

    sys = get_system("lorenz63")
    traj = sys.generate(params=sys.default_params(),
                        ic=sys.default_ic(), t_max=30.0, dt=0.01)

    # Each system declares its metadata for the identification pipeline:
    sys.symmetries        # ['time_translation']
    sys.conservation_laws # []  (Lorenz dissipates)
    sys.mode_count        # 1   (single chaotic attractor)
    sys.smoothness_class  # 'analytic'

Discipline
----------

Every canonical system MUST declare:
  - domain, dynamics_doc, state_dim, observable_dim
  - parameter ranges (for the curriculum to sample within)
  - symmetries + conservation_laws (string identifiers, from
    standardized vocabulary in workbench.types)
  - smoothness_class + mode_count
  - default_params() + default_ic() for reproducible defaults

Adding a system requires:
  - Subclassing CanonicalSystem and registering via `register(...)`
  - A unit test verifying generator correctness against analytical or
    well-known statistical properties
  - Updating Section 4 of the Stage 0 design note's canonical list

Removing or modifying a system requires:
  - A CHANGELOG entry explaining why (per the experimental-subpackage
    discipline pattern; the workbench is methodology-load-bearing)
"""
from __future__ import annotations

from .types import (
    Trajectory,
    InformationRequirements,
    CanonicalSystem,
    SYMMETRY_VOCABULARY,
    CONSERVATION_VOCABULARY,
    SMOOTHNESS_CLASSES,
)
from .systems import REGISTRY, get_system, list_systems, register

__all__ = [
    "Trajectory",
    "InformationRequirements",
    "CanonicalSystem",
    "SYMMETRY_VOCABULARY",
    "CONSERVATION_VOCABULARY",
    "SMOOTHNESS_CLASSES",
    "REGISTRY",
    "get_system",
    "list_systems",
    "register",
]
