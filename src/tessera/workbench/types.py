"""Type definitions for the methodology workbench.

See `docs/research/methodology_workbench_and_library.md` Section 5
("workbench API contract") for design rationale. The Trajectory,
InformationRequirements, and CanonicalSystem types defined here are
the data contract between workbench / signatures / library /
identification pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np


class ModelClass(str, Enum):
    """What kind of mathematical object generated the data.

    See `docs/research/model_class_taxonomy.md` for the full taxonomy
    + the y vs y' vs y'' conflation it addresses.

    ALGEBRAIC    — pure function y = f(x); inputs are iid in no
                   particular order; no time/state structure.
    DISCRETE_MAP — y_{n+1} = f(y_n); iterated map.
    ODE          — dx/dt = f(x, t) on state x (higher-order ODEs
                   reduce to first-order on extended state).
    PDE          — dt(u) = f(u, dx(u), dx^2(u), ...); local update
                   rule on a spatial grid.

    Each model class implies a different `target_form(trajectory)`
    transformation. Stage 5 identification first classifies the model
    class via signature tests before within-class SR.
    """
    ALGEBRAIC = "algebraic"
    DISCRETE_MAP = "discrete_map"
    ODE = "ode"
    PDE = "pde"


# Standardized vocabulary for cross-system signature comparisons.
# Stage 2 signature extractors test for these explicitly; the strings
# must agree across systems.

SYMMETRY_VOCABULARY = frozenset({
    "time_translation",     # f(t) is invariant under t -> t + a
    "time_reversal",        # dynamics invariant under t -> -t (conservative)
    "space_translation",    # PDE invariant under x -> x + a (or rotation in higher dim)
    "rotation_so2",         # 2D rotational symmetry
    "rotation_so3",         # 3D rotational symmetry
    "reflection",           # parity x -> -x
    "scaling",              # invariance under x -> a*x for some a
})

CONSERVATION_VOCABULARY = frozenset({
    "energy",               # H = T + V is conserved
    "angular_momentum",     # L = r x p
    "linear_momentum",
    "phase_volume",         # Liouville's theorem; Hamiltonian flow
    "l2_norm",              # PDE L^2 norm conserved (advection without dissipation)
})

SMOOTHNESS_CLASSES = ("analytic", "c_infty", "c_2", "lipschitz", "general")


@dataclass
class InformationRequirements:
    """Minimum data conditions for a system to be identifiable.

    Calibrated empirically in Stage 3 via sample-complexity sweeps. The
    defaults are conservative placeholders; identification pipelines
    must check `data_meets_requirements(trajectory, info_min)` before
    attempting to fit.
    """
    min_samples: int = 100
    min_trajectories: int = 1
    observable_subset: Optional[list[str]] = None
    """If not None: only these state variables need to be observed.
    None means all state variables must be observable."""
    noise_max: float = 1e-2
    """Maximum noise standard deviation (relative to signal) at which
    identification is expected to succeed."""
    excitation_requirements: list[str] = field(default_factory=list)
    """Free-form notes on what input/IC richness is required, e.g.
    'non-equilibrium IC required' or 'multi-frequency input excitation'."""
    identifiability_proof: Optional[str] = None
    """Citation if a formal identifiability result exists. None means
    requirements are empirically calibrated only."""

    # Per Stage 0.5 note (model_class_taxonomy.md §5.4): record the
    # discrete-vs-continuum gap explicitly for ODE/PDE systems.
    min_dt: Optional[float] = None
    """Maximum time step (None for algebraic — no time-stepping).
    For ODE/PDE, identification is impossible if simulator dt > min_dt
    because truncation error dominates noise."""
    min_dx: Optional[float] = None
    """Maximum spatial step (PDE only; None otherwise)."""
    grid_floor: Optional[dict] = None
    """PDE-specific resolution constraints beyond (min_dt, min_dx).
    Example: {'cfl_max': 0.5, 'stencil_width': 3}. None for non-PDE."""


@dataclass
class Trajectory:
    """A single simulated trajectory plus provenance metadata.

    For ODE/SDE systems: state has shape (n_t, state_dim).
    For PDE systems: state has shape (n_t, *spatial_dims).

    `observable` is what the identification pipeline gets to see, which
    may be a subset of `state` (under-observation curriculum). For Stage 1
    `observable == state`; observable subsetting is a Stage 7 feature.
    """
    t: np.ndarray                       # shape (n_t,)
    state: np.ndarray                   # shape (n_t, ...) — full ground-truth
    observable: np.ndarray              # shape (n_t, ...) — what we measure
    system_id: str                      # which canonical it came from
    params: dict[str, float]            # parameters used for this trajectory
    ic: np.ndarray                      # initial condition (state at t[0])
    noise_std: float = 0.0              # noise applied to observable
    seed: Optional[int] = None          # RNG seed for reproducibility
    meta: dict = field(default_factory=dict)  # free-form extras


class CanonicalSystem(ABC):
    """Abstract base for a canonical dynamical system in the workbench.

    Concrete subclasses must:
      - Set class attributes: id, model_class, dynamics_doc,
        canonical_target_form, state_dim, observable_dim, parameters,
        symmetries, conservation_laws, smoothness_class, mode_count,
        info_min
      - Implement default_params() -> dict[str, float]
      - Implement default_ic() -> np.ndarray
      - Implement generate(...) -> Trajectory
      - Implement target_form(traj) -> (features, target)

    Per Stage 0.5 design contract (`docs/research/model_class_taxonomy.md`),
    `model_class` is the first-class concept declaring what mathematical
    object generated the data; `target_form()` is the model-class-specific
    transformation producing SR-ready (features, target) arrays.

    Implementations should keep generators deterministic given (params, ic,
    seed). The noise_std argument controls observation noise applied to
    `observable`; `state` always remains the ground-truth (noise-free).
    """

    # Class attributes (each subclass sets these)
    id: str
    model_class: ModelClass        # NEW (Stage 0.5) — first-class
    dynamics_doc: str
    canonical_target_form: str     # NEW — one-line description of the
    """Short human-readable description of the canonical target form
    (e.g., 'dt(u) = f(u, dx(u), dx^2(u))' for heat equation)."""
    state_dim: int
    observable_dim: int
    parameters: dict[str, tuple[float, float]]
    """Parameter name -> (min, max) range. For curriculum sampling."""
    symmetries: tuple[str, ...]
    conservation_laws: tuple[str, ...]
    smoothness_class: str
    mode_count: int
    info_min: InformationRequirements

    # Backward-compat alias for one transition window. New code should
    # use model_class; this exposes the same information in the older
    # vocabulary (DEPRECATED — will be removed in a later cleanup).
    @property
    def domain(self) -> str:
        return self.model_class.value

    @abstractmethod
    def default_params(self) -> dict[str, float]:
        """Canonical parameter values (typically the most-studied regime)."""

    @abstractmethod
    def default_ic(self) -> np.ndarray:
        """Canonical initial condition. For PDEs, the initial field."""

    @abstractmethod
    def generate(
        self,
        *,
        params: Optional[dict[str, float]] = None,
        ic: Optional[np.ndarray] = None,
        t_max: float = 30.0,
        dt: float = 0.01,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Trajectory:
        """Simulate the system. Concrete subclasses override.

        Defaults
        --------
        params: self.default_params() if not provided
        ic: self.default_ic() if not provided
        t_max, dt: per-system suitable defaults; can be overridden
        noise_std: Gaussian observation noise std applied to `observable`
        seed: RNG seed for reproducibility

        Returns
        -------
        Trajectory dataclass with state, observable, and provenance.
        """

    @abstractmethod
    def target_form(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
        """Transform a trajectory into (features, target) for SR.

        Model-class-specific:
          ALGEBRAIC    — (inputs, y) where inputs are in traj.meta['inputs']
          DISCRETE_MAP — (state[:-1], state[1:])
          ODE          — (state[:-1], finite-difference of state)
          PDE          — (u[:-1], u[1:] - u[:-1])  where u is the field

        Returns
        -------
        features : np.ndarray
            What the SR target is a function of. Shape depends on model class.
        target : np.ndarray
            What SR is trying to predict. Shape matches features along
            the sample axis.
        """

    # Convenience: declared metadata as a dict (useful for logging,
    # signature comparison, library construction).
    def metadata(self) -> dict:
        return {
            "id": self.id,
            "model_class": self.model_class.value,
            "canonical_target_form": self.canonical_target_form,
            "state_dim": self.state_dim,
            "observable_dim": self.observable_dim,
            "symmetries": list(self.symmetries),
            "conservation_laws": list(self.conservation_laws),
            "smoothness_class": self.smoothness_class,
            "mode_count": self.mode_count,
            "parameter_ranges": dict(self.parameters),
        }

    def __repr__(self) -> str:
        return (f"<CanonicalSystem {self.id} "
                f"({self.model_class.value}, state_dim={self.state_dim})>")


def validate_symmetries(symmetries) -> tuple[str, ...]:
    """Check that all declared symmetries are in the standardized vocab."""
    syms = tuple(symmetries)
    bad = [s for s in syms if s not in SYMMETRY_VOCABULARY]
    if bad:
        raise ValueError(
            f"Unknown symmetry tokens: {bad}. "
            f"Allowed: {sorted(SYMMETRY_VOCABULARY)}"
        )
    return syms


def validate_conservation(laws) -> tuple[str, ...]:
    """Check that all declared conservation laws are in the standardized vocab."""
    ls = tuple(laws)
    bad = [c for c in ls if c not in CONSERVATION_VOCABULARY]
    if bad:
        raise ValueError(
            f"Unknown conservation tokens: {bad}. "
            f"Allowed: {sorted(CONSERVATION_VOCABULARY)}"
        )
    return ls


def validate_smoothness(smoothness: str) -> str:
    if smoothness not in SMOOTHNESS_CLASSES:
        raise ValueError(
            f"Unknown smoothness class {smoothness!r}. "
            f"Allowed: {SMOOTHNESS_CLASSES}"
        )
    return smoothness
