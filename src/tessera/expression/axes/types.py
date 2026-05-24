"""Axis types for tessera.

An `Axis` declares one dimension of a variable with:
  - a name (e.g., "time", "height", "width", "asset")
  - a size (integer length)
  - an invariance group acting on it

A `TypedVar` is a Var with a tuple of Axis declarations matching the
shape of the variable's data array. The number of axes must equal the
data's ndim.

Invariance groups
-----------------
TRANSLATION — full translation in both directions (used for spatial
              dims of images: shifting the image doesn't change its
              class)
CAUSAL_TRANSLATION — translation but only in one direction (used for
                     TIME: past affects future, never the reverse;
                     tessera's existing Measure has lag ≥ 0 which
                     matches this)
PERMUTATION — any reordering preserves the function (used for
              multi-asset baskets, point clouds, set-valued features)
CYCLIC — rotation on a discrete ring (angles, days-of-week,
         spectral bins)
LOG_TRANSLATION — scale changes (pitch shifts on a log-frequency axis,
                  multiplicative scaling of a positive variable)
ROTATION — continuous rotation (SO(2) for 2-D vector fields,
           SO(3) for 3-D spatial data)
GRAPH — automorphism group of an explicit graph; carries an edge list
        in its parameters
NONE — no assumed invariance; treat as a sequence of independent
       features
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Invariance(Enum):
    """The symmetry group acting on an axis."""
    TRANSLATION = "translation"
    CAUSAL_TRANSLATION = "causal_translation"
    PERMUTATION = "permutation"
    CYCLIC = "cyclic"
    LOG_TRANSLATION = "log_translation"
    ROTATION = "rotation"
    GRAPH = "graph"
    NONE = "none"


@dataclass(frozen=True)
class Axis:
    """Declares one dimension of a TypedVar.

    Examples:
        Axis("time", 1000, Invariance.CAUSAL_TRANSLATION)
        Axis("height", 28, Invariance.TRANSLATION)
        Axis("width", 28, Invariance.TRANSLATION)
        Axis("asset", 8, Invariance.PERMUTATION)
        Axis("hour_of_day", 24, Invariance.CYCLIC)
    """
    name: str
    size: int
    invariance: Invariance

    def __post_init__(self):
        if self.size < 1:
            raise ValueError(f"Axis size must be >= 1, got {self.size}")

    def __repr__(self) -> str:
        return f"Axis({self.name!r}, size={self.size}, "\
               f"inv={self.invariance.value})"


@dataclass(frozen=True)
class TypedVar:
    """A variable with axis-semantic declarations.

    Wraps a `name` (matching tessera.expression.Var.name) with a tuple
    of `Axis` declarations. The order of axes corresponds to the data
    array's dimensions.

    Usage
    -----
        # Time series of price returns
        ts = TypedVar("returns",
                      axes=(Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),))

        # MNIST image (single sample)
        img = TypedVar("image",
                        axes=(Axis("height", 28, Invariance.TRANSLATION),
                              Axis("width", 28, Invariance.TRANSLATION)))

        # Multi-asset basket time series
        basket = TypedVar("prices",
                          axes=(Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),
                                Axis("asset", 8, Invariance.PERMUTATION)))

    The corresponding raw numpy data should have shape matching
    `tuple(a.size for a in self.axes)`.
    """
    name: str
    axes: tuple[Axis, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.name:
            raise ValueError("TypedVar name cannot be empty")

    @property
    def shape(self) -> tuple[int, ...]:
        """Expected data shape derived from axis sizes."""
        return tuple(a.size for a in self.axes)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def __repr__(self) -> str:
        axes_str = ", ".join(repr(a) for a in self.axes)
        return f"TypedVar({self.name!r}, axes=[{axes_str}])"
