"""Functional — n-ary wrapper over Measure.

A Measure handles a single linear functional (Kx)(t) = ∫ x(t−s) dμ(s).
Two natural extensions cover the bulk of useful "second-order" structure:

  1. Bilinear / cross-variable functionals
       F(x, y)(t) = ∫∫ K(s, τ) x(t−s) y(t−τ) ds dτ
     Examples: rolling correlation / covariance between two series.

  2. Quadratic / Volterra-2 functionals on a single series
       F(x)(t)    = ∫∫ K(s, τ) x(t−s) x(t−τ) ds dτ
     Examples: realised variance, autocorrelation features.

When the kernel factors  K(s, τ) = κ_a(s) · κ_b(τ)  (the "separable" case),
Fubini's theorem (Reznikov §3.7, Thm 143) gives:

    F(x, y)(t) = (∫ κ_a(s) x(t−s) ds) · (∫ κ_b(τ) y(t−τ) dτ)
               = (μ_a · x)(t) · (μ_b · y)(t)

I.e. the n-D functional decomposes into a product of two 1-D measure
applications. This makes separable bilinear / Volterra functionals
extremely cheap given the FunctionalCache — both halves are already
cached subexpressions of the surrounding tree.

Non-separable kernels (general 2D measures on lag²) are NOT yet
implemented here; they require true 2D convolution. Most useful applied
functionals are separable, so we ship that case first.

Subclasses
----------
- LinearFunctional   : n=1, wraps a Measure
- SeparableBilinear  : n=2, product of two 1D measures
- Volterra2          : n=1, self-product with two different measures

All subclasses are frozen dataclasses (hashable, comparable by value),
so they slot into the GP search and FunctionalCache cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .measure import Measure


# ---------------- Base ----------------

@dataclass(frozen=True)
class Functional:
    """Abstract base. Use a concrete subclass.

    Subclass contract:
      - frozen + hashable (so it can be a cache key / GP node)
      - n_inputs : int (= arity of apply)
      - apply(*xs) -> np.ndarray  same length as input(s)
    """
    name: str = ""

    @property
    def n_inputs(self) -> int:
        raise NotImplementedError("subclass must declare n_inputs")

    def apply(self, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name or self.__class__.__name__


# ---------------- n=1: Linear ----------------

@dataclass(frozen=True)
class LinearFunctional(Functional):
    """A pure linear functional on one series: y(t) = (μ * x)(t).

    Thin wrapper around Measure. Exists so the n-ary type hierarchy is
    uniform — GP trees and the cache can treat all functionals
    interchangeably.
    """
    measure: Measure = field(default_factory=Measure)

    @property
    def n_inputs(self) -> int:
        return 1

    def apply(self, x: np.ndarray, fill_warmup: float | None = np.nan,
              backend: str = "auto") -> np.ndarray:
        return self.measure.apply(x, fill_warmup=fill_warmup, backend=backend)

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"L[{self.measure}]"


# ---------------- n=2: Separable Bilinear ----------------

@dataclass(frozen=True)
class SeparableBilinear(Functional):
    """Two-input bilinear functional with separable kernel.

    Mathematics
    -----------
    Kernel factors as K(s, τ) = κ_a(s) · κ_b(τ). By Fubini (§3.7):

        F(x, y)(t) = (μ_a * x)(t) · (μ_b * y)(t)

    Both 1D pieces can be cached independently of the bilinear wrapping,
    so the cost of this functional is essentially "one elementwise
    multiply of two cached arrays."

    Common instantiations
    ---------------------
    - μ_a = μ_b = roll_mean(w) gives "rolling product mean" (≈ rolling
      cross-product, the raw building block for correlation).
    - μ_a = δ(0), μ_b = ema(h) gives "current x times smoothed y" —
      a regime-conditioning primitive.
    - μ_a = diff(1), μ_b = diff(1) gives "returns × returns" — sign-
      coincidence indicator.
    """
    measure_a: Measure = field(default_factory=Measure)
    measure_b: Measure = field(default_factory=Measure)

    @property
    def n_inputs(self) -> int:
        return 2

    def apply(self, x: np.ndarray, y: np.ndarray,
              fill_warmup: float | None = np.nan,
              backend: str = "auto") -> np.ndarray:
        a = self.measure_a.apply(x, fill_warmup=fill_warmup, backend=backend)
        b = self.measure_b.apply(y, fill_warmup=fill_warmup, backend=backend)
        # Pointwise product. Where either side is NaN (warmup), result is NaN
        # which is the correct propagation.
        return a * b

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"B[{self.measure_a}, {self.measure_b}]"


# ---------------- n=1: Volterra-2 (self-product) ----------------

@dataclass(frozen=True)
class Volterra2(Functional):
    """Quadratic Volterra functional on one series: F(x) = (μ_a · x) · (μ_b · x).

    The "diagonal" of a separable bilinear with both inputs = x. Lets the
    search discover auto-product structure (realised variance, lagged
    self-correlation) without explicit cross-variable inputs.

    Example: Volterra2(measure_diff(1), measure_diff(1)) on a price
    series gives r(t)² — the squared return — a textbook volatility
    proxy.
    """
    measure_a: Measure = field(default_factory=Measure)
    measure_b: Measure = field(default_factory=Measure)

    @property
    def n_inputs(self) -> int:
        return 1

    def apply(self, x: np.ndarray, fill_warmup: float | None = np.nan,
              backend: str = "auto") -> np.ndarray:
        a = self.measure_a.apply(x, fill_warmup=fill_warmup, backend=backend)
        b = self.measure_b.apply(x, fill_warmup=fill_warmup, backend=backend)
        return a * b

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"V2[{self.measure_a}, {self.measure_b}]"


# ---------------- Cache-aware helpers ----------------

def apply_with_cache(
    functional: Functional,
    cache,                       # FunctionalCache; not typed to avoid circular import
    *,
    var_ids: tuple[str, ...] | str,
    xs: tuple[np.ndarray, ...] | np.ndarray,
    fill_warmup: float | None = np.nan,
    backend: str = "auto",
) -> np.ndarray:
    """Cache-aware apply that memoizes each 1D measure piece.

    For LinearFunctional this is just `cache.get_or_compute(measure, var_id, x)`.

    For SeparableBilinear / Volterra2 we Fubini-decompose: each of the
    two 1D measure applications is fetched (or computed) from the cache
    independently, then multiplied. This is why caching pays off so
    strongly here — the same `ema(x, h=24)` array is reused across
    many functionals that involve it.

    Parameters
    ----------
    functional : Functional (LinearFunctional / SeparableBilinear / Volterra2)
    cache      : FunctionalCache
    var_ids    : single str if n_inputs==1, tuple of strs if n_inputs==2
    xs         : matching 1 or 2 input arrays
    """
    # Normalise to tuple form
    if isinstance(xs, np.ndarray):
        xs = (xs,)
    if isinstance(var_ids, str):
        var_ids = (var_ids,)
    if len(xs) != functional.n_inputs:
        raise ValueError(
            f"{functional.__class__.__name__} expects {functional.n_inputs} inputs, got {len(xs)}"
        )
    if len(var_ids) != functional.n_inputs:
        raise ValueError(
            f"need {functional.n_inputs} var_ids, got {len(var_ids)}"
        )

    if isinstance(functional, LinearFunctional):
        return cache.get_or_compute(
            functional.measure, var_ids[0], xs[0],
            fill_warmup=fill_warmup, backend=backend,
        )

    if isinstance(functional, SeparableBilinear):
        a = cache.get_or_compute(
            functional.measure_a, var_ids[0], xs[0],
            fill_warmup=fill_warmup, backend=backend,
        )
        b = cache.get_or_compute(
            functional.measure_b, var_ids[1], xs[1],
            fill_warmup=fill_warmup, backend=backend,
        )
        return a * b

    if isinstance(functional, Volterra2):
        # Both halves apply to the SAME input series, so they hit the
        # same cache entries if the two measures coincide.
        a = cache.get_or_compute(
            functional.measure_a, var_ids[0], xs[0],
            fill_warmup=fill_warmup, backend=backend,
        )
        b = cache.get_or_compute(
            functional.measure_b, var_ids[0], xs[0],
            fill_warmup=fill_warmup, backend=backend,
        )
        return a * b

    # Generic fallback (no caching): defer to the functional's own apply.
    return functional.apply(*xs, fill_warmup=fill_warmup, backend=backend)


__all__ = [
    "Functional", "LinearFunctional", "SeparableBilinear", "Volterra2",
    "apply_with_cache",
]
