"""Measure-theoretic core for the expression_layer.

A `Measure` is a signed measure on non-negative integer lags (discrete-time
shift positions). By the Lebesgue decomposition theorem, any signed measure
on a Polish space decomposes uniquely into:

    μ = atomic + absolutely_continuous + singular

We support the first two (singular measures are Cantor-set-like and
useless in practice):

  • Atomic part:    Σᵢ wᵢ · δ(lagᵢ)    — finite weighted sum of Dirac masses
                                       gives: lag, diff, signed-sum kernels
  • AC part:        ∫ κ(s) δ(·−s) ds  — convolution with a density
                                       gives: ema, gaussian, power-law,
                                       rectangular (roll_mean), etc.

A Functional (Kx)(t) = ∫ x(t − s) dμ(s) is then specified entirely by μ.
For our discrete time series this becomes:

    (Kx)[t] = Σₖ κ[k] · x[t − k]    for k = 0, 1, …, support_max

where κ is the discrete kernel built from atoms + density (see `to_kernel()`).

Mathematical foundations
------------------------
The reference for this design is measure theory à la Royden / Reznikov
(MAA-5616 lecture notes, §3.3 "Integral as a Measure", §3.7 "Product
Measures", §5.1 "Signed Measures").

Key facts we rely on:

  Thm 99 (Reznikov §3.3): For f ≥ 0 measurable, A ↦ ∫_A f dμ is itself a
  measure φ_f.  → Each density family registered here induces a measure.

  Example 102 (§3.3): On X = ℕ with counting measure, ∫ g dμ = Σₙ gₙ.
  → Infinite sums are integrals against the counting measure; supported
    naturally by the atomic part with arbitrary lag positions.

  Lemma 105 (§3.3) + Lemma 101: For ∫f dμ to be well-defined when f can be
  negative, Σ|wᵢ| < ∞ (absolute summability). Enforced by
  `is_absolutely_summable()`.

  Product measures (§3.7, Fubini): μ_X ⊗ μ_Y on X × Y. Used by higher-
  arity Functionals (bilinear roll_corr, 2D space-time PDE kernels) — not
  implemented in this file, see `functional.py`.

Out of scope
------------
- Nonlinear functionals (Volterra series). Need a 2D-kernel representation
  on lag² space; will live in `functional.py`.
- Time-varying kernels K(s, t). Not shift-invariant — outside the
  measure-theoretic framework we set up here.
- Continuous-time integrals (∫dt). We discretize first. The σ-finite
  condition (§3.7 Def. 136) restricts us to discrete or σ-finite continuous
  measures; counting on ℝ is *not* σ-finite. Working on a discrete grid
  sidesteps this entirely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------- Atomic part: weighted Dirac masses ----------------

@dataclass(frozen=True)
class Atom:
    """A single Dirac mass: weight · δ(lag).

    Causal kernels have lag ≥ 0 (looking into the past). Non-causal lag
    is allowed but the application function handles only causal ops by
    default; non-causal use needs explicit two-sided padding.
    """
    weight: float
    lag: int

    def __post_init__(self):
        if not isinstance(self.lag, int) or self.lag < 0:
            raise ValueError(f"Atom lag must be non-negative int, got {self.lag}")
        if not math.isfinite(self.weight):
            raise ValueError(f"Atom weight must be finite, got {self.weight}")

    def __repr__(self) -> str:
        return f"{self.weight:g}·δ({self.lag})"


# ---------------- Density family registry ----------------

DensityKernelFn = Callable[[dict, int], np.ndarray]
"""Signature: (params: dict, support_max: int) → kernel array κ[0..support_max].

Implementations must:
  - Return κ[k] = ∫_{[k, k+1)} κ_continuous(s) ds (i.e., bucket-integrate),
    OR κ[k] = κ_continuous(k) (point-sample) — pick one and document.
    The discrete kernel is then applied via causal convolution.
  - Be absolutely summable when support_max → ∞ (or document if not).
"""

DENSITY_FAMILIES: dict[str, DensityKernelFn] = {}


def register_density(name: str) -> Callable[[DensityKernelFn], DensityKernelFn]:
    """Decorator to register a density family under the given name."""
    def deco(fn: DensityKernelFn) -> DensityKernelFn:
        if name in DENSITY_FAMILIES:
            raise ValueError(f"density family {name!r} already registered")
        DENSITY_FAMILIES[name] = fn
        return fn
    return deco


@register_density("exponential")
def _density_exponential(params: dict, support_max: int) -> np.ndarray:
    """κ(s) = α · (1−α)^s for s ∈ {0,1,…}, where α = 1 − 2^(−1/h).

    Discrete EMA with halflife h: weight halves every h periods (matches
    pandas .ewm(halflife=h, adjust=False)). Σκ = 1 (probability measure).
    Absolutely summable for any h > 0.

    Note on conventions: "halflife" h means κ(h) = κ(0)/2, so α = 1 − 2^(−1/h).
    If you instead want κ(s) ∝ e^{−s/h} (i.e. h = "decay time"), use the
    "exponential_e" family.
    """
    h = float(params["halflife"])
    if h <= 0:
        raise ValueError(f"halflife must be > 0, got {h}")
    alpha = 1.0 - 2.0 ** (-1.0 / h)
    s = np.arange(support_max + 1, dtype=np.float64)
    return alpha * (1.0 - alpha) ** s


@register_density("exponential_e")
def _density_exponential_e(params: dict, support_max: int) -> np.ndarray:
    """κ(s) ∝ e^{−s/τ} on s ≥ 0, with τ as the e-folding decay time.

    Different parameterization from "exponential" (halflife). Mass-1
    normalised over the truncated support.
    """
    tau = float(params["tau"])
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    s = np.arange(support_max + 1, dtype=np.float64)
    raw = np.exp(-s / tau)
    return raw / raw.sum()


@register_density("power_law")
def _density_power_law(params: dict, support_max: int) -> np.ndarray:
    """κ(s) = C · (1 + s/h)^{−α} for s ≥ 0, with normalizing C s.t. Σκ = 1.

    Absolutely summable iff α > 1. Long-memory / fat-tailed (slow decay
    compared to exponential).
    """
    h = float(params["scale"])
    alpha = float(params["alpha"])
    if h <= 0:
        raise ValueError(f"scale must be > 0, got {h}")
    if alpha <= 1.0:
        raise ValueError(f"power_law density requires α > 1 for summability, got {alpha}")
    s = np.arange(support_max + 1, dtype=np.float64)
    raw = (1.0 + s / h) ** (-alpha)
    return raw / raw.sum()   # normalise to mass 1 on the truncated support


@register_density("gaussian_half")
def _density_gaussian_half(params: dict, support_max: int) -> np.ndarray:
    """One-sided Gaussian kernel: κ(s) ∝ exp(−s²/(2σ²)) for s ≥ 0.

    Used for one-sided smoothing. Σκ ≈ 1 (after normalisation on the
    truncated support).
    """
    sigma = float(params["sigma"])
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    s = np.arange(support_max + 1, dtype=np.float64)
    raw = np.exp(-(s ** 2) / (2.0 * sigma * sigma))
    return raw / raw.sum()


@register_density("rectangular")
def _density_rectangular(params: dict, support_max: int) -> np.ndarray:
    """Uniform window: κ(s) = 1/w for 0 ≤ s < w, else 0.

    Computes a w-period moving average. Mass 1.
    """
    w = int(params["window"])
    if w < 1:
        raise ValueError(f"window must be ≥ 1, got {w}")
    k = np.zeros(support_max + 1, dtype=np.float64)
    k[: min(w, support_max + 1)] = 1.0 / w
    return k


@register_density("delta_minus_exp")
def _density_delta_minus_exp(params: dict, support_max: int) -> np.ndarray:
    """High-pass: δ(0) − exponential. Signed density.

    Captures "residual from the EMA" in a single primitive. Equivalent to
    x − ema(x, h), but exposed as a single measure so the GP can search it.
    Uses the same halflife convention as "exponential".
    """
    h = float(params["halflife"])
    if h <= 0:
        raise ValueError(f"halflife must be > 0, got {h}")
    alpha = 1.0 - 2.0 ** (-1.0 / h)
    s = np.arange(support_max + 1, dtype=np.float64)
    k = -alpha * (1.0 - alpha) ** s
    k[0] += 1.0
    return k


# ---------------- Measure: atomic ⊕ AC decomposition ----------------

@dataclass(frozen=True)
class Measure:
    """Signed measure on ℕ (non-negative integer lags).

    Lebesgue-decomposed: μ = atoms + density. Either part may be empty
    (resulting in a pure atomic, pure AC, or zero measure).

    Attributes
    ----------
    atoms : tuple of Atom
        Finite list of weighted Dirac masses. Order does not matter.
    density_family : str | None
        Key in DENSITY_FAMILIES, or None if no AC part.
    density_params : tuple of (str, float) pairs
        Frozen dict of parameters for the density family. Stored as
        tuple-of-tuples to keep the dataclass hashable.
    support_max : int
        Maximum lag at which the kernel is materialised. For infinite-
        support kernels (exponential, power-law), truncation here decides
        the tail mass discarded. Default 256 ≈ 7 halflives at h=24h.

    Invariants
    ----------
    - All atom lags must be ≤ support_max (else they're silently dropped
      by to_kernel() — checked in __post_init__).
    - Density family if specified must be registered.
    """
    atoms: tuple[Atom, ...] = ()
    density_family: str | None = None
    density_params: tuple[tuple[str, float], ...] = ()
    support_max: int = 256

    def __post_init__(self):
        if self.support_max < 0:
            raise ValueError(f"support_max must be ≥ 0, got {self.support_max}")
        for atom in self.atoms:
            if atom.lag > self.support_max:
                raise ValueError(
                    f"atom lag {atom.lag} exceeds support_max {self.support_max}"
                )
        # Canonicalise the atoms tuple at construction so two semantically
        # identical Measures hash equal. Required for the FunctionalCache
        # to recognise mathematically-equivalent measures across mutations
        # (per docs/research_notes/measure_theory_and_perfect_info.md §3.1
        # on Lebesgue decomposition uniqueness).
        #
        # Three operations:
        #   1. Merge atoms with the same lag (sum their weights)
        #   2. Drop atoms with effectively-zero weight (|w| < 1e-12)
        #   3. Sort the survivors by lag ascending (canonical order)
        canonical = self._canonicalise_atoms(self.atoms)
        if canonical != self.atoms:
            # Frozen dataclass: must use object.__setattr__ to mutate.
            object.__setattr__(self, "atoms", canonical)
        if self.density_family is not None:
            if self.density_family not in DENSITY_FAMILIES:
                raise ValueError(
                    f"unknown density family {self.density_family!r}; "
                    f"registered: {list(DENSITY_FAMILIES)}"
                )
            # Eager validation: probe the density with the actual params so
            # malformed params (e.g. power_law α ≤ 1) fail at construction
            # rather than first apply().
            try:
                DENSITY_FAMILIES[self.density_family](dict(self.density_params), min(self.support_max, 32))
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"density family {self.density_family!r} failed validation with "
                    f"params {dict(self.density_params)}: {e}"
                ) from e

    # ---- Canonicalisation ----

    @staticmethod
    def _canonicalise_atoms(
        atoms: tuple[Atom, ...],
        zero_tol: float = 1e-12,
    ) -> tuple[Atom, ...]:
        """Canonical form of an atoms tuple: merge by lag, drop zeros, sort.

        Pure function on the tuple — no `self` access — so it's safe to
        call from `__post_init__` before the dataclass is fully built.

        Steps:
          1. Group atoms by `lag`; sum the weights within each group.
          2. Drop groups whose summed weight is |w| < zero_tol.
          3. Sort the surviving (weight, lag) pairs by lag ascending.

        Returns a NEW tuple; does not mutate the input.
        """
        if not atoms:
            return ()
        merged: dict[int, float] = {}
        for a in atoms:
            merged[a.lag] = merged.get(a.lag, 0.0) + a.weight
        # Drop near-zero weights, sort by lag
        out = sorted(
            ((w, lag) for lag, w in merged.items() if abs(w) >= zero_tol),
            key=lambda wl: wl[1],
        )
        return tuple(Atom(weight=w, lag=lag) for w, lag in out)

    # ---- Inspection ----

    @property
    def has_atomic(self) -> bool:
        return len(self.atoms) > 0

    @property
    def has_density(self) -> bool:
        return self.density_family is not None

    @property
    def density_params_dict(self) -> dict:
        return dict(self.density_params)

    def total_atomic_mass(self) -> float:
        """Σ wᵢ across atoms (algebraic sum, can be negative or zero)."""
        return sum(a.weight for a in self.atoms)

    def is_absolutely_summable(self) -> bool:
        """∫|dμ| < ∞ ?  Lemma 105 in Reznikov §3.3.

        Atomic part is finite ⇒ always summable. Density part is checked
        by materialising the kernel and summing |κ[k]| on the truncated
        support, plus a guard against NaN/Inf.

        Note: this checks summability of the *truncated* kernel; for
        registered families with documented summability (exponential,
        power_law with α>1, gaussian) this is always true. Custom
        densities may need tighter analytical checks.
        """
        # Atoms: weight finiteness already enforced in Atom.__post_init__
        if not all(math.isfinite(a.weight) for a in self.atoms):
            return False
        if self.has_density:
            try:
                k = DENSITY_FAMILIES[self.density_family](
                    self.density_params_dict, self.support_max
                )
            except Exception:
                return False
            l1 = float(np.abs(k).sum())
            if not math.isfinite(l1):
                return False
        return True

    # ---- Composition (convolution of measures) ----

    def compose(self, other: "Measure", zero_tol: float = 1e-12) -> "Measure":
        """Measure convolution: μ * ν.

        Mathematically (Reznikov §3.4):
            ∫ x(t - τ) d(μ*ν)(τ)
              = ∫∫ x(t - τ_1 - τ_2) dμ(τ_1) dν(τ_2)

        Operationally: applying `μ.compose(ν)` to x gives the same
        result as applying ν first, then applying μ to the result.

        That is the identity that lets the SR search collapse
        `L_μ(L_ν(x)) → L_{μ*ν}(x)` — see the §3.3 (measure-algebra
        identities) of docs/research_notes/measure_theory_and_perfect_info.md.

        Implementation: discrete convolution of the two kernels via
        `np.convolve`. The resulting kernel is then sparsified into
        an atomic representation (one atom per nonzero coefficient).
        We lose the compact density representation, but keep semantic
        identity. For dense convolutions, this can produce many atoms;
        in practice tessera's search keeps measures small enough that
        the atomic form stays tractable.

        Returns
        -------
        A new Measure representing μ*ν. Atomic-only; the support is
        the sum of the inputs' supports.
        """
        k_a = self.to_kernel()
        k_b = other.to_kernel()
        k_compose = np.convolve(k_a, k_b)
        atoms = tuple(
            Atom(weight=float(w), lag=int(lag))
            for lag, w in enumerate(k_compose) if abs(w) >= zero_tol
        )
        support_max = max(0, len(k_compose) - 1)
        return Measure(atoms=atoms, support_max=support_max)

    # ---- Materialisation ----

    def to_kernel(self) -> np.ndarray:
        """Build the discrete kernel array κ[0..support_max].

        (Kx)[t] = Σ_{k=0}^{support_max} κ[k] · x[t − k].
        """
        N = self.support_max + 1
        kernel = np.zeros(N, dtype=np.float64)
        if self.has_density:
            d = DENSITY_FAMILIES[self.density_family](self.density_params_dict, self.support_max)
            kernel[: len(d)] += d
        for atom in self.atoms:
            kernel[atom.lag] += atom.weight
        return kernel

    def apply(
        self,
        x,
        fill_warmup: float | None = np.nan,
        *,
        backend: str = "auto",
    ):
        """Apply the measure as a causal convolution on a 1-D series.

        Returns y of length len(x). Rows where the kernel reaches beyond
        the start of the series (`t < max_nonzero_lag`) are warmup and
        filled with `fill_warmup` (NaN by default; set to 0 to zero-pad).

        Input dispatch
        --------------
        If `x` is a JAX array (detected by type), routes to a JAX-native
        kernel-materialise + jnp.convolve path. The output is a JAX array
        on the same device. Numba and FFT fast-paths are bypassed.

        If `x` is a numpy array (default), uses the original numba/FFT
        fast-path family:
          - **Recursive EMA** (O(N)) when the measure is *purely* a single
            exponential density with no atoms — bypasses the truncated
            convolution and matches pandas' .ewm exactly.
          - **Atomic shift-and-accumulate** (O(N · #atoms)) for atomic-
            only measures.
          - **Direct conv via Numba** (O(N·K)) for short kernels (K ≤ 64).
          - **FFT conv** (O(N log N)) for long kernels.

        `backend="kernel"` forces materialising kernel + direct conv. Used
        by tests to verify the routing matches.

        Note on warmup
        --------------
        Warmup = largest lag at which the kernel is non-zero, NOT
        support_max. A sparse atomic kernel with a single Dirac at lag=5
        inside support_max=200 has only 5 warmup rows.
        """
        # JAX-array fast path: route through jnp.convolve. No numba.
        if type(x).__module__.startswith("jax"):
            return self._apply_jax(x, fill_warmup)

        from ._numba_kernels import (
            ema_recursive, atomic_apply, conv_causal,
        )

        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return np.empty(0, dtype=np.float64)

        fwm = np.nan if fill_warmup is None else float(fill_warmup)

        if backend == "auto":
            # Fast path: pure exponential, no atoms → recursive
            if (not self.has_atomic
                    and self.density_family == "exponential"
                    and len(self.density_params) == 1):
                h = float(dict(self.density_params)["halflife"])
                return ema_recursive(x, h, fill_warmup=fwm, warmup_rows=0)

            # Fast path: pure atomic → shift-and-accumulate
            if self.has_atomic and not self.has_density:
                lags = np.array([a.lag for a in self.atoms], dtype=np.int64)
                weights = np.array([a.weight for a in self.atoms], dtype=np.float64)
                return atomic_apply(x, lags, weights, fill_warmup=fwm)

            # General path: materialise the kernel, conv (FFT or direct)
            kernel = self.to_kernel()
            return conv_causal(x, kernel, fill_warmup=fwm)

        elif backend == "kernel":
            # Forced kernel path — used for golden comparisons in tests
            kernel = self.to_kernel()
            return conv_causal(x, kernel, fill_warmup=fwm)

        else:
            raise ValueError(f"unknown backend {backend!r}")

    def _apply_jax(self, x, fill_warmup):
        """JAX path: materialize kernel, run jnp.convolve, apply causal warmup.

        Always uses kernel-materialise + same-mode conv. Doesn't try to
        use the recursive-EMA fast-path (would need jax.lax.scan; deferred
        to a later tier for jit-friendly perf). Sufficient for end-to-end
        correctness and ~10-100x faster than numpy at large N on GPU.
        """
        import jax.numpy as jnp

        x = jnp.asarray(x).ravel()
        if x.size == 0:
            return jnp.empty(0, dtype=jnp.float64)

        # Build kernel as JAX array (kernel is small; materialising is cheap)
        kernel_np = self.to_kernel()
        kernel = jnp.asarray(kernel_np, dtype=x.dtype)
        K = kernel.shape[0]
        N = x.shape[0]

        # Causal convolution: y[t] = sum_{k=0..K-1} kernel[k] * x[t-k]
        # This equals np.convolve(x, kernel, mode="full")[:N] when both
        # are 1-D with kernel oriented as [k=0, k=1, ...]. jnp.convolve
        # has the same semantics as np.convolve.
        full = jnp.convolve(x, kernel, mode="full")
        y = full[:N]

        # Warmup mask: first (max_nonzero_lag) rows
        # Find max non-zero lag in the kernel.
        if K > 0:
            # max_nonzero_lag = K-1 if kernel is dense; for sparse kernels,
            # detect from atoms (kernel positions with non-zero weight)
            max_lag = int(K - 1)
        else:
            max_lag = 0
        if max_lag > 0 and fill_warmup is not None:
            fwm = float(fill_warmup) if not (isinstance(fill_warmup, float) and np.isnan(fill_warmup)) else jnp.nan
            t = jnp.arange(N)
            y = jnp.where(t < max_lag, fwm, y)
        return y

    # ---- Pretty-printing ----

    def __str__(self) -> str:
        parts: list[str] = []
        for a in self.atoms:
            parts.append(str(a))
        if self.has_density:
            ps = ",".join(f"{k}={v:g}" for k, v in self.density_params)
            parts.append(f"{self.density_family}({ps})[0,{self.support_max}]")
        return " + ".join(parts) if parts else "0"


# ---------------- Convenience constructors for common kernels ----------------

def measure_lag(k: int) -> Measure:
    """Pure lag-by-k:  δ(s − k)."""
    return Measure(atoms=(Atom(1.0, int(k)),), support_max=max(int(k), 1))


def measure_diff(k: int = 1) -> Measure:
    """Finite difference at lag k:  δ(0) − δ(k)."""
    k = int(k)
    return Measure(atoms=(Atom(1.0, 0), Atom(-1.0, k)), support_max=max(k, 1))


def measure_ema(halflife: float, support_max: int | None = None) -> Measure:
    """Exponential moving average. Density α(1−α)^s with α = 1 − e^{−1/h}."""
    h = float(halflife)
    if support_max is None:
        support_max = max(int(round(7 * h)) + 1, 8)
    return Measure(
        density_family="exponential",
        density_params=(("halflife", h),),
        support_max=int(support_max),
    )


def measure_roll_mean(window: int) -> Measure:
    """Rectangular window average over `window` lags."""
    w = int(window)
    return Measure(
        density_family="rectangular",
        density_params=(("window", float(w)),),
        support_max=w,
    )


def measure_power_law(scale: float, alpha: float, support_max: int = 256) -> Measure:
    """Power-law (long-memory) kernel. Requires α > 1 for summability."""
    return Measure(
        density_family="power_law",
        density_params=(("scale", float(scale)), ("alpha", float(alpha))),
        support_max=int(support_max),
    )


def measure_signed_sum(weights_and_lags: Sequence[tuple[float, int]]) -> Measure:
    """Arbitrary finite signed sum of Dirac masses.

    The most general atomic measure with finite support. Subsumes lag,
    diff, k-step diff, and any custom Stieltjes-step kernel.
    """
    atoms = tuple(Atom(float(w), int(k)) for w, k in weights_and_lags)
    max_lag = max((a.lag for a in atoms), default=0)
    return Measure(atoms=atoms, support_max=max(max_lag, 1))


__all__ = [
    "Atom", "Measure", "DENSITY_FAMILIES", "register_density",
    "measure_lag", "measure_diff", "measure_ema", "measure_roll_mean",
    "measure_power_law", "measure_signed_sum",
]
