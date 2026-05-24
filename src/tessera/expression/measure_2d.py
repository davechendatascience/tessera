"""2D measures — extends the 1D Measure to space-time / image-like fields.

Where 1D measure handled
    (K·x)(t) = Σ_s κ(s) x(t − s)
2D handles
    (K·U)(t, x) = Σ_{s_t, s_x} κ(s_t, s_x) U(t − s_t, x − s_x)

Lebesgue decomposition (same as 1D)
-----------------------------------
A Measure2D is `atomic ⊕ separable density` where:

  • Atomic part: finite weighted sum of 2D Dirac masses
        Σᵢ wᵢ δ(s_t − tᵢ, s_x − xᵢ)
    Used for finite-difference operators (Laplacian, Sobel gradients, etc.).

  • Separable density: μ_t(s_t) · μ_x(s_x), product of two 1D Measures.
    By Fubini (§3.7 Thm 143) the apply decomposes into two 1D convs:
        (K·U)(t, x) = ((κ_t along time of (κ_x along space of U))(t, x)
    Used for Gaussian blur, smoothing, EMA-in-time-with-Gaussian-in-space.

Non-separable continuous densities (where κ(s_t, s_x) doesn't factor) are
NOT in this v0. They'd require materialising the full 2D kernel and using
scipy.signal.fftconvolve(mode='same'); add when a use case arises.

Convention on lags
------------------
  • lag_t  ∈ {0, 1, 2, …}     causal in time (look only into the past)
  • lag_x  ∈ ℤ                signed in space (left, centre, right
                              neighbours all meaningful — Laplacians need
                              ±1 lag_x; pure spatial gradients need ±1).

This asymmetry reflects the physical reality: time has a direction (we
can't look into the future), space generally doesn't (PDE operators are
local in both directions). The user can still force causal-in-space by
clamping lag_x ≥ 0 in their Atom2D constructions.

Input arrays
------------
A 2D measure applies to a 2D numpy array U of shape (T, X) where T is
the time axis (axis 0) and X is the space axis (axis 1). Warmup rows
(t < max lag_t) are filled with `fill_warmup` (NaN by default); spatial
boundary rows depend on the lag_x range and are handled by causal+ /
zero-padding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .measure import Measure


# ---------------- Atom2D ----------------

@dataclass(frozen=True)
class Atom2D:
    """A 2D Dirac mass: weight · δ(s_t − lag_t, s_x − lag_x).

    lag_t must be ≥ 0 (causal in time). lag_x is signed (centred or
    one-sided spatial operators are both valid).
    """
    weight: float
    lag_t: int
    lag_x: int

    def __post_init__(self):
        if not isinstance(self.lag_t, int) or self.lag_t < 0:
            raise ValueError(f"Atom2D lag_t must be non-negative int, got {self.lag_t}")
        if not isinstance(self.lag_x, int):
            raise ValueError(f"Atom2D lag_x must be int, got {self.lag_x}")
        if not math.isfinite(self.weight):
            raise ValueError(f"Atom2D weight must be finite, got {self.weight}")

    def __repr__(self) -> str:
        return f"{self.weight:g}·δ({self.lag_t},{self.lag_x})"


# ---------------- Measure2D ----------------

@dataclass(frozen=True)
class Measure2D:
    """Signed measure on ℕ × ℤ (causal time, signed space).

    Decomposed into atomic + separable-density parts. Either part may be
    empty:
      - atoms-only: finite-difference operators (Laplacian, gradients)
      - density-only: blur/smooth/EMA-in-time × Gaussian-in-space
      - both: combined transforms

    Attributes
    ----------
    atoms : tuple of Atom2D
        Finite list of weighted 2D Dirac masses. Order doesn't matter.
    sep_t : Measure | None
        1D measure to apply along the time axis (separable density part).
    sep_x : Measure | None
        1D measure to apply along the space axis. The pair (sep_t, sep_x)
        encodes the separable density κ(s_t, s_x) = κ_t(s_t) · κ_x(s_x).
        Both must be present together, or both absent.
    """
    atoms: tuple[Atom2D, ...] = ()
    sep_t: Measure | None = None
    sep_x: Measure | None = None

    def __post_init__(self):
        if (self.sep_t is None) != (self.sep_x is None):
            raise ValueError(
                "Measure2D separable density needs BOTH sep_t and sep_x, "
                f"got sep_t={self.sep_t!r}, sep_x={self.sep_x!r}"
            )

    # ---- Inspection ----

    @property
    def has_atomic(self) -> bool:
        return len(self.atoms) > 0

    @property
    def has_density(self) -> bool:
        return self.sep_t is not None

    def max_lag_t(self) -> int:
        m = max((a.lag_t for a in self.atoms), default=0)
        if self.has_density:
            m = max(m, self.sep_t.support_max)
        return m

    def max_abs_lag_x(self) -> int:
        m = max((abs(a.lag_x) for a in self.atoms), default=0)
        if self.has_density:
            # sep_x is causal-in-space by default (Measure has lag ≥ 0); a
            # user can still build centred spatial smoothers by sticking
            # appropriate atoms in the atomic part.
            m = max(m, self.sep_x.support_max)
        return m

    def is_absolutely_summable(self) -> bool:
        """Atoms: finite, so always summable. Density: defer to 1D check."""
        if not all(math.isfinite(a.weight) for a in self.atoms):
            return False
        if self.has_density:
            return self.sep_t.is_absolutely_summable() and self.sep_x.is_absolutely_summable()
        return True

    # ---- Apply ----

    def apply(self, U, fill_warmup: float | None = np.nan):
        """Apply the 2D measure to a (T, X) field.

        (K·U)(t, x) = atomic part + separable density part
                    = Σᵢ wᵢ U(t − tᵢ, x − xᵢ)
                      + (κ_t along time of (κ_x along space of U))(t, x)

        Input dispatch
        --------------
        If `U` is a JAX array (detected by type), routes to `_apply_jax`,
        which is a faithful jit-friendly rewrite of the numpy path using
        `jax.numpy` + `Y.at[...].add(...)` for the atomic shift-and-
        accumulate and `jax.scipy.signal.convolve2d` for the separable
        density. Output is a JAX array on the same device.

        Returns
        -------
        np.ndarray (or jax.Array) of shape (T, X). Time-warmup rows
        (t < max_lag_t) and space-boundary columns are filled with
        `fill_warmup`.
        """
        # JAX-array fast path
        if type(U).__module__.startswith("jax"):
            return self._apply_jax(U, fill_warmup)

        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2:
            raise ValueError(f"Measure2D requires a 2D input, got shape {U.shape}")
        T, X = U.shape
        Y = np.zeros((T, X), dtype=np.float64)

        # Atomic part: shift-and-accumulate
        for a in self.atoms:
            # Apply weight · U[t - lag_t, x - lag_x]; out-of-bounds positions
            # become "warmup" cells which we mark after the loop.
            if a.weight == 0.0:
                continue
            t_src_start = max(0, a.lag_t)
            t_dst_start = max(0, a.lag_t)
            # In time: causal — shift down by lag_t
            # In space: signed — shift left if lag_x>0, right if lag_x<0
            x_src_start = max(0, -a.lag_x)
            x_dst_start = max(0, a.lag_x)
            x_len = X - abs(a.lag_x)
            if x_len <= 0:
                continue
            t_len = T - a.lag_t
            if t_len <= 0:
                continue
            Y[t_dst_start: t_dst_start + t_len, x_dst_start: x_dst_start + x_len] += (
                a.weight * U[
                    t_src_start - a.lag_t: t_src_start - a.lag_t + t_len,
                    x_src_start: x_src_start + x_len,
                ]
            )

        # Separable density part: apply sep_x along axis 1 (each time-row),
        # then sep_t along axis 0 (each space-column).
        if self.has_density:
            mid = np.empty_like(U)
            for ti in range(T):
                mid[ti] = self.sep_x.apply(U[ti], fill_warmup=0.0)
            sep_t_part = np.empty_like(U)
            for xj in range(X):
                sep_t_part[:, xj] = self.sep_t.apply(mid[:, xj], fill_warmup=0.0)
            Y += sep_t_part

        # Mark warmup region
        warmup_t = self.max_lag_t()
        warmup_x_left = max((a.lag_x for a in self.atoms), default=0)   # positive lag_x → left warmup
        warmup_x_right = max((-a.lag_x for a in self.atoms), default=0) # negative lag_x → right warmup
        if self.has_density:
            warmup_x_left = max(warmup_x_left, self.sep_x.support_max)

        if fill_warmup is None or math.isnan(float(fill_warmup)):
            fwm = np.nan
        else:
            fwm = float(fill_warmup)

        if warmup_t > 0:
            Y[:warmup_t, :] = fwm
        if warmup_x_left > 0:
            Y[:, :warmup_x_left] = fwm
        if warmup_x_right > 0:
            Y[:, X - warmup_x_right:] = fwm

        return Y

    def _apply_jax(self, U, fill_warmup):
        """JAX-native rewrite of Measure2D.apply.

        Mirrors the numpy logic structurally: atomic shift-and-accumulate
        via `Y.at[...].add(...)`, separable density via
        `jax.scipy.signal.convolve2d` with the materialised outer-product
        kernel, then warmup mask via `Y.at[...].set(fwm)`.

        Output dtype follows U's dtype (typically float32 under JAX).
        """
        import jax.numpy as jnp

        if U.ndim != 2:
            raise ValueError(
                f"Measure2D requires a 2D input, got shape {U.shape}"
            )
        T, X = U.shape
        Y = jnp.zeros((T, X), dtype=U.dtype)

        # Atomic part: shift-and-accumulate via functional .at[] updates
        for a in self.atoms:
            if a.weight == 0.0:
                continue
            t_src_start = max(0, a.lag_t)
            t_dst_start = max(0, a.lag_t)
            x_src_start = max(0, -a.lag_x)
            x_dst_start = max(0, a.lag_x)
            x_len = X - abs(a.lag_x)
            if x_len <= 0:
                continue
            t_len = T - a.lag_t
            if t_len <= 0:
                continue
            # t_src_start - a.lag_t = max(0, lag_t) - lag_t; for lag_t >= 0
            # (causal in time) this is 0
            src = U[t_src_start - a.lag_t: t_src_start - a.lag_t + t_len,
                    x_src_start: x_src_start + x_len]
            Y = Y.at[t_dst_start: t_dst_start + t_len,
                     x_dst_start: x_dst_start + x_len].add(a.weight * src)

        # Separable density part: causal conv in time x causal conv in space.
        # Materialise the 1D kernels as JAX arrays, take outer product, run
        # one convolve2d. Crop the "causal" corner (top-left starting at
        # K_t-1, K_x-1) so output has shape (T, X) matching the input.
        if self.has_density:
            # Use fftconvolve (not convolve2d) because JAX's convolve2d
            # rejects cases where the kernel is larger than the input in
            # any dimension; that can legitimately happen for separable
            # measures with long-tail densities (e.g., EMA support_max=100
            # applied to a small field). fftconvolve handles all sizes.
            from jax.scipy.signal import fftconvolve

            sep_t_kernel = jnp.asarray(self.sep_t.to_kernel(), dtype=U.dtype)
            sep_x_kernel = jnp.asarray(self.sep_x.to_kernel(), dtype=U.dtype)
            density_kernel = jnp.outer(sep_t_kernel, sep_x_kernel)
            full = fftconvolve(U, density_kernel, mode="full")
            # Causal in both axes: y[i, j] = Σ K[s, t] U[i-s, j-t] for
            # s ∈ [0, T_k-1], t ∈ [0, X_k-1] is the first [T, X] window
            # of the mode='full' output (matches 1D convention:
            # np.convolve(x, k, mode='full')[:N] is the causal output).
            sep_t_part = full[:T, :X]
            Y = Y + sep_t_part

        # Warmup mask. fill_warmup may be Python float (incl. NaN); pick a
        # concrete fill value.
        warmup_t = self.max_lag_t()
        warmup_x_left = max((a.lag_x for a in self.atoms), default=0)
        warmup_x_right = max((-a.lag_x for a in self.atoms), default=0)
        if self.has_density:
            warmup_x_left = max(warmup_x_left, self.sep_x.support_max)

        if fill_warmup is None or (isinstance(fill_warmup, float)
                                   and math.isnan(fill_warmup)):
            fwm = jnp.nan
        else:
            fwm = float(fill_warmup)

        if warmup_t > 0:
            Y = Y.at[:warmup_t, :].set(fwm)
        if warmup_x_left > 0:
            Y = Y.at[:, :warmup_x_left].set(fwm)
        if warmup_x_right > 0:
            Y = Y.at[:, X - warmup_x_right:].set(fwm)
        return Y

    # ---- Pretty-print ----

    def __str__(self) -> str:
        parts: list[str] = []
        for a in self.atoms:
            parts.append(str(a))
        if self.has_density:
            parts.append(f"({self.sep_t}) ⊗ ({self.sep_x})")
        return " + ".join(parts) if parts else "0"


# ---------------- Convenience constructors ----------------

def measure_2d_atomic(atoms: list[tuple[float, int, int]]) -> Measure2D:
    """From a list of (weight, lag_t, lag_x) tuples. Most common use:
    finite-difference operators."""
    return Measure2D(atoms=tuple(Atom2D(float(w), int(t), int(x)) for w, t, x in atoms))


def measure_2d_separable(measure_t: Measure, measure_x: Measure) -> Measure2D:
    """Product measure: μ_t ⊗ μ_x. Apply = two 1D convolutions (Fubini)."""
    return Measure2D(sep_t=measure_t, sep_x=measure_x)


# ----- PDE operators (atomic 2D measures) -----

def measure_2d_laplacian_5pt() -> Measure2D:
    """5-point discrete Laplacian on the SPATIAL axis at time-lag 0:
        ∇²u(x) ≈ u(x−1) − 2 u(x) + u(x+1).
    Time-stencil = δ(0). Used for parabolic / elliptic PDE discovery."""
    return measure_2d_atomic([
        (1.0,  0, -1),
        (-2.0, 0, 0),
        (1.0,  0, +1),
    ])


def measure_2d_diff_t(lag_t: int = 1) -> Measure2D:
    """Temporal difference at lag k: u(t) − u(t−k). Standard for ∂/∂t."""
    return measure_2d_atomic([
        (1.0,  0,      0),
        (-1.0, int(lag_t), 0),
    ])


def measure_2d_grad_x() -> Measure2D:
    """Centred first-difference in space: (u(x+1) − u(x−1)) / 2.
    Standard for ∂/∂x.

    Convention reminder: (K·U)(t,x) = Σ wᵢ U(t−tᵢ, x−xᵢ), so reading
    U(t, x+1) requires lag_x = −1, and reading U(t, x−1) requires lag_x = +1.
    """
    return measure_2d_atomic([
        ( 0.5, 0, -1),   # +0.5 · U(t, x+1)
        (-0.5, 0, +1),   # -0.5 · U(t, x-1)
    ])


def measure_2d_sobel_x() -> Measure2D:
    """Sobel horizontal-edge kernel, causal-in-time form.

    Standard non-causal Sobel-X kernel (rows top-to-bottom):
         [[-1, 0, +1],
          [-2, 0, +2],
          [-1, 0, +1]]
    where positive response means intensity increases left-to-right.

    Causal-in-time form maps the kernel's top/middle/bottom rows to
    lag_t = 2 / 1 / 0 (current row at the bottom). Combined with the
    (K·U)(t,x) = Σ wᵢ U(t−tᵢ, x−xᵢ) convention (lag_x=−1 reads U(t, x+1)),
    we get the atoms below.
    """
    return measure_2d_atomic([
        (-1.0, 2, +1), (+1.0, 2, -1),
        (-2.0, 1, +1), (+2.0, 1, -1),
        (-1.0, 0, +1), (+1.0, 0, -1),
    ])


def measure_2d_sobel_y() -> Measure2D:
    """Sobel vertical-edge kernel, causal-in-time form.

    Standard kernel (rows top-to-bottom):
         [[+1, +2, +1],
          [ 0,  0,  0],
          [-1, -2, -1]]

    Causal mapping: top row → lag_t=2, bottom row → lag_t=0.
    """
    return measure_2d_atomic([
        (+1.0, 2, +1), (+2.0, 2, 0), (+1.0, 2, -1),
        (-1.0, 0, +1), (-2.0, 0, 0), (-1.0, 0, -1),
    ])


__all__ = [
    "Atom2D", "Measure2D",
    "measure_2d_atomic", "measure_2d_separable",
    "measure_2d_laplacian_5pt", "measure_2d_diff_t", "measure_2d_grad_x",
    "measure_2d_sobel_x", "measure_2d_sobel_y",
]
