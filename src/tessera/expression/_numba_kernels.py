"""Numba-compiled and NumPy-optimised hot kernels for Measure.apply().

This module is the *implementation backend* for Measure operations. The
public Measure API in measure.py routes through here. Three speed tiers:

  1. **Recursive fast-path** for purely exponential AC kernels: O(N) using
     the y[t] = α·x[t] + (1−α)·y[t−1] recursion. Avoids materialising the
     decaying tail at all. Matches pandas .ewm(adjust=False) exactly.

  2. **Atom-only fast-path** for pure-atomic measures: vectorised
     shift-and-accumulate. O(N · #atoms), perfectly fine for typical
     N >> #atoms.

  3. **General convolution**: scipy.signal.fftconvolve for kernels longer
     than ~64 samples (O(N log N)), Numba-compiled direct loop otherwise.
     Both handle arbitrary signed kernels.

Numba JIT is mandatory — install in your venv:  pip install numba

The functions here are pure (no Measure-class dependencies); they take
plain numpy arrays + scalar params, return numpy arrays. This makes them
straightforward to test in isolation and to call from a future GP loop.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.signal import fftconvolve


# ---------------- Recursive EMA (the big speedup) ----------------

@njit(cache=True, fastmath=False)
def ema_recursive(x: np.ndarray, halflife: float, fill_warmup: float = np.nan,
                  warmup_rows: int = 0) -> np.ndarray:
    """y[t] = α·x[t] + (1−α)·y[t−1], with α = 1 − 2^(−1/halflife).

    Matches pandas .ewm(halflife=h, adjust=False).mean() exactly (modulo
    fp ordering). The first `warmup_rows` outputs are filled with
    `fill_warmup` (NaN by default). Set warmup_rows=0 to keep the full
    recursive output starting from y[0] = x[0].

    O(N) — orders of magnitude faster than convolving the materialised
    α(1−α)^s kernel out to support_max.
    """
    N = x.shape[0]
    alpha = 1.0 - 2.0 ** (-1.0 / halflife)
    one_minus_alpha = 1.0 - alpha
    y = np.empty(N, dtype=np.float64)
    if N == 0:
        return y
    y[0] = x[0]
    for t in range(1, N):
        y[t] = alpha * x[t] + one_minus_alpha * y[t - 1]
    if warmup_rows > 0:
        m = min(warmup_rows, N)
        for i in range(m):
            y[i] = fill_warmup
    return y


# ---------------- Atomic-only kernel (sparse Diracs) ----------------

def atomic_apply(
    x: np.ndarray,
    lags: np.ndarray,         # int64 array of non-negative lag positions
    weights: np.ndarray,      # float64 array of weights (same length)
    fill_warmup: float = np.nan,
) -> np.ndarray:
    """Apply a sparse atomic measure: y[t] = Σᵢ wᵢ · x[t − lagᵢ].

    Vectorised in NumPy (no Numba — shift-and-accumulate is already fast
    enough since #atoms is typically small).
    """
    N = x.shape[0]
    y = np.zeros(N, dtype=np.float64)
    max_lag = 0
    for k, w in zip(lags, weights):
        if k == 0:
            y += w * x
        elif k < N:
            y[k:] += w * x[: N - k]
        max_lag = max(max_lag, int(k))
    if max_lag > 0 and not math.isnan(fill_warmup):
        y[:max_lag] = fill_warmup
    elif max_lag > 0:
        y[:max_lag] = np.nan
    return y


# ---------------- General causal convolution ----------------

@njit(cache=True, fastmath=True)
def _conv_causal_direct(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Direct O(N·K) causal convolution. Best when K is small (≤ ~64)."""
    N = x.shape[0]
    K = kernel.shape[0]
    y = np.zeros(N, dtype=np.float64)
    for k in range(K):
        coef = kernel[k]
        if coef == 0.0:
            continue
        if k == 0:
            for t in range(N):
                y[t] += coef * x[t]
        else:
            for t in range(k, N):
                y[t] += coef * x[t - k]
    return y


def conv_causal(
    x: np.ndarray,
    kernel: np.ndarray,
    fill_warmup: float = np.nan,
) -> np.ndarray:
    """Causal convolution y[t] = Σ_k kernel[k] · x[t−k]. Auto-picks the best
    implementation based on kernel length.

    For kernels with many zeros, this is suboptimal — use atomic_apply
    on the non-zero positions instead.
    """
    K = len(kernel)
    if K <= 64:
        y = _conv_causal_direct(x, kernel)
    else:
        # FFT convolve gives O(N log N), much faster for long kernels
        # `full` mode yields len(x)+K-1; we crop to causal length len(x).
        full = fftconvolve(x, kernel, mode="full")
        y = full[: len(x)].astype(np.float64, copy=False)

    # Apply warmup mask using the max non-zero kernel index
    nonzero = np.nonzero(kernel)[0]
    if len(nonzero) > 0:
        warmup = int(nonzero[-1])
        if warmup > 0:
            y = y.copy()  # FFT output may be read-only view in edge cases
            y[:warmup] = fill_warmup
    return y


# ---------------- Optional benchmark helper ----------------

def benchmark_apply_paths(N: int = 100_000, halflife: float = 24.0,
                          support_max: int | None = None, repeats: int = 5):
    """Micro-benchmark: recursive EMA vs direct conv vs FFT conv.

    Useful for verifying the routing in Measure.apply() picks the fastest
    path. Returns (ns_recursive, ns_direct, ns_fft).
    """
    import time
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)
    if support_max is None:
        support_max = max(int(round(15 * halflife)), 32)

    # Build the EMA kernel for comparison
    alpha = 1.0 - 2.0 ** (-1.0 / halflife)
    s = np.arange(support_max + 1, dtype=np.float64)
    kernel = alpha * (1.0 - alpha) ** s

    # Warm Numba JITs
    _ = ema_recursive(x[:100].astype(np.float64), halflife)
    _ = _conv_causal_direct(x[:100].astype(np.float64), kernel[:8])

    def time_it(fn):
        best = 1e18
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            fn()
            best = min(best, time.perf_counter_ns() - t0)
        return best

    ns_rec = time_it(lambda: ema_recursive(x, halflife, warmup_rows=0))
    ns_dir = time_it(lambda: _conv_causal_direct(x, kernel))
    ns_fft = time_it(lambda: fftconvolve(x, kernel, mode="full")[: len(x)])
    return ns_rec, ns_dir, ns_fft


__all__ = [
    "ema_recursive", "atomic_apply", "conv_causal", "benchmark_apply_paths",
]
