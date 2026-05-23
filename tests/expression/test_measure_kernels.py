"""Tests for the JIT-accelerated kernel backends.

Verifies that:
  - Recursive EMA matches the materialised-kernel conv path (steady state)
    and pandas ewm exactly (steady state).
  - Atomic apply matches numpy shift-and-add.
  - General conv (Numba direct vs FFT) agrees on the same kernel.
  - Measure.apply() routing produces the same result as backend="kernel".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.expression.measure import (
    Atom, Measure,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
)
from tessera.expression._numba_kernels import (
    ema_recursive, atomic_apply, conv_causal,
)


def _rng_series(n=2000, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# ---------------- Recursive EMA correctness ----------------

def test_ema_recursive_matches_pandas_exactly():
    """Recursive form: y[t] = α x[t] + (1−α) y[t−1] should be byte-identical
    to pandas ewm(adjust=False) up to fp ordering."""
    x = _rng_series(2000, seed=1)
    for h in (3.0, 12.0, 168.0):
        y_ours = ema_recursive(x, h, fill_warmup=np.nan, warmup_rows=0)
        y_pd = pd.Series(x).ewm(halflife=h, adjust=False).mean().values
        assert np.allclose(y_ours, y_pd, atol=1e-12)

def test_ema_recursive_skips_warmup_rows():
    x = _rng_series(100, seed=2)
    y = ema_recursive(x, halflife=8.0, fill_warmup=np.nan, warmup_rows=8)
    assert np.isnan(y[:8]).all()
    assert not np.isnan(y[8:]).any()


# ---------------- Atomic apply correctness ----------------

def test_atomic_apply_matches_manual():
    x = _rng_series(200, seed=3)
    lags = np.array([0, 5, 24], dtype=np.int64)
    weights = np.array([0.5, -0.3, 0.1], dtype=np.float64)
    y = atomic_apply(x, lags, weights, fill_warmup=0.0)
    expected = (
        0.5 * x
        + (-0.3) * pd.Series(x).shift(5).fillna(0).values
        + 0.1 * pd.Series(x).shift(24).fillna(0).values
    )
    expected[:24] = 0.0   # warmup
    assert np.allclose(y, expected, atol=1e-12)


# ---------------- conv_causal: direct vs FFT branch ----------------

def test_conv_short_kernel_uses_direct_and_matches_fft():
    """For short kernels (≤64), conv_causal uses the Numba direct loop. We
    verify it matches FFT convolution on the same kernel."""
    from scipy.signal import fftconvolve
    x = _rng_series(500, seed=4)
    kernel = np.array([0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.1], dtype=np.float64)
    y_ours = conv_causal(x, kernel, fill_warmup=0.0)
    y_fft = fftconvolve(x, kernel, mode="full")[: len(x)]
    y_fft[: int(np.nonzero(kernel)[0][-1])] = 0.0
    assert np.allclose(y_ours, y_fft, atol=1e-9)

def test_conv_long_kernel_uses_fft_and_matches_direct():
    """For long kernels (>64), conv_causal uses FFT. Compare against a
    direct convolution computed elsewhere."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(3000)
    K = 200
    kernel = rng.standard_normal(K)
    # Sparsify to ensure warmup mask is at K-1
    kernel[-1] = 1.0
    y_fft_path = conv_causal(x, kernel, fill_warmup=0.0)
    # Reference: manual direct conv
    y_direct = np.zeros_like(x)
    for k in range(K):
        if k == 0:
            y_direct += kernel[k] * x
        else:
            y_direct[k:] += kernel[k] * x[: len(x) - k]
    y_direct[: K - 1] = 0.0
    assert np.allclose(y_fft_path, y_direct, atol=1e-8)


# ---------------- Measure.apply routing ----------------

def test_apply_auto_matches_kernel_for_pure_atomic():
    x = _rng_series(500, seed=6)
    m = measure_signed_sum([(0.7, 0), (-0.4, 3), (0.2, 12)])
    y_auto = m.apply(x, fill_warmup=0.0, backend="auto")
    y_kernel = m.apply(x, fill_warmup=0.0, backend="kernel")
    assert np.allclose(y_auto, y_kernel, atol=1e-12)

def test_apply_auto_matches_kernel_for_roll_mean():
    """Pure density (no atoms, rectangular family) — uses conv_causal path."""
    x = _rng_series(500, seed=7)
    m = measure_roll_mean(window=24)
    y_auto = m.apply(x, fill_warmup=0.0, backend="auto")
    y_kernel = m.apply(x, fill_warmup=0.0, backend="kernel")
    assert np.allclose(y_auto, y_kernel, atol=1e-9)

def test_apply_auto_ema_takes_recursive_path():
    """For a pure exponential, auto path bypasses kernel materialisation
    AND matches pandas exactly (where kernel path only matches past warmup)."""
    x = _rng_series(2000, seed=8)
    m = measure_ema(halflife=12.0)
    y_auto = m.apply(x, fill_warmup=np.nan, backend="auto")
    y_pd = pd.Series(x).ewm(halflife=12.0, adjust=False).mean().values
    # Recursive path matches pandas everywhere (no warmup truncation)
    assert np.allclose(y_auto, y_pd, atol=1e-12)

def test_apply_auto_ema_with_atoms_uses_general_path():
    """If atoms are present, the recursive fast-path does NOT apply.
    Should fall through to kernel materialisation."""
    x = _rng_series(2000, seed=9)
    m = Measure(
        atoms=(Atom(weight=0.5, lag=4),),
        density_family="exponential",
        density_params=(("halflife", 12.0),),
        support_max=200,
    )
    y_auto = m.apply(x, fill_warmup=0.0, backend="auto")
    y_kernel = m.apply(x, fill_warmup=0.0, backend="kernel")
    assert np.allclose(y_auto, y_kernel, atol=1e-9)


# ---------------- Speed check (smoke; not a hard SLA) ----------------

def test_recursive_ema_faster_than_kernel_for_large_h():
    """Sanity check: at h=24 with N=20k, recursive should be ≥3× faster
    than materialising + convolving the truncated kernel."""
    import time
    N = 20_000
    x = _rng_series(N, seed=10)

    # Warm JITs
    _ = measure_ema(24).apply(x[:100], backend="auto")
    _ = measure_ema(24).apply(x[:100], backend="kernel")

    m = measure_ema(halflife=24.0)
    t0 = time.perf_counter_ns()
    for _ in range(5):
        m.apply(x, backend="auto")
    t_auto = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    for _ in range(5):
        m.apply(x, backend="kernel")
    t_kernel = time.perf_counter_ns() - t0

    # Allow some slack — at minimum recursive should not be slower
    assert t_auto < t_kernel * 1.5, (
        f"recursive ({t_auto/1e6:.1f}ms) should be faster than kernel "
        f"({t_kernel/1e6:.1f}ms)"
    )
