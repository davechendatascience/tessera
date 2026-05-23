"""Tests for the Measure abstraction.

We verify two things:
  (a) The measure primitives produce kernels that match the existing
      hand-written operators (lag, ema, diff, roll_mean) numerically.
  (b) The Lebesgue decomposition holds in the obvious way (atomic + AC
      is just the sum of contributions; summability checks fire).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tessera.expression.measure import (
    Atom, Measure, DENSITY_FAMILIES,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
)


# ---------------- Atom / construction guards ----------------

def test_atom_rejects_negative_lag():
    with pytest.raises(ValueError):
        Atom(weight=1.0, lag=-1)

def test_atom_rejects_nonfinite_weight():
    with pytest.raises(ValueError):
        Atom(weight=float("nan"), lag=0)

def test_measure_rejects_atom_beyond_support():
    with pytest.raises(ValueError):
        Measure(atoms=(Atom(1.0, 50),), support_max=10)

def test_measure_rejects_unknown_density():
    with pytest.raises(ValueError):
        Measure(density_family="bogus", support_max=10)


# ---------------- Inspection ----------------

def test_zero_measure():
    m = Measure(support_max=8)
    assert not m.has_atomic and not m.has_density
    assert m.is_absolutely_summable()
    assert m.total_atomic_mass() == 0.0
    assert str(m) == "0"

def test_signed_sum_str():
    m = measure_signed_sum([(0.5, 1), (-1.0, 3)])
    assert "0.5·δ(1)" in str(m)
    assert "-1·δ(3)" in str(m)


# ---------------- Lag reproduces shift ----------------

def test_lag_matches_pandas_shift():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    for k in (1, 3, 7, 24, 60):
        y = measure_lag(k).apply(x)
        expected = pd.Series(x).shift(k).values
        # Both should match where defined (after warmup)
        valid = ~np.isnan(y) & ~np.isnan(expected)
        assert np.allclose(y[valid], expected[valid])
        # Warmup rows are NaN
        assert np.isnan(y[:k]).all()


# ---------------- Diff reproduces pandas .diff ----------------

def test_diff_matches_pandas_diff():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    for k in (1, 6, 24):
        y = measure_diff(k).apply(x)
        expected = pd.Series(x).diff(k).values
        valid = ~np.isnan(y) & ~np.isnan(expected)
        assert np.allclose(y[valid], expected[valid])


# ---------------- EMA matches pandas .ewm (within numerical tolerance) ----------------

def test_ema_matches_pandas_ewm():
    """Discrete EMA: our truncated convolution α(1−α)^s vs pandas' recursive
    ewm(adjust=False).

    The kernel is non-zero across the full support, so the warmup region
    equals support_max. We compare on the steady-state region past warmup.
    The two forms differ at finite t by ≈ (1−α)^t · x[0] which is well
    below 1e-3 past support_max.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3000)
    for h in (6.0, 24.0, 168.0):
        m = measure_ema(halflife=h)   # default support_max ≈ 7h
        smax = m.support_max
        if smax + 20 >= len(x):
            continue
        y = m.apply(x, fill_warmup=np.nan)
        expected = pd.Series(x).ewm(halflife=h, adjust=False).mean().values
        valid_start = smax + 20
        max_err = float(np.max(np.abs(y[valid_start:] - expected[valid_start:])))
        assert max_err < 1e-2, f"EMA mismatch at h={h}: max abs err={max_err}"


# ---------------- Roll-mean matches pandas .rolling().mean() ----------------

def test_roll_mean_matches_pandas_rolling():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    for w in (3, 12, 48):
        m = measure_roll_mean(window=w)
        y = m.apply(x)
        expected = pd.Series(x).rolling(window=w).mean().values
        # measure_roll_mean is left-aligned (causal): the support is [0, w),
        # so the value at index t uses x[t], x[t-1], ..., x[t-w+1].
        # pandas .rolling() with min_periods=window is also left-aligned and
        # NaN for t < w-1. They should agree past the warmup.
        valid = ~np.isnan(y) & ~np.isnan(expected)
        assert np.allclose(y[valid], expected[valid], atol=1e-9), \
            f"roll_mean mismatch at w={w}"


# ---------------- Signed-sum is the new primitive ----------------

def test_signed_sum_freeform():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100)
    # Three Diracs: 0.5·δ(0) − 0.3·δ(6) + 0.1·δ(24)
    m = measure_signed_sum([(0.5, 0), (-0.3, 6), (0.1, 24)])
    y = m.apply(x)
    expected = (
        0.5 * x
        + (-0.3) * pd.Series(x).shift(6).values
        + 0.1 * pd.Series(x).shift(24).values
    )
    valid = ~np.isnan(y) & ~np.isnan(expected)
    assert np.allclose(y[valid], expected[valid])

def test_signed_sum_includes_diff():
    """diff(k) = signed_sum([(1, 0), (-1, k)])."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(50)
    k = 7
    a = measure_diff(k).apply(x)
    b = measure_signed_sum([(1.0, 0), (-1.0, k)]).apply(x)
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


# ---------------- Composition: atomic + density ----------------

def test_atomic_plus_density_is_sum():
    """Composite (atom + density) applied = atom_apply + density_apply on the
    steady-state region (past the common max warmup).

    We force backend="kernel" for all three so the truncated-kernel
    convention matches; the recursive EMA fast path uses pandas-style
    initial conditions which differ slightly at the warmup boundary.
    """
    rng = np.random.default_rng(6)
    x = rng.standard_normal(500)
    h, lag, w_lag = 24.0, 12, 0.5
    smax = 200
    composite = Measure(
        atoms=(Atom(weight=w_lag, lag=lag),),
        density_family="exponential",
        density_params=(("halflife", h),),
        support_max=smax,
    )
    y_composite = composite.apply(x, fill_warmup=np.nan, backend="kernel")
    y_atomic_only = Measure(atoms=(Atom(w_lag, lag),), support_max=smax).apply(
        x, fill_warmup=np.nan, backend="kernel"
    )
    y_density_only = measure_ema(halflife=h, support_max=smax).apply(
        x, fill_warmup=np.nan, backend="kernel"
    )
    start = smax + 1
    assert np.allclose(
        y_composite[start:],
        y_atomic_only[start:] + y_density_only[start:],
        atol=1e-9,
    )


# ---------------- Power-law summability ----------------

def test_power_law_requires_alpha_gt_1():
    with pytest.raises(ValueError):
        measure_power_law(scale=10.0, alpha=0.8)
    # α > 1 passes
    m = measure_power_law(scale=10.0, alpha=1.5, support_max=200)
    assert m.is_absolutely_summable()

def test_power_law_decays_slower_than_exp():
    """At lag s = 5h, exp halves ~7x but power-law (α=1.5) halves ~2.8x."""
    h, alpha, smax = 24.0, 1.5, 256
    exp_k = DENSITY_FAMILIES["exponential"]({"halflife": h}, smax)
    pl_k = DENSITY_FAMILIES["power_law"]({"scale": h, "alpha": alpha}, smax)
    s = int(5 * h)
    assert exp_k[s] < pl_k[s], "exp should decay faster than power-law"


# ---------------- Absolute summability ----------------

def test_summability_holds_for_standard_kernels():
    assert measure_lag(7).is_absolutely_summable()
    assert measure_diff(24).is_absolutely_summable()
    assert measure_ema(24).is_absolutely_summable()
    assert measure_roll_mean(48).is_absolutely_summable()
    assert measure_power_law(scale=24, alpha=1.5).is_absolutely_summable()
    assert measure_signed_sum([(1, 0), (-1, 5), (0.5, 24)]).is_absolutely_summable()


# ---------------- High-pass via delta_minus_exp ----------------

def test_delta_minus_exp_equals_residual():
    """δ(0) − ema kernel should give x − ema(x). One primitive, two ops.

    Force backend="kernel" for the EMA comparison so both sides use the
    same truncated-kernel convention.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000)
    h = 24.0
    smax = int(15 * h)
    m_highpass = Measure(
        density_family="delta_minus_exp",
        density_params=(("halflife", h),),
        support_max=smax,
    )
    y = m_highpass.apply(x, fill_warmup=np.nan)
    y_alt = x - measure_ema(halflife=h, support_max=smax).apply(
        x, fill_warmup=np.nan, backend="kernel"
    )
    valid = ~np.isnan(y) & ~np.isnan(y_alt)
    assert np.allclose(y[valid], y_alt[valid], atol=1e-9)


# ---------------- Pretty print smoke ----------------

def test_str_compact():
    assert str(measure_lag(5)) == "1·δ(5)"
    assert "exponential" in str(measure_ema(24))
    assert "rectangular" in str(measure_roll_mean(7))
