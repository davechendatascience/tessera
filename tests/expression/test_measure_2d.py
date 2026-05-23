"""Tests for tessera.expression.measure_2d."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.expression.measure import measure_ema, measure_diff
from tessera.expression.measure_2d import (
    Atom2D, Measure2D,
    measure_2d_atomic, measure_2d_separable,
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
    measure_2d_sobel_x, measure_2d_sobel_y,
)


# ---------------- Construction guards ----------------

def test_atom_2d_rejects_negative_lag_t():
    with pytest.raises(ValueError):
        Atom2D(weight=1.0, lag_t=-1, lag_x=0)

def test_atom_2d_allows_negative_lag_x():
    a = Atom2D(weight=1.0, lag_t=0, lag_x=-1)
    assert a.lag_x == -1

def test_atom_2d_rejects_non_finite():
    with pytest.raises(ValueError):
        Atom2D(weight=float("inf"), lag_t=0, lag_x=0)

def test_measure_2d_density_requires_both_or_neither():
    m_ema = measure_ema(12)
    with pytest.raises(ValueError):
        Measure2D(sep_t=m_ema, sep_x=None)
    with pytest.raises(ValueError):
        Measure2D(sep_t=None, sep_x=m_ema)


# ---------------- Apply: pure atomic (finite differences) ----------------

def test_diff_t_matches_manual():
    """temporal diff at lag 1 should produce U[t] - U[t-1] interior."""
    U = np.arange(40, dtype=np.float64).reshape(8, 5)   # rows 0..7, cols 0..4
    m = measure_2d_diff_t(lag_t=1)
    Y = m.apply(U, fill_warmup=0.0)
    # Row 0 is warmup → 0
    assert np.all(Y[0] == 0.0)
    # Row t ≥ 1: Y[t] = U[t] - U[t-1]; for arange-reshape this is 5 per row
    expected = np.full((8, 5), 5.0)
    expected[0] = 0.0
    assert np.allclose(Y, expected)


def test_grad_x_matches_manual():
    """Centred spatial diff: (u(x+1) - u(x-1)) / 2."""
    U = np.arange(40, dtype=np.float64).reshape(4, 10)
    m = measure_2d_grad_x()
    Y = m.apply(U, fill_warmup=0.0)
    # Interior columns: per row, U is [10k, 10k+1, ..., 10k+9]; gradient = 1
    assert np.allclose(Y[:, 1:-1], 1.0)
    # Boundary columns 0 and -1 are warmup → 0
    assert np.all(Y[:, 0] == 0.0)
    assert np.all(Y[:, -1] == 0.0)


def test_laplacian_on_quadratic_field_is_constant_2():
    """∇² of u(x) = x² is exactly 2 in the discrete 5-point sense too:
        u(x-1) - 2u(x) + u(x+1) = (x-1)² - 2x² + (x+1)² = 2
    """
    T, X = 4, 12
    xs = np.arange(X, dtype=np.float64)
    U = np.tile(xs ** 2, (T, 1))
    m = measure_2d_laplacian_5pt()
    Y = m.apply(U, fill_warmup=0.0)
    # Interior columns should be exactly 2
    assert np.allclose(Y[:, 1:-1], 2.0)


def test_laplacian_on_linear_field_is_zero():
    """∇² of u(x) = αx + β is 0 in the discrete sense."""
    T, X = 3, 10
    U = np.tile(2.0 * np.arange(X, dtype=np.float64) + 3.0, (T, 1))
    m = measure_2d_laplacian_5pt()
    Y = m.apply(U, fill_warmup=0.0)
    assert np.allclose(Y[:, 1:-1], 0.0, atol=1e-12)


def test_sobel_x_on_horizontal_step_responds():
    """Sobel-X should respond at the location of a horizontal edge."""
    U = np.zeros((5, 10), dtype=np.float64)
    U[:, 5:] = 1.0   # step from 0 to 1 between col 4 and col 5
    m = measure_2d_sobel_x()
    Y = m.apply(U, fill_warmup=0.0)
    # Mark valid region: t ∈ [2, T-1], x ∈ [1, X-2]
    # At x=4 and x=5, expect non-zero; elsewhere near zero
    valid = Y[2:-1, 1:-1]
    assert np.abs(valid[:, 3]).max() > 0.5 or np.abs(valid[:, 4]).max() > 0.5


# ---------------- Apply: separable density ----------------

def test_separable_density_ema_t_times_rect_x():
    """μ_t = EMA(12) along time, μ_x = rect(3) along space. Verify that
    applying it equals two 1D applies in sequence."""
    rng = np.random.default_rng(0)
    U = rng.standard_normal((100, 30))
    m_t = measure_ema(halflife=12.0)
    from tessera.expression.measure import measure_roll_mean
    m_x = measure_roll_mean(window=3)
    m2d = measure_2d_separable(m_t, m_x)

    Y = m2d.apply(U, fill_warmup=0.0)

    # Compute expected manually: apply m_x along axis 1, then m_t along axis 0
    mid = np.array([m_x.apply(U[i], fill_warmup=0.0) for i in range(U.shape[0])])
    expected = np.array([m_t.apply(mid[:, j], fill_warmup=0.0) for j in range(U.shape[1])]).T

    # Compare past warmup (max of m_t.support_max time-rows and m_x.support_max space-cols)
    t_warmup = m_t.support_max
    x_warmup = m_x.support_max
    assert np.allclose(
        Y[t_warmup:, x_warmup:],
        expected[t_warmup:, x_warmup:],
        atol=1e-9,
    )


# ---------------- Atomic + density combination ----------------

def test_atomic_plus_density_sums():
    """Apply of (atomic + density) measure = atomic-only apply + density-only apply."""
    rng = np.random.default_rng(1)
    U = rng.standard_normal((50, 20))
    composite = Measure2D(
        atoms=(Atom2D(weight=0.5, lag_t=2, lag_x=0),),
        sep_t=measure_ema(8),
        sep_x=measure_diff(1),
    )
    y_full = composite.apply(U, fill_warmup=0.0)
    y_atom = Measure2D(atoms=(Atom2D(weight=0.5, lag_t=2, lag_x=0),)).apply(U, fill_warmup=0.0)
    y_dens = measure_2d_separable(measure_ema(8), measure_diff(1)).apply(U, fill_warmup=0.0)

    # Past the max warmup, sums should agree
    t_warmup = max(2, measure_ema(8).support_max)
    x_warmup = measure_diff(1).support_max
    assert np.allclose(
        y_full[t_warmup:, x_warmup:],
        y_atom[t_warmup:, x_warmup:] + y_dens[t_warmup:, x_warmup:],
        atol=1e-9,
    )


# ---------------- Heat equation forward step (PDE-discovery preview) ----------------

def test_heat_equation_step_consistency():
    """Run a small heat-equation simulation: U_{t+1} = U_t + α · ∇²U_t.

    Then check that applying (diff_t at lag 1) to the trajectory equals α ·
    (Laplacian applied to U at the same time step).
    """
    T, X = 50, 32
    alpha = 0.05
    U = np.zeros((T, X), dtype=np.float64)
    U[0, X // 2] = 1.0   # impulse in the middle

    laplacian = measure_2d_laplacian_5pt()
    diff_t = measure_2d_diff_t(lag_t=1)

    # Forward Euler simulation
    for t in range(1, T):
        Lap_prev = laplacian.apply(U[t-1:t+1], fill_warmup=0.0)[1]   # take "current" row
        # We need the Laplacian of U[t-1, :] not of the 2-row slice;
        # easier to just compute it directly along the spatial axis
        prev = U[t-1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        U[t] = prev + alpha * lap

    # Verify with our 2D primitives:
    #   diff_t(U)[t, x] = U[t] - U[t-1] ≈ α · laplacian(U)[t-1, x]
    lhs = diff_t.apply(U, fill_warmup=0.0)
    rhs_lap = laplacian.apply(U, fill_warmup=0.0)
    # rhs at row t is lap of U[t]; we want lap of U[t-1], so shift rhs by 1 row
    rhs_shifted = np.roll(rhs_lap, shift=1, axis=0)
    rhs_shifted[0] = 0.0
    rhs = alpha * rhs_shifted

    # Compare interior cells where both have valid values
    valid_t = slice(2, T)
    valid_x = slice(2, X - 2)
    assert np.allclose(lhs[valid_t, valid_x], rhs[valid_t, valid_x], atol=1e-12), \
        "diff_t(U) should equal α · Laplacian(U[t-1]) for heat-equation trajectory"


# ---------------- Inspection / pretty print ----------------

def test_str_compact():
    s = str(measure_2d_grad_x())
    assert "δ(0,1)" in s or "δ(0,+1)" in s

def test_is_absolutely_summable_holds():
    assert measure_2d_laplacian_5pt().is_absolutely_summable()
    assert measure_2d_separable(measure_ema(12), measure_diff(1)).is_absolutely_summable()
