"""Tests for the n-ary Functional wrappers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.expression.measure import (
    Measure, measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_signed_sum,
)
from tessera.expression.functional import (
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
    apply_with_cache,
)
from tessera.expression.cache import FunctionalCache


def _rng(n=500, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# ---------------- LinearFunctional ----------------

def test_linear_functional_matches_measure_apply():
    x = _rng(500, 0)
    m = measure_ema(24)
    lf = LinearFunctional(measure=m)
    assert lf.n_inputs == 1
    y_lf = lf.apply(x)
    y_m = m.apply(x)
    assert np.array_equal(np.nan_to_num(y_lf), np.nan_to_num(y_m))


def test_linear_functional_hashable_and_value_equal():
    a = LinearFunctional(measure=measure_ema(24))
    b = LinearFunctional(measure=measure_ema(24))
    assert a == b
    assert hash(a) == hash(b)


# ---------------- SeparableBilinear ----------------

def test_separable_bilinear_factors_as_product():
    """B(x, y) = (μa·x) · (μb·y) — by Fubini, equals pointwise product
    of two 1D applies."""
    x = _rng(500, 1)
    y = _rng(500, 2)
    bil = SeparableBilinear(
        measure_a=measure_roll_mean(window=12),
        measure_b=measure_ema(halflife=24),
    )
    assert bil.n_inputs == 2
    out = bil.apply(x, y)
    expected = measure_roll_mean(12).apply(x) * measure_ema(24).apply(y)
    # equal_nan handles warmup regions
    assert np.allclose(out, expected, equal_nan=True)


def test_bilinear_diff_diff_captures_signed_coincidence():
    """(diff(1)*x) · (diff(1)*y) is x and y returns multiplied — basic
    sign-coincidence indicator."""
    x = _rng(200, 3)
    y = _rng(200, 4)
    bil = SeparableBilinear(
        measure_a=measure_diff(1),
        measure_b=measure_diff(1),
    )
    out = bil.apply(x, y)
    expected = (pd.Series(x).diff(1).values * pd.Series(y).diff(1).values)
    # Both share warmup, so where NaN, both should agree
    np.testing.assert_array_almost_equal(
        np.nan_to_num(out), np.nan_to_num(expected), decimal=10,
    )


# ---------------- Volterra2 ----------------

def test_volterra2_diff_squared_is_realised_variance_proxy():
    """V2(diff(1), diff(1)) on x is r(t)² — squared returns."""
    x = _rng(500, 5)
    v2 = Volterra2(
        measure_a=measure_diff(1),
        measure_b=measure_diff(1),
    )
    assert v2.n_inputs == 1
    out = v2.apply(x)
    expected = pd.Series(x).diff(1).values ** 2
    np.testing.assert_array_almost_equal(
        np.nan_to_num(out), np.nan_to_num(expected), decimal=10,
    )


def test_volterra2_distinct_measures_capture_cross_scale_product():
    """V2(ema(12), ema(168)) on x: short-EMA times long-EMA — typical
    'momentum × trend' interaction."""
    x = _rng(2000, 6)
    v2 = Volterra2(measure_a=measure_ema(12), measure_b=measure_ema(168))
    out = v2.apply(x)
    expected = measure_ema(12).apply(x) * measure_ema(168).apply(x)
    np.testing.assert_array_almost_equal(
        np.nan_to_num(out), np.nan_to_num(expected), decimal=10,
    )


# ---------------- Cache integration ----------------

def test_cache_aware_linear():
    x = _rng(500, 7)
    cache = FunctionalCache(mem_size=16)
    lf = LinearFunctional(measure=measure_ema(24))
    out1 = apply_with_cache(lf, cache, var_ids="x", xs=x)
    out2 = apply_with_cache(lf, cache, var_ids="x", xs=x)
    assert np.allclose(out1, out2, equal_nan=True)
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 1


def test_cache_aware_bilinear_reuses_1d_subexpressions():
    """The 1D pieces of a bilinear are cached individually — two
    bilinears that share a 1D piece should reuse it."""
    x = _rng(500, 8)
    y = _rng(500, 9)
    cache = FunctionalCache(mem_size=16)
    bil_a = SeparableBilinear(
        measure_a=measure_ema(24),
        measure_b=measure_ema(168),
    )
    bil_b = SeparableBilinear(
        measure_a=measure_ema(24),       # same as bil_a's measure_a
        measure_b=measure_diff(1),       # different
    )
    apply_with_cache(bil_a, cache, var_ids=("x", "y"), xs=(x, y))
    apply_with_cache(bil_b, cache, var_ids=("x", "y"), xs=(x, y))

    # After 2 bilinears with 2 measures each:
    #   bil_a misses on (ema24, "x") and (ema168, "y")    → 2 misses
    #   bil_b misses on (diff1, "y") but HITS on (ema24, "x") → 1 miss, 1 hit
    # 3 total misses, 1 mem hit. n_mem = 3.
    assert cache.n_mem == 3
    assert cache.stats["misses"] == 3
    assert cache.stats["mem_hits"] == 1


def test_cache_aware_volterra2_both_pieces_on_same_var():
    """V2 applies to ONE series; the two measure pieces share the
    same var_id and so both go into the cache under that var_id."""
    x = _rng(500, 10)
    cache = FunctionalCache(mem_size=16)
    v2 = Volterra2(measure_a=measure_ema(12), measure_b=measure_ema(168))
    apply_with_cache(v2, cache, var_ids="x", xs=x)
    # Two different measures on same var → 2 misses, 0 hits
    assert cache.stats["misses"] == 2
    # Re-apply: both hits
    apply_with_cache(v2, cache, var_ids="x", xs=x)
    assert cache.stats["mem_hits"] == 2


def test_volterra2_same_measure_twice_caches_once():
    """V2(μ, μ) on x: applying with cache hits the same entry twice."""
    x = _rng(500, 11)
    cache = FunctionalCache(mem_size=16)
    v2 = Volterra2(measure_a=measure_ema(24), measure_b=measure_ema(24))
    apply_with_cache(v2, cache, var_ids="x", xs=x)
    # First: miss, then hit (same measure, same var) → 1 miss + 1 hit
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 1
    assert cache.n_mem == 1


# ---------------- Construction guards ----------------

def test_apply_with_cache_arity_mismatch():
    x = _rng(100, 12)
    bil = SeparableBilinear(
        measure_a=measure_ema(12),
        measure_b=measure_ema(24),
    )
    cache = FunctionalCache(mem_size=8)
    # bil expects 2 inputs; pass 1
    with pytest.raises(ValueError):
        apply_with_cache(bil, cache, var_ids="x", xs=x)


def test_str_repr_meaningful():
    lf = LinearFunctional(measure=measure_ema(24))
    bil = SeparableBilinear(
        measure_a=measure_roll_mean(12),
        measure_b=measure_diff(1),
    )
    v2 = Volterra2(measure_a=measure_diff(1), measure_b=measure_diff(1))
    assert "L[" in str(lf) and "exponential" in str(lf)
    assert "B[" in str(bil)
    assert "V2[" in str(v2)
