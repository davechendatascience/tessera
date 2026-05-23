"""Tests for FunctionalCache.

Verifies:
  - Compute correctness (matches direct measure.apply)
  - Memory hit on second call
  - LRU eviction
  - Disk persistence (write, clear mem, re-read from disk)
  - Different measures / var_ids / fill_warmups → different cache slots
  - Mutation safety guidance (we return views; user must copy to mutate)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera.expression.cache import FunctionalCache
from tessera.expression.measure import (
    measure_ema, measure_lag, measure_diff, measure_signed_sum, measure_roll_mean,
)


def _rng(n=500, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# ---------------- Correctness ----------------

def test_get_or_compute_matches_direct_apply():
    x = _rng(500, 0)
    cache = FunctionalCache(mem_size=128)
    m = measure_ema(24)
    y_cached = cache.get_or_compute(m, "v0", x)
    y_direct = m.apply(x, fill_warmup=np.nan)
    assert np.allclose(y_cached, y_direct, atol=1e-12, equal_nan=True)
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 0

def test_second_call_is_memory_hit():
    x = _rng(500, 1)
    cache = FunctionalCache(mem_size=128)
    m = measure_diff(3)
    _ = cache.get_or_compute(m, "v0", x)
    _ = cache.get_or_compute(m, "v0", x)
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 1
    assert cache.hit_rate() == 0.5


# ---------------- Key uniqueness ----------------

def test_different_measures_get_different_slots():
    x = _rng(500, 2)
    cache = FunctionalCache(mem_size=128)
    cache.get_or_compute(measure_ema(12), "v0", x)
    cache.get_or_compute(measure_ema(24), "v0", x)
    cache.get_or_compute(measure_ema(48), "v0", x)
    assert cache.n_mem == 3
    assert cache.stats["misses"] == 3

def test_different_var_ids_get_different_slots():
    x = _rng(500, 3)
    y = _rng(500, 4)
    cache = FunctionalCache(mem_size=128)
    cache.get_or_compute(measure_ema(24), "var_x", x)
    cache.get_or_compute(measure_ema(24), "var_y", y)
    assert cache.n_mem == 2

def test_different_fill_warmups_get_different_slots():
    x = _rng(500, 5)
    cache = FunctionalCache(mem_size=128)
    cache.get_or_compute(measure_ema(24), "v", x, fill_warmup=np.nan)
    cache.get_or_compute(measure_ema(24), "v", x, fill_warmup=0.0)
    assert cache.n_mem == 2

def test_same_measure_different_construction_same_slot():
    """Two `measure_ema(24)` calls produce equal Measures → same slot."""
    x = _rng(500, 6)
    cache = FunctionalCache(mem_size=128)
    cache.get_or_compute(measure_ema(24), "v", x)
    cache.get_or_compute(measure_ema(24), "v", x)
    assert cache.n_mem == 1
    assert cache.stats["mem_hits"] == 1


# ---------------- LRU eviction ----------------

def test_lru_evicts_oldest():
    x = _rng(500, 7)
    cache = FunctionalCache(mem_size=2)
    cache.get_or_compute(measure_ema(6), "v", x)
    cache.get_or_compute(measure_ema(12), "v", x)
    cache.get_or_compute(measure_ema(24), "v", x)   # this should evict h=6
    assert cache.n_mem == 2
    # h=6 entry is now a miss again
    cache.get_or_compute(measure_ema(6), "v", x)
    assert cache.stats["misses"] == 4

def test_lru_touch_on_hit():
    x = _rng(500, 8)
    cache = FunctionalCache(mem_size=2)
    cache.get_or_compute(measure_ema(6), "v", x)
    cache.get_or_compute(measure_ema(12), "v", x)
    # Touch the h=6 entry — it should NOT be evicted next
    cache.get_or_compute(measure_ema(6), "v", x)
    cache.get_or_compute(measure_ema(24), "v", x)   # should evict h=12, not h=6
    assert cache.n_mem == 2
    # h=6 still cached
    n_misses_before = cache.stats["misses"]
    cache.get_or_compute(measure_ema(6), "v", x)
    assert cache.stats["misses"] == n_misses_before    # no new miss


# ---------------- Disk persistence ----------------

def test_disk_persistence(tmp_path: Path):
    x = _rng(500, 9)
    disk_dir = tmp_path / "feat_cache"

    cache_1 = FunctionalCache(mem_size=4, disk_dir=disk_dir)
    y_orig = cache_1.get_or_compute(measure_ema(24), "v", x)
    assert cache_1.n_disk == 1

    # Simulate a fresh run: new cache instance, no memory state
    cache_2 = FunctionalCache(mem_size=4, disk_dir=disk_dir)
    y_loaded = cache_2.get_or_compute(measure_ema(24), "v", x)
    assert np.allclose(y_orig, y_loaded, atol=1e-12, equal_nan=True)
    assert cache_2.stats["disk_hits"] == 1
    assert cache_2.stats["misses"] == 0

def test_clear_disk_then_recompute(tmp_path: Path):
    x = _rng(500, 10)
    disk_dir = tmp_path / "fc"
    cache = FunctionalCache(mem_size=2, disk_dir=disk_dir)
    cache.get_or_compute(measure_ema(24), "v", x)
    assert cache.n_disk == 1
    cache.clear_disk()
    assert cache.n_disk == 0
    # Memory still holds the entry
    assert cache.n_mem == 1

def test_clear_both_tiers(tmp_path: Path):
    x = _rng(500, 11)
    cache = FunctionalCache(mem_size=4, disk_dir=tmp_path / "fc")
    cache.get_or_compute(measure_ema(24), "v", x)
    cache.clear()
    assert cache.n_mem == 0
    assert cache.n_disk == 0


# ---------------- Atomic + signed_sum + roll_mean all cache ----------------

def test_all_kernel_kinds_cache():
    x = _rng(500, 12)
    cache = FunctionalCache(mem_size=16)
    cache.get_or_compute(measure_lag(5), "v", x)
    cache.get_or_compute(measure_diff(24), "v", x)
    cache.get_or_compute(measure_ema(48), "v", x)
    cache.get_or_compute(measure_roll_mean(12), "v", x)
    cache.get_or_compute(measure_signed_sum([(0.5, 0), (-0.3, 7)]), "v", x)
    assert cache.n_mem == 5
    # Re-call each → all memory hits
    for m in [measure_lag(5), measure_diff(24), measure_ema(48),
              measure_roll_mean(12), measure_signed_sum([(0.5, 0), (-0.3, 7)])]:
        cache.get_or_compute(m, "v", x)
    assert cache.stats["mem_hits"] == 5


# ---------------- Repr smoke ----------------

def test_repr_includes_stats(tmp_path: Path):
    cache = FunctionalCache(mem_size=10, disk_dir=tmp_path / "fc")
    cache.get_or_compute(measure_ema(24), "v", _rng(100, 13))
    s = repr(cache)
    assert "mem=1/10" in s
    assert "disk=on" in s
    assert "hit_rate" in s
