"""FunctionalCache — memo store for measure.apply() results.

Why
---
GP search over symbolic expressions evaluates many candidate trees per
generation. Most subtrees repeat across candidates (a population of 200
expressions might contain `ema(temperature_2m, h=24)` 50× in different
parent contexts). Computing each one once and reusing it makes the GP
loop dramatically cheaper.

Key design
----------
Each cached entry is keyed by:

    (hash(measure), var_id, fill_warmup_value)

  - `measure` is a Measure — frozen dataclass, already hashable. Two
    `measure_ema(24)` objects compare equal and hash equally.
  - `var_id` is a stable string the caller provides — typically a feature
    name like "temperature_2m" or a derived subtree identifier like
    "temperature_2m::diff(k=1)". The CALLER is responsible for keeping
    var_ids consistent: same identifier ⇒ same underlying data.
  - `fill_warmup` is included because the same measure with different
    warmup fills (NaN vs 0.0) yields different result arrays.

Two-tier storage
----------------
- **Memory (LRU):** OrderedDict-based, capped at `mem_size` entries.
  Eviction is least-recently-used.
- **Disk (optional):** one `.npy` per entry, named by a hash of the key.
  Persistent across runs; safe to delete the directory anytime.

Concurrency
-----------
NOT thread-safe. Wrap with a lock if you need parallel GP eval. The
inner numpy ops release the GIL so a simple lock around get_or_compute
suffices.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .measure import Measure


CacheKey = tuple[int, str, float | None]


class FunctionalCache:
    """Two-tier (mem + disk) cache for Measure.apply() results.

    Parameters
    ----------
    mem_size : int
        Max number of arrays held in memory before LRU eviction. Each
        array is len(x) * 8 bytes (float64), so e.g. 10_000 entries ×
        50k samples × 8 B ≈ 4 GB. Tune based on RAM budget.
    disk_dir : Path | None
        Optional directory for persistent storage. If None, cache is
        memory-only. If given, computed arrays are also saved as .npy.

    Usage
    -----
        cache = FunctionalCache(mem_size=10_000, disk_dir="cache/feat")
        y = cache.get_or_compute(
            measure=measure_ema(24),
            var_id="temperature_2m",
            x=temp_array,
        )
    """

    def __init__(self, mem_size: int = 10_000, disk_dir: str | Path | None = None):
        if mem_size < 0:
            raise ValueError(f"mem_size must be ≥ 0, got {mem_size}")
        self.mem: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self.mem_size = int(mem_size)
        self.disk_dir: Path | None = Path(disk_dir) if disk_dir is not None else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"mem_hits": 0, "disk_hits": 0, "misses": 0}

    # ---- Key construction ----

    @staticmethod
    def _make_key(measure: Measure, var_id: str, fill_warmup: float | None) -> CacheKey:
        """Build the lookup key. Measure is hashable; var_id is a string."""
        fwm = None if fill_warmup is None else float(fill_warmup)
        return (hash(measure), str(var_id), fwm)

    def _disk_path(self, key: CacheKey) -> Path:
        """Map a key to a deterministic on-disk filename."""
        if self.disk_dir is None:
            raise RuntimeError("disk cache not configured")
        # Use blake2b for a short, collision-resistant filename
        # We hash the *string repr* of the key (including measure hash + var_id + fwm)
        h = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=12).hexdigest()
        return self.disk_dir / f"feat_{h}.npy"

    # ---- Insertion / eviction ----

    def _memo(self, key: CacheKey, arr: np.ndarray) -> None:
        """LRU insert / refresh."""
        if key in self.mem:
            self.mem.move_to_end(key)
            return
        self.mem[key] = arr
        if self.mem_size > 0 and len(self.mem) > self.mem_size:
            self.mem.popitem(last=False)   # evict oldest

    # ---- Public API ----

    def get_or_compute(
        self,
        measure: Measure,
        var_id: str,
        x: np.ndarray,
        fill_warmup: float | None = np.nan,
        *,
        backend: str = "auto",
    ) -> np.ndarray:
        """Look up or compute measure.apply(x).

        Returns a *view* of the cached array — DO NOT MUTATE. If you need
        to modify the result, copy it first.
        """
        key = self._make_key(measure, var_id, fill_warmup)

        # Tier 1: memory
        if key in self.mem:
            self.mem.move_to_end(key)
            self.stats["mem_hits"] += 1
            return self.mem[key]

        # Tier 2: disk
        if self.disk_dir is not None:
            disk_path = self._disk_path(key)
            if disk_path.exists():
                arr = np.load(disk_path)
                self._memo(key, arr)
                self.stats["disk_hits"] += 1
                return arr

        # Miss: compute
        arr = measure.apply(x, fill_warmup=fill_warmup, backend=backend)
        self._memo(key, arr)
        if self.disk_dir is not None:
            np.save(self._disk_path(key), arr)
        self.stats["misses"] += 1
        return arr

    # ---- Maintenance ----

    def clear_mem(self) -> None:
        """Evict all memory entries. Disk untouched."""
        self.mem.clear()

    def clear_disk(self) -> None:
        """Remove all disk entries. Memory untouched."""
        if self.disk_dir is None or not self.disk_dir.exists():
            return
        for p in self.disk_dir.glob("feat_*.npy"):
            p.unlink()

    def clear(self) -> None:
        """Clear both tiers."""
        self.clear_mem()
        self.clear_disk()

    # ---- Inspection ----

    @property
    def n_mem(self) -> int:
        return len(self.mem)

    @property
    def n_disk(self) -> int:
        if self.disk_dir is None or not self.disk_dir.exists():
            return 0
        return sum(1 for _ in self.disk_dir.glob("feat_*.npy"))

    def hit_rate(self) -> float:
        total = sum(self.stats.values())
        if total == 0:
            return 0.0
        return (self.stats["mem_hits"] + self.stats["disk_hits"]) / total

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0

    def __repr__(self) -> str:
        hr = self.hit_rate()
        return (
            f"FunctionalCache(mem={self.n_mem}/{self.mem_size}, "
            f"disk={'on' if self.disk_dir else 'off'} ({self.n_disk} files), "
            f"hit_rate={hr:.1%}, stats={dict(self.stats)})"
        )


__all__ = ["FunctionalCache"]
