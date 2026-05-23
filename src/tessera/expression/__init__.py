"""tessera.expression — measure-theoretic operators and symbolic search primitives.

Public API
----------

Measures (Lebesgue-decomposed signed measures on non-negative lags):
    Atom, Measure, DENSITY_FAMILIES, register_density
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum

Functionals (n-ary wrappers; bilinear and Volterra-2):
    Functional, LinearFunctional, SeparableBilinear, Volterra2
    apply_with_cache

Caching:
    FunctionalCache

Numba-JIT kernels (internal, exposed for direct benchmarking):
    from tessera.expression._numba_kernels import (
        ema_recursive, atomic_apply, conv_causal, benchmark_apply_paths,
    )
"""
from __future__ import annotations

from .measure import (
    Atom, Measure, DENSITY_FAMILIES, register_density,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
)
from .cache import FunctionalCache
from .functional import (
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
    apply_with_cache,
)

__all__ = [
    # measure
    "Atom", "Measure", "DENSITY_FAMILIES", "register_density",
    "measure_lag", "measure_diff", "measure_ema", "measure_roll_mean",
    "measure_power_law", "measure_signed_sum",
    # cache
    "FunctionalCache",
    # functional
    "Functional", "LinearFunctional", "SeparableBilinear", "Volterra2",
    "apply_with_cache",
]
