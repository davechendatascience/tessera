"""tessera — applied-math primitives for compositional ML.

Top-level convenience re-exports from `tessera.expression`. For finer-
grained imports, use the submodule directly:

    from tessera.expression.measure import Measure, measure_ema
    from tessera.expression.cache import FunctionalCache
    from tessera.expression.functional import SeparableBilinear, Volterra2
"""
from __future__ import annotations

__version__ = "0.1.0"

# Re-export the expression submodule's public API for convenience.
# Heavier deps (numba) are imported lazily inside the modules; this top-
# level import stays cheap.
from .expression import (  # noqa: F401
    # measure
    Atom, Measure, DENSITY_FAMILIES, register_density,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
    # cache
    FunctionalCache,
    # functional
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
    apply_with_cache,
)
from .backend import (  # noqa: F401
    Backend, NumpyBackend, JaxBackend,
    set_backend, get_backend, current,
)

__all__ = [
    "__version__",
    "Atom", "Measure", "DENSITY_FAMILIES", "register_density",
    "measure_lag", "measure_diff", "measure_ema", "measure_roll_mean",
    "measure_power_law", "measure_signed_sum",
    "FunctionalCache",
    "Functional", "LinearFunctional", "SeparableBilinear", "Volterra2",
    "apply_with_cache",
    # Backend
    "Backend", "NumpyBackend", "JaxBackend",
    "set_backend", "get_backend", "current",
]
