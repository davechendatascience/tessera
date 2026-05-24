"""Switchable CPU / GPU backend for tessera.

Public API
----------
    from tessera.backend import set_backend, get_backend, current

    set_backend("numpy")    # default; CPU via numpy + numba
    set_backend("jax")       # GPU via jax.numpy (when fully implemented)

    arr = current().asarray([1.0, 2.0, 3.0])
    out = current().convolve(arr, kernel)

Status (2026-05-25)
-------------------
- **NumpyBackend**: fully functional. Wraps the existing numpy/numba
  code paths. This is the default.
- **JaxBackend**: SKELETON. Imports jax lazily; arrays can be moved
  to JAX form via `asarray`. Most operators raise NotImplementedError
  pending the full Stage-1 backend port (per
  docs/research/gpu_and_cv_via_sr.md §3 stage 1).

Why this exists as a public API NOW, before the full implementation
is done: the *interface commitment* shapes the rest of the codebase.
Code that touches `current().asarray(...)` instead of `np.asarray(...)`
will work cleanly when the JAX backend ships. Code that hardcodes
numpy will need rewriting. Establishing the contract early avoids
that churn.

What needs to happen for full GPU support
-----------------------------------------
See `docs/shipped/gpu_backend.md`. Headline summary:

1. `JaxBackend.asarray`, `.convolve`, `.where`, etc. — wrap jax.numpy
2. `Measure.apply` — dispatch to backend; provide a JAX
   convolution path
3. `FunctionalCache` — handle JAX arrays as cached values
4. Operator dispatch tables (`BIN_OP_FNS`, `UN_OP_FNS`) — replace
   `np.` calls with `current().` for the few non-broadcastable ops
   (`step`, `gt`, etc.). Most lambdas already work via JAX's
   `__array_function__` protocol.
5. Batched-population evaluation for GP — the main GPU performance
   win; see invariance_in_sr.md §8.

Roughly 6 weeks of focused work per the gpu_and_cv_via_sr.md scoping.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Minimal backend interface.

    A backend wraps the array library (numpy or jax.numpy) and the
    convolution / FFT routines tessera needs. Operators that are
    safe under JAX's `__array_function__` protocol (i.e., most numpy
    ufuncs) don't need explicit backend support — they auto-dispatch.

    What DOES need explicit support:
      - Array construction (asarray, zeros, full_like)
      - Convolution routines (direct convolve, FFT convolve)
      - Recursive EMA (numba kernel needs a JAX equivalent)
      - Device transfers (to_device, from_device)
    """
    name: str
    """Backend identifier, e.g. 'numpy' or 'jax'."""

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert input to the backend's native array type."""
        ...

    def zeros(self, shape, dtype=None) -> Any:
        """Create a zero array."""
        ...

    def full_like(self, a, fill_value, dtype=None) -> Any:
        """Create an array of the same shape/dtype as `a`."""
        ...

    def convolve(self, a, v, mode: str = "full") -> Any:
        """1-D convolution (same semantics as np.convolve)."""
        ...

    def is_available(self) -> bool:
        """Return True iff the backend's underlying library is importable."""
        ...


# ---------------- NumpyBackend (default) ----------------

class NumpyBackend:
    """Default backend. Wraps numpy + numba (existing tessera behaviour)."""
    name = "numpy"

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype if dtype is not None else np.float64)

    def full_like(self, a, fill_value, dtype=None):
        return np.full_like(a, fill_value, dtype=dtype)

    def convolve(self, a, v, mode="full"):
        return np.convolve(a, v, mode=mode)

    def is_available(self) -> bool:
        return True


# ---------------- JaxBackend (skeleton; stage-1 work pending) ----------------

class JaxBackend:
    """JAX backend skeleton. Functional for `asarray` and `zeros` /
    `full_like` / `convolve`; raises informative errors for routines
    that still need porting.

    Activate via `set_backend("jax")`. JAX must be installed:
        pip install jax jaxlib

    See `docs/shipped/gpu_backend.md` for the full porting roadmap.
    """
    name = "jax"

    def __init__(self) -> None:
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
            self._jnp = jnp
            self._available = True
        except ImportError:
            self._jnp = None
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def _require(self) -> None:
        if not self._available:
            raise ImportError(
                "JaxBackend requires the `jax` package. Install with:\n"
                "    pip install jax jaxlib\n"
                "On GPU: pip install jax[cuda12_pip]"
            )

    def asarray(self, x, dtype=None):
        self._require()
        return self._jnp.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        self._require()
        return self._jnp.zeros(shape, dtype=dtype if dtype is not None
                                else self._jnp.float64)

    def full_like(self, a, fill_value, dtype=None):
        self._require()
        return self._jnp.full_like(a, fill_value, dtype=dtype)

    def convolve(self, a, v, mode="full"):
        self._require()
        # jax.numpy.convolve has the same signature as np.convolve
        return self._jnp.convolve(a, v, mode=mode)


# ---------------- Global state ----------------

_CURRENT: Backend = NumpyBackend()

_REGISTRY: dict[str, type] = {
    "numpy": NumpyBackend,
    "jax": JaxBackend,
}


def set_backend(name: str) -> Backend:
    """Switch the active tessera backend.

    Parameters
    ----------
    name : {'numpy', 'jax'}
        Which backend to activate.

    Returns
    -------
    The new active backend instance.

    Raises
    ------
    ValueError  : unknown backend name
    ImportError : the backend's underlying library is not installed
    """
    global _CURRENT
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown backend {name!r}; available: {list(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    instance = cls()
    if not instance.is_available():
        raise ImportError(
            f"backend {name!r} is not available "
            "(underlying library not importable)"
        )
    _CURRENT = instance
    return _CURRENT


def get_backend() -> Backend:
    """Return the currently-active backend instance."""
    return _CURRENT


def current() -> Backend:
    """Shorter alias for `get_backend()`. Idiomatic call site:
        from tessera.backend import current
        arr = current().asarray([1, 2, 3])
    """
    return _CURRENT


# ---------------- Array-module dispatch (used by ops + Measure.apply) ----------------

def array_module(x):
    """Return the array library (numpy or jax.numpy) appropriate for x.

    Used by backend-polymorphic op implementations:

        xp = array_module(x)
        return (xp.asarray(a) > xp.asarray(b)).astype(xp.float64)

    Detection is by type-introspection (no eager jax import). If the input
    is a JAX array (regardless of which backend is active), returns
    `jax.numpy`. Otherwise returns `numpy`. This means trees can be
    evaluated on JAX arrays even with the default numpy backend active,
    and vice versa -- the array's own type drives the dispatch.

    Falls back to numpy if jax can't be imported.
    """
    # Cheap duck-type test: JAX arrays expose .device_buffer or live in jax.* modules
    if type(x).__module__.startswith("jax") or type(x).__module__.startswith("jaxlib"):
        try:
            import jax.numpy as jnp
            return jnp
        except ImportError:
            return np
    return np


__all__ = [
    "Backend", "NumpyBackend", "JaxBackend",
    "set_backend", "get_backend", "current",
    "array_module",
]
