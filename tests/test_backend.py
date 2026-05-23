"""Tests for the switchable CPU/GPU backend abstraction."""
import numpy as np
import pytest

from tessera.backend import (
    Backend, NumpyBackend, JaxBackend,
    set_backend, get_backend, current,
)


# ---------------- Default backend ----------------

def test_default_backend_is_numpy():
    b = get_backend()
    assert b.name == "numpy"
    assert isinstance(b, NumpyBackend)


def test_current_alias():
    assert current() is get_backend()


# ---------------- NumpyBackend behaviour ----------------

def test_numpy_backend_asarray_returns_numpy():
    b = NumpyBackend()
    arr = b.asarray([1.0, 2.0, 3.0])
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


def test_numpy_backend_zeros():
    b = NumpyBackend()
    arr = b.zeros((3, 3))
    assert arr.shape == (3, 3)
    assert (arr == 0).all()


def test_numpy_backend_convolve_matches_np_convolve():
    b = NumpyBackend()
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, -1.0])
    out = b.convolve(a, v)
    expected = np.convolve(a, v)
    np.testing.assert_array_equal(out, expected)


def test_numpy_backend_always_available():
    assert NumpyBackend().is_available() is True


# ---------------- Set/get ----------------

def test_set_backend_numpy_returns_instance():
    b = set_backend("numpy")
    assert b.name == "numpy"
    assert get_backend() is b


def test_set_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        set_backend("nonexistent")


def test_set_backend_jax_when_unavailable_raises():
    # If JAX is not installed in this environment, set_backend("jax")
    # should raise ImportError with a helpful message.
    jax_backend = JaxBackend()
    if jax_backend.is_available():
        pytest.skip("JAX is installed in this env; can't test the unavailable path")
    with pytest.raises(ImportError, match="not available"):
        set_backend("jax")
    # The current backend should be unchanged after a failed switch
    assert get_backend().name == "numpy"


# ---------------- JaxBackend skeleton ----------------

def test_jax_backend_construction_is_safe_without_jax():
    """JaxBackend() does NOT raise if jax isn't installed; it just
    reports is_available() = False. Errors come at use site."""
    b = JaxBackend()  # should not raise
    if not b.is_available():
        # Calling a backend method should raise
        with pytest.raises(ImportError, match="requires the `jax` package"):
            b.asarray([1.0, 2.0])
    else:
        # JAX is installed — the call should succeed
        arr = b.asarray([1.0, 2.0])
        assert arr.shape == (2,)


# ---------------- Protocol check ----------------

def test_numpy_backend_satisfies_backend_protocol():
    assert isinstance(NumpyBackend(), Backend)


def test_jax_backend_satisfies_backend_protocol():
    assert isinstance(JaxBackend(), Backend)


# ---------------- Cleanup ----------------

def teardown_function():
    """Reset to default after each test."""
    set_backend("numpy")
