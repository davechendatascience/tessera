"""Tier 3-C: 2D Measure JAX path.

Tests that Measure2D.apply(jax_array) returns a JAX array on the same
device and matches the numpy path output within float32 tolerance.

Required for the MNIST experiment (image features need 2D measures).

Skipped if jax is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax

from tessera.expression.measure import measure_lag, measure_diff, measure_ema
from tessera.expression.measure_2d import (
    Measure2D, Atom2D,
    measure_2d_atomic, measure_2d_separable,
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
    measure_2d_sobel_x, measure_2d_sobel_y,
)
from tessera.expression.tree import Var, FunctionalOp2D, evaluate
from tessera.expression.functional import LinearFunctional


# ---------------- Direct Measure2D.apply on JAX inputs ----------------

def test_measure_2d_atomic_apply_jax_matches_numpy():
    """Atomic 2D measure on a small field; JAX output should match numpy."""
    m = measure_2d_atomic([(1.0, 0, 0), (-1.0, 1, 0)])   # time-difference
    rng = np.random.default_rng(0)
    U_np = rng.standard_normal((10, 10)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    assert type(Y_jax).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_measure_2d_laplacian_apply_jax():
    """5-point spatial Laplacian — the MNIST workhorse."""
    m = measure_2d_laplacian_5pt()
    rng = np.random.default_rng(1)
    U_np = rng.standard_normal((20, 20)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_measure_2d_diff_t_apply_jax():
    """Time-difference 2D measure."""
    m = measure_2d_diff_t()
    rng = np.random.default_rng(2)
    U_np = rng.standard_normal((15, 8)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_measure_2d_grad_x_apply_jax():
    m = measure_2d_grad_x()
    rng = np.random.default_rng(3)
    U_np = rng.standard_normal((10, 12)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_measure_2d_sobel_x_apply_jax():
    m = measure_2d_sobel_x()
    rng = np.random.default_rng(4)
    U_np = rng.standard_normal((10, 12)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_measure_2d_separable_apply_jax():
    """Separable density: first-diff in time × first-diff in space.

    Use atomic-only sep_t (not EMA) to avoid the documented recursive-EMA
    vs kernel-conv divergence: numpy's measure_ema goes through a
    recursive fast-path that preserves the initial value with weight
    (1-α)^t, while the JAX path uses kernel-conv (truncated kernel,
    weight α·(1-α)^t at the boundary). For atomic measures the two paths
    agree exactly within float precision."""
    m = measure_2d_separable(
        measure_t=measure_diff(1),
        measure_x=measure_diff(1),
    )
    rng = np.random.default_rng(5)
    U_np = rng.standard_normal((20, 20)).astype(np.float32)
    U_jax = jnp.asarray(U_np)

    Y_np = m.apply(U_np, fill_warmup=0.0)
    Y_jax = m.apply(U_jax, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=5e-3, atol=1e-4)


# ---------------- evaluate(FunctionalOp2D) on JAX env ----------------

def test_evaluate_functional_op_2d_on_jax_env():
    """Tree with a FunctionalOp2D evaluated on a JAX env should produce
    a JAX array output."""
    m = measure_2d_laplacian_5pt()
    tree = FunctionalOp2D(m, Var("U"))
    rng = np.random.default_rng(0)
    U_np = rng.standard_normal((28, 28)).astype(np.float32)

    Y_np = evaluate(tree, {"U": U_np})
    Y_jax = evaluate(tree, {"U": jnp.asarray(U_np)})
    assert type(Y_jax).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)


def test_evaluate_pointwise_plus_2d_on_jax_env():
    """Tree mixing pointwise + 2D ops on JAX env. y = |Laplacian(U)|."""
    from tessera.expression.tree import UnOp
    m = measure_2d_laplacian_5pt()
    tree = UnOp("abs", FunctionalOp2D(m, Var("U")))
    rng = np.random.default_rng(0)
    U_np = rng.standard_normal((10, 10)).astype(np.float32)

    Y_np = evaluate(tree, {"U": U_np})
    Y_jax = evaluate(tree, {"U": jnp.asarray(U_np)})
    assert type(Y_jax).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(Y_jax), Y_np, rtol=1e-3, atol=1e-4)
