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


# ---------------- compile_image_predictor (Tier 3-D batched vmap) ----------------

def test_compile_image_predictor_basic():
    """Compiled per-sample fn applied to a batch of images returns [N]
    mean-pooled scalars matching the per-image loop."""
    from tessera.expression.tree import UnOp
    from tessera.expression import compile_image_predictor, clear_image_predictor_cache

    m = measure_2d_laplacian_5pt()
    tree = UnOp("abs", FunctionalOp2D(m, Var("image")))

    rng = np.random.default_rng(0)
    N, H, W = 20, 14, 14
    images = rng.standard_normal((N, H, W)).astype(np.float32)

    # Reference: per-image Python loop with mean-pool
    def loop_predict(tree, X):
        preds = np.zeros(len(X))
        for i, img in enumerate(X):
            out = np.asarray(evaluate(tree, {"image": img}), dtype=np.float64)
            mask = np.isfinite(out)
            preds[i] = out[mask].mean() if mask.any() else float("nan")
        return preds
    pred_loop = loop_predict(tree, images)

    # jit+vmap batched
    clear_image_predictor_cache()
    fn = compile_image_predictor(tree, batch_var="image", reduce="mean")
    pred_jit = np.asarray(fn(jnp.asarray(images)))

    np.testing.assert_allclose(pred_jit, pred_loop, rtol=1e-3, atol=1e-4)


def test_compile_image_predictor_caches():
    from tessera.expression.tree import UnOp
    from tessera.expression import (
        compile_image_predictor, clear_image_predictor_cache,
        image_predictor_cache_size,
    )

    clear_image_predictor_cache()
    assert image_predictor_cache_size() == 0

    m = measure_2d_laplacian_5pt()
    tree = UnOp("abs", FunctionalOp2D(m, Var("image")))
    fn1 = compile_image_predictor(tree)
    fn2 = compile_image_predictor(tree)
    assert fn1 is fn2
    assert image_predictor_cache_size() == 1


def test_compile_image_predictor_reduce_modes():
    """All reduce= modes return finite scalars on a clean input."""
    from tessera.expression.tree import UnOp
    from tessera.expression import compile_image_predictor, clear_image_predictor_cache

    m = measure_2d_laplacian_5pt()
    tree = UnOp("abs", FunctionalOp2D(m, Var("image")))

    rng = np.random.default_rng(1)
    images = rng.standard_normal((5, 10, 10)).astype(np.float32)
    images_jax = jnp.asarray(images)

    for reduce in ["mean", "max", "sum"]:
        clear_image_predictor_cache()
        fn = compile_image_predictor(tree, reduce=reduce)
        out = np.asarray(fn(images_jax))
        assert out.shape == (5,), f"{reduce}: shape {out.shape}"
        assert np.all(np.isfinite(out)), f"{reduce}: non-finite output"
