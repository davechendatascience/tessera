"""Tier 1 GPU backend port — verification that `evaluate(tree, env)` and
`Measure.apply` work end-to-end with JAX arrays.

What Tier 1 covers (per docs/shipped/gpu_backend.md):
- `array_module(x)` returns `jax.numpy` for JAX inputs, `numpy` for numpy
- All BIN_OP_FNS / UN_OP_FNS produce JAX arrays when given JAX inputs
- `Measure.apply(jax_array)` returns a JAX array (kernel-materialise path)
- `evaluate(tree, env_jax)` end-to-end returns a JAX array
- Outputs match the numpy path within float32/float64 tolerance

What Tier 1 does NOT do:
- jit-compile tree evaluation (Tier 2)
- batched-population eval (Tier 3)
- jax.grad const-opt (separate)

Skipped if jax is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax

from tessera.backend import array_module
from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, evaluate,
)
from tessera.expression.measure import (
    measure_lag, measure_diff, measure_ema, measure_signed_sum,
)
from tessera.expression.functional import LinearFunctional


# ---------------- array_module dispatch ----------------

def test_array_module_numpy():
    x = np.array([1.0, 2.0])
    assert array_module(x).__name__ == "numpy"


def test_array_module_jax():
    x = jnp.array([1.0, 2.0])
    assert array_module(x).__name__ == "jax.numpy"


# ---------------- Pointwise op tables on JAX inputs ----------------

@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "min", "max"])
def test_bin_op_returns_jax_array(op):
    from tessera.expression.tree import BIN_OP_FNS
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([2.0, 1.0, 4.0])
    y = BIN_OP_FNS[op](a, b)
    assert type(y).__module__.startswith("jax"), \
        f"{op}: got {type(y).__module__}"


@pytest.mark.parametrize("op", ["gt", "lt", "ge", "le"])
def test_indicator_op_returns_jax_array(op):
    from tessera.expression.tree import BIN_OP_FNS
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([2.0, 2.0, 2.0])
    y = BIN_OP_FNS[op](a, b)
    assert type(y).__module__.startswith("jax"), \
        f"{op}: got {type(y).__module__}"
    np.testing.assert_array_equal(
        np.asarray(y),
        np.asarray(BIN_OP_FNS[op](np.asarray(a), np.asarray(b)))
    )


def test_pow_returns_jax_array():
    from tessera.expression.tree import BIN_OP_FNS
    a = jnp.array([2.0, 3.0])
    b = jnp.array([2.0, 0.5])
    y = BIN_OP_FNS["pow"](a, b)
    assert type(y).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(y), [4.0, np.sqrt(3.0)], rtol=1e-5)


@pytest.mark.parametrize("op", ["tanh", "abs", "sign", "neg", "step",
                                "sqrt", "log", "exp"])
def test_un_op_returns_jax_array(op):
    from tessera.expression.tree import UN_OP_FNS
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    y = UN_OP_FNS[op](x)
    assert type(y).__module__.startswith("jax"), \
        f"{op}: got {type(y).__module__}"


@pytest.mark.parametrize("op", ["reduce_mean", "reduce_max", "reduce_sum", "reduce_std"])
def test_reduce_op_works_on_jax(op):
    """Reduce ops return scalar arrays (numpy float64 or jax scalar)
    rather than Python float, so that the result is jit-traceable.
    Downstream BinOp/UnOp broadcasts handle this transparently."""
    from tessera.expression.tree import UN_OP_FNS
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = UN_OP_FNS[op](x)
    # Result should be a 0-d JAX array or scalar
    assert hasattr(y, "shape"), f"{op}: expected an array-like, got {type(y)}"
    expected = UN_OP_FNS[op](np.asarray(x))
    np.testing.assert_allclose(float(y), float(expected), rtol=1e-5)


# ---------------- Measure.apply on JAX inputs ----------------

def test_measure_lag_apply_jax():
    """measure_lag(3) shifts the series by 3."""
    m = measure_lag(3)
    x_np = np.arange(20, dtype=np.float64)
    x_jax = jnp.asarray(x_np)
    y_jax = m.apply(x_jax, fill_warmup=0.0)
    assert type(y_jax).__module__.startswith("jax")
    y_np = m.apply(x_np, fill_warmup=0.0)
    # JAX defaults to float32; loosen rtol to match
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-3)


def test_measure_diff_apply_jax():
    """measure_diff(1) gives first differences."""
    m = measure_diff(1)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(100)
    x_jax = jnp.asarray(x_np)
    y_jax = m.apply(x_jax, fill_warmup=0.0)
    y_np = m.apply(x_np, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-5, atol=1e-7)


def test_measure_ema_apply_jax():
    """EMA via kernel materialisation on JAX. Output should agree with
    numpy path to within numerical tolerance (the recursive vs convolve
    paths differ in float ordering)."""
    m = measure_ema(halflife=8.0, support_max=50)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(200)
    x_jax = jnp.asarray(x_np)
    y_jax = m.apply(x_jax, fill_warmup=0.0)
    # numpy path uses recursive EMA; JAX path uses kernel-conv truncated
    # at support_max=50. They agree only approximately at long timescales.
    # Compare against the numpy *kernel* path for an exact match.
    y_np_kernel = m.apply(x_np, fill_warmup=0.0, backend="kernel")
    np.testing.assert_allclose(np.asarray(y_jax), y_np_kernel,
                               rtol=1e-5, atol=1e-7)


def test_measure_signed_sum_apply_jax():
    m = measure_signed_sum([(0.5, 0), (-0.3, 5), (0.1, 12)])
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(100)
    x_jax = jnp.asarray(x_np)
    y_jax = m.apply(x_jax, fill_warmup=0.0)
    y_np = m.apply(x_np, fill_warmup=0.0)
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-5, atol=1e-7)


# ---------------- evaluate(tree, env_jax) end-to-end ----------------

def test_evaluate_pointwise_tree_jax():
    """y = tanh(a + b) - c on JAX arrays."""
    tree = BinOp("sub",
                 UnOp("tanh", BinOp("add", Var("a"), Var("b"))),
                 Var("c"))
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50); b = rng.standard_normal(50); c = rng.standard_normal(50)

    y_np = evaluate(tree, {"a": a, "b": b, "c": c})
    y_jax = evaluate(tree, {
        "a": jnp.asarray(a), "b": jnp.asarray(b), "c": jnp.asarray(c),
    })
    assert type(y_jax).__module__.startswith("jax")
    # JAX defaults to float32; loosen rtol to match
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-3)


def test_evaluate_with_indicator_jax():
    """y = (a > b) * c on JAX arrays — exercises gt indicator."""
    tree = BinOp("mul",
                 BinOp("gt", Var("a"), Var("b")),
                 Var("c"))
    rng = np.random.default_rng(1)
    a = rng.standard_normal(50); b = rng.standard_normal(50); c = rng.standard_normal(50)

    y_np = evaluate(tree, {"a": a, "b": b, "c": c})
    y_jax = evaluate(tree, {
        "a": jnp.asarray(a), "b": jnp.asarray(b), "c": jnp.asarray(c),
    })
    assert type(y_jax).__module__.startswith("jax")
    # JAX defaults to float32; loosen rtol to match
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-3)


def test_evaluate_with_transcendental_jax():
    """y = sqrt(|a|) + log(|b| + 1) - exp(c/10) on JAX arrays."""
    tree = BinOp("sub",
                 BinOp("add",
                       UnOp("sqrt", Var("a")),
                       UnOp("log", BinOp("add", Var("b"), Const(1.0)))),
                 UnOp("exp", BinOp("div", Var("c"), Const(10.0))))
    rng = np.random.default_rng(2)
    a = rng.standard_normal(50); b = np.abs(rng.standard_normal(50))
    c = rng.standard_normal(50)

    y_np = evaluate(tree, {"a": a, "b": b, "c": c})
    y_jax = evaluate(tree, {
        "a": jnp.asarray(a), "b": jnp.asarray(b), "c": jnp.asarray(c),
    })
    assert type(y_jax).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(y_jax), y_np, rtol=1e-4)


def test_evaluate_with_linear_functional_jax():
    """y = LinearFunctional(diff)(x) on JAX arrays — exercises Measure.apply
    routing through the JAX path."""
    f = LinearFunctional(measure=measure_diff(1))
    tree = FunctionalOp(f, (Var("x"),))
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)

    y_np = evaluate(tree, {"x": x})
    y_jax = evaluate(tree, {"x": jnp.asarray(x)})
    assert type(y_jax).__module__.startswith("jax")
    # First row is warmup (NaN by default); compare from row 1 onward
    np.testing.assert_allclose(
        np.asarray(y_jax)[1:], y_np[1:], rtol=1e-5, atol=1e-7
    )
