"""Tests for the trigonometric primitives: sin, cos.

Unlike sqrt/exp/log/pow, sin/cos are bounded everywhere on the reals,
so no "protected" form is needed. The tests verify:
- Op-table registration (UN_OPS, UN_OP_FNS, interval bounds)
- Evaluation correctness on numpy + JAX
- Interval soundness: outputs within [-1, 1] for any finite input
- Simplifier handles them (constant fold via numpy compat)
- Op-swap group {sin, cos} works
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, UnOp, UN_OPS, UN_OP_FNS, evaluate,
)
from tessera.expression.interval import (
    Interval, interval_evaluate, _UN_IVAL_FNS,
)
from tessera.expression.simplify import simplify


# ---------------- Op-table registration ----------------

def test_sin_cos_in_un_ops():
    assert "sin" in UN_OPS
    assert "cos" in UN_OPS
    assert "sin" in UN_OP_FNS
    assert "cos" in UN_OP_FNS
    assert "sin" in _UN_IVAL_FNS
    assert "cos" in _UN_IVAL_FNS


def test_unop_constructor_accepts_sin_cos():
    UnOp("sin", Var("x"))   # must not raise
    UnOp("cos", Var("x"))


# ---------------- Evaluation correctness ----------------

def test_sin_numpy_correctness():
    env = {"x": np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])}
    y = evaluate(UnOp("sin", Var("x")), env)
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0, -1.0], atol=1e-7)


def test_cos_numpy_correctness():
    env = {"x": np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])}
    y = evaluate(UnOp("cos", Var("x")), env)
    np.testing.assert_allclose(y, [1.0, 0.0, -1.0, 0.0], atol=1e-7)


def test_sin_cos_evaluated_in_compound_tree():
    """y = sin(a)² + cos(a)² == 1 (the Pythagorean identity)."""
    from tessera.expression.tree import BinOp
    tree = BinOp(
        "add",
        BinOp("mul", UnOp("sin", Var("a")), UnOp("sin", Var("a"))),
        BinOp("mul", UnOp("cos", Var("a")), UnOp("cos", Var("a"))),
    )
    rng = np.random.default_rng(0)
    env = {"a": rng.uniform(-10, 10, 200)}
    y = evaluate(tree, env)
    np.testing.assert_allclose(y, 1.0, atol=1e-7)


# ---------------- Interval bounds ----------------

def test_sin_cos_interval_is_bounded_minus_one_one():
    """Conservative bound: sin/cos lie in [-1, 1] for any input."""
    huge = Interval(-1e6, 1e6)
    assert _UN_IVAL_FNS["sin"](huge).lo == -1.0
    assert _UN_IVAL_FNS["sin"](huge).hi == 1.0
    assert _UN_IVAL_FNS["cos"](huge).lo == -1.0
    assert _UN_IVAL_FNS["cos"](huge).hi == 1.0


def test_interval_sin_soundness():
    """For any sample value in the input interval, sin(value) must lie
    in the output interval. We use random samples to confirm."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        lo = rng.uniform(-5, 5)
        hi = lo + rng.uniform(0, 5)
        iv_in = Interval(lo, hi)
        iv_out = _UN_IVAL_FNS["sin"](iv_in)
        samples = rng.uniform(lo, hi, 100)
        vals = np.sin(samples)
        assert iv_out.lo <= vals.min() + 1e-9
        assert vals.max() <= iv_out.hi + 1e-9


# ---------------- Simplifier (constant folding via UN_OP_FNS path) ----------------

def test_simplify_folds_sin_of_const():
    """simplify uses UN_OP_FNS[node.op](a.value) to fold constant unary
    inputs. Should work for sin/cos transparently."""
    folded = simplify(UnOp("sin", Const(0.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value - 0.0) < 1e-10

    folded = simplify(UnOp("cos", Const(0.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value - 1.0) < 1e-10

    folded = simplify(UnOp("sin", Const(math.pi / 2)))
    assert isinstance(folded, Const)
    assert abs(folded.value - 1.0) < 1e-10


# ---------------- Op-swap mutation group ----------------

def test_op_swap_can_swap_sin_to_cos():
    """{sin, cos} is in the op_swap groups; mutation may swap one to the
    other."""
    import random
    from tessera.expression.mutation import op_swap

    tree = UnOp("sin", Var("x"))
    # Try many seeds; over many calls, at least one should swap to cos
    seen_swapped = False
    for seed in range(50):
        rng = random.Random(seed)
        new_tree = op_swap(tree, rng)
        if isinstance(new_tree, UnOp) and new_tree.op == "cos":
            seen_swapped = True
            break
    assert seen_swapped, "op_swap never produced cos from sin in 50 attempts"


# ---------------- JAX path ----------------

def test_sin_cos_on_jax_array():
    """sin/cos must be jit-compatible (no Python branches on traced
    values). The simplest test: evaluate produces a JAX array on JAX
    input."""
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.array([0.0, math.pi / 2, math.pi])
    y_sin = evaluate(UnOp("sin", Var("x")), {"x": x})
    y_cos = evaluate(UnOp("cos", Var("x")), {"x": x})
    assert type(y_sin).__module__.startswith("jax")
    assert type(y_cos).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(y_sin), [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(y_cos), [1.0, 0.0, -1.0], atol=1e-5)


def test_sin_cos_jit_compile():
    """sin/cos must work inside jax.jit (no Python branches on traced)."""
    pytest.importorskip("jax.numpy")
    from tessera.expression.jit import compile_tree, clear_jit_cache, evaluate_jit
    import jax.numpy as jnp
    clear_jit_cache()
    tree = UnOp("sin", Var("x"))
    x = jnp.linspace(0, 2 * math.pi, 50)
    y = evaluate_jit(tree, {"x": x})
    assert type(y).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(y), np.sin(np.asarray(x)), atol=1e-5)
