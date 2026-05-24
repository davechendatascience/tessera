"""Tests for atan2 / acos / asin — the inverse-trig primitives shipped
to close the IK Tier-D gap (per `docs/planned/roadmap.md` §2.3).

Same test pattern as test_sin_cos.py (lifecycle ship #1).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, BIN_OPS, UN_OPS, BIN_OP_FNS, UN_OP_FNS, evaluate,
)
from tessera.expression.interval import (
    Interval, _BIN_IVAL_FNS, _UN_IVAL_FNS,
)
from tessera.expression.simplify import simplify


# ---------------- Op-table registration ----------------

def test_atan2_acos_asin_in_op_tables():
    assert "atan2" in BIN_OPS
    assert "atan2" in BIN_OP_FNS
    assert "atan2" in _BIN_IVAL_FNS
    assert "acos" in UN_OPS
    assert "asin" in UN_OPS
    assert "acos" in UN_OP_FNS
    assert "asin" in UN_OP_FNS
    assert "acos" in _UN_IVAL_FNS
    assert "asin" in _UN_IVAL_FNS


def test_constructors_accept_new_ops():
    BinOp("atan2", Var("y"), Var("x"))
    UnOp("acos", Var("x"))
    UnOp("asin", Var("x"))


# ---------------- Evaluation correctness ----------------

def test_atan2_numpy_basic_quadrants():
    """atan2 is quadrant-aware. Verify against known angles."""
    y = np.array([0.0, 1.0, 0.0, -1.0])
    x = np.array([1.0, 0.0, -1.0, 0.0])
    out = evaluate(BinOp("atan2", Var("y"), Var("x")), {"y": y, "x": x})
    expected = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_acos_protected_clips_out_of_domain():
    """acos input clipped to [-1, 1]; values outside the domain don't
    produce NaN."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])   # outside-domain values present
    out = evaluate(UnOp("acos", Var("x")), {"x": x})
    assert np.all(np.isfinite(out))   # no NaN due to clipping
    # acos(clip(-2, -1, 1)) = acos(-1) = π
    # acos(clip(2, -1, 1)) = acos(1) = 0
    np.testing.assert_allclose(out, [math.pi, math.pi, math.pi / 2, 0.0, 0.0], atol=1e-7)


def test_asin_protected_clips_out_of_domain():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = evaluate(UnOp("asin", Var("x")), {"x": x})
    assert np.all(np.isfinite(out))
    # asin(clip(-2, -1, 1)) = asin(-1) = -π/2
    np.testing.assert_allclose(out, [-math.pi / 2, -math.pi / 2, 0.0,
                                      math.pi / 2, math.pi / 2], atol=1e-7)


def test_inverse_identity_for_trig_pairs():
    """sin(asin(x)) = x for x in [-1, 1]; cos(acos(x)) = x.
    These are the analytical identities; verify on samples."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 100)
    env = {"x": x}
    sin_asin = evaluate(UnOp("sin", UnOp("asin", Var("x"))), env)
    cos_acos = evaluate(UnOp("cos", UnOp("acos", Var("x"))), env)
    np.testing.assert_allclose(sin_asin, x, atol=1e-6)
    np.testing.assert_allclose(cos_acos, x, atol=1e-6)


def test_atan2_inverse_of_sin_cos():
    """atan2(sin(θ), cos(θ)) = θ for θ ∈ (-π, π]."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(-math.pi + 0.01, math.pi - 0.01, 100)   # avoid the ±π wrap
    env = {"t": theta}
    out = evaluate(BinOp("atan2",
                          UnOp("sin", Var("t")),
                          UnOp("cos", Var("t"))),
                    env)
    np.testing.assert_allclose(out, theta, atol=1e-6)


# ---------------- Interval bounds ----------------

def test_atan2_interval_bounded():
    """atan2 output always in [-π, π]."""
    # Any input → conservative [-π, π]
    iv_y = Interval(-100.0, 100.0)
    iv_x = Interval(-100.0, 100.0)
    out = _BIN_IVAL_FNS["atan2"](iv_y, iv_x)
    assert abs(out.lo + math.pi) < 1e-9
    assert abs(out.hi - math.pi) < 1e-9


def test_acos_interval_bounded():
    """acos output always in [0, π]."""
    out = _UN_IVAL_FNS["acos"](Interval(-1e6, 1e6))
    assert abs(out.lo) < 1e-9
    assert abs(out.hi - math.pi) < 1e-9


def test_asin_interval_bounded():
    """asin output always in [-π/2, π/2]."""
    out = _UN_IVAL_FNS["asin"](Interval(-1e6, 1e6))
    assert abs(out.lo + math.pi / 2) < 1e-9
    assert abs(out.hi - math.pi / 2) < 1e-9


# ---------------- Simplifier (constant folding via UN/BIN_OP_FNS path) ----------------

def test_simplify_folds_acos_of_one():
    folded = simplify(UnOp("acos", Const(1.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value) < 1e-10


def test_simplify_folds_acos_of_minus_one():
    folded = simplify(UnOp("acos", Const(-1.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value - math.pi) < 1e-10


def test_simplify_folds_asin_of_zero():
    folded = simplify(UnOp("asin", Const(0.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value) < 1e-10


def test_simplify_folds_atan2_known_corners():
    """atan2(0, 1) = 0; atan2(1, 0) = π/2."""
    folded = simplify(BinOp("atan2", Const(0.0), Const(1.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value) < 1e-10
    folded = simplify(BinOp("atan2", Const(1.0), Const(0.0)))
    assert isinstance(folded, Const)
    assert abs(folded.value - math.pi / 2) < 1e-10


# ---------------- Op-swap mutation group ----------------

def test_op_swap_acos_to_asin():
    """{acos, asin} is an op_swap group; mutation can swap one for the other."""
    import random
    from tessera.expression.mutation import op_swap

    tree = UnOp("acos", Var("x"))
    seen_swap = False
    for seed in range(50):
        rng = random.Random(seed)
        new_tree = op_swap(tree, rng)
        if isinstance(new_tree, UnOp) and new_tree.op == "asin":
            seen_swap = True
            break
    assert seen_swap, "op_swap never produced asin from acos in 50 attempts"


# ---------------- JAX path ----------------

def test_atan2_acos_asin_on_jax_array():
    jnp = pytest.importorskip("jax.numpy")
    y_jax = jnp.array([0.0, 1.0, -1.0])
    x_jax = jnp.array([1.0, 0.0, 0.0])
    out_atan2 = evaluate(BinOp("atan2", Var("y"), Var("x")), {"y": y_jax, "x": x_jax})
    out_acos = evaluate(UnOp("acos", Var("x")), {"x": jnp.array([1.0, 0.0, -1.0])})
    out_asin = evaluate(UnOp("asin", Var("x")), {"x": jnp.array([1.0, 0.0, -1.0])})
    assert type(out_atan2).__module__.startswith("jax")
    assert type(out_acos).__module__.startswith("jax")
    assert type(out_asin).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(out_atan2), [0.0, math.pi / 2, -math.pi / 2], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_acos), [0.0, math.pi / 2, math.pi], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_asin), [math.pi / 2, 0.0, -math.pi / 2], atol=1e-5)
