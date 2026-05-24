"""Tests for the new protected transcendentals: sqrt, exp, log, pow.

Each op is PySR-style protected:
  sqrt(x) := sqrt(|x|)
  log(x)  := log(max(|x|, 1e-12))
  exp(x)  := exp(clip(x, ±50))
  pow(a, b) := pow(max(|a|, 1e-12), clip(b, ±8))   then 0 for non-finite

These contracts must hold for the GP search to use them safely.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, BIN_OPS, UN_OPS, evaluate,
)
from tessera.expression.interval import (
    Interval, interval_evaluate, env_intervals_from_arrays,
)
from tessera.expression.simplify import simplify


# ---------------- Op-table registration ----------------

def test_new_ops_in_tables():
    for name in ("sqrt", "log", "exp"):
        assert name in UN_OPS, f"{name} missing from UN_OPS"
    assert "pow" in BIN_OPS


# ---------------- Evaluation correctness ----------------

def test_sqrt_protected_negative():
    """sqrt(|x|) on negative input must not produce NaN."""
    env = {"x": np.array([-4.0, -1.0, 0.0, 1.0, 4.0])}
    y = evaluate(UnOp("sqrt", Var("x")), env)
    np.testing.assert_allclose(y, [2.0, 1.0, 0.0, 1.0, 2.0])


def test_log_protected_zero_and_negative():
    """log(max(|x|, eps)) — no NaN, no -inf."""
    env = {"x": np.array([-1.0, 0.0, 1.0, math.e])}
    y = evaluate(UnOp("log", Var("x")), env)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y[2], 0.0, atol=1e-10)
    np.testing.assert_allclose(y[3], 1.0, atol=1e-10)


def test_exp_protected_large_input():
    """exp(x) clipped at x=±50 — no overflow."""
    env = {"x": np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])}
    y = evaluate(UnOp("exp", Var("x")), env)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y[2], 1.0)
    np.testing.assert_allclose(y[3], math.e)


def test_pow_protected_negative_base():
    """pow(|a|, clip(b, ±8)) — works on negative base."""
    env = {"a": np.array([-2.0, 2.0, -3.0]), "b": np.array([2.0, 3.0, 0.5])}
    y = evaluate(BinOp("pow", Var("a"), Var("b")), env)
    np.testing.assert_allclose(y, [4.0, 8.0, math.sqrt(3.0)])


def test_pow_protected_extreme_exponent():
    """pow exponent clipped to ±8 — no overflow."""
    env = {"a": np.array([2.0]), "b": np.array([100.0])}
    y = evaluate(BinOp("pow", Var("a"), Var("b")), env)
    np.testing.assert_allclose(y, [2.0 ** 8])


def test_pow_zero_to_negative():
    """pow(0, -k) — floor saves it; result is huge but finite."""
    env = {"a": np.array([0.0]), "b": np.array([-2.0])}
    y = evaluate(BinOp("pow", Var("a"), Var("b")), env)
    assert np.all(np.isfinite(y))


# ---------------- Interval bounds (soundness) ----------------

def _check_sound(tree, env, iv_env, n_samples=200):
    """Sound = every actual sample is inside [lo, hi]."""
    iv = interval_evaluate(tree, iv_env)
    y = evaluate(tree, env)
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    # Permit small numerical slack at the bounds.
    slack = 1e-6 + abs(iv.hi - iv.lo) * 1e-6
    assert iv.lo - slack <= y.min(), f"sample {y.min()} below iv.lo {iv.lo}"
    assert y.max() <= iv.hi + slack, f"sample {y.max()} above iv.hi {iv.hi}"


def test_interval_sqrt_sound():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 200)
    env = {"x": x}
    _check_sound(UnOp("sqrt", Var("x")), env, env_intervals_from_arrays(env))


def test_interval_log_sound():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    env = {"x": x}
    _check_sound(UnOp("log", Var("x")), env, env_intervals_from_arrays(env))


def test_interval_exp_sound():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 200)
    env = {"x": x}
    _check_sound(UnOp("exp", Var("x")), env, env_intervals_from_arrays(env))


def test_interval_pow_sound():
    rng = np.random.default_rng(0)
    a = rng.uniform(-2, 2, 200)
    b = rng.uniform(-3, 3, 200)
    env = {"a": a, "b": b}
    _check_sound(BinOp("pow", Var("a"), Var("b")),
                 env, env_intervals_from_arrays(env))


# ---------------- Simplifier folds ----------------

def test_simplify_log_exp_inverse():
    """log(exp(x)) → x."""
    t = UnOp("log", UnOp("exp", Var("x")))
    assert simplify(t) == Var("x")


def test_simplify_exp_log_to_abs():
    """exp(log(x)) → |x| under protected semantics."""
    t = UnOp("exp", UnOp("log", Var("x")))
    assert simplify(t) == UnOp("abs", Var("x"))


def test_simplify_sqrt_abs_idempotent():
    """sqrt(|x|) = sqrt(x) — drop redundant abs."""
    t = UnOp("sqrt", UnOp("abs", Var("x")))
    assert simplify(t) == UnOp("sqrt", Var("x"))


def test_simplify_pow_zero_exponent():
    t = BinOp("pow", Var("x"), Const(0.0))
    assert simplify(t) == Const(1.0)


def test_simplify_pow_one_exponent():
    """pow(x, 1) → |x| (protected pow drops sign)."""
    t = BinOp("pow", Var("x"), Const(1.0))
    assert simplify(t) == UnOp("abs", Var("x"))


def test_simplify_pow_zero_base():
    t = BinOp("pow", Const(0.0), Var("x"))
    assert simplify(t) == Const(0.0)


# ---------------- Discovery sanity (very small GP) ----------------

def test_gp_can_use_sqrt_in_population():
    """GP run with new ops should not crash and should find SOMETHING."""
    from tessera.search import GP, GPConfig

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 5.0, 300)
    y = np.sqrt(x)

    cfg = GPConfig(pop_size=40, n_gens=10, seed=42, pointwise_only=True)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])
    # Should at least find a finite-loss candidate.
    assert all(math.isfinite(c.train_loss) for c in front)
