"""Tests for the tessera.expression.tree module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp,
    BIN_OPS, UN_OPS,
    complexity, depth, used_features, iter_subtrees, replace_at, evaluate,
)
from tessera.expression.measure import (
    measure_lag, measure_diff, measure_ema, measure_signed_sum,
)
from tessera.expression.functional import (
    LinearFunctional, SeparableBilinear, Volterra2,
)
from tessera.expression.cache import FunctionalCache


def _rng_env(n=500, seed=0, keys=("x", "y")):
    """Build an env dict of named random series."""
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(n) for k in keys}


# ---------------- Construction + str repr ----------------

def test_var_const_str():
    assert str(Var("temp")) == "temp"
    assert str(Const(0.5)) == "0.5"
    assert str(Const(1.234567891)) == "1.23457"   # 6 sig figs

def test_binop_infix_print():
    t = BinOp("add", Var("x"), Const(1.0))
    assert "x + 1" in str(t)

def test_binop_minmax_prefix_print():
    t = BinOp("min", Var("x"), Var("y"))
    assert str(t).startswith("min(")

def test_unop_rejects_unknown():
    with pytest.raises(ValueError):
        UnOp("nope", Var("x"))

def test_binop_rejects_unknown():
    with pytest.raises(ValueError):
        BinOp("xor", Var("x"), Var("y"))

def test_functional_op_arity_check():
    bil = SeparableBilinear(measure_a=measure_ema(24), measure_b=measure_diff(1))
    # bil expects 2 args
    with pytest.raises(ValueError):
        FunctionalOp(bil, (Var("x"),))
    FunctionalOp(bil, (Var("x"), Var("y")))   # ok


# ---------------- Structural helpers ----------------

def test_complexity_and_depth():
    leaf = Var("x")
    assert complexity(leaf) == 1
    assert depth(leaf) == 1

    t = BinOp("add", Var("x"), BinOp("mul", Var("y"), Const(2.0)))
    assert complexity(t) == 5
    assert depth(t) == 3

def test_complexity_with_functional():
    lf = LinearFunctional(measure=measure_ema(24))
    t = UnOp("tanh", FunctionalOp(lf, (Var("x"),)))
    # tanh + funcop + var = 3 nodes
    assert complexity(t) == 3
    assert depth(t) == 3

def test_used_features():
    t = BinOp("mul",
              Var("temp"),
              UnOp("tanh", BinOp("add", Var("pressure"), Const(0.5))))
    assert used_features(t) == {"temp", "pressure"}

def test_iter_subtrees():
    t = BinOp("add", Var("x"), Const(1.0))
    subs = list(iter_subtrees(t))
    assert len(subs) == 3
    assert subs[0] is t
    assert subs[1] == Var("x")
    assert subs[2] == Const(1.0)

def test_replace_at():
    t = BinOp("add", Var("x"), Const(1.0))
    # Index 1 is the Var, index 2 is the Const
    t2 = replace_at(t, 1, Var("y"))
    assert t2 == BinOp("add", Var("y"), Const(1.0))
    # Original unchanged
    assert t == BinOp("add", Var("x"), Const(1.0))


# ---------------- Hashability + value equality ----------------

def test_equal_trees_hash_equally():
    a = BinOp("add", Var("x"), Const(1.0))
    b = BinOp("add", Var("x"), Const(1.0))
    assert a == b
    assert hash(a) == hash(b)
    # And as dict keys
    d = {a: "val"}
    assert d[b] == "val"

def test_functional_op_hashable():
    lf = LinearFunctional(measure=measure_ema(24))
    a = FunctionalOp(lf, (Var("x"),))
    b = FunctionalOp(lf, (Var("x"),))
    assert a == b
    assert hash(a) == hash(b)


# ---------------- Evaluation: pointwise ----------------

def test_eval_var():
    env = _rng_env(50, 0)
    y = evaluate(Var("x"), env)
    assert np.array_equal(y, env["x"])

def test_eval_const_broadcasts_in_binop():
    env = _rng_env(50, 0)
    t = BinOp("add", Var("x"), Const(1.0))
    y = evaluate(t, env)
    assert np.allclose(y, env["x"] + 1.0)

def test_eval_safe_div():
    env = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([1.0, 0.0, 2.0])}
    t = BinOp("div", Var("x"), Var("y"))
    y = evaluate(t, env)
    assert np.allclose(y, [1.0, 0.0, 1.5])   # zero-denom → 0

def test_eval_unop_chain():
    env = _rng_env(20, 1)
    t = UnOp("abs", UnOp("neg", Var("x")))
    y = evaluate(t, env)
    assert np.allclose(y, np.abs(env["x"]))


# ---------------- Evaluation: functional ----------------

def test_eval_linear_functional():
    env = _rng_env(500, 2)
    lf = LinearFunctional(measure=measure_ema(24))
    t = FunctionalOp(lf, (Var("x"),))
    y = evaluate(t, env)
    expected = measure_ema(24).apply(env["x"])
    assert np.allclose(y, expected, equal_nan=True)

def test_eval_bilinear_functional():
    env = _rng_env(500, 3)
    bil = SeparableBilinear(measure_a=measure_diff(1), measure_b=measure_ema(24))
    t = FunctionalOp(bil, (Var("x"), Var("y")))
    y = evaluate(t, env)
    expected = (measure_diff(1).apply(env["x"])
                * measure_ema(24).apply(env["y"]))
    assert np.allclose(y, expected, equal_nan=True)

def test_eval_volterra2():
    env = _rng_env(500, 4)
    v2 = Volterra2(measure_a=measure_diff(1), measure_b=measure_diff(1))
    t = FunctionalOp(v2, (Var("x"),))
    y = evaluate(t, env)
    expected = measure_diff(1).apply(env["x"]) ** 2
    np.testing.assert_array_almost_equal(
        np.nan_to_num(y), np.nan_to_num(expected), decimal=10,
    )

def test_eval_nested_functional_with_zero_warmup():
    """tanh(ema(diff(x, 1), 24)) — composed functionals through tree.

    When composing through functionals, set fill_warmup=0 to avoid the
    recursive EMA propagating NaN forever (once any input is NaN, the
    α·x[t]+(1−α)·y[t−1] recursion stays NaN).
    """
    env = _rng_env(500, 5)
    lf_diff = LinearFunctional(measure=measure_diff(1))
    lf_ema = LinearFunctional(measure=measure_ema(24))
    inner = FunctionalOp(lf_diff, (Var("x"),))
    middle = FunctionalOp(lf_ema, (inner,))
    outer = UnOp("tanh", middle)
    y = evaluate(outer, env, fill_warmup=0.0)
    # Expected: tanh of ema applied to diff-with-zero-warmup
    diffed = measure_diff(1).apply(env["x"], fill_warmup=0.0)
    expected = np.tanh(measure_ema(24).apply(diffed))
    assert np.allclose(y, expected, atol=1e-10, equal_nan=True)


# ---------------- Cache integration ----------------

def test_eval_with_cache_hits_on_repeat():
    """Same FunctionalOp evaluated twice → second call is a cache hit."""
    env = _rng_env(500, 6)
    cache = FunctionalCache(mem_size=16)
    lf = LinearFunctional(measure=measure_ema(24))
    t = FunctionalOp(lf, (Var("x"),))
    _ = evaluate(t, env, cache)
    _ = evaluate(t, env, cache)
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 1

def test_eval_cache_shares_subexpressions_across_different_outer_trees():
    """ema(x, 24) computed inside two different outer trees → one cache slot."""
    env = _rng_env(500, 7)
    cache = FunctionalCache(mem_size=16)
    lf = LinearFunctional(measure=measure_ema(24))
    inner = FunctionalOp(lf, (Var("x"),))
    t1 = UnOp("tanh", inner)
    t2 = UnOp("abs", inner)
    evaluate(t1, env, cache)
    evaluate(t2, env, cache)
    # Same inner subtree → same var_id → 1 miss + 1 hit
    assert cache.stats["misses"] == 1
    assert cache.stats["mem_hits"] == 1

def test_eval_without_cache():
    env = _rng_env(200, 8)
    lf = LinearFunctional(measure=measure_ema(12))
    t = FunctionalOp(lf, (Var("x"),))
    # No cache provided
    y = evaluate(t, env, cache=None)
    expected = measure_ema(12).apply(env["x"])
    assert np.allclose(y, expected, equal_nan=True)


# ---------------- Error handling ----------------

def test_eval_unknown_var_raises():
    env = {"x": np.array([1.0, 2.0, 3.0])}
    t = Var("missing")
    with pytest.raises(KeyError):
        evaluate(t, env)
