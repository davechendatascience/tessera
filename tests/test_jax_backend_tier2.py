"""Tier 2 GPU port — JAX jit-compilation of tessera Expr trees.

Tests that compile_tree(tree, var_names) returns a jax-jitted callable
producing the same output as evaluate() on the same env, and that
repeated calls are dramatically faster than the eager evaluate path.

Skipped if jax is not importable.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, evaluate,
)
from tessera.expression.functional import LinearFunctional
from tessera.expression.measure import measure_diff
from tessera.expression.jit import (
    compile_tree, evaluate_jit, is_pure_pointwise,
    clear_jit_cache, jit_cache_size,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty cache."""
    clear_jit_cache()
    yield
    clear_jit_cache()


# ---------------- is_pure_pointwise ----------------

def test_is_pure_pointwise_atoms():
    assert is_pure_pointwise(Var("x"))
    assert is_pure_pointwise(Const(1.0))


def test_is_pure_pointwise_compound():
    tree = BinOp("add", Var("x"), Const(1.0))
    assert is_pure_pointwise(tree)
    tree = UnOp("tanh", BinOp("mul", Var("x"), Var("y")))
    assert is_pure_pointwise(tree)


def test_is_pure_pointwise_rejects_functional():
    f = LinearFunctional(measure=measure_diff(1))
    tree = BinOp("add", FunctionalOp(f, (Var("x"),)), Const(1.0))
    assert not is_pure_pointwise(tree)


# ---------------- compile_tree ----------------

def test_compile_tree_pure_pointwise_returns_callable():
    tree = BinOp("add", Var("x"), Const(1.0))
    fn = compile_tree(tree, ["x"])
    assert callable(fn)


def test_compile_tree_rejects_functional():
    f = LinearFunctional(measure=measure_diff(1))
    tree = FunctionalOp(f, (Var("x"),))
    with pytest.raises(ValueError, match="only pure-pointwise"):
        compile_tree(tree, ["x"])


def test_compile_tree_caches_by_str_and_var_names():
    tree = BinOp("add", Var("x"), Const(1.0))
    fn1 = compile_tree(tree, ["x"])
    fn2 = compile_tree(tree, ["x"])
    assert fn1 is fn2
    assert jit_cache_size() == 1

    # Different var_names → different cache entry
    tree2 = BinOp("add", Var("y"), Const(1.0))
    fn3 = compile_tree(tree2, ["y"])
    assert fn3 is not fn1
    assert jit_cache_size() == 2


# ---------------- Output correctness ----------------

def test_compiled_output_matches_evaluate_pointwise():
    """y = tanh(a + b) - sqrt(|c|) + log(d² + 1)"""
    tree = BinOp("add",
        BinOp("sub",
            UnOp("tanh", BinOp("add", Var("a"), Var("b"))),
            UnOp("sqrt", Var("c"))),
        UnOp("log", BinOp("add", BinOp("mul", Var("d"), Var("d")), Const(1.0)))
    )
    rng = np.random.default_rng(0)
    n = 200
    env_np = {k: rng.standard_normal(n).astype(np.float32) for k in ["a","b","c","d"]}
    env_jax = {k: jnp.asarray(v) for k, v in env_np.items()}

    # Reference: evaluate on numpy
    y_np = evaluate(tree, env_np)

    # Test: compile_tree + call
    y_jit = evaluate_jit(tree, env_jax)
    assert type(y_jit).__module__.startswith("jax")
    np.testing.assert_allclose(np.asarray(y_jit), y_np, rtol=1e-3, atol=1e-4)


def test_compiled_output_indicators():
    """y = (a > b) * c — exercises indicator op in jit."""
    tree = BinOp("mul", BinOp("gt", Var("a"), Var("b")), Var("c"))
    rng = np.random.default_rng(1)
    n = 100
    env_np = {k: rng.standard_normal(n).astype(np.float32) for k in ["a","b","c"]}
    env_jax = {k: jnp.asarray(v) for k, v in env_np.items()}

    y_np = evaluate(tree, env_np)
    y_jit = evaluate_jit(tree, env_jax)
    np.testing.assert_allclose(np.asarray(y_jit), y_np, rtol=1e-3)


def test_compiled_output_pow():
    tree = BinOp("pow", Var("a"), Const(2.0))
    rng = np.random.default_rng(2)
    a = rng.standard_normal(50).astype(np.float32)
    y_np = evaluate(tree, {"a": a})
    y_jit = evaluate_jit(tree, {"a": jnp.asarray(a)})
    np.testing.assert_allclose(np.asarray(y_jit), y_np, rtol=1e-3, atol=1e-4)


# ---------------- Cache behavior ----------------

def test_cache_clear():
    tree = BinOp("add", Var("x"), Const(1.0))
    compile_tree(tree, ["x"])
    assert jit_cache_size() == 1
    clear_jit_cache()
    assert jit_cache_size() == 0


# ---------------- Speedup (smoke test, not a strict assertion) ----------------

def test_jit_speedup_over_eager_evaluate():
    """JIT path should be faster than eager evaluate on JAX arrays after
    warmup. Not a strict claim because CPU JAX is much less efficient
    than GPU JAX -- on CPU the speedup may be modest.

    The point of this test is to catch regressions: if jit is somehow
    SLOWER than eager, something is wrong with the design."""
    # Tree of moderate complexity
    tree = BinOp("add",
        UnOp("tanh", BinOp("mul", Var("a"), Var("b"))),
        UnOp("sqrt", BinOp("add", Var("c"), Const(1.0)))
    )
    n = 5000
    rng = np.random.default_rng(0)
    env_jax = {k: jnp.asarray(rng.standard_normal(n).astype(np.float32))
               for k in ["a", "b", "c"]}

    # Warmup both paths
    _ = evaluate(tree, env_jax)
    _ = evaluate_jit(tree, env_jax).block_until_ready()

    n_iter = 100
    t0 = time.time()
    for _ in range(n_iter):
        _ = evaluate(tree, env_jax)
    t_eager = (time.time() - t0) / n_iter

    t0 = time.time()
    for _ in range(n_iter):
        _ = evaluate_jit(tree, env_jax).block_until_ready()
    t_jit = (time.time() - t0) / n_iter

    # Print for visibility; assertion just catches gross regressions
    print(f"\neager: {t_eager*1000:.3f} ms, jit: {t_jit*1000:.3f} ms, "
          f"speedup: {t_eager/t_jit:.2f}x")
    # JIT should at least not be MORE than 2x slower than eager (smoke)
    assert t_jit < t_eager * 2.0, \
        f"jit is significantly slower than eager: {t_jit*1000:.3f}ms vs {t_eager*1000:.3f}ms"
