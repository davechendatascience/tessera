"""Tier 3 GPU port — batched-population vmap evaluation.

Tests that evaluate_population() returns the same per-tree outputs as
running evaluate_jit() in a loop, with topology clustering and vmap
under the hood.

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
from tessera.expression.jit import evaluate_jit, clear_jit_cache
from tessera.expression.batched import (
    topology_key, extract_constants, n_constants,
    compile_topology, evaluate_population,
    clear_topo_cache, topo_cache_size,
)


@pytest.fixture(autouse=True)
def _clean_caches():
    clear_jit_cache()
    clear_topo_cache()
    yield
    clear_jit_cache()
    clear_topo_cache()


# ---------------- topology_key ----------------

def test_topology_key_const_erasure():
    """Two trees that differ only in Const values share a topology."""
    t1 = BinOp("add", Var("x"), Const(1.0))
    t2 = BinOp("add", Var("x"), Const(5.0))
    assert topology_key(t1) == topology_key(t2)
    assert topology_key(t1) == "add(x,C)"


def test_topology_key_different_var_positions():
    """Swapping Var for Const changes the topology."""
    t1 = BinOp("add", Var("x"), Const(1.0))
    t2 = BinOp("add", Const(1.0), Var("x"))
    assert topology_key(t1) != topology_key(t2)


def test_topology_key_nested():
    t = UnOp("tanh", BinOp("mul", Var("x"), Const(2.0)))
    assert topology_key(t) == "tanh(mul(x,C))"


# ---------------- extract_constants ----------------

def test_extract_constants_preorder():
    """Constants appear in pre-order tree-walk order."""
    t = BinOp("add",
              BinOp("mul", Const(2.0), Var("x")),
              Const(5.0))
    assert extract_constants(t) == [2.0, 5.0]


def test_extract_constants_empty():
    t = BinOp("add", Var("x"), Var("y"))
    assert extract_constants(t) == []
    assert n_constants(t) == 0


# ---------------- compile_topology ----------------

def test_compile_topology_outputs_match_per_tree():
    """A topology with K different constant settings, batched, matches
    K separate jit-compiled evaluations."""
    template = BinOp("add", BinOp("mul", Var("x"), Const(0.0)), Const(0.0))

    # Three concrete trees with different constants
    t1 = BinOp("add", BinOp("mul", Var("x"), Const(2.0)), Const(5.0))
    t2 = BinOp("add", BinOp("mul", Var("x"), Const(3.0)), Const(-1.0))
    t3 = BinOp("add", BinOp("mul", Var("x"), Const(0.5)), Const(0.0))

    # Same topology
    assert topology_key(t1) == topology_key(t2) == topology_key(t3) == topology_key(template)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float32)
    env = {"x": jnp.asarray(x)}

    # Reference: each tree via evaluate_jit
    y1 = evaluate_jit(t1, env)
    y2 = evaluate_jit(t2, env)
    y3 = evaluate_jit(t3, env)

    # Batched: one vmap call
    consts_batch = jnp.asarray([extract_constants(t) for t in [t1, t2, t3]])
    fn = compile_topology(template, ["x"])
    y_batch = fn((jnp.asarray(x),), consts_batch)
    assert y_batch.shape == (3, 100)

    np.testing.assert_allclose(np.asarray(y_batch[0]), np.asarray(y1), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(y_batch[1]), np.asarray(y2), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(y_batch[2]), np.asarray(y3), rtol=1e-4)


def test_compile_topology_caches():
    template = BinOp("add", Var("x"), Const(1.0))
    fn1 = compile_topology(template, ["x"])
    # Different tree, same topology
    other = BinOp("add", Var("x"), Const(42.0))
    fn2 = compile_topology(other, ["x"])
    assert fn1 is fn2
    assert topo_cache_size() == 1


def test_compile_topology_rejects_functional():
    f = LinearFunctional(measure=measure_diff(1))
    t = FunctionalOp(f, (Var("x"),))
    with pytest.raises(ValueError, match="pure-pointwise"):
        compile_topology(t, ["x"])


# ---------------- evaluate_population ----------------

def test_evaluate_population_single_tree():
    tree = BinOp("add", Var("x"), Const(1.0))
    x = jnp.asarray(np.linspace(0, 1, 20).astype(np.float32))
    results = evaluate_population([tree], {"x": x})
    assert len(results) == 1
    expected = evaluate_jit(tree, {"x": x})
    np.testing.assert_allclose(np.asarray(results[0]), np.asarray(expected), rtol=1e-4)


def test_evaluate_population_multi_topology():
    """A mixed population: some trees share topology, some don't.
    All outputs must match the per-tree evaluate_jit baseline."""
    trees = [
        BinOp("add", Var("x"), Const(1.0)),         # topo A
        BinOp("add", Var("x"), Const(2.0)),         # topo A (same)
        BinOp("mul", Var("x"), Const(3.0)),         # topo B
        UnOp("tanh", Var("x")),                      # topo C (no consts)
        BinOp("add", Var("x"), Const(5.0)),         # topo A (same as 1,2)
    ]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(50).astype(np.float32))

    # Run via evaluate_population
    batch_results = evaluate_population(trees, {"x": x})

    # Reference via evaluate_jit per tree
    expected = [evaluate_jit(t, {"x": x}) for t in trees]

    assert len(batch_results) == len(trees)
    for i, (got, exp) in enumerate(zip(batch_results, expected)):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(exp), rtol=1e-4,
            err_msg=f"tree {i} mismatch")

    # Should have only 3 topologies cached (A, B, C)
    assert topo_cache_size() == 3


def test_evaluate_population_no_consts():
    """Trees with no Consts — topology has M=0."""
    trees = [
        UnOp("tanh", Var("x")),
        UnOp("tanh", Var("x")),    # identical tree
    ]
    x = jnp.asarray(np.linspace(-1, 1, 20).astype(np.float32))
    batch_results = evaluate_population(trees, {"x": x})

    expected = evaluate_jit(trees[0], {"x": x})
    for r in batch_results:
        np.testing.assert_allclose(np.asarray(r), np.asarray(expected), rtol=1e-4)


def test_evaluate_population_preserves_order():
    """Output order matches input tree order, even though internal
    grouping may reorder."""
    # Trees in shuffled topology order
    trees = [
        BinOp("mul", Var("x"), Const(1.0)),   # B
        BinOp("add", Var("x"), Const(1.0)),   # A
        BinOp("mul", Var("x"), Const(2.0)),   # B
        BinOp("add", Var("x"), Const(2.0)),   # A
        BinOp("mul", Var("x"), Const(3.0)),   # B
    ]
    x = jnp.asarray(np.arange(10, dtype=np.float32))
    batch_results = evaluate_population(trees, {"x": x})

    expected = [evaluate_jit(t, {"x": x}) for t in trees]
    for i, (got, exp) in enumerate(zip(batch_results, expected)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-4,
                                   err_msg=f"order mismatch at index {i}")


def test_evaluate_population_rejects_functional():
    f = LinearFunctional(measure=measure_diff(1))
    trees = [
        BinOp("add", Var("x"), Const(1.0)),
        FunctionalOp(f, (Var("x"),)),
    ]
    x = jnp.asarray(np.arange(10, dtype=np.float32))
    with pytest.raises(ValueError, match="has FunctionalOp"):
        evaluate_population(trees, {"x": x})


# ---------------- Speedup smoke test ----------------

def test_population_eval_correctness_realistic_pop():
    """Smoke test: batched eval on a realistic mixed population.

    NOTE on perf: vmap is a GPU optimization. On CPU JAX, vmap'd code
    typically runs SLOWER than the equivalent per-tree jit loop because
    CPU has no SIMD lanes to fill — the extra batch dimension is pure
    overhead. The win shows up on GPU where the whole batch becomes one
    kernel launch.

    This test asserts CORRECTNESS only (outputs match per-tree path).
    For the actual GPU speedup, see notebooks/tessera_jax_tier3.ipynb.
    """
    # Build a population of 20 trees, mostly sharing 2-3 topologies
    rng = np.random.default_rng(0)
    population = []
    for _ in range(20):
        # Three topologies, randomly chosen
        choice = rng.integers(0, 3)
        if choice == 0:
            t = BinOp("add",
                      BinOp("mul", Var("x"), Const(float(rng.standard_normal()))),
                      Const(float(rng.standard_normal())))
        elif choice == 1:
            t = UnOp("tanh", BinOp("mul", Var("x"), Const(float(rng.standard_normal()))))
        else:
            t = BinOp("sub", Var("x"), Const(float(rng.standard_normal())))
        population.append(t)

    n = 5000
    env = {"x": jnp.asarray(rng.standard_normal(n).astype(np.float32))}

    # Warmup both paths
    _ = [evaluate_jit(t, env).block_until_ready() for t in population]
    _ = [r.block_until_ready() for r in evaluate_population(population, env)]

    # Time per-tree jit loop
    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        outs = [evaluate_jit(t, env).block_until_ready() for t in population]
    t_per_tree = (time.time() - t0) / n_iter

    # Time batched population eval
    t0 = time.time()
    for _ in range(n_iter):
        outs = evaluate_population(population, env)
        for r in outs:
            r.block_until_ready()
    t_batched = (time.time() - t0) / n_iter

    print(f"\nper-tree jit loop: {t_per_tree*1000:.2f} ms / pop")
    print(f"batched (vmap):    {t_batched*1000:.2f} ms / pop")
    print(f"speedup: {t_per_tree / t_batched:.2f}x  (expect <1x on CPU; >1x on GPU)")

    # Correctness check
    outs_per_tree = [evaluate_jit(t, env) for t in population]
    outs_batched = evaluate_population(population, env)
    for i, (got, exp) in enumerate(zip(outs_batched, outs_per_tree)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-4,
                                   err_msg=f"correctness mismatch at index {i}")
