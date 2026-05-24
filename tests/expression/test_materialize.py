"""Tests for tessera.expression.materialize — cross-tree subexpression
materialization.

Covers:
- Subtrees shared by >= threshold trees get pre-evaluated
- Rewritten trees reference synthetic Vars instead of the original subtree
- Augmented env contains the materialized arrays
- Output of evaluate(rewritten_tree, augmented_env) matches
  evaluate(original_tree, env)
- Extension points: is_cacheable, canonical_key, persistent cache
- Edge cases: empty population, no shareable subtrees, threshold not met
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D, evaluate,
)
from tessera.expression.measure import measure_diff, measure_ema
from tessera.expression.measure_2d import (
    measure_2d_laplacian_5pt, measure_2d_grad_x,
)
from tessera.expression.functional import LinearFunctional
from tessera.expression.materialize import (
    materialize_shared_subtrees, default_is_cacheable,
)


# ---------------- Basic behaviour ----------------

def test_no_shared_subtrees_returns_population_unchanged():
    """If no subtree is shared, no materialisation happens."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    f_ema = LinearFunctional(measure=measure_ema(halflife=4, support_max=20))
    trees = [
        FunctionalOp(f_diff, (Var("x"),)),
        FunctionalOp(f_ema, (Var("x"),)),
    ]
    env = {"x": np.arange(50.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees(trees, env)
    assert stats["n_materialized"] == 0
    assert stats["n_replacements"] == 0
    assert rewritten == list(trees)
    assert set(aug_env) == set(env)


def test_threshold_2_default_caches_shared_functional():
    """A FunctionalOp appearing in 2 trees gets materialised once."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    diff_sub = FunctionalOp(f_diff, (Var("x"),))
    trees = [
        BinOp("add", diff_sub, Const(1.0)),
        BinOp("mul", diff_sub, Const(2.0)),
    ]
    env = {"x": np.arange(20.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees(trees, env)
    assert stats["n_materialized"] == 1
    assert stats["n_replacements"] == 2
    # Augmented env should have one synthetic var
    new_vars = set(aug_env) - set(env)
    assert len(new_vars) == 1


def test_rewritten_trees_evaluate_to_same_result():
    """Critical correctness check: pre-evaluation + rewrite preserves
    the original computation."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    diff_sub = FunctionalOp(f_diff, (Var("x"),))
    trees = [
        BinOp("add", diff_sub, Var("x")),
        BinOp("sub", Var("x"), diff_sub),
    ]
    rng = np.random.default_rng(0)
    env = {"x": rng.standard_normal(50)}

    # Evaluate originals
    y_orig = [evaluate(t, env, fill_warmup=0.0) for t in trees]

    # Materialise + evaluate rewritten on augmented env
    rewritten, aug_env, _ = materialize_shared_subtrees(trees, env)
    y_new = [evaluate(t, aug_env, fill_warmup=0.0) for t in rewritten]

    for a, b in zip(y_orig, y_new):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_threshold_not_met_skips_materialisation():
    """Threshold=3 on a subtree appearing in 2 trees -> no caching."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    diff_sub = FunctionalOp(f_diff, (Var("x"),))
    trees = [
        BinOp("add", diff_sub, Const(1.0)),
        BinOp("mul", diff_sub, Const(2.0)),
    ]
    env = {"x": np.arange(20.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees(
        trees, env, threshold=3
    )
    assert stats["n_materialized"] == 0
    assert rewritten == list(trees)


def test_count_distinct_trees_not_occurrences_within_one_tree():
    """`f(x) + f(x)` in ONE tree counts f(x) once, not twice."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    diff_sub = FunctionalOp(f_diff, (Var("x"),))
    trees = [
        BinOp("add", diff_sub, diff_sub),  # f(x) appears twice IN THIS tree
    ]
    env = {"x": np.arange(20.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees(
        trees, env, threshold=2
    )
    # Only ONE distinct tree has f(x), so count = 1, threshold=2 not met
    assert stats["n_materialized"] == 0


# ---------------- 2D / image-like ----------------

def test_image_laplacian_shared_across_population():
    """MNIST-shape scenario: Laplacian(image) appears in many trees."""
    m = measure_2d_laplacian_5pt()
    laplacian = FunctionalOp2D(m, Var("image"))
    trees = [
        UnOp("abs", laplacian),
        BinOp("add", laplacian, Const(0.5)),
        UnOp("tanh", laplacian),
        # Some trees use a different op (won't be shared)
        FunctionalOp2D(measure_2d_grad_x(), Var("image")),
    ]
    rng = np.random.default_rng(0)
    env = {"image": rng.standard_normal((10, 10))}
    rewritten, aug_env, stats = materialize_shared_subtrees(trees, env)
    assert stats["n_materialized"] == 1   # only Laplacian shared >=2
    assert stats["n_replacements"] == 3   # in 3 trees
    # Correctness
    for orig, new in zip(trees, rewritten):
        y_orig = evaluate(orig, env, fill_warmup=0.0)
        y_new = evaluate(new, aug_env, fill_warmup=0.0)
        np.testing.assert_allclose(y_orig, y_new, rtol=1e-8)


# ---------------- Extension points ----------------

def test_custom_is_cacheable_includes_pointwise():
    """User can extend cacheability to compound pointwise expressions."""
    sub = BinOp("mul", Var("x"), Var("x"))   # x*x — expensive compound
    trees = [
        BinOp("add", sub, Const(1.0)),
        BinOp("sub", sub, Var("x")),
    ]
    env = {"x": np.arange(20.0)}

    # Default: no caching (BinOp not cacheable)
    _, _, stats_default = materialize_shared_subtrees(trees, env)
    assert stats_default["n_materialized"] == 0

    # Custom predicate: BinOp("mul", ...) is cacheable
    def is_mul_cacheable(node):
        return isinstance(node, BinOp) and node.op == "mul"

    rewritten, aug_env, stats = materialize_shared_subtrees(
        trees, env, is_cacheable=is_mul_cacheable,
    )
    assert stats["n_materialized"] == 1
    # Correctness on rewrites
    for orig, new in zip(trees, rewritten):
        np.testing.assert_allclose(
            evaluate(orig, env), evaluate(new, aug_env), rtol=1e-10
        )


def test_persistent_cache_reused_across_calls():
    """An external cache dict accumulates across calls."""
    f_diff = LinearFunctional(measure=measure_diff(1))
    diff_sub = FunctionalOp(f_diff, (Var("x"),))
    trees = [
        BinOp("add", diff_sub, Const(1.0)),
        BinOp("mul", diff_sub, Const(2.0)),
    ]
    env = {"x": np.arange(20.0)}
    cache: dict = {}

    # First call populates cache
    _, _, stats1 = materialize_shared_subtrees(trees, env, cache=cache)
    assert stats1["cache_hits"] == 0
    assert len(cache) == 1

    # Second call reuses cache
    _, _, stats2 = materialize_shared_subtrees(trees, env, cache=cache)
    assert stats2["cache_hits"] == 1
    assert stats2["n_materialized"] == 1   # still 1 materialised this call


def test_custom_canonical_key():
    """User can supply their own identity function (e.g., for AC-norm
    matching that defaults wouldn't catch)."""
    # Build two trees with equivalent expressions in different orderings
    f_diff = LinearFunctional(measure=measure_diff(1))
    f_a = FunctionalOp(f_diff, (Var("x"),))

    trees = [
        BinOp("add", f_a, Const(1.0)),
        BinOp("add", Const(1.0), f_a),   # same expression, different order
    ]
    env = {"x": np.arange(20.0)}

    # Default str: the OUTER BinOps differ, but the inner f_a subtree
    # appears in both -> still gets cached.
    _, _, stats = materialize_shared_subtrees(trees, env)
    assert stats["n_materialized"] == 1


# ---------------- Edge cases ----------------

def test_empty_population():
    env = {"x": np.arange(20.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees([], env)
    assert rewritten == []
    assert stats["n_materialized"] == 0


def test_single_tree_no_sharing_possible():
    f_diff = LinearFunctional(measure=measure_diff(1))
    tree = BinOp("add", FunctionalOp(f_diff, (Var("x"),)), Const(1.0))
    env = {"x": np.arange(20.0)}
    rewritten, aug_env, stats = materialize_shared_subtrees([tree], env)
    assert stats["n_materialized"] == 0   # threshold=2 default
    assert rewritten[0] == tree


def test_default_is_cacheable_excludes_pointwise_and_vars():
    assert default_is_cacheable(FunctionalOp(
        LinearFunctional(measure=measure_diff(1)), (Var("x"),)
    ))
    assert default_is_cacheable(FunctionalOp2D(
        measure_2d_laplacian_5pt(), Var("U")
    ))
    assert not default_is_cacheable(Var("x"))
    assert not default_is_cacheable(Const(1.0))
    assert not default_is_cacheable(BinOp("add", Var("x"), Var("y")))
    assert not default_is_cacheable(UnOp("tanh", Var("x")))
