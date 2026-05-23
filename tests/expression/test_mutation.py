"""Tests for random tree generation and mutation operators."""
from __future__ import annotations

import random

import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp,
    complexity, depth, used_features, iter_subtrees,
)
from tessera.expression.measure import (
    Measure, measure_ema, measure_diff, measure_lag,
)
from tessera.expression.functional import (
    LinearFunctional, SeparableBilinear, Volterra2,
)
from tessera.expression.mutation import (
    MAX_DEPTH, MAX_COMPLEXITY, validate_tree,
    random_measure, random_functional, random_tree,
    subtree_swap, subtree_crossover, constant_jitter,
    term_insert, term_delete, op_swap, measure_mutate,
    OP_WEIGHTS, mutate,
)


FEATURES = ["temp", "pressure", "humidity", "wind", "dew_point"]


# ---------------- Validators ----------------

def test_validate_accepts_simple_tree():
    t = BinOp("add", Var("temp"), Const(0.5))
    assert validate_tree(t, set(FEATURES)) is None

def test_validate_rejects_overdepth():
    t = Var("temp")
    for _ in range(MAX_DEPTH):
        t = BinOp("add", t, Const(1.0))
    err = validate_tree(t, set(FEATURES))
    assert err is not None and "depth" in err

def test_validate_rejects_unknown_feature():
    t = Var("not_a_feature")
    err = validate_tree(t, set(FEATURES))
    assert err is not None and "unknown features" in err


# ---------------- random_measure ----------------

def test_random_measure_returns_valid_measure():
    rng = random.Random(0)
    for _ in range(30):
        m = random_measure(rng)
        assert isinstance(m, Measure)
        assert m.is_absolutely_summable()

def test_random_measure_covers_multiple_families():
    """Over many samples, multiple density families + atomic forms should appear."""
    rng = random.Random(1)
    families = set()
    has_atomic_only = False
    has_signed_sum = False
    for _ in range(200):
        m = random_measure(rng)
        if m.has_density:
            families.add(m.density_family)
        if m.has_atomic and not m.has_density:
            has_atomic_only = True
            if len(m.atoms) >= 2:
                has_signed_sum = True
    assert "exponential" in families
    assert "rectangular" in families
    assert has_atomic_only
    assert has_signed_sum


# ---------------- random_functional ----------------

def test_random_functional_returns_valid_n_inputs():
    rng = random.Random(2)
    for _ in range(50):
        fn = random_functional(rng)
        assert fn.n_inputs in (1, 2)


def test_random_functional_covers_all_three_kinds():
    rng = random.Random(3)
    seen = set()
    for _ in range(200):
        fn = random_functional(rng)
        seen.add(type(fn).__name__)
    assert seen == {"LinearFunctional", "SeparableBilinear", "Volterra2"}


# ---------------- random_tree ----------------

def test_random_tree_is_valid():
    rng = random.Random(4)
    for _ in range(30):
        t = random_tree(rng, FEATURES, max_depth=4)
        err = validate_tree(t, set(FEATURES))
        assert err is None, f"invalid tree: {err}"

def test_random_tree_uses_only_known_features():
    rng = random.Random(5)
    for _ in range(30):
        t = random_tree(rng, FEATURES, max_depth=4)
        assert used_features(t).issubset(set(FEATURES))

def test_random_tree_can_produce_functional_ops():
    rng = random.Random(6)
    has_funcop = False
    for _ in range(40):
        t = random_tree(rng, FEATURES, max_depth=4)
        if any(isinstance(s, FunctionalOp) for s in iter_subtrees(t)):
            has_funcop = True
            break
    assert has_funcop


# ---------------- Individual mutations ----------------

def _sample_seed_tree(rng=None):
    """Build a deterministic non-trivial seed tree to mutate."""
    return BinOp(
        "mul",
        FunctionalOp(
            LinearFunctional(measure=measure_ema(24)),
            (Var("temp"),),
        ),
        BinOp("sub", Var("pressure"), Const(1013.0)),
    )

def test_subtree_swap_returns_different_tree():
    rng = random.Random(7)
    parent = _sample_seed_tree()
    different = 0
    for _ in range(10):
        child = subtree_swap(parent, rng, FEATURES)
        if child != parent:
            different += 1
    # Should be different in most cases
    assert different >= 8

def test_subtree_crossover_can_mix_parents():
    rng = random.Random(8)
    a = _sample_seed_tree()
    b = BinOp("add", Var("humidity"), UnOp("tanh", Var("wind")))
    found = False
    for _ in range(30):
        child = subtree_crossover(a, b, rng)
        # If "humidity" or "wind" appears in child but wasn't in `a`,
        # crossover did its job
        if {"humidity", "wind"} & used_features(child):
            found = True
            break
    assert found

def test_constant_jitter_changes_constants_not_structure():
    rng = random.Random(9)
    parent = _sample_seed_tree()
    child = constant_jitter(parent, rng)
    assert complexity(parent) == complexity(child)
    # Same Var leaves
    assert used_features(parent) == used_features(child)
    # But at least one constant should differ
    p_consts = [n.value for n in iter_subtrees(parent) if isinstance(n, Const)]
    c_consts = [n.value for n in iter_subtrees(child) if isinstance(n, Const)]
    assert len(p_consts) == len(c_consts)
    assert any(abs(p - c) > 1e-10 for p, c in zip(p_consts, c_consts))

def test_term_insert_grows_complexity():
    rng = random.Random(10)
    parent = _sample_seed_tree()
    child = term_insert(parent, rng, FEATURES)
    assert complexity(child) > complexity(parent)

def test_term_delete_does_not_grow():
    rng = random.Random(11)
    parent = BinOp("add", Var("temp"), BinOp("sub", Var("pressure"), Const(1.0)))
    for _ in range(20):
        child = term_delete(parent, rng)
        assert complexity(child) <= complexity(parent)

def test_op_swap_preserves_complexity():
    rng = random.Random(12)
    parent = _sample_seed_tree()
    child = op_swap(parent, rng)
    assert complexity(parent) == complexity(child)

def test_measure_mutate_changes_a_functional():
    """Apply measure_mutate to a tree with a FunctionalOp; the measure
    inside that FunctionalOp should change."""
    rng = random.Random(13)
    parent = _sample_seed_tree()
    parent_measures = {
        s.functional for s in iter_subtrees(parent)
        if isinstance(s, FunctionalOp)
    }
    different = 0
    for _ in range(20):
        child = measure_mutate(parent, rng)
        child_measures = {
            s.functional for s in iter_subtrees(child)
            if isinstance(s, FunctionalOp)
        }
        if child_measures != parent_measures:
            different += 1
    # Most of the time mutate should produce a different functional
    assert different > 10


# ---------------- mutate() dispatcher ----------------

def test_mutate_returns_valid_offspring():
    rng = random.Random(14)
    a = _sample_seed_tree()
    b = BinOp("add", Var("humidity"), Const(0.5))
    n_valid = 0
    n_attempts = 100
    for _ in range(n_attempts):
        child = mutate([a, b], rng, FEATURES)
        if child is not None:
            assert validate_tree(child, set(FEATURES)) is None
            n_valid += 1
    assert n_valid > n_attempts * 0.7, f"only {n_valid}/{n_attempts} valid"

def test_mutate_deterministic_given_seed():
    a = _sample_seed_tree()
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    c1 = mutate([a], rng1, FEATURES)
    c2 = mutate([a], rng2, FEATURES)
    assert c1 == c2

def test_mutate_op_weights_sum_to_one():
    s = sum(OP_WEIGHTS.values())
    assert abs(s - 1.0) < 1e-9, f"OP_WEIGHTS sum = {s}"
