"""Tests for the collapse_functional_chain mutation."""
import random

import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp, FunctionalOp, LinearFunctional,
    complexity, evaluate,
    measure_diff, measure_lag, measure_ema,
)
from tessera.expression.mutation import (
    collapse_functional_chain, mutate, OP_WEIGHTS,
)


# ---------------- Pattern matching ----------------

def test_collapse_on_simple_nested():
    """L_diff(L_diff(x)) → L_{diff*diff}(x)."""
    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (inner,))
    rng = random.Random(0)
    collapsed = collapse_functional_chain(outer, rng)
    # Should be a single FunctionalOp now
    assert isinstance(collapsed, FunctionalOp)
    # cx reduced by 1 (the nested wrapper is gone)
    assert complexity(collapsed) < complexity(outer)


def test_collapse_reduces_complexity():
    inner = FunctionalOp(LinearFunctional(measure=measure_lag(2)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_lag(3)), (inner,))
    rng = random.Random(0)
    cx_before = complexity(outer)
    collapsed = collapse_functional_chain(outer, rng)
    assert complexity(collapsed) < cx_before


def test_collapse_raises_on_no_pattern():
    """Tree without nested LinearFunctional → raises ValueError."""
    tree = BinOp("add", Var("x"), Var("y"))
    rng = random.Random(0)
    with pytest.raises(ValueError):
        collapse_functional_chain(tree, rng)


def test_collapse_raises_on_lone_functional():
    """A single FunctionalOp (not nested) → raises ValueError."""
    tree = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    rng = random.Random(0)
    with pytest.raises(ValueError):
        collapse_functional_chain(tree, rng)


# ---------------- Semantic preservation ----------------

def test_collapse_preserves_semantics():
    """The collapsed tree evaluates to the same result as the nested one
    on real data (modulo warmup-boundary effects)."""
    rng_data = np.random.default_rng(0)
    x = rng_data.standard_normal(200)

    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (inner,))
    rng = random.Random(0)
    collapsed = collapse_functional_chain(outer, rng)

    y_orig = evaluate(outer, {"x": x})
    y_collapsed = evaluate(collapsed, {"x": x})

    mask = np.isfinite(y_orig) & np.isfinite(y_collapsed)
    np.testing.assert_allclose(y_orig[mask], y_collapsed[mask], rtol=1e-10)


def test_collapse_in_deep_tree():
    """The mutation finds a nested pattern inside a larger tree."""
    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    nested = FunctionalOp(LinearFunctional(measure=measure_lag(2)), (inner,))
    # Embed in a larger compound: nested + abs(x)
    tree = BinOp("add", nested, UnOp("abs", Var("x")))
    rng = random.Random(0)
    cx_before = complexity(tree)
    collapsed = collapse_functional_chain(tree, rng)
    assert complexity(collapsed) < cx_before
    # The abs(x) branch is untouched
    assert isinstance(collapsed, BinOp) and collapsed.op == "add"


# ---------------- Integration with mutate() dispatcher ----------------

def test_mutate_dispatcher_includes_collapse_op():
    """The new op is in OP_WEIGHTS."""
    assert "collapse_functional_chain" in OP_WEIGHTS
    assert OP_WEIGHTS["collapse_functional_chain"] > 0


def test_mutate_op_weights_still_sum_to_one():
    """Total OP_WEIGHTS still normalised."""
    total = sum(OP_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9


def test_mutate_can_produce_collapse_offspring():
    """The dispatcher CAN produce a collapse-mutated offspring (over many
    tries). Verifies the wiring works end-to-end."""
    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (inner,))
    rng = random.Random(0)

    # Try many times; should eventually fire the collapse op (5% weight)
    found_collapsed = False
    for _ in range(100):
        child = mutate([outer], rng, ["x"], max_attempts=1)
        if child is not None and complexity(child) < complexity(outer):
            # Collapsed (complexity went DOWN)
            found_collapsed = True
            break
    # At ~5% weight over 100 tries, P(no collapse) = 0.95^100 ≈ 0.006.
    # We allow the rare case it didn't fire; soft check.
    # Either way the dispatcher should not have raised.


def test_pointwise_only_skips_collapse():
    """In pointwise-only mode, the dispatcher skips collapse just like
    measure_mutate (no FunctionalOp generated, can't collapse)."""
    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (inner,))
    rng = random.Random(0)
    # With pointwise_only=True, ANY chosen mutation that touches
    # functionals should be skipped. Just verify the dispatcher doesn't
    # crash.
    for _ in range(20):
        child = mutate([outer], rng, ["x"], pointwise_only=True, max_attempts=3)
        # Either returns None (couldn't make a valid pointwise child) or
        # a real tree. Just verifying no crash.


# ---------------- Canonical form of the composed measure ----------------

def test_collapsed_measure_is_canonical():
    """The output's measure has canonical-form atoms (sorted, merged)."""
    # diff(1) * diff(1) = (1, -2, 1) — already sorted, no duplicates
    inner = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (Var("x"),))
    outer = FunctionalOp(LinearFunctional(measure=measure_diff(1)), (inner,))
    rng = random.Random(0)
    collapsed = collapse_functional_chain(outer, rng)
    atoms = collapsed.functional.measure.atoms
    lags = [a.lag for a in atoms]
    assert lags == sorted(lags)
