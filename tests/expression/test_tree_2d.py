"""Tests for FunctionalOp2D in the tree evaluator + 2D-mode GP."""
from __future__ import annotations

import random

import numpy as np
import pytest

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp2D, evaluate,
    complexity, depth, iter_subtrees, used_features, replace_at,
)
from tessera.expression.measure_2d import (
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
    measure_2d_separable,
)
from tessera.expression.measure import measure_ema, measure_lag
from tessera.expression.mutation import (
    random_measure_2d, random_tree, measure_2d_mutate, validate_tree,
)
from tessera.expression.gp import GP, GPConfig


def _heat_eq_trajectory(T=60, X=32, alpha=0.05, seed=0):
    """Simulate a small heat-equation trajectory for tests."""
    rng = np.random.default_rng(seed)
    U = np.zeros((T, X), dtype=np.float64)
    # Initial condition: a Gaussian bump
    xs = np.arange(X) - X / 2
    U[0] = np.exp(-(xs ** 2) / (2 * 4.0 ** 2))
    for t in range(1, T):
        prev = U[t-1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        U[t] = prev + alpha * lap + 0.001 * rng.standard_normal(X)
    return U


# ---------------- FunctionalOp2D construction + structure ----------------

def test_functionalop2d_basic_attributes():
    m = measure_2d_laplacian_5pt()
    f = FunctionalOp2D(m, Var("U"))
    assert f.measure_2d is m
    assert f.arg == Var("U")

def test_functionalop2d_appears_in_iter_subtrees():
    m = measure_2d_laplacian_5pt()
    t = BinOp("add", FunctionalOp2D(m, Var("U")), Const(0.5))
    subs = list(iter_subtrees(t))
    assert any(isinstance(s, FunctionalOp2D) for s in subs)

def test_functionalop2d_complexity_counts_correctly():
    t = FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))
    assert complexity(t) == 2   # functional + var
    t2 = UnOp("tanh", t)
    assert complexity(t2) == 3
    assert depth(t2) == 3

def test_functionalop2d_used_features_finds_inner_var():
    t = FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))
    assert used_features(t) == {"U"}

def test_functionalop2d_replace_at_works():
    """Replace the inner Var with a different one."""
    t = FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))
    # Pre-order indices: 0=func, 1=var; replace var with const
    t2 = replace_at(t, 1, Const(1.0))
    assert isinstance(t2, FunctionalOp2D)
    assert t2.arg == Const(1.0)


# ---------------- evaluate() on 2D fields ----------------

def test_evaluate_functionalop2d_on_heat_field():
    """Tree: laplacian_5pt(U). Compare against measure_2d.apply directly."""
    U = _heat_eq_trajectory()
    env = {"U": U}
    tree = FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))
    y_tree = evaluate(tree, env, fill_warmup=0.0)
    y_direct = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    assert np.allclose(y_tree, y_direct, atol=1e-12)

def test_evaluate_binop_of_2d_fields():
    """U + 0.5*∇²U pointwise. Mix of FunctionalOp2D and BinOp."""
    U = _heat_eq_trajectory()
    env = {"U": U}
    tree = BinOp("add",
                 Var("U"),
                 BinOp("mul", Const(0.5),
                       FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))))
    y = evaluate(tree, env, fill_warmup=0.0)
    expected = U + 0.5 * measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    assert np.allclose(y, expected, atol=1e-12)

def test_evaluate_separable_2d_density_via_tree():
    """ema_t × roll_mean_x as a separable Measure2D, applied in a tree."""
    from tessera.expression.measure import measure_roll_mean
    U = _heat_eq_trajectory()
    env = {"U": U}
    m2d = measure_2d_separable(measure_ema(8), measure_roll_mean(3))
    tree = FunctionalOp2D(m2d, Var("U"))
    y = evaluate(tree, env, fill_warmup=0.0)
    y_direct = m2d.apply(U, fill_warmup=0.0)
    assert np.allclose(y, y_direct, atol=1e-12)


# ---------------- Mutation ----------------

def test_random_tree_2d_mode_produces_funcop2d():
    """With enable_2d=True the random_tree path can yield FunctionalOp2D."""
    rng = random.Random(0)
    has_2d = False
    for _ in range(30):
        t = random_tree(rng, ["U"], max_depth=4, enable_2d=True)
        if any(isinstance(s, FunctionalOp2D) for s in iter_subtrees(t)):
            has_2d = True
            break
    assert has_2d, "enable_2d=True should produce trees with FunctionalOp2D"

def test_random_tree_1d_mode_never_produces_funcop2d():
    """Default enable_2d=False: trees should never contain FunctionalOp2D."""
    rng = random.Random(1)
    for _ in range(30):
        t = random_tree(rng, ["x"], max_depth=4, enable_2d=False)
        assert not any(isinstance(s, FunctionalOp2D) for s in iter_subtrees(t))

def test_measure_2d_mutate_changes_measure():
    rng = random.Random(2)
    t = FunctionalOp2D(measure_2d_laplacian_5pt(), Var("U"))
    different = 0
    for _ in range(20):
        t2 = measure_2d_mutate(t, rng)
        if t2 != t:
            different += 1
    assert different > 10


# ---------------- GP on a PDE-discovery toy problem ----------------

def test_gp_with_enable_2d_runs_and_finds_low_loss():
    """Toy: target = ∇²(U) where U is a heat-equation trajectory.
    Tessera's GP with enable_2d should converge to low train loss quickly."""
    U = _heat_eq_trajectory(T=80, X=24, alpha=0.05, seed=7)
    target = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)

    # Slightly higher budget than the original pop=40/gens=15 to make the
    # test more robust to seed-dependent stochasticity (the
    # FunctionalOp(Const) and FunctionalOp2D(Const) const-folds added in
    # 2026-05-24 change the mutation trajectory just enough that the
    # original budget sometimes lands above the 50%-of-var threshold).
    cfg = GPConfig(
        pop_size=60, n_gens=25, parsimony=0.005, verbose=False,
        seed=42, enable_2d=True, fill_warmup=0.0,
    )
    gp = GP(cfg)
    front = gp.run({"U": U}, target, ["U"])
    assert len(front) >= 1
    best = min(c.train_loss for c in front)
    # Target IS reachable exactly (∇² is in the alphabet). Expect very low
    # but not 0 in 25 gens because the GP has to find it.
    target_var = float(np.var(target))
    # A loss < 50% of target variance is "GP is doing useful work"
    assert best < 0.5 * target_var, (
        f"GP failed to fit ∇²(U); best loss={best:.4g}, var(target)={target_var:.4g}"
    )
