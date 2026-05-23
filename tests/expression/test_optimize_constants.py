"""Tests for the constant-optimisation polish step."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp,
    collect_const_values, set_const_values,
    optimize_constants, mse_loss, FunctionalCache,
    GP, GPConfig,
)


# ---------------- Const helpers (tree.py) ----------------

def test_collect_const_values_preorder():
    # (3 + (x * -1.5)) — pre-order constants: 3, -1.5
    tree = BinOp("add", Const(3.0), BinOp("mul", Var("x"), Const(-1.5)))
    assert collect_const_values(tree) == [3.0, -1.5]


def test_collect_no_consts():
    assert collect_const_values(Var("x")) == []
    assert collect_const_values(UnOp("abs", Var("x"))) == []


def test_set_const_values_replaces_in_preorder():
    tree = BinOp("add", Const(3.0), BinOp("mul", Var("x"), Const(-1.5)))
    new = set_const_values(tree, [10.0, 99.0])
    assert collect_const_values(new) == [10.0, 99.0]


def test_set_const_values_fewer_values_keeps_extras():
    tree = BinOp("add", Const(3.0), BinOp("mul", Const(2.0), Const(1.0)))
    new = set_const_values(tree, [99.0])
    assert collect_const_values(new) == [99.0, 2.0, 1.0]


def test_set_const_values_preserves_structure():
    """Replacing constants preserves the tree skeleton (Vars, Ops, shape)."""
    tree = BinOp("add", Const(3.0), BinOp("mul", Var("x"), Const(-1.5)))
    new = set_const_values(tree, [7.0, 2.0])
    # Var stays at the same path; ops stay the same
    assert isinstance(new, BinOp) and new.op == "add"
    assert isinstance(new.b, BinOp) and new.b.op == "mul"
    assert new.b.a == Var("x")


# ---------------- optimize_constants ----------------

def test_optimize_constants_finds_true_linear_coefficients():
    """y = 2*x + 0.5; start at 0.1*x + 0; optimizer must reach near-truth."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 2.0 * x + 0.5
    tree = BinOp("add",
                 BinOp("mul", Const(0.1), Var("x")),
                 Const(0.0))
    cache = FunctionalCache(mem_size=100)
    new_tree, loss = optimize_constants(
        tree, {"x": x}, y, mse_loss, cache,
        method="Nelder-Mead", maxiter=200,
    )
    a, b = collect_const_values(new_tree)
    assert abs(a - 2.0) < 1e-3, f"slope not recovered: got {a}"
    assert abs(b - 0.5) < 1e-3, f"intercept not recovered: got {b}"
    assert loss < 1e-9


def test_optimize_constants_no_consts_returns_tree_unchanged():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = x ** 2
    tree = Var("x")
    cache = FunctionalCache(mem_size=100)
    new_tree, loss = optimize_constants(
        tree, {"x": x}, y, mse_loss, cache,
    )
    assert new_tree == tree
    assert np.isfinite(loss)


def test_optimize_constants_handles_broken_tree():
    """If the initial constants give an invalid prediction (mostly NaN),
    the polish step gives up and returns inf — doesn't crash."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    y = x
    # Tree: const_0 + Const(0.0), with min_valid_frac at default 0.9.
    # We force an invalid path by feeding an NaN-heavy input.
    cache = FunctionalCache(mem_size=10)
    x_nan = x.copy(); x_nan[:45] = np.nan
    tree = BinOp("add", Const(0.0), Var("x"))
    new_tree, loss = optimize_constants(
        tree, {"x": x_nan}, y, mse_loss, cache, min_valid_frac=0.9,
    )
    assert loss == float("inf")


def test_optimize_constants_doesnt_make_things_worse():
    """If optimisation can't improve, return the original tree's loss
    (not a worse one)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = np.sin(x)   # nonlinear; (a*x + b) can't fit it perfectly
    tree = BinOp("add",
                 BinOp("mul", Const(0.7), Var("x")),
                 Const(0.0))
    cache = FunctionalCache(mem_size=100)
    initial_loss = mse_loss(
        np.asarray(BinOp("add",
                          BinOp("mul", Const(0.7), Var("x")),
                          Const(0.0)).a.a.value) * x + 0.0,
        y,
    )
    new_tree, loss = optimize_constants(
        tree, {"x": x}, y, mse_loss, cache, maxiter=20,
    )
    # Either improved or stayed at original; never worse
    assert loss <= initial_loss + 1e-9


# ---------------- GP integration ----------------

def test_gp_with_const_optimization_beats_without():
    """Same GP run with optimize_constants_every=2 should beat one with
    optimize_constants_every=0 on a problem where constants matter."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    # Target: 2.371 * tanh(0.473 * x) + 0.183
    # SR can find the structure tanh(c1*x) easily; the value of c1 + outer
    # scale/offset is where const optimisation pays off.
    y = 2.371 * np.tanh(0.473 * x) + 0.183

    cfg_off = GPConfig(pop_size=40, n_gens=8, verbose=False, seed=1,
                       optimize_constants_every=0)
    cfg_on = GPConfig(pop_size=40, n_gens=8, verbose=False, seed=1,
                      optimize_constants_every=2, optimize_constants_maxiter=30)

    front_off = GP(cfg_off).run({"x": x}, y, ["x"])
    front_on = GP(cfg_on).run({"x": x}, y, ["x"])

    best_off = min(c.train_loss for c in front_off)
    best_on = min(c.train_loss for c in front_on)
    assert best_on <= best_off, (
        f"const-opt should not regress; got off={best_off:.4g} on={best_on:.4g}"
    )
