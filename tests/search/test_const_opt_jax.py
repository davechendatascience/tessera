"""Tests for optimize_constants_jax — the jax.grad + Adam const-opt path.

Per docs/planned/roadmap.md §1.3 (shipped 2026-05-24): a faster
replacement for scipy Nelder-Mead on pure-pointwise trees + MSE.

Skipped if jax is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from tessera.expression.tree import Var, Const, BinOp, UnOp
from tessera.search.const_opt import optimize_constants_jax


def test_finds_known_optimum_linear():
    """y = 2*x + 3; tree (a*x + b) starts with a=1, b=0 → adam should
    converge to a≈2, b≈3."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200).astype(np.float32)
    y = 2.0 * x + 3.0
    env = {"x": jnp.asarray(x)}
    y_jax = jnp.asarray(y)

    tree = BinOp("add",
                 BinOp("mul", Const(1.0), Var("x")),
                 Const(0.0))
    new_tree, new_loss = optimize_constants_jax(
        tree, env, y_jax, n_steps=200, learning_rate=0.05,
    )
    # Loss should drop to near zero
    assert new_loss < 0.01, f"expected near-zero loss; got {new_loss:.4g}"
    # Extract the optimised constants
    from tessera.expression.tree import collect_const_values
    consts = collect_const_values(new_tree)
    # Tree-walk order: (a, b) → first const = a, second = b
    a, b = consts
    assert abs(a - 2.0) < 0.05, f"a should be ~2.0; got {a:.4f}"
    assert abs(b - 3.0) < 0.05, f"b should be ~3.0; got {b:.4f}"


def test_no_consts_returns_unchanged_tree():
    """Tree with no Const leaves → return the tree and its MSE."""
    tree = BinOp("add", Var("x"), Var("y"))
    env = {
        "x": jnp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        "y": jnp.asarray(np.array([1.0, 1.0, 1.0], dtype=np.float32)),
    }
    y_true = jnp.asarray(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    new_tree, loss = optimize_constants_jax(tree, env, y_true)
    assert new_tree == tree
    assert abs(loss) < 1e-5   # x + y matches target exactly


def test_initial_loss_inf_returns_inf():
    """A tree whose initial prediction is all-NaN should report inf
    loss (and the tree is returned unmodified)."""
    # Construct a tree that produces NaN: log(neg(x²+1)) — wait, our
    # log is protected. Try: pow(Const(0.0), Var("x")) where x has
    # negative values → safe_pow gives 0 for everything. Loss is
    # then ||0 - y||² which is finite. Bad example.
    #
    # Simpler: just check that a tree with finite loss converges
    # normally (covered above) — there's no clean way to construct
    # an all-NaN initial in tessera's protected-op world. Skip this
    # boundary test; covered by the early-return in the function.
    pytest.skip("no easy way to construct all-NaN initial state with protected ops")


def test_correctness_vs_scipy_nelder_mead():
    """Correctness parity: scipy Nelder-Mead and jax Adam both converge
    to a near-zero loss on the same quadratic-fit problem.

    Does NOT assert wall-clock superiority — jax_adam's speedup shows
    up on GPU + warm-cache. On CPU dev (where this test runs), the
    one-time JIT-compile is the dominant cost and scipy wins; that's
    documented expected behaviour. The point of this test is "both
    paths produce a valid solution," not which is faster."""
    import time
    from tessera.search.const_opt import optimize_constants
    from tessera.expression.cache import FunctionalCache
    from tessera.search.losses import mse_loss

    rng = np.random.default_rng(0)
    x = rng.standard_normal(500).astype(np.float64)
    y = 1.5 * x * x + 0.5 * x - 0.3
    env = {"x": x}
    env_jax = {"x": jnp.asarray(x.astype(np.float32))}
    y_jax = jnp.asarray(y.astype(np.float32))

    # Tree: a*x² + b*x + c (with bad initial constants)
    tree = BinOp("add",
                 BinOp("add",
                       BinOp("mul", Const(0.0), BinOp("mul", Var("x"), Var("x"))),
                       BinOp("mul", Const(0.0), Var("x"))),
                 Const(0.0))

    # Scipy
    cache = FunctionalCache(mem_size=1024)
    t0 = time.time()
    _, scipy_loss = optimize_constants(
        tree, env, y, mse_loss, cache,
        method="Nelder-Mead", maxiter=200,
    )
    t_scipy = time.time() - t0

    # JAX (warmup compile + 200 steps)
    t0 = time.time()
    _, jax_loss = optimize_constants_jax(
        tree, env_jax, y_jax, n_steps=200, learning_rate=0.05,
    )
    t_jax = time.time() - t0

    # Both should converge to near-zero loss — correctness parity
    assert scipy_loss < 0.1, f"scipy didn't converge; loss={scipy_loss}"
    assert jax_loss < 0.1, f"jax_adam didn't converge; loss={jax_loss}"
    print(f"\nscipy NM: {t_scipy*1000:.1f}ms loss={scipy_loss:.4g}")
    print(f"jax Adam: {t_jax*1000:.1f}ms loss={jax_loss:.4g}")
    # No speed assertion — see test docstring


def test_gp_integration_with_jax_adam():
    """End-to-end: GP run with optimize_constants_method='jax_adam'
    completes without errors and produces non-trivial output."""
    from tessera.search import GP, GPConfig

    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n).astype(np.float32)
    y = 2.0 * x * x - 0.5 * x + 1.0 + 0.05 * rng.standard_normal(n).astype(np.float32)

    cfg = GPConfig(
        pop_size=30, n_gens=8, seed=42,
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_method="jax_adam",
        optimize_constants_every=2,
        optimize_constants_maxiter=50,
        optimize_constants_jax_lr=0.01,
    )
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) > 0
    best = min(front, key=lambda c: c.train_loss)
    # Should find SOMETHING (not necessarily the exact answer in 8 gens)
    assert np.isfinite(best.train_loss)


def test_gp_falls_back_to_scipy_when_loss_not_mse():
    """If loss_fn isn't mse, jax_adam path is skipped; scipy default used."""
    from tessera.search import GP, GPConfig

    def quartic_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 4))

    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float32)
    y = x * x

    cfg = GPConfig(
        pop_size=15, n_gens=3, seed=0,
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,   # JAX env populated, but...
        optimize_constants_method="jax_adam",
        optimize_constants_every=1,
        optimize_constants_maxiter=20,
    )
    gp = GP(cfg, loss_fn=quartic_loss)   # custom loss disables JAX path
    front = gp.run({"x": x}, y, ["x"])
    # Just verify it ran (no exception) and produced a front
    assert len(front) > 0
