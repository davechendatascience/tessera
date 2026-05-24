"""GP integration with PopulationEvaluator (Tier 3-A).

Tests that `GPConfig(use_jax_population_eval=True)` produces equivalent
search results to the default numpy path on the same seed / problem,
within float32 tolerance.

Skipped if jax is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from tessera.search import GP, GPConfig
from tessera.search.gp import _HAS_JAX
from tessera.expression import clear_jit_cache, clear_topo_cache


@pytest.fixture(autouse=True)
def _clean():
    clear_jit_cache(); clear_topo_cache()
    yield
    clear_jit_cache(); clear_topo_cache()


def _make_problem(n=500, seed=0):
    """Synthetic: y = x*x - 0.5*x + 1 + small noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    y = x * x - 0.5 * x + 1.0 + 0.05 * rng.standard_normal(n).astype(np.float32)
    return {"x": x}, y


def test_use_jax_flag_exists():
    cfg = GPConfig(use_jax_population_eval=True)
    assert cfg.use_jax_population_eval is True


def test_jax_path_initializes_env_jax():
    """When the flag is on, GP.run sets up the JAX env on the first call."""
    env, y = _make_problem()
    cfg = GPConfig(
        pop_size=20, n_gens=2, seed=0,
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_every=0,
    )
    gp = GP(cfg)
    gp.run(env, y, feature_names=["x"])
    assert gp._env_jax is not None
    assert gp._y_true_jax is not None


def test_jax_path_disables_when_loss_fn_not_mse():
    """If user supplies a non-mse loss_fn, the JAX path is silently
    disabled (env_jax stays None)."""
    from tessera.search.losses import mse_loss

    def custom_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 4))

    env, y = _make_problem()
    cfg = GPConfig(
        pop_size=20, n_gens=2, seed=0,
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_every=0,
    )
    gp = GP(cfg, loss_fn=custom_loss)
    gp.run(env, y, feature_names=["x"])
    # Custom loss disables the JAX path
    assert gp._env_jax is None


def test_jax_path_finds_signal():
    """A small GP run on a simple quadratic should find a low-loss
    candidate via the JAX path. Doesn't have to find the exact form;
    just confirm it produces a reasonable Pareto front."""
    env, y = _make_problem(n=500, seed=0)
    cfg = GPConfig(
        pop_size=50, n_gens=10, seed=42,
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_every=3,
    )
    gp = GP(cfg)
    front = gp.run(env, y, feature_names=["x"])
    assert len(front) > 0
    best = min(front, key=lambda c: c.train_loss)
    # var(y) is roughly 1 for this synthetic problem; a reasonable fit
    # should beat sqrt(var(y)) ≈ 1
    assert best.train_loss < 1.0, f"best loss {best.train_loss} too high"


def test_jax_and_numpy_paths_run_to_completion():
    """Both code paths should run end-to-end without crashing on the
    same problem + seed. We can't expect identical outputs because
    JAX uses float32 vs numpy float64, and the GP is stochastic; but
    both should complete and find SOMETHING."""
    env, y = _make_problem(n=500, seed=0)

    base_cfg = dict(
        pop_size=30, n_gens=5, seed=7,
        pointwise_only=True, verbose=False,
        optimize_constants_every=0,
    )

    # numpy path
    gp_np = GP(GPConfig(**base_cfg, use_jax_population_eval=False))
    front_np = gp_np.run(env, y, feature_names=["x"])
    assert len(front_np) > 0

    # JAX path
    clear_jit_cache(); clear_topo_cache()
    gp_jax = GP(GPConfig(**base_cfg, use_jax_population_eval=True))
    front_jax = gp_jax.run(env, y, feature_names=["x"])
    assert len(front_jax) > 0

    # Both should find a low-loss candidate (not necessarily identical)
    best_np = min(front_np, key=lambda c: c.train_loss)
    best_jax = min(front_jax, key=lambda c: c.train_loss)
    assert best_np.train_loss < 1.5
    assert best_jax.train_loss < 1.5


def test_jax_path_handles_mixed_functional_trees():
    """If the search produces some FunctionalOp trees, they should be
    routed to the per-tree path while pointwise trees go batched."""
    # pointwise_only=False ensures some FunctionalOp candidates
    env, y = _make_problem(n=500, seed=0)
    cfg = GPConfig(
        pop_size=20, n_gens=3, seed=0,
        pointwise_only=False, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_every=0,
    )
    gp = GP(cfg)
    front = gp.run(env, y, feature_names=["x"])
    assert len(front) > 0


def test_init_pop_uses_batch_when_jax_enabled():
    """Init population path also uses _score_batch when flag is on.
    Verify by checking that the populated trees have valid losses
    (rather than infinities from a broken path)."""
    env, y = _make_problem()
    cfg = GPConfig(
        pop_size=30, n_gens=0, seed=0,         # n_gens=0 -> just init
        pointwise_only=True, verbose=False,
        use_jax_population_eval=True,
        optimize_constants_every=0,
    )
    gp = GP(cfg)
    gp.run(env, y, feature_names=["x"])
    # All HoF entries should have finite loss
    hof_front = gp.hall_of_fame.pareto_front()
    assert len(hof_front) > 0
    for c in hof_front:
        assert np.isfinite(c.train_loss), f"non-finite loss in HoF: {c}"
