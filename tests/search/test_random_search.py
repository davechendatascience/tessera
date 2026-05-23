"""Tests for RandomSearch baseline."""
import numpy as np
import pytest

from tessera.search import RandomSearch, RSConfig, Candidate


def test_random_search_runs_and_returns_pareto_front():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = x * x

    cfg = RSConfig(n_trees=200, seed=1, verbose=False)
    rs = RandomSearch(cfg)
    front = rs.run({"x": x}, y, ["x"])

    assert len(front) >= 1
    assert all(isinstance(c, Candidate) for c in front)
    losses = [c.train_loss for c in front]
    cxs = [c.complexity for c in front]
    assert cxs == sorted(cxs)
    assert all(losses[i] >= losses[i+1] for i in range(len(losses) - 1))


def test_random_search_history_records_summary():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    y = x
    cfg = RSConfig(n_trees=50, seed=0, verbose=False)
    rs = RandomSearch(cfg)
    rs.run({"x": x}, y, ["x"])

    assert len(rs.history) == 1
    h = rs.history[0]
    for key in ("n_trees", "n_attempts", "pareto_size", "elapsed", "best_loss"):
        assert key in h


def test_random_search_finds_x_on_easy_problem():
    """y = x: random search with 500 draws will sometimes find a tree
    close to Var('x'). At least beats constant baseline."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = x

    cfg = RSConfig(n_trees=500, seed=2026, verbose=False)
    rs = RandomSearch(cfg)
    front = rs.run({"x": x}, y, ["x"])

    best = min(front, key=lambda c: c.train_loss)
    assert best.train_loss < y.var() * 0.5
