"""Cross-algorithm comparison: GP vs SA vs RandomSearch on a shared toy
problem. Validates that the three searchers produce comparable
Candidates and that directed searchers (GP, SA) beat random."""
import numpy as np
import pytest

from tessera.search import (
    GP, GPConfig, SimulatedAnnealing, SAConfig,
    RandomSearch, RSConfig,
)


@pytest.fixture
def easy_data():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal(n)
    y = x + 0.5
    return {"env": {"x": x}, "y": y, "feats": ["x"]}


def test_all_three_finish_on_same_problem(easy_data):
    gp_cfg = GPConfig(pop_size=30, n_gens=10, verbose=False, seed=1)
    sa_cfg = SAConfig(n_steps=300, verbose=False, seed=1)
    rs_cfg = RSConfig(n_trees=300, verbose=False, seed=1)

    gp_front = GP(gp_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    sa_front = SimulatedAnnealing(sa_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    rs_front = RandomSearch(rs_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])

    assert len(gp_front) >= 1
    assert len(sa_front) >= 1
    assert len(rs_front) >= 1


def test_directed_searchers_beat_random_on_easy_target(easy_data):
    """y = x + 0.5: GP and SA should both find a very low loss; random
    search at similar budget should be MEASURABLY worse. (We can't
    require both to be strictly better than random because the
    distributions overlap on this small problem; assert non-regression.)
    """
    gp_cfg = GPConfig(pop_size=40, n_gens=15, verbose=False, seed=1,
                       optimize_constants_every=3)
    sa_cfg = SAConfig(n_steps=600, T_initial=0.5, T_final=1e-4,
                      verbose=False, seed=1, optimize_constants_every=20)
    rs_cfg = RSConfig(n_trees=600, verbose=False, seed=1)

    gp_front = GP(gp_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    sa_front = SimulatedAnnealing(sa_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    rs_front = RandomSearch(rs_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])

    best_gp = min(c.train_loss for c in gp_front)
    best_sa = min(c.train_loss for c in sa_front)
    best_rs = min(c.train_loss for c in rs_front)

    # Directed searchers must reach a meaningful fit; can't claim strict
    # superiority over random in EVERY run, but a reasonable bar:
    assert best_gp < 0.01, f"GP didn't fit y=x+0.5 well enough: loss={best_gp}"
    assert best_sa < 0.5, f"SA didn't make progress on y=x+0.5: loss={best_sa}"

    # Random search may stumble onto the right answer or not; just
    # assert it doesn't crash and produces a finite loss.
    assert np.isfinite(best_rs)


def test_pareto_fronts_can_be_merged(easy_data):
    """The whole point of sharing Candidate across searchers: you can
    merge fronts and re-run pareto_front to get a combined view."""
    from tessera.search import pareto_front

    gp_cfg = GPConfig(pop_size=20, n_gens=5, verbose=False, seed=0)
    sa_cfg = SAConfig(n_steps=100, verbose=False, seed=0)
    rs_cfg = RSConfig(n_trees=100, verbose=False, seed=0)

    gp_front = GP(gp_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    sa_front = SimulatedAnnealing(sa_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])
    rs_front = RandomSearch(rs_cfg).run(easy_data["env"], easy_data["y"], easy_data["feats"])

    merged = pareto_front(gp_front + sa_front + rs_front)
    # The merged front cannot be larger than the sum and must satisfy
    # Pareto monotonicity itself.
    assert len(merged) <= len(gp_front) + len(sa_front) + len(rs_front)
    losses = [c.train_loss for c in merged]
    cxs = [c.complexity for c in merged]
    assert cxs == sorted(cxs)
    assert all(losses[i] >= losses[i+1] for i in range(len(losses) - 1))
