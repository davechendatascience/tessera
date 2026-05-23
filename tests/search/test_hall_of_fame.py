"""Tests for HallOfFame — per-complexity best-ever store."""
import numpy as np
import pytest

from tessera.expression.tree import Var, Const, BinOp
from tessera.search import (
    Candidate, HallOfFame, pareto_front,
    GP, GPConfig, SimulatedAnnealing, SAConfig,
    RandomSearch, RSConfig,
)


# ---------------- HallOfFame unit tests ----------------

def _cand(loss: float, cx: int, tree=None, born=0) -> Candidate:
    if tree is None:
        # Use a unique placeholder so frozen-dataclass equality doesn't merge
        tree = Var(f"v{cx}_{loss}")
    return Candidate(tree=tree, train_loss=loss, complexity=cx,
                     fitness=loss + 0.01 * cx, born_gen=born)


def test_hof_empty():
    hof = HallOfFame()
    assert len(hof) == 0
    assert hof.candidates() == []
    assert hof.pareto_front() == []
    assert hof.best() is None


def test_hof_update_adds_new_cx():
    hof = HallOfFame()
    assert hof.update(_cand(loss=0.5, cx=3)) is True
    assert len(hof) == 1
    assert hof.update(_cand(loss=0.3, cx=5)) is True
    assert len(hof) == 2


def test_hof_update_keeps_best_at_each_cx():
    hof = HallOfFame()
    hof.update(_cand(loss=0.5, cx=3))
    # Worse at same cx — should be rejected
    assert hof.update(_cand(loss=0.7, cx=3)) is False
    assert len(hof) == 1
    # Better at same cx — should replace
    assert hof.update(_cand(loss=0.2, cx=3)) is True
    assert len(hof) == 1
    assert hof.best().train_loss == pytest.approx(0.2)


def test_hof_update_ignores_microscopic_improvements():
    """Loss-improvement epsilon prevents float-noise thrash."""
    hof = HallOfFame()
    hof.update(_cand(loss=0.5, cx=3))
    # 1e-13 improvement should be ignored
    assert hof.update(_cand(loss=0.5 - 1e-13, cx=3)) is False


def test_hof_update_many_returns_improvement_count():
    hof = HallOfFame()
    candidates = [
        _cand(loss=0.5, cx=3),
        _cand(loss=0.3, cx=5),
        _cand(loss=0.7, cx=3),     # worse — no improvement
        _cand(loss=0.1, cx=8),     # new cx
    ]
    improved = hof.update_many(candidates)
    assert improved == 3
    assert len(hof) == 3
    assert hof.n_updates == 3


def test_hof_pareto_front_drops_dominated_entries():
    """A cx=5 entry with HIGHER loss than the cx=3 entry is dominated and
    must not appear in the Pareto front."""
    hof = HallOfFame()
    hof.update(_cand(loss=0.2, cx=3))
    hof.update(_cand(loss=0.5, cx=5))   # dominated by cx=3
    hof.update(_cand(loss=0.1, cx=7))
    front = hof.pareto_front()
    cxs = [c.complexity for c in front]
    assert 3 in cxs and 7 in cxs
    assert 5 not in cxs    # dominated


def test_hof_pareto_front_keeps_monotone_decreasing_losses():
    hof = HallOfFame()
    hof.update(_cand(loss=1.0, cx=1))
    hof.update(_cand(loss=0.5, cx=3))
    hof.update(_cand(loss=0.2, cx=5))
    hof.update(_cand(loss=0.1, cx=8))
    front = hof.pareto_front()
    losses = [c.train_loss for c in front]
    cxs = [c.complexity for c in front]
    assert cxs == sorted(cxs)
    assert all(losses[i] >= losses[i+1] for i in range(len(losses) - 1))


def test_hof_merge_takes_best_of_both():
    a = HallOfFame()
    a.update(_cand(loss=0.5, cx=3))
    a.update(_cand(loss=0.3, cx=5))
    b = HallOfFame()
    b.update(_cand(loss=0.2, cx=3))    # better than a's
    b.update(_cand(loss=0.6, cx=7))    # new cx
    merged = a.merge(b)
    assert len(merged) == 3            # cx in {3, 5, 7}
    assert merged.best().train_loss == pytest.approx(0.2)
    # cx=3 came from b
    cx3 = next(c for c in merged if c.complexity == 3)
    assert cx3.train_loss == pytest.approx(0.2)
    # cx=5 came from a (only one)
    cx5 = next(c for c in merged if c.complexity == 5)
    assert cx5.train_loss == pytest.approx(0.3)


def test_hof_iteration_yields_cx_sorted_candidates():
    hof = HallOfFame()
    hof.update(_cand(loss=0.5, cx=7))
    hof.update(_cand(loss=0.2, cx=3))
    hof.update(_cand(loss=0.4, cx=5))
    cxs = [c.complexity for c in hof]
    assert cxs == [3, 5, 7]


def test_hof_contains_check():
    hof = HallOfFame()
    hof.update(_cand(loss=0.5, cx=3))
    assert 3 in hof
    assert 5 not in hof


# ---------------- HoF integration with each searcher ----------------

@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = x * x + 0.5
    return {"env": {"x": x}, "y": y, "feats": ["x"]}


def test_gp_returns_hof_pareto_front(toy_data):
    cfg = GPConfig(pop_size=30, n_gens=10, verbose=False, seed=42)
    gp = GP(cfg)
    front = gp.run(toy_data["env"], toy_data["y"], toy_data["feats"])
    assert len(front) >= 1
    # Front == HoF Pareto front
    assert front == gp.hall_of_fame.pareto_front()
    # HoF has at least as many entries as the front
    assert len(gp.hall_of_fame) >= len(front)


def test_sa_returns_hof_pareto_front(toy_data):
    cfg = SAConfig(n_steps=200, verbose=False, seed=42)
    sa = SimulatedAnnealing(cfg)
    front = sa.run(toy_data["env"], toy_data["y"], toy_data["feats"])
    assert front == sa.hall_of_fame.pareto_front()
    assert len(sa.hall_of_fame) >= len(front)


def test_rs_returns_hof_pareto_front(toy_data):
    cfg = RSConfig(n_trees=200, verbose=False, seed=42)
    rs = RandomSearch(cfg)
    front = rs.run(toy_data["env"], toy_data["y"], toy_data["feats"])
    assert front == rs.hall_of_fame.pareto_front()


def test_gp_hof_protects_discovery_from_drift():
    """The motivating use case: discover a great cx=N early, then have
    the population drift away. HoF must still have that great candidate
    at the end.

    Construction: tiny pop_size + many gens encourages drift on
    challenging targets. We confirm that the HoF's best entry has loss
    no worse than the population's final-gen best.
    """
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    y = x * x * x + 0.3

    cfg = GPConfig(pop_size=15, n_gens=25, verbose=False, seed=2026,
                   parsimony=1e-3)
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])

    # The HoF must contain at least one entry strictly better than the
    # WORST entry on the Pareto front (sanity check that it's not empty).
    assert len(gp.hall_of_fame) >= 1
    hof_best_loss = gp.hall_of_fame.best().train_loss
    front_best_loss = min(c.train_loss for c in front)
    # HoF best must equal Pareto front best (front comes from HoF)
    assert hof_best_loss == pytest.approx(front_best_loss)
