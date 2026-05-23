"""Test that the search submodule exposes its public API + backwards
compatibility via tessera.expression."""
import pytest


def test_search_submodule_public_api():
    from tessera.search import (
        Candidate,
        GP, GPConfig,
        SimulatedAnnealing, SAConfig,
        RandomSearch, RSConfig,
        mse_loss, pareto_front, optimize_constants,
    )
    # Just confirm they're all importable; class identity is checked elsewhere.
    assert all(c is not None for c in [
        Candidate, GP, GPConfig, SimulatedAnnealing, SAConfig,
        RandomSearch, RSConfig,
    ])


def test_backwards_compat_via_expression():
    """Existing code: `from tessera.expression import GP, GPConfig, ...`
    must continue to work after the refactor."""
    from tessera.expression import GP, GPConfig, mse_loss, pareto_front
    # And the deeper path used by tests:
    from tessera.expression.gp import (
        Candidate, GP as GP2, GPConfig as GPConfig2,
        mse_loss as mse_loss2, pareto_front as pareto_front2,
        _prediction_is_valid, _evaluate_tree,
    )
    # Re-exports must be the same objects (no accidental shadowing)
    assert GP is GP2
    assert GPConfig is GPConfig2
    assert mse_loss is mse_loss2
    assert pareto_front is pareto_front2


def test_candidate_shared_across_searchers():
    """All three searchers must produce the SAME Candidate type so
    Pareto fronts are mergeable across algorithms (test_compare uses this)."""
    from tessera.search import Candidate as C1
    from tessera.search.gp import Candidate as C2
    from tessera.search.sa import Candidate as C3
    from tessera.search.random_search import Candidate as C4
    assert C1 is C2 is C3 is C4
