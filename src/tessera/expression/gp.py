"""Backwards-compatibility shim.

The GP machinery moved to `tessera.search` in v0.2 so multiple search
algorithms (GP, SA, RandomSearch, future QUBO/annealing variants) can
share the same scoring + Pareto + simplifier infrastructure.

This module re-exports the public symbols from their new locations so
existing code like `from tessera.expression.gp import GP, GPConfig,
mse_loss, pareto_front` continues to work unchanged.

New code should import directly from `tessera.search`:

    from tessera.search import GP, GPConfig
    from tessera.search import SimulatedAnnealing, SAConfig
    from tessera.search import RandomSearch, RSConfig
    from tessera.search import mse_loss, pareto_front, optimize_constants
"""
from __future__ import annotations

from tessera.search.base import Candidate
from tessera.search.losses import mse_loss, _prediction_is_valid
from tessera.search.scoring import _evaluate_tree
from tessera.search.pareto import pareto_front
from tessera.search.const_opt import optimize_constants
from tessera.search.gp import (
    GP, GPConfig,
    _init_worker, _score_in_worker,
    _WORKER_ENV, _WORKER_Y, _WORKER_CACHE,
    _WORKER_FILL_WARMUP, _WORKER_PARSIMONY,
    _WORKER_LOSS_FN, _WORKER_MIN_VALID_FRAC, _WORKER_SIMPLIFY,
)

__all__ = [
    "GPConfig", "Candidate", "GP",
    "mse_loss", "pareto_front", "optimize_constants",
    "_prediction_is_valid", "_evaluate_tree",
]
