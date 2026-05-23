"""tessera.search — search algorithms over tessera Expr trees.

Each search algorithm produces a Pareto-front of `Candidate` objects in
(complexity, train_loss) space. The submodule lets you swap the search
strategy while keeping evaluation, scoring, simplification, and constant
optimisation shared.

Available searchers
-------------------
    GP                   — population-based evolutionary search
                           (μ+λ ES with tournament selection)
    SimulatedAnnealing   — single-state annealing with metropolis
                           acceptance
    RandomSearch         — i.i.d. random-tree baseline (proves the
                           directed searchers are worth their cost)

Shared infrastructure
---------------------
    Candidate            — tree + (train_loss, complexity, fitness, born)
    pareto_front         — non-dominated set in (cx, loss) space
    mse_loss             — default loss
    _prediction_is_valid — NaN-fraction precheck (called before loss_fn)
    _evaluate_tree       — score one tree (the chokepoint)
    optimize_constants   — scipy-based Const-leaf refinement

Backwards compatibility
-----------------------
`from tessera.expression import GP, GPConfig, mse_loss, pareto_front`
keeps working — those symbols are re-exported from this submodule via
`tessera.expression.__init__`.
"""
from __future__ import annotations

from .base import Candidate
from .losses import mse_loss, _prediction_is_valid
from .losses_trading import pnl_loss_hard, pnl_loss_smooth
from .scoring import _evaluate_tree
from .pareto import pareto_front
from .const_opt import optimize_constants
from .hall_of_fame import HallOfFame
from .bounds import mse_lower_bound, pareto_threshold
from .gp import GP, GPConfig
from .sa import SimulatedAnnealing, SAConfig
from .random_search import RandomSearch, RSConfig

__all__ = [
    # Shared
    "Candidate",
    "mse_loss", "_prediction_is_valid", "_evaluate_tree",
    "pnl_loss_hard", "pnl_loss_smooth",
    "pareto_front", "optimize_constants",
    "HallOfFame",
    "mse_lower_bound", "pareto_threshold",
    # Searchers
    "GP", "GPConfig",
    "SimulatedAnnealing", "SAConfig",
    "RandomSearch", "RSConfig",
]
