"""Constant optimisation — PySR-style polish step.

Refines the Const leaves of a tree via scipy.optimize.minimize. Used by
GP every K generations on Pareto-front candidates; also available
standalone for use by SA, RandomSearch, or any future searcher.
"""
from __future__ import annotations
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import (
    Node, collect_const_values, set_const_values,
)
from .scoring import _evaluate_tree


def optimize_constants(
    tree: Node,
    env: dict[str, np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    cache: FunctionalCache,
    *,
    fill_warmup: float = 0.0,
    min_valid_frac: float = 0.9,
    method: str = "Nelder-Mead",
    maxiter: int = 50,
) -> tuple[Node, float]:
    """Refine the Const leaves of `tree` via scipy.optimize.minimize.

    Returns (tree, loss). If the tree has no Const leaves OR optimisation
    raises OR optimisation doesn't improve, returns the input tree with
    its evaluated loss (caller can decide whether to replace or keep).

    This is the polish step that closes most of the TRAIN-loss gap
    between tessera and PySR. PySR runs an equivalent BFGS pass on every
    Pareto candidate every generation. Tessera by default does it every
    `GPConfig.optimize_constants_every` generations to amortise the cost.

    Method recommendations:
      - 'Nelder-Mead' (default): gradient-free, robust to non-smooth
        losses (PnL+flip_rate etc.). Slowest but most reliable.
      - 'BFGS': uses finite-difference gradient; faster on smooth
        losses; can struggle on discontinuities.
      - 'Powell': another gradient-free option, often faster than
        Nelder-Mead on convex problems.
    """
    # scipy is in tessera's hard deps (FFT path). Import here to defer
    # the cost until the polish step is actually used.
    from scipy.optimize import minimize  # type: ignore

    initial = collect_const_values(tree)
    if not initial:
        # Nothing to optimise; just return the current evaluation.
        loss = _evaluate_tree(tree, env, y_true, cache, fill_warmup,
                              loss_fn, min_valid_frac)
        return tree, loss

    x0 = np.array(initial, dtype=np.float64)

    def objective(x: np.ndarray) -> float:
        new_tree = set_const_values(tree, x.tolist())
        loss = _evaluate_tree(new_tree, env, y_true, cache,
                              fill_warmup, loss_fn, min_valid_frac)
        # Guard against scipy stumbling on inf/nan by mapping to a finite
        # large value — keeps the minimiser moving instead of aborting.
        if not np.isfinite(loss):
            return 1e18
        return float(loss)

    initial_loss = objective(x0)
    if not np.isfinite(initial_loss) or initial_loss >= 1e18:
        return tree, float("inf")

    try:
        result = minimize(
            objective, x0,
            method=method,
            options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-9)
                    if method == "Nelder-Mead"
                    else dict(maxiter=maxiter),
        )
    except Exception:
        return tree, initial_loss

    if result.success or result.fun < initial_loss:
        if result.fun < initial_loss:
            new_tree = set_const_values(tree, result.x.tolist())
            return new_tree, float(result.fun)
    return tree, initial_loss
