"""Tree scoring — used by every search algorithm.

`_evaluate_tree` is the single chokepoint that turns a candidate Node
into a scalar loss, handling:
  1. Evaluation via tessera.expression.tree.evaluate (with cache)
  2. NaN-fraction validity precheck
  3. Scalar-to-array broadcast (if the tree evaluates to a constant)
  4. Application of the user-provided loss_fn

Keeping this in one function means every searcher gets identical
behaviour on edge cases (overflow, all-NaN predictions, broken trees).
"""
from __future__ import annotations
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import Node, evaluate
from .losses import _prediction_is_valid


def _evaluate_tree(
    tree: Node,
    env: dict[str, np.ndarray],
    y_true: np.ndarray,
    cache: FunctionalCache,
    fill_warmup: float,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    min_valid_frac: float = 0.9,
) -> float:
    try:
        y_pred = evaluate(tree, env, cache=cache, fill_warmup=fill_warmup)
    except Exception:
        return float("inf")
    if not _prediction_is_valid(y_pred, y_true, min_valid_frac):
        return float("inf")
    if np.isscalar(y_pred):
        y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
    return loss_fn(y_pred, y_true)
