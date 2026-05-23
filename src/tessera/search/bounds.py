"""Loss lower bounds for branch-and-bound candidate pruning.

Given an interval bound `[pred_lo, pred_hi]` on a candidate tree's
predictions (from `tessera.expression.interval.interval_evaluate`),
compute a sound lower bound on the loss. If that lower bound exceeds
the Pareto-front's loss at the candidate's complexity, the candidate
is provably suboptimal and can be pruned without full evaluation.

This is the search-side operationalisation of the
"energy-minimisation with known landscape" framing: SR has full
information from the dataset, so dataset-derived bounds on the
candidate's possible loss let us prune large regions of expression
space without evaluating them.

What's here
-----------
    mse_lower_bound(pred_lo, pred_hi, y_true)
        Tight closed-form lower bound for MSE. The optimal prediction
        within [pred_lo, pred_hi] for each y_true sample is the
        clipped value `clip(y_true, pred_lo, pred_hi)`; the bound is
        the mean squared distance from y_true to the clipped value.

    pareto_threshold(front, cx)
        Returns the minimum loss to beat for a new candidate at
        complexity `cx` to be Pareto-relevant. A candidate is
        provably suboptimal iff its loss lower bound >= this
        threshold.

What's NOT here yet
-------------------
    pnl_loss_lower_bound — non-trivial: PnL is a sum over positions
    which depend on sign(pred), so the bound is on a stochastic
    combinatorial structure. Could be derived but deferred.

    Per-row interval evaluation — the current bound uses a single
    scalar interval for the whole prediction. Per-row intervals would
    give a tighter bound at the cost of O(N) interval arithmetic.
"""
from __future__ import annotations
import math

import numpy as np

from .base import Candidate


def mse_lower_bound(
    pred_lo: float,
    pred_hi: float,
    y_true: np.ndarray,
) -> float:
    """Tight MSE lower bound given a scalar prediction-interval and
    known targets.

    For each sample i, the optimal prediction within [pred_lo, pred_hi]
    is `clip(y_true[i], pred_lo, pred_hi)`. The squared distance from
    y_true[i] to that clip is the per-sample minimum contribution to
    MSE; summing and averaging gives the bound.

    If the interval is unbounded, the bound is 0 (no information).
    """
    if not math.isfinite(pred_lo) or not math.isfinite(pred_hi):
        return 0.0
    if pred_lo > pred_hi:
        # Degenerate (shouldn't happen with valid Interval; defensive)
        return 0.0
    mask = np.isfinite(y_true)
    if not mask.any():
        return 0.0
    y = y_true[mask]
    clipped = np.clip(y, pred_lo, pred_hi)
    err = y - clipped
    return float(np.mean(err ** 2))


def pareto_threshold(front: list[Candidate], cx: int) -> float:
    """Loss threshold a new candidate at complexity `cx` must beat to
    be Pareto-relevant.

    A new candidate at complexity `cx` with loss L dominates an existing
    front member iff there exists a member at complexity <= cx with
    loss > L. Equivalently: the candidate is Pareto-relevant iff
    L < min(c.train_loss for c in front if c.complexity <= cx).

    If no front member has complexity <= cx (e.g. empty front, or this
    candidate is the smallest yet seen), returns +inf — no pruning.

    Use:
        if lower_bound >= pareto_threshold(front, cx):
            prune  # provably can't beat the front at this cx
    """
    eligible = [c.train_loss for c in front if c.complexity <= cx]
    return min(eligible) if eligible else float("inf")
