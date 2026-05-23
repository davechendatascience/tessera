"""Pareto-front maintenance in (complexity, train_loss) space.

Shared across all searchers — every algorithm builds a Pareto front
over its evaluated candidates and returns it as the final output.
"""
from __future__ import annotations
from .base import Candidate


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    """Return the non-dominated set in (complexity, train_loss) space.

    c1 dominates c2 iff c1.complexity ≤ c2.complexity AND
    c1.train_loss ≤ c2.train_loss AND at least one is strict.

    Returns a list sorted by complexity ascending (loss is monotone
    non-increasing on a true Pareto front).
    """
    # Sort by complexity ascending; then sweep keeping the running min loss
    by_cx = sorted(candidates, key=lambda c: (c.complexity, c.train_loss))
    front: list[Candidate] = []
    best_loss = float("inf")
    for c in by_cx:
        if c.train_loss < best_loss:
            front.append(c)
            best_loss = c.train_loss
    return front
