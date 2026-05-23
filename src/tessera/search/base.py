"""Shared types for the search submodule.

A `Candidate` is the unit of comparison across all search algorithms in
`tessera.search` — GP, simulated annealing, random search, future
QUBO/quantum-annealing variants. Each algorithm's per-step loop produces
candidates; the Pareto front, the history log, and any caller-facing
output return them.

Why a single shared dataclass: a Pareto front pulled from a GP run
should be directly comparable to one from an SA run, both for testing
(`test_compare.py`) and for benchmark reports. If each searcher invents
its own struct, that comparison gets fiddly.
"""
from __future__ import annotations
from dataclasses import dataclass
from tessera.expression.tree import Node


@dataclass(frozen=True)
class Candidate:
    """One element of a search trajectory (a population, an SA chain, etc.).

    Fields
    ------
    tree : Node
        The tessera Expr tree this candidate represents. Immutable;
        Pareto-front members can be safely shared across populations.
    train_loss : float
        Raw loss on the training data, BEFORE parsimony penalty. The
        Pareto front is computed in (complexity, train_loss) space.
    complexity : int
        Number of nodes in the (post-simplification) tree. Acts as the
        regularisation axis on the Pareto front.
    fitness : float
        Loss + parsimony * complexity. Lower is better. Used as the
        scalar selection criterion in GP and as the acceptance objective
        in SA. The Pareto front uses train_loss + complexity instead,
        ignoring fitness — `fitness` is for INTRA-algorithm comparisons
        (tournament selection, SA acceptance).
    born_gen : int
        For GP: the generation that produced this candidate. For SA:
        the step. For random search: 0 (or the proposal index).
    """
    tree: Node
    train_loss: float
    complexity: int
    fitness: float
    born_gen: int

    def __repr__(self) -> str:
        tree_str = str(self.tree)
        if len(tree_str) > 60:
            tree_str = tree_str[:57] + "..."
        return (
            f"Candidate(cx={self.complexity}, loss={self.train_loss:.4g}, "
            f"fit={self.fitness:.4g}, born@{self.born_gen}, tree={tree_str})"
        )
