"""Random-tree search — the obvious baseline.

Generates N random trees independently, scores each, returns the Pareto
front. No mutation, no selection, no annealing. Exists for two reasons:

1. **Baseline for comparison.** Any directed searcher (GP, SA) should
   beat random search on a comparable budget; if it doesn't, the
   directed search is broken or the problem is too small for it to
   matter.
2. **Sanity check.** On trivial targets (`y = x`, `y = x*x`), random
   search will sometimes find the right answer in a few hundred draws.
   Confirms that the tree generator + scoring path actually work.

Wall-clock equivalent of `GP(pop=120, gens=N)` is about `RandomSearch(
n_trees = 120*N)` since no mutation overhead, no tournament, no Pareto
maintenance per gen.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import complexity
from tessera.expression.simplify import simplify
from tessera.expression.mutation import random_tree, validate_tree

from .base import Candidate
from .losses import mse_loss
from .scoring import _evaluate_tree
from .pareto import pareto_front
from .hall_of_fame import HallOfFame


@dataclass(frozen=True)
class RSConfig:
    """Knobs for random search."""
    n_trees: int = 5000
    init_max_depth: int = 4
    parsimony: float = 0.005
    seed: int = 0
    verbose: bool = True

    fill_warmup: float = 0.0
    cache_mem_size: int = 4096
    min_valid_frac: float = 0.9
    simplify_trees: bool = True
    enable_2d: bool = False
    pointwise_only: bool = False


class RandomSearch:
    """Sample N random trees, return the Pareto front.

    No directed search — pure i.i.d. sampling from the random-tree
    generator. Used as a baseline.
    """

    def __init__(self, cfg: RSConfig | None = None,
                 loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None):
        self.cfg = cfg or RSConfig()
        self.loss_fn = loss_fn or mse_loss
        self.rng = random.Random(self.cfg.seed)
        self.cache = FunctionalCache(mem_size=self.cfg.cache_mem_size)
        self.history: list[dict] = []
        self.hall_of_fame = HallOfFame()

    def run(
        self,
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[Candidate]:
        if feature_names is None:
            feature_names = list(env.keys())

        t0 = time.time()
        candidates: list[Candidate] = []
        attempts = 0
        max_attempts = self.cfg.n_trees * 10
        log_every = max(1, self.cfg.n_trees // 20)

        while len(candidates) < self.cfg.n_trees and attempts < max_attempts:
            attempts += 1
            tree = random_tree(
                self.rng, feature_names,
                max_depth=self.cfg.init_max_depth,
                enable_2d=self.cfg.enable_2d,
                pointwise_only=self.cfg.pointwise_only,
            )
            if validate_tree(tree, set(feature_names)) is not None:
                continue
            if self.cfg.simplify_trees:
                tree = simplify(tree)
            cx = complexity(tree)
            loss = _evaluate_tree(
                tree, env, y_true, self.cache,
                fill_warmup=self.cfg.fill_warmup,
                loss_fn=self.loss_fn,
                min_valid_frac=self.cfg.min_valid_frac,
            )
            fitness = loss + self.cfg.parsimony * cx
            cand = Candidate(
                tree=tree, train_loss=loss, complexity=cx,
                fitness=fitness, born_gen=len(candidates),
            )
            candidates.append(cand)
            self.hall_of_fame.update(cand)

            if self.cfg.verbose and len(candidates) % log_every == 0:
                best = min(candidates, key=lambda c: c.fitness)
                print(f"[rs] n={len(candidates):5d} attempts={attempts} "
                      f"best loss={best.train_loss:.4g} cx={best.complexity}")

        elapsed = time.time() - t0
        front = self.hall_of_fame.pareto_front()
        self.history.append(dict(
            n_trees=len(candidates), n_attempts=attempts,
            pareto_size=len(front), elapsed=elapsed,
            best_loss=min(c.train_loss for c in candidates) if candidates else float("inf"),
        ))
        if self.cfg.verbose:
            print(f"[rs] done in {elapsed:.2f}s: {len(candidates)} valid trees, "
                  f"Pareto |F|={len(front)}")
        return front
