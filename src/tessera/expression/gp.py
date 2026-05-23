"""GP loop: population-based search over Expr trees with a Pareto front.

This is the search driver that ties together:
  - `tree.py`       — Node types + evaluate()
  - `mutation.py`   — random_tree + mutation operators
  - `cache.py`      — FunctionalCache for subexpression reuse

Algorithm
---------
A (μ + λ) evolution strategy with tournament selection and Pareto-front
maintenance:

  1. **Initialise** a population of `pop_size` random trees.
  2. **Evaluate** each on (X, y) via the cache. Score by train MSE
     (or any user-supplied loss).
  3. **Maintain a Pareto front** in (complexity, loss) space — the set
     of trees that are not dominated by any other (lower complexity AND
     lower loss).
  4. **Breed** `pop_size` offspring via the mutation dispatcher, sampling
     parents by tournament (size 3).
  5. **Survival** keeps the best μ from (parents ∪ offspring) by loss,
     with elitism guaranteeing the current Pareto front survives.
  6. **Stop** when (a) generations exhausted, (b) loss plateaus for N
     generations, or (c) user calls stop().

The cache hit rate typically climbs from ~30% in early generations to
~70-80% once the search settles into productive regions — this is the
amortised payoff of subexpression sharing.

Public API
----------
    GPConfig                     — knobs (pop_size, n_gens, parsimony, ...)
    Candidate                    — (tree, loss, complexity) tuple
    GP                           — the search engine
    GP.run(X, y, features)       — fit; returns the Pareto front
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .cache import FunctionalCache
from .tree import Node, complexity, depth, evaluate, used_features
from .mutation import (
    MAX_COMPLEXITY, MAX_DEPTH, OP_WEIGHTS,
    mutate, random_tree, validate_tree,
)


# ---------------- Config ----------------

@dataclass(frozen=True)
class GPConfig:
    """Knobs for the GP search.

    Defaults are tuned for medium-scale problems (≤10⁵ samples, ~10
    features). For PySR-scale benchmarks, raise pop_size + n_gens.
    """
    pop_size: int = 100
    n_gens: int = 50
    init_max_depth: int = 4
    mutation_max_depth: int = 3
    tournament_size: int = 3
    parsimony: float = 0.005    # added penalty per complexity unit
    elitism_keep_pareto: bool = True
    early_stop_patience: int = 10   # gens without best-fitness improvement
    seed: int = 0
    fill_warmup: float = 0.0   # zero-fill warmup rows during eval
    cache_mem_size: int = 4096
    verbose: bool = True


# ---------------- Candidate ----------------

@dataclass(frozen=True)
class Candidate:
    """One element of the population. Frozen so it can be a Pareto-front member."""
    tree: Node
    train_loss: float          # raw MSE (no parsimony penalty)
    complexity: int
    fitness: float             # = train_loss + parsimony * complexity (lower is better)
    born_gen: int

    def __repr__(self) -> str:
        return (
            f"Candidate(cx={self.complexity}, loss={self.train_loss:.4g}, "
            f"fit={self.fitness:.4g}, born@{self.born_gen}, "
            f"tree={str(self.tree)[:60]}{'...' if len(str(self.tree)) > 60 else ''})"
        )


# ---------------- Loss + evaluation ----------------

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error, NaN-safe.

    Returns inf if shapes are incompatible OR fewer than half the
    samples are finite (we count a prediction as "failed" if it's
    mostly NaN — typical of overflowing constants or pathological
    measure parameters).
    """
    if np.isscalar(y_pred):
        # Broadcast scalar to match y_true
        y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_true.shape:
        try:
            y_pred = np.broadcast_to(y_pred, y_true.shape)
        except ValueError:
            return float("inf")
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    n_valid = int(mask.sum())
    # Need at least 50% of samples valid (and at least 2 absolute)
    if n_valid < max(2, len(y_true) // 2):
        return float("inf")
    err = y_pred[mask] - y_true[mask]
    return float(np.mean(err ** 2))


def _evaluate_tree(
    tree: Node,
    env: dict[str, np.ndarray],
    y_true: np.ndarray,
    cache: FunctionalCache,
    fill_warmup: float,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    try:
        y_pred = evaluate(tree, env, cache=cache, fill_warmup=fill_warmup)
    except Exception:
        return float("inf")
    if np.isscalar(y_pred):
        y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
    return loss_fn(y_pred, y_true)


# ---------------- Pareto front ----------------

def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    """Return the non-dominated set in (complexity, train_loss) space.

    c1 dominates c2 iff c1.complexity ≤ c2.complexity AND
    c1.train_loss ≤ c2.train_loss AND at least one is strict.
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


# ---------------- The GP engine ----------------

class GP:
    """Population-based symbolic regression on tessera Expr trees.

    Usage
    -----
        cfg = GPConfig(pop_size=200, n_gens=50)
        gp = GP(cfg)
        front = gp.run(env={"x": x, "y": y, ...},
                       y_true=target_array,
                       feature_names=["x", "y", ...])
        # `front` is a sorted list[Candidate] from low-cx low-quality
        # to high-cx high-quality. Walk it for interpretability vs
        # accuracy trade-offs.
    """

    def __init__(self, cfg: GPConfig | None = None,
                 loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None):
        self.cfg = cfg or GPConfig()
        self.loss_fn = loss_fn or mse_loss
        self.rng = random.Random(self.cfg.seed)
        self.cache = FunctionalCache(mem_size=self.cfg.cache_mem_size)
        self.history: list[dict] = []   # per-gen best-loss, pareto-size, etc.
        self._stop_requested = False

    # ---- public ----

    def stop(self) -> None:
        """Signal the run loop to exit at the next generation boundary."""
        self._stop_requested = True

    def run(
        self,
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[Candidate]:
        """Run the search; return the final Pareto front."""
        if feature_names is None:
            feature_names = list(env.keys())

        # Initialise population
        pop = self._init_population(feature_names, env, y_true)
        front = pareto_front(pop)

        best = min(pop, key=lambda c: c.fitness)
        self._log_gen(0, pop, front, best, t0=time.time())

        last_best_loss = min(c.train_loss for c in pop)
        gens_without_improvement = 0

        # Generation loop
        for gen in range(1, self.cfg.n_gens + 1):
            if self._stop_requested:
                if self.cfg.verbose:
                    print(f"[gp] stop requested at gen {gen}, exiting")
                break

            t_gen = time.time()
            offspring = self._breed(pop, feature_names, env, y_true, gen)
            pop = self._survive(pop, offspring, front)
            front = pareto_front(pop)
            best = min(pop, key=lambda c: c.fitness)

            self._log_gen(gen, pop, front, best, t0=t_gen)

            # Early stop check
            current_best_loss = min(c.train_loss for c in pop)
            if current_best_loss + 1e-9 < last_best_loss:
                last_best_loss = current_best_loss
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
            if gens_without_improvement >= self.cfg.early_stop_patience:
                if self.cfg.verbose:
                    print(f"[gp] early stop at gen {gen} (no improvement for "
                          f"{gens_without_improvement} gens)")
                break

        return front

    # ---- internals ----

    def _init_population(
        self,
        feature_names: list[str],
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> list[Candidate]:
        pop: list[Candidate] = []
        attempts = 0
        max_attempts = self.cfg.pop_size * 10
        while len(pop) < self.cfg.pop_size and attempts < max_attempts:
            attempts += 1
            tree = random_tree(self.rng, feature_names, max_depth=self.cfg.init_max_depth)
            if validate_tree(tree, set(feature_names)) is not None:
                continue
            pop.append(self._score(tree, env, y_true, born_gen=0))
        if len(pop) < self.cfg.pop_size:
            raise RuntimeError(
                f"could not initialise {self.cfg.pop_size} valid trees in {max_attempts} attempts"
            )
        return pop

    def _score(
        self,
        tree: Node,
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
        born_gen: int,
    ) -> Candidate:
        cx = complexity(tree)
        loss = _evaluate_tree(
            tree, env, y_true, self.cache,
            fill_warmup=self.cfg.fill_warmup, loss_fn=self.loss_fn,
        )
        fitness = loss + self.cfg.parsimony * cx
        return Candidate(tree=tree, train_loss=loss, complexity=cx,
                         fitness=fitness, born_gen=born_gen)

    def _tournament(self, pop: list[Candidate]) -> Candidate:
        """Pick the best (lowest fitness) of `tournament_size` random candidates."""
        contestants = self.rng.sample(pop, k=min(self.cfg.tournament_size, len(pop)))
        return min(contestants, key=lambda c: c.fitness)

    def _breed(
        self,
        pop: list[Candidate],
        feature_names: list[str],
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
        gen: int,
    ) -> list[Candidate]:
        offspring: list[Candidate] = []
        while len(offspring) < self.cfg.pop_size:
            a = self._tournament(pop)
            b = self._tournament(pop)
            child_tree = mutate([a.tree, b.tree], self.rng, feature_names)
            if child_tree is None:
                continue
            offspring.append(self._score(child_tree, env, y_true, born_gen=gen))
        return offspring

    def _survive(
        self,
        pop: list[Candidate],
        offspring: list[Candidate],
        front: list[Candidate],
    ) -> list[Candidate]:
        """(μ + λ) survival: keep the best pop_size from parents+offspring.

        If elitism is on, ensure every Pareto-front member survives.
        """
        combined = list(pop) + list(offspring)
        combined.sort(key=lambda c: c.fitness)
        survivors = combined[: self.cfg.pop_size]

        if self.cfg.elitism_keep_pareto:
            survivor_set = set(survivors)
            for elite in front:
                if elite not in survivor_set:
                    # Bump the worst surviving non-elite
                    for i in range(len(survivors) - 1, -1, -1):
                        if survivors[i] not in front:
                            survivors[i] = elite
                            survivor_set.discard(survivors[i])
                            survivor_set.add(elite)
                            break
        return survivors

    def _log_gen(self, gen: int, pop: list[Candidate], front: list[Candidate],
                 best: Candidate, t0: float) -> None:
        """Record per-gen stats. Always appends to self.history; prints
        only when verbose."""
        elapsed = time.time() - t0
        self.history.append(dict(
            gen=gen, best_loss=best.train_loss, best_cx=best.complexity,
            pareto_size=len(front), hit_rate=self.cache.hit_rate(),
            n_cache=self.cache.n_mem, elapsed=elapsed,
        ))
        if self.cfg.verbose:
            print(
                f"[gp] gen {gen:3d} | best loss={best.train_loss:.4g} cx={best.complexity:2d} "
                f"| Pareto |F|={len(front)} | hit_rate={self.cache.hit_rate():.1%} "
                f"| n_cache={self.cache.n_mem} | {elapsed:.2f}s"
            )


__all__ = [
    "GPConfig", "Candidate", "GP",
    "mse_loss", "pareto_front",
]
