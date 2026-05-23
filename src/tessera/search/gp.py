"""GP loop: population-based search over Expr trees with a Pareto front.

This is the search driver that ties together:
  - `tessera.expression.tree`       — Node types + evaluate()
  - `tessera.expression.mutation`   — random_tree + mutation operators
  - `tessera.expression.cache`      — FunctionalCache for subexpression reuse
  - `tessera.search.scoring`        — _evaluate_tree (loss + validity)
  - `tessera.search.const_opt`      — optimize_constants polish step

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
  6. **Polish** the Pareto front's Const leaves every K gens via
     scipy.optimize (PySR-style refinement).
  7. **Stop** when (a) generations exhausted, (b) loss plateaus for N
     generations, or (c) user calls stop().

Public API
----------
    GPConfig                     — knobs (pop_size, n_gens, parsimony, ...)
    GP                           — the search engine
    GP.run(env, y, features)     — fit; returns the Pareto front
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import Node, complexity
from tessera.expression.simplify import simplify_canonical as simplify
from tessera.expression.mutation import (
    mutate, random_tree, validate_tree,
)

from .base import Candidate
from .losses import mse_loss
from .scoring import _evaluate_tree
from .pareto import pareto_front
from .const_opt import optimize_constants
from .hall_of_fame import HallOfFame


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

    # 2-D / PDE mode
    enable_2d: bool = False
    """If True, random_tree generates FunctionalOp2D nodes (for PDE-style
    search over 2-D fields). The caller is responsible for env having
    2-D arrays as values."""

    # Pointwise-only mode (for pure ODE rediscovery)
    pointwise_only: bool = False
    """If True, no FunctionalOp / FunctionalOp2D nodes are generated.
    Suitable for SR on closed-form analytic targets like Lorenz-63 or
    Feynman benchmarks where the answer is a polynomial in the input
    variables (no temporal/functional structure needed)."""

    # Multiprocessing
    n_workers: int = 1
    """1 = sequential (no MP); >1 spawns a ProcessPoolExecutor.

    Honest performance note
    -----------------------
    MP at the population level gives modest speedup (≈1.2-1.5× on
    N≥50 000 samples, pop≥120, on a typical 8-core Windows desktop).
    On smaller problems the per-worker spawn cost (~1-2 s on Windows)
    + per-task pickling exceeds the parallelism gain — sequential
    (n_workers=1) is then faster.

    Default n_workers=1 keeps tests deterministic + portable. Tune up only
    for production runs where the per-eval cost dominates.

    Note: when n_workers > 1, the loss_fn passed to GP() must be picklable
    (importable top-level function or a `functools.partial` of one). Inline
    `lambda` or factory-returned closures fail to pickle under the default
    spawn start method and will raise at pool start. Use n_workers=1 if your
    loss_fn isn't picklable.
    """

    # Constant optimisation (PySR-style polish)
    optimize_constants_every: int = 5
    """Every K generations, run `scipy.optimize.minimize` on the Const
    leaves of each Pareto-front candidate. Closest tessera equivalent
    of PySR's `optimize_constants` inner loop. Set 0 to disable."""

    optimize_constants_method: str = "Nelder-Mead"
    """Scipy minimiser. Defaults to Nelder-Mead because the most
    interesting tessera loss functions (PnL+flip_rate) are non-smooth
    due to `sign`/`diff(sign)` discontinuities; gradient-based methods
    struggle. For smooth losses, 'BFGS' converges faster."""

    optimize_constants_maxiter: int = 50

    # Algebraic simplification
    simplify_trees: bool = True
    """If True (default), each scored tree is passed through `simplify()`
    before evaluation. Folds X-X→0, X*0→0, X/0→0 (safe-divide),
    constant arithmetic, double-neg. Pareto front shows EFFECTIVE
    complexity rather than nominal node count."""

    # NaN robustness
    min_valid_frac: float = 0.9
    """Reject any candidate whose prediction has < min_valid_frac finite
    entries. Raised from 0.5 to 0.9 in v0.1.2 to close the
    (x-x)/(x-x) divide-by-zero pathology."""


# ---------------- Worker-side helpers for multiprocessing ----------------
#
# These live at module level so they're picklable by spawned workers.
# Each worker holds its own env/y/cache via globals set in _init_worker.

_WORKER_ENV: dict[str, np.ndarray] = {}
_WORKER_Y: np.ndarray = np.array([])
_WORKER_CACHE: FunctionalCache | None = None
_WORKER_FILL_WARMUP: float = 0.0
_WORKER_PARSIMONY: float = 0.005
_WORKER_LOSS_FN: Callable[[np.ndarray, np.ndarray], float] = mse_loss
_WORKER_MIN_VALID_FRAC: float = 0.9
_WORKER_SIMPLIFY: bool = True


def _init_worker(
    env: dict[str, np.ndarray],
    y_true: np.ndarray,
    cache_mem_size: int,
    fill_warmup: float,
    parsimony: float,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    min_valid_frac: float,
    simplify_trees: bool,
) -> None:
    """Run once per worker on pool startup. Pickles env+y+loss_fn across
    just once rather than per-task."""
    global _WORKER_ENV, _WORKER_Y, _WORKER_CACHE
    global _WORKER_FILL_WARMUP, _WORKER_PARSIMONY
    global _WORKER_LOSS_FN, _WORKER_MIN_VALID_FRAC, _WORKER_SIMPLIFY
    _WORKER_ENV = env
    _WORKER_Y = y_true
    _WORKER_CACHE = FunctionalCache(mem_size=cache_mem_size)
    _WORKER_FILL_WARMUP = fill_warmup
    _WORKER_PARSIMONY = parsimony
    _WORKER_LOSS_FN = loss_fn
    _WORKER_MIN_VALID_FRAC = min_valid_frac
    _WORKER_SIMPLIFY = simplify_trees


def _score_in_worker(tree_and_gen: tuple[Node, int]) -> Candidate:
    """Score a candidate inside the worker process. Returns a fully-formed
    Candidate."""
    tree, born_gen = tree_and_gen
    if _WORKER_SIMPLIFY:
        tree = simplify(tree)
    cx = complexity(tree)
    loss = _evaluate_tree(
        tree, _WORKER_ENV, _WORKER_Y, _WORKER_CACHE,
        fill_warmup=_WORKER_FILL_WARMUP,
        loss_fn=_WORKER_LOSS_FN,
        min_valid_frac=_WORKER_MIN_VALID_FRAC,
    )
    fitness = loss + _WORKER_PARSIMONY * cx
    return Candidate(tree=tree, train_loss=loss, complexity=cx,
                     fitness=fitness, born_gen=born_gen)


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
    """

    def __init__(self, cfg: GPConfig | None = None,
                 loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None):
        self.cfg = cfg or GPConfig()
        self.loss_fn = loss_fn or mse_loss
        self.rng = random.Random(self.cfg.seed)
        self.cache = FunctionalCache(mem_size=self.cfg.cache_mem_size)
        self.history: list[dict] = []
        self.hall_of_fame = HallOfFame()
        self._stop_requested = False
        self._pool: ProcessPoolExecutor | None = None

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

        if self.cfg.n_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.cfg.n_workers,
                initializer=_init_worker,
                initargs=(env, y_true, self.cfg.cache_mem_size,
                          self.cfg.fill_warmup, self.cfg.parsimony,
                          self.loss_fn, self.cfg.min_valid_frac,
                          self.cfg.simplify_trees),
            )

        pop = self._init_population(feature_names, env, y_true)
        self.hall_of_fame.update_many(pop)
        front = pareto_front(pop)
        best = min(pop, key=lambda c: c.fitness)
        self._log_gen(0, pop, front, best, t0=time.time())

        last_best_loss = min(c.train_loss for c in pop)
        gens_without_improvement = 0

        for gen in range(1, self.cfg.n_gens + 1):
            if self._stop_requested:
                if self.cfg.verbose:
                    print(f"[gp] stop requested at gen {gen}, exiting")
                break

            t_gen = time.time()
            offspring = self._breed(pop, feature_names, env, y_true, gen)
            # Update HoF with all newly-evaluated offspring -- this is the
            # critical step that protects discoveries from mutation drift.
            self.hall_of_fame.update_many(offspring)
            pop = self._survive(pop, offspring, front)
            front = pareto_front(pop)

            if (self.cfg.optimize_constants_every > 0
                and gen % self.cfg.optimize_constants_every == 0):
                pop = self._polish_pareto_constants(pop, front, env, y_true, gen)
                # Polished candidates may be HoF improvements too
                self.hall_of_fame.update_many(pop)
                front = pareto_front(pop)

            best = min(pop, key=lambda c: c.fitness)
            self._log_gen(gen, pop, front, best, t0=t_gen)

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

        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

        # Return the Hall-of-Fame Pareto front, NOT the population front.
        # HoF protects per-cx best-ever from being lost to mutation drift,
        # so the final output is more honest about what the search
        # discovered. (Population front is available via pareto_front(pop)
        # if you need it — pop is internal but self.hall_of_fame is public.)
        return self.hall_of_fame.pareto_front()

    # ---- internals ----

    def _init_population(self, feature_names, env, y_true):
        pop: list[Candidate] = []
        attempts = 0
        max_attempts = self.cfg.pop_size * 10
        while len(pop) < self.cfg.pop_size and attempts < max_attempts:
            attempts += 1
            tree = random_tree(
                self.rng, feature_names,
                max_depth=self.cfg.init_max_depth,
                enable_2d=self.cfg.enable_2d,
                pointwise_only=self.cfg.pointwise_only,
            )
            if validate_tree(tree, set(feature_names)) is not None:
                continue
            pop.append(self._score(tree, env, y_true, born_gen=0))
        if len(pop) < self.cfg.pop_size:
            raise RuntimeError(
                f"could not initialise {self.cfg.pop_size} valid trees "
                f"in {max_attempts} attempts"
            )
        return pop

    def _score(self, tree, env, y_true, born_gen):
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
        return Candidate(tree=tree, train_loss=loss, complexity=cx,
                         fitness=fitness, born_gen=born_gen)

    def _polish_pareto_constants(self, pop, front, env, y_true, gen):
        """Run scipy.optimize on the Const leaves of each Pareto-front
        candidate. If improved, splice back into pop by tree identity."""
        if not front:
            return pop
        improved: dict[int, Candidate] = {}
        for c in front:
            new_tree, new_loss = optimize_constants(
                c.tree, env, y_true, self.loss_fn, self.cache,
                fill_warmup=self.cfg.fill_warmup,
                min_valid_frac=self.cfg.min_valid_frac,
                method=self.cfg.optimize_constants_method,
                maxiter=self.cfg.optimize_constants_maxiter,
            )
            if new_loss < c.train_loss - 1e-12:
                cx = complexity(new_tree)
                fitness = new_loss + self.cfg.parsimony * cx
                improved[id(c.tree)] = Candidate(
                    tree=new_tree, train_loss=new_loss, complexity=cx,
                    fitness=fitness, born_gen=gen,
                )
        if not improved:
            return pop
        return [improved.get(id(c.tree), c) for c in pop]

    def _tournament(self, pop):
        contestants = self.rng.sample(pop, k=min(self.cfg.tournament_size, len(pop)))
        return min(contestants, key=lambda c: c.fitness)

    def _breed(self, pop, feature_names, env, y_true, gen):
        new_trees: list[Node] = []
        while len(new_trees) < self.cfg.pop_size:
            a = self._tournament(pop)
            b = self._tournament(pop)
            child_tree = mutate(
                [a.tree, b.tree], self.rng, feature_names,
                pointwise_only=self.cfg.pointwise_only,
                enable_2d=self.cfg.enable_2d,
            )
            if child_tree is None:
                continue
            new_trees.append(child_tree)

        if self._pool is not None:
            tasks = [(t, gen) for t in new_trees]
            return list(self._pool.map(_score_in_worker, tasks))
        return [self._score(t, env, y_true, born_gen=gen) for t in new_trees]

    def _survive(self, pop, offspring, front):
        combined = list(pop) + list(offspring)
        combined.sort(key=lambda c: c.fitness)
        survivors = combined[: self.cfg.pop_size]
        if self.cfg.elitism_keep_pareto:
            survivor_set = set(survivors)
            for elite in front:
                if elite not in survivor_set:
                    for i in range(len(survivors) - 1, -1, -1):
                        if survivors[i] not in front:
                            survivors[i] = elite
                            survivor_set.discard(survivors[i])
                            survivor_set.add(elite)
                            break
        return survivors

    def _log_gen(self, gen, pop, front, best, t0):
        elapsed = time.time() - t0
        self.history.append(dict(
            gen=gen, best_loss=best.train_loss, best_cx=best.complexity,
            pareto_size=len(front), hit_rate=self.cache.hit_rate(),
            n_cache=self.cache.n_mem, elapsed=elapsed,
        ))
        if self.cfg.verbose:
            print(
                f"[gp] gen {gen:3d} | best loss={best.train_loss:.4g} "
                f"cx={best.complexity:2d} | Pareto |F|={len(front)} "
                f"| hit_rate={self.cache.hit_rate():.1%} "
                f"| n_cache={self.cache.n_mem} | {elapsed:.2f}s"
            )
