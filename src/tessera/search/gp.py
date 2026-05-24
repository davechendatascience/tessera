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
from tessera.expression.interval import (
    interval_evaluate, env_intervals_from_arrays,
)
from .bounds import mse_lower_bound, pareto_threshold
from tessera.expression.mutation import (
    mutate, random_tree, validate_tree,
)

from .base import Candidate
from .losses import mse_loss
from .scoring import _evaluate_tree
from .pareto import pareto_front
from .const_opt import optimize_constants, optimize_constants_jax
from .hall_of_fame import HallOfFame

# Tier 3 imports: optional; only used when cfg.use_jax_population_eval=True.
try:
    from tessera.expression.jit import is_pure_pointwise
    from tessera.expression.batched import evaluate_population_stacked
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    is_pure_pointwise = None              # type: ignore
    evaluate_population_stacked = None    # type: ignore


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
    parsimony_schedule: Callable[[int, int], float] | None = None
    """Optional non-monotone parsimony schedule. When set, called as
    `schedule(current_gen, total_gens) -> float` each generation; the
    returned value overrides `parsimony` for THAT gen's fitness term.

    Use case: the climb-then-simplify path problem (per
    `docs/research/benchmark_difficulty_and_climb_then_simplify.md` §2).
    Low parsimony during early gens lets the GP explore complex trees
    that might compose useful operator combinations (e.g., atan2 in IK);
    annealing to normal parsimony in later gens drives the discovered
    structure down to its simplest form.

    Idiom: pair with `tessera.search.climb_then_anneal_parsimony`:
        cfg = GPConfig(
            parsimony=0.005,
            parsimony_schedule=climb_then_anneal_parsimony(
                climb_until=0.3, climb_value=0.0001, final_value=0.005,
            ),
        )

    Note: schedules do NOT propagate through the multiprocess worker
    path. With n_workers>1, only the static `parsimony` is used."""
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
    """Optimiser for the polish step. Options:

    Scipy methods (always available):
      'Nelder-Mead'   — default; gradient-free; robust to non-smooth
                        losses (PnL+flip_rate etc.).
      'BFGS'          — finite-difference gradient; faster on smooth.
      'Powell'        — gradient-free alternative.

    JAX method (requires jax installed + use_jax_population_eval=True):
      'jax_adam'      — jax.grad + hand-rolled Adam. ~10-50× per-tree
                        vs Nelder-Mead on the GPU. Only applicable to
                        pure-pointwise trees + mse_loss; mixed trees
                        and non-MSE losses silently fall back to the
                        configured scipy method.

    Per `docs/planned/roadmap.md` §1.3."""

    optimize_constants_maxiter: int = 50
    """Steps. For scipy methods this is the maxiter passed to
    scipy.optimize.minimize. For 'jax_adam' this is the Adam step
    count; default 50 works for most MSE losses."""

    optimize_constants_jax_lr: float = 1e-2
    """Adam learning rate for the 'jax_adam' optimiser (ignored
    otherwise). Tune up if losses plateau; down if they diverge."""

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

    # JAX batched population evaluation (Tier 3 integration)
    use_jax_population_eval: bool = False
    """If True, evaluate the pure-pointwise subset of each generation as
    a single batched JAX call via
    `tessera.expression.batched.evaluate_population_stacked`.

    Requires `jax` to be installed. Trees containing FunctionalOp /
    FunctionalOp2D fall back to the per-tree numpy path. Only `mse_loss`
    is supported in the batched path; other losses (PnL etc.) fall back.

    On CPU, batched vmap is overhead, not speedup -- the gain is on GPU.
    Typical Colab T4: ~25x over numpy at MNIST scale (N=60K, K=200).
    See benchmarks in notebooks/tessera_jax_tier3.ipynb.

    When enabled, the FunctionalCache is bypassed for the batched
    subset; JAX's own topology-keyed jit cache handles equivalent reuse.
    """

    # Branch-and-bound pruning (interval arithmetic)
    prune_by_lower_bound: bool = False
    """If True, before full evaluation, compute an interval-arithmetic
    bound on the candidate's predictions and a corresponding MSE loss
    lower bound. If the bound exceeds the Pareto-front loss at the
    candidate's complexity, skip full evaluation.

    Only MSE bounds are supported currently; other loss functions
    (PnL+flip, etc.) get no pruning even with this flag on. Trees
    containing FunctionalOp / FunctionalOp2D get loose bounds
    (interval evaluator is conservative on functionals); pruning is
    most effective on pure pointwise trees.

    Default off because the tight-fit case (where pruning matters
    most) typically already has a low-loss incumbent; the bound is
    rarely tighter than just running the eval. Turn on for SR tasks
    with large parsimony gap between cx levels.
    """


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
        # Branch-and-bound state. Populated in run() if pruning is enabled.
        self._env_intervals: dict = {}
        self._current_front: list[Candidate] = []
        self.prune_stats = dict(n_pruned=0, n_evaluated=0)
        self._stop_requested = False
        self._pool: ProcessPoolExecutor | None = None
        # Tier-3 batched-eval state (populated in run() if enabled)
        self._env_jax: dict | None = None
        self._y_true_jax = None
        self._feature_names_tuple: tuple = ()
        # Current generation's parsimony coefficient. Updated each gen
        # if `cfg.parsimony_schedule` is set; otherwise stays at the
        # static `cfg.parsimony` value.
        self._current_parsimony: float = self.cfg.parsimony

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

        # Branch-and-bound: enable only with mse_loss + n_workers=1 +
        # the config flag. Pruning the worker path is harder (workers
        # don't see the parent's Pareto front); deferred.
        if (self.cfg.prune_by_lower_bound
            and self.loss_fn is mse_loss
            and self.cfg.n_workers == 1):
            self._env_intervals = env_intervals_from_arrays(env)
        else:
            self._env_intervals = {}
        self.prune_stats = dict(n_pruned=0, n_evaluated=0)

        # Tier-3 setup: convert env + y_true to JAX once. The batched
        # scorer reads these via self._env_jax / self._y_true_jax.
        if self.cfg.use_jax_population_eval and _HAS_JAX:
            if self.loss_fn is not mse_loss:
                if self.cfg.verbose:
                    print("[gp] use_jax_population_eval=True but loss_fn is not "
                          "mse_loss; batched JAX path disabled.")
            else:
                import jax.numpy as jnp
                self._env_jax = {k: jnp.asarray(v) for k, v in env.items()}
                self._y_true_jax = jnp.asarray(y_true)
                self._feature_names_tuple = tuple(feature_names)
        else:
            self._env_jax = None
            self._y_true_jax = None

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

            # Non-monotone parsimony schedule, if configured. The
            # returned value is used by every scoring path this gen
            # (_score, _score_batch, _score_no_simplify, polish).
            if self.cfg.parsimony_schedule is not None:
                self._current_parsimony = float(
                    self.cfg.parsimony_schedule(gen, self.cfg.n_gens)
                )

            # Expose the current Pareto front to _score so it can prune
            # using the pareto_threshold.
            self._current_front = front

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
        # Collect valid trees first, then score in a batch
        trees: list[Node] = []
        attempts = 0
        max_attempts = self.cfg.pop_size * 10
        while len(trees) < self.cfg.pop_size and attempts < max_attempts:
            attempts += 1
            tree = random_tree(
                self.rng, feature_names,
                max_depth=self.cfg.init_max_depth,
                enable_2d=self.cfg.enable_2d,
                pointwise_only=self.cfg.pointwise_only,
            )
            if validate_tree(tree, set(feature_names)) is not None:
                continue
            trees.append(tree)
        if len(trees) < self.cfg.pop_size:
            raise RuntimeError(
                f"could not initialise {self.cfg.pop_size} valid trees "
                f"in {max_attempts} attempts"
            )
        if self._env_jax is not None:
            return self._score_batch(trees, env, y_true, born_gen=0)
        return [self._score(t, env, y_true, born_gen=0) for t in trees]

    def _score_batch(self, trees, env, y_true, born_gen):
        """Tier-3 batched scoring with cross-tree subexpression caching.

        Pipeline per generation:
        1. Simplify trees (canonical form for cache-key matching)
        2. Materialize shared subtrees: pre-evaluate FunctionalOp / 2D
           subtrees that appear in >= 2 trees ONCE, bind to synthetic
           Vars. After this, many trees become pure-pointwise.
        3. Partition: pure-pointwise (batched JAX) vs mixed (per-tree).
        4. Batched subset: one vmapped jit call.
        5. Mixed subset: per-tree numpy path via _score_no_simplify.

        The Candidate stored downstream uses the ORIGINAL (pre-rewrite)
        tree — materialization is purely an evaluation-time optimisation
        and must not affect HoF or selection. The rewritten tree is
        equivalent on `env` but is not the user-facing artefact.
        """
        if self._env_jax is None:
            # Batched eval disabled or not applicable — fall back per-tree
            return [self._score(t, env, y_true, born_gen) for t in trees]

        # Simplify first (matches _score behavior; also normalises trees
        # so the materialize step's str-based canonical key matches more
        # subtrees).
        if self.cfg.simplify_trees:
            trees = [simplify(t) for t in trees]

        # Cross-tree subexpression materialization (the FunctionalCache
        # bridge into the batched JAX path). Pre-evaluates shared
        # FunctionalOp/2D subtrees once on the JAX env; returns
        # rewritten trees that may now be pure-pointwise.
        from tessera.expression.materialize import materialize_shared_subtrees
        rewritten, augmented_env_jax, mat_stats = materialize_shared_subtrees(
            trees, self._env_jax, threshold=2,
        )
        self.prune_stats.setdefault("materialized", 0)
        self.prune_stats.setdefault("substitutions", 0)
        self.prune_stats["materialized"] += mat_stats["n_materialized"]
        self.prune_stats["substitutions"] += mat_stats["n_replacements"]

        # Partition the REWRITTEN trees (not the originals): after
        # materialization, many trees that contained FunctionalOp are
        # now pure-pointwise and eligible for batched eval.
        results: list = [None] * len(rewritten)
        batch_idx: list[int] = []
        for i, t in enumerate(rewritten):
            if is_pure_pointwise(t):
                batch_idx.append(i)

        # Per-tree fallback for still-mixed trees. Pass the ORIGINAL
        # (pre-materialization) tree + the original env, since the
        # numpy _score path doesn't know about synthetic Vars.
        non_batch = set(range(len(rewritten))) - set(batch_idx)
        for i in non_batch:
            results[i] = self._score_no_simplify(
                trees[i], env, y_true, born_gen
            )

        if not batch_idx:
            return results

        # Batched JAX eval on the augmented env (original vars +
        # synthetic _cached_* vars from materialization).
        import jax.numpy as jnp
        batch_trees = [rewritten[i] for i in batch_idx]
        # Complexity is reported from the ORIGINAL tree (the user-facing
        # artefact), not the rewritten one.
        cxs = [complexity(trees[i]) for i in batch_idx]

        var_names = tuple(augmented_env_jax.keys())
        preds = evaluate_population_stacked(
            batch_trees, augmented_env_jax, var_names=var_names
        )                                                     # [K, N]
        # NaN-safe MSE per row + min_valid_frac filter
        finite = jnp.isfinite(preds)
        n_valid = finite.sum(axis=1)                          # [K]
        N = preds.shape[1]
        # Replace non-finite predictions with the corresponding y_true so
        # they contribute 0 to the squared error (then we divide by n_valid
        # to get per-row mean over valid entries only).
        safe_preds = jnp.where(finite, preds, self._y_true_jax)
        sq_err = (safe_preds - self._y_true_jax) ** 2         # [K, N]
        mse = sq_err.sum(axis=1) / jnp.maximum(n_valid, 1)    # [K]
        valid_frac = n_valid / N
        loss_vec = jnp.where(
            (n_valid > 0) & (valid_frac >= self.cfg.min_valid_frac),
            mse,
            jnp.inf,
        )
        losses_np = np.asarray(loss_vec.block_until_ready())  # one sync

        self.prune_stats["n_evaluated"] += len(batch_idx)
        for j, i in enumerate(batch_idx):
            l = float(losses_np[j])
            cx = cxs[j]
            fitness = l + self._current_parsimony * cx
            results[i] = Candidate(
                tree=trees[i], train_loss=l, complexity=cx,
                fitness=fitness, born_gen=born_gen,
            )
        return results

    def _score_no_simplify(self, tree, env, y_true, born_gen):
        """Like _score but skips the simplify step (caller already did it)."""
        cx = complexity(tree)
        if self._env_intervals and self._current_front:
            try:
                pred_iv = interval_evaluate(tree, self._env_intervals)
                if pred_iv.lo > -float("inf") or pred_iv.hi < float("inf"):
                    lb = mse_lower_bound(pred_iv.lo, pred_iv.hi, y_true)
                    threshold = pareto_threshold(self._current_front, cx)
                    if lb >= threshold:
                        self.prune_stats["n_pruned"] += 1
                        return Candidate(
                            tree=tree, train_loss=float("inf"),
                            complexity=cx, fitness=float("inf"),
                            born_gen=born_gen,
                        )
            except Exception:
                pass
        self.prune_stats["n_evaluated"] += 1
        loss = _evaluate_tree(
            tree, env, y_true, self.cache,
            fill_warmup=self.cfg.fill_warmup,
            loss_fn=self.loss_fn,
            min_valid_frac=self.cfg.min_valid_frac,
        )
        fitness = loss + self._current_parsimony * cx
        return Candidate(tree=tree, train_loss=loss, complexity=cx,
                         fitness=fitness, born_gen=born_gen)

    def _score(self, tree, env, y_true, born_gen):
        if self.cfg.simplify_trees:
            tree = simplify(tree)
        cx = complexity(tree)

        # Branch-and-bound prune: cheap interval-arithmetic bound +
        # MSE lower bound + Pareto threshold check. Sound: a pruned
        # candidate provably can't beat the current front at its cx.
        if self._env_intervals and self._current_front:
            try:
                pred_iv = interval_evaluate(tree, self._env_intervals)
                if pred_iv.lo > -float("inf") or pred_iv.hi < float("inf"):
                    lb = mse_lower_bound(pred_iv.lo, pred_iv.hi, y_true)
                    threshold = pareto_threshold(self._current_front, cx)
                    if lb >= threshold:
                        self.prune_stats["n_pruned"] += 1
                        return Candidate(
                            tree=tree, train_loss=float("inf"),
                            complexity=cx, fitness=float("inf"),
                            born_gen=born_gen,
                        )
            except Exception:
                # Interval eval failing is non-fatal; just fall through.
                pass

        self.prune_stats["n_evaluated"] += 1
        loss = _evaluate_tree(
            tree, env, y_true, self.cache,
            fill_warmup=self.cfg.fill_warmup,
            loss_fn=self.loss_fn,
            min_valid_frac=self.cfg.min_valid_frac,
        )
        fitness = loss + self._current_parsimony * cx
        return Candidate(tree=tree, train_loss=loss, complexity=cx,
                         fitness=fitness, born_gen=born_gen)

    def _polish_pareto_constants(self, pop, front, env, y_true, gen):
        """Run scipy.optimize on the Const leaves of each Pareto-front
        candidate. If improved, splice back into pop by tree identity."""
        if not front:
            return pop
        improved: dict[int, Candidate] = {}

        # Dispatch decision: use jax_adam path when applicable.
        # Applicability: (1) method explicitly set to 'jax_adam',
        # (2) loss is mse, (3) JAX env is populated (which requires
        # use_jax_population_eval=True and jax installed).
        use_jax_path = (
            self.cfg.optimize_constants_method == "jax_adam"
            and self.loss_fn is mse_loss
            and self._env_jax is not None
            and self._y_true_jax is not None
        )

        for c in front:
            # Tree-level applicability: jax_adam requires pure-pointwise.
            if use_jax_path and is_pure_pointwise is not None \
                    and is_pure_pointwise(c.tree):
                new_tree, new_loss = optimize_constants_jax(
                    c.tree, self._env_jax, self._y_true_jax,
                    n_steps=self.cfg.optimize_constants_maxiter,
                    learning_rate=self.cfg.optimize_constants_jax_lr,
                    min_valid_frac=self.cfg.min_valid_frac,
                )
            else:
                # Scipy fallback: applies when (a) jax_adam not requested,
                # (b) loss isn't mse, (c) tree contains FunctionalOp, or
                # (d) JAX env not populated.
                method = self.cfg.optimize_constants_method
                if method == "jax_adam":
                    # User requested jax_adam but conditions not met;
                    # fall back to a scipy default.
                    method = "Nelder-Mead"
                new_tree, new_loss = optimize_constants(
                    c.tree, env, y_true, self.loss_fn, self.cache,
                    fill_warmup=self.cfg.fill_warmup,
                    min_valid_frac=self.cfg.min_valid_frac,
                    method=method,
                    maxiter=self.cfg.optimize_constants_maxiter,
                )
            if new_loss < c.train_loss - 1e-12:
                cx = complexity(new_tree)
                fitness = new_loss + self._current_parsimony * cx
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
        if self._env_jax is not None:
            return self._score_batch(new_trees, env, y_true, born_gen=gen)
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
