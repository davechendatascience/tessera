"""Simulated annealing over Expr trees.

Single-state search with Metropolis acceptance:
  1. Start from a random tree.
  2. At each step, propose a mutation via the tessera mutation dispatcher.
  3. Score the proposal.
  4. Accept with probability `min(1, exp(-Δfitness / T))`.
     If accepted, the proposal becomes the current state.
  5. Cool T over `n_steps` (exponential or linear schedule).
  6. Optionally polish the current state's constants every K steps.
  7. Track every accepted-AND-improving state into a "trace"; return
     the Pareto front of the trace.

When SA over GP
---------------
Use SA when:
  - You have a smooth loss and a small search budget (SA's directed
    walk often beats GP's population-spread).
  - You want to study how a specific loss landscape navigates (the
    trace is one trajectory, easier to debug than a population).
  - You want a search method with formal convergence guarantees
    (Geman & Geman 1984: under log-cooling, SA converges in
    probability to the global optimum — GP has no such guarantee).

Use GP when:
  - You want population diversity (SA can get stuck in basins).
  - You want subexpression caching to amortise across many candidates
    (the cache pays off less in single-state SA than in pop-based GP).
  - You want parallelism (SA is inherently sequential at the state
    level).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tessera.expression.cache import FunctionalCache
from tessera.expression.tree import Node, complexity
from tessera.expression.simplify import simplify_canonical as simplify
from tessera.expression.mutation import mutate, random_tree, validate_tree

from .base import Candidate
from .losses import mse_loss
from .scoring import _evaluate_tree
from .pareto import pareto_front
from .const_opt import optimize_constants
from .hall_of_fame import HallOfFame


@dataclass(frozen=True)
class SAConfig:
    """Knobs for the simulated-annealing search.

    Defaults are tuned to match GP's per-evaluation cost on a typical
    problem (4-5 raw features, ~10k samples) — about the same wall-clock
    on a 4000-step SA run as a pop=120 gens=33 GP run.
    """
    n_steps: int = 2000
    """Total proposal evaluations. SA's wall-clock is exactly n_steps
    times the per-candidate cost (no per-step overhead)."""

    T_initial: float = 1.0
    """Initial temperature. Should be ~order(initial_loss) so the very
    first proposals have a real chance of accepting moves that worsen
    the loss. Auto-scaling from the initial state is a known
    improvement; left manual for now."""

    T_final: float = 1e-4
    """Final temperature. Should be small enough that only strictly-
    improving moves are accepted at the end."""

    cooling: str = "exponential"
    """'exponential' (T_k = T_initial * (T_final/T_initial)^(k/n_steps))
    or 'linear' (T_k = T_initial - k * (T_initial - T_final) / n_steps).
    Exponential is the standard choice."""

    n_restarts: int = 1
    """Number of independent SA chains. The final Pareto front merges
    across all restarts. Multistart helps when the loss landscape has
    multiple basins of attraction."""

    init_max_depth: int = 4

    parsimony: float = 0.005

    early_stop_patience: int = 500
    """Stop a chain if no improvement for this many steps. Set 0 to
    disable."""

    seed: int = 0
    verbose: bool = True

    fill_warmup: float = 0.0
    cache_mem_size: int = 4096
    min_valid_frac: float = 0.9
    simplify_trees: bool = True
    enable_2d: bool = False
    pointwise_only: bool = False

    optimize_constants_every: int = 0
    """Every K accepted moves, run scipy.optimize on the current state's
    Const leaves. Default 0 (off) because SA's natural acceptance
    rule already explores constants; turn on for smooth losses."""
    optimize_constants_method: str = "Nelder-Mead"
    optimize_constants_maxiter: int = 50


class SimulatedAnnealing:
    """Single-state simulated-annealing search over Expr trees.

    Usage
    -----
        cfg = SAConfig(n_steps=2000, T_initial=1.0, T_final=1e-4)
        sa = SimulatedAnnealing(cfg)
        front = sa.run(env={"x": x, ...}, y_true=y, feature_names=["x"])
    """

    def __init__(self, cfg: SAConfig | None = None,
                 loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None):
        self.cfg = cfg or SAConfig()
        self.loss_fn = loss_fn or mse_loss
        self.rng = random.Random(self.cfg.seed)
        self.cache = FunctionalCache(mem_size=self.cfg.cache_mem_size)
        self.history: list[dict] = []
        self.hall_of_fame = HallOfFame()
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        env: dict[str, np.ndarray],
        y_true: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[Candidate]:
        """Run all SA restarts; return Pareto front of all visited states."""
        if feature_names is None:
            feature_names = list(env.keys())

        all_visited: list[Candidate] = []
        for restart in range(self.cfg.n_restarts):
            if self._stop_requested:
                break
            chain_trace = self._run_chain(restart, feature_names, env, y_true)
            all_visited.extend(chain_trace)
            # Every visited candidate (accepted OR rejected proposal) is
            # a HoF candidate -- rejection doesn't make a tree's loss
            # invalid, it just means the chain didn't move there.
            self.hall_of_fame.update_many(chain_trace)

        return self.hall_of_fame.pareto_front()

    # ---- internals ----

    def _score(self, tree: Node, env, y_true, born_step: int) -> Candidate:
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
                         fitness=fitness, born_gen=born_step)

    def _random_initial_tree(self, feature_names) -> Node:
        """Sample one valid initial tree."""
        for _ in range(100):
            tree = random_tree(
                self.rng, feature_names,
                max_depth=self.cfg.init_max_depth,
                enable_2d=self.cfg.enable_2d,
                pointwise_only=self.cfg.pointwise_only,
            )
            if validate_tree(tree, set(feature_names)) is None:
                return tree
        raise RuntimeError("could not sample valid initial tree in 100 attempts")

    def _propose(self, current: Node, feature_names) -> Node | None:
        """Propose a mutation. Returns None if no valid proposal."""
        return mutate(
            [current], self.rng, feature_names,
            pointwise_only=self.cfg.pointwise_only,
            enable_2d=self.cfg.enable_2d,
        )

    def _temperature(self, k: int) -> float:
        """Cooling schedule at step k ∈ [0, n_steps)."""
        n = self.cfg.n_steps
        if n <= 1:
            return self.cfg.T_initial
        u = k / max(1, n - 1)
        if self.cfg.cooling == "linear":
            return self.cfg.T_initial + u * (self.cfg.T_final - self.cfg.T_initial)
        # exponential (default)
        if self.cfg.T_initial <= 0 or self.cfg.T_final <= 0:
            return max(self.cfg.T_final, 1e-12)
        ratio = self.cfg.T_final / self.cfg.T_initial
        return self.cfg.T_initial * (ratio ** u)

    def _accept(self, delta_fitness: float, T: float) -> bool:
        """Metropolis acceptance: always accept improvements; accept
        worsening moves with probability exp(-Δ/T)."""
        if delta_fitness < 0:
            return True
        if T <= 0:
            return False
        p = math.exp(-delta_fitness / T)
        return self.rng.random() < p

    def _run_chain(self, restart_idx: int, feature_names, env, y_true):
        """One SA chain (one full restart). Returns the trace of every
        visited candidate (including duplicates from rejected proposals).
        """
        # Each restart re-seeds for reproducibility but with a different offset.
        chain_rng = random.Random(self.cfg.seed + 1000 * restart_idx)
        self.rng = chain_rng

        t0 = time.time()
        current_tree = self._random_initial_tree(feature_names)
        current = self._score(current_tree, env, y_true, born_step=0)
        trace: list[Candidate] = [current]

        best_so_far = current
        steps_without_improvement = 0
        n_accepted = 0
        n_proposed = 0

        for step in range(1, self.cfg.n_steps + 1):
            if self._stop_requested:
                break

            T = self._temperature(step)
            proposal_tree = self._propose(current.tree, feature_names)
            if proposal_tree is None:
                continue
            n_proposed += 1

            proposal = self._score(proposal_tree, env, y_true, born_step=step)
            trace.append(proposal)

            delta = proposal.fitness - current.fitness
            if self._accept(delta, T):
                current = proposal
                n_accepted += 1

                # Optional const-opt polish on accepted states
                if (self.cfg.optimize_constants_every > 0
                    and n_accepted % self.cfg.optimize_constants_every == 0):
                    new_tree, new_loss = optimize_constants(
                        current.tree, env, y_true, self.loss_fn, self.cache,
                        fill_warmup=self.cfg.fill_warmup,
                        min_valid_frac=self.cfg.min_valid_frac,
                        method=self.cfg.optimize_constants_method,
                        maxiter=self.cfg.optimize_constants_maxiter,
                    )
                    if new_loss < current.train_loss - 1e-12:
                        cx = complexity(new_tree)
                        fitness = new_loss + self.cfg.parsimony * cx
                        current = Candidate(
                            tree=new_tree, train_loss=new_loss, complexity=cx,
                            fitness=fitness, born_gen=step,
                        )
                        trace.append(current)

            if current.fitness + 1e-9 < best_so_far.fitness:
                best_so_far = current
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if (self.cfg.early_stop_patience > 0
                and steps_without_improvement >= self.cfg.early_stop_patience):
                if self.cfg.verbose:
                    print(f"[sa] restart {restart_idx} early stop at step {step} "
                          f"(no improvement for {steps_without_improvement})")
                break

            if self.cfg.verbose and step % max(1, self.cfg.n_steps // 20) == 0:
                acc_rate = n_accepted / max(1, n_proposed)
                print(f"[sa] restart={restart_idx} step={step:5d} "
                      f"T={T:.3g} best_loss={best_so_far.train_loss:.4g} "
                      f"cx={best_so_far.complexity} acc={acc_rate:.1%}")

        elapsed = time.time() - t0
        self.history.append(dict(
            restart=restart_idx, n_proposed=n_proposed, n_accepted=n_accepted,
            best_loss=best_so_far.train_loss, best_cx=best_so_far.complexity,
            elapsed=elapsed,
        ))
        if self.cfg.verbose:
            print(f"[sa] restart {restart_idx} done in {elapsed:.2f}s, "
                  f"accepted {n_accepted}/{n_proposed} "
                  f"({n_accepted/max(1,n_proposed):.1%}), "
                  f"best loss={best_so_far.train_loss:.4g} "
                  f"cx={best_so_far.complexity}")
        return trace
