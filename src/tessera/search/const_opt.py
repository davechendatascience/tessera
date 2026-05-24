"""Constant optimisation — PySR-style polish step.

Refines the Const leaves of a tree via scipy.optimize.minimize. Used by
GP every K generations on Pareto-front candidates; also available
standalone for use by SA, RandomSearch, or any future searcher.

Also hosts the parsimony-schedule factory used by GP's climb-then-
simplify support (per
`docs/research/benchmark_difficulty_and_climb_then_simplify.md` §2).
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


def optimize_constants_jax(
    tree: Node,
    env_jax: dict,
    y_true_jax,
    *,
    n_steps: int = 100,
    learning_rate: float = 1e-2,
    min_valid_frac: float = 0.9,
) -> tuple[Node, float]:
    """jax.grad + Adam constant optimisation for pure-pointwise trees + MSE.

    Per `docs/planned/roadmap.md` §1.3: replaces scipy's derivative-free
    Nelder-Mead path with a jit'd autodiff loop. For tasks where it's
    applicable (pure-pointwise + MSE), this is ~10-50× faster per tree
    and the resulting jit can be vmap'd over a population for further
    speedup (a follow-up after this lands).

    Applicability is checked by the caller (see GP._polish_pareto_constants
    in tessera.search.gp). When the tree is not pure-pointwise or the
    loss isn't MSE, fall back to `optimize_constants` (scipy).

    Implementation notes:
    - Hand-rolled Adam (no optax dependency) to keep tessera's hard-dep
      surface minimal. JAX itself is already optional.
    - Builds the loss function via `_build_parametric_fn` from
      `tessera.expression.batched` — same primitive used by the Tier-3
      batched evaluator, so the jit graph is consistent across the
      eval and const-opt paths.
    - Output is broadcast to the input length (handles constant-only
      and reduce-collapsed subtrees the same way `compile_tree` does).

    Parameters
    ----------
    tree : Node
        Pure-pointwise tree. If it contains FunctionalOp or
        FunctionalOp2D, this function will fail to compile during the
        jax.grad call — caller must filter.
    env_jax : dict[str, jax_array]
        Variable arrays. JAX arrays required.
    y_true_jax : jax_array
        Target. JAX array required.
    n_steps : int
        Adam steps. Default 100 is a reasonable midpoint between scipy's
        50 Nelder-Mead iters and the full convergence Adam can hit.
    learning_rate : float
        Adam learning rate. Default 1e-2 works for most MSE losses on
        normalised inputs; tune up/down if losses diverge or plateau.
    min_valid_frac : float
        Reject candidates whose prediction has < min_valid_frac finite
        entries (post-NaN-substitution). Matches scipy path's contract.

    Returns
    -------
    (tree_with_optimized_consts, final_loss). If the tree has no Const
    leaves, returns the input tree + its initial MSE.
    """
    import jax
    import jax.numpy as jnp

    from tessera.expression.batched import _build_parametric_fn

    initial_consts = collect_const_values(tree)
    if not initial_consts:
        # No consts to optimise — just compute and return the current loss
        var_names = sorted(env_jax.keys())
        var_idx = {v: i for i, v in enumerate(var_names)}
        args = tuple(env_jax[v] for v in var_names)
        counter = [0]
        raw_fn = _build_parametric_fn(tree, var_idx, counter)
        empty_consts = jnp.zeros(0, dtype=args[0].dtype)
        preds = raw_fn(args, empty_consts)
        preds = jnp.broadcast_to(preds, args[0].shape)
        finite = jnp.isfinite(preds)
        n_valid = finite.sum()
        N = preds.shape[0]
        safe = jnp.where(finite, preds, y_true_jax)
        mse = ((safe - y_true_jax) ** 2).sum() / jnp.maximum(n_valid, 1)
        valid_frac = n_valid / N
        loss = float(jnp.where(valid_frac >= min_valid_frac, mse, jnp.inf))
        return tree, loss

    var_names = sorted(env_jax.keys())
    var_idx = {v: i for i, v in enumerate(var_names)}
    args = tuple(env_jax[v] for v in var_names)
    counter = [0]
    raw_fn = _build_parametric_fn(tree, var_idx, counter)
    N = args[0].shape[0]
    target_shape = args[0].shape
    min_n_valid = jnp.asarray(min_valid_frac * N)

    def loss_fn(consts):
        preds = raw_fn(args, consts)
        preds = jnp.broadcast_to(preds, target_shape)
        finite = jnp.isfinite(preds)
        n_valid = finite.sum()
        safe = jnp.where(finite, preds, y_true_jax)
        sq_err = (safe - y_true_jax) ** 2
        mse = sq_err.sum() / jnp.maximum(n_valid, 1)
        # If too many non-finite predictions, push loss to large value
        # (not inf — keeps the gradient defined).
        return jnp.where(n_valid >= min_n_valid, mse, 1e18)

    jit_loss = jax.jit(loss_fn)
    jit_grad = jax.jit(jax.grad(loss_fn))

    # Hand-rolled Adam (no optax dependency)
    consts = jnp.asarray(initial_consts, dtype=args[0].dtype)
    initial_loss = float(jit_loss(consts))
    if not np.isfinite(initial_loss) or initial_loss >= 1e17:
        return tree, float("inf")

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = jnp.zeros_like(consts)
    v = jnp.zeros_like(consts)
    best_consts = consts
    best_loss = initial_loss

    for step in range(1, n_steps + 1):
        grads = jit_grad(consts)
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads * grads
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        consts = consts - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        loss = float(jit_loss(consts))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_consts = consts

    new_tree = set_const_values(tree, [float(c) for c in best_consts])
    return new_tree, best_loss


# ---------------- Parsimony schedules ----------------

def climb_then_anneal_parsimony(
    climb_until: float = 0.3,
    climb_value: float = 0.0001,
    final_value: float = 0.005,
) -> Callable[[int, int], float]:
    """Climb-then-anneal parsimony schedule.

    Returns a callable suitable for `GPConfig.parsimony_schedule`:

        schedule(gen, n_gens) -> float

    Behaviour:
      - For the first `climb_until` fraction of generations, returns
        `climb_value` (use a small POSITIVE value like 0.0001 to barely
        penalise complexity during exploration; avoid going negative
        unless you want runaway complexity).
      - From `climb_until` to 1.0, linearly anneals from `climb_value`
        up to `final_value` (the normal parsimony pressure).

    Use case: the climb-then-simplify path problem (per
    `docs/research/benchmark_difficulty_and_climb_then_simplify.md` §2).
    Lets the GP explore high-cx trees that combine new operators
    (e.g., atan2 in IK) before parsimony pressure forces simplification.

    Parameters
    ----------
    climb_until : float
        Fraction of generations to keep at climb_value. Default 0.3.
    climb_value : float
        Parsimony during the climb phase. Default 0.0001 (50× less
        than the typical 0.005 final value). Recommended >= 0; negative
        values cause runaway complexity since GP fitness rewards
        bigger cx.
    final_value : float
        Parsimony at gen = n_gens. Matches `GPConfig.parsimony` typical
        value (0.005).
    """
    def schedule(gen: int, n_gens: int) -> float:
        if n_gens <= 0:
            return final_value
        progress = gen / n_gens
        if progress < climb_until:
            return climb_value
        # Linear anneal from climb_value at progress=climb_until to
        # final_value at progress=1.0
        anneal_fraction = (progress - climb_until) / max(1.0 - climb_until, 1e-9)
        return climb_value + (final_value - climb_value) * anneal_fraction
    return schedule


__all__ = [
    "optimize_constants",
    "optimize_constants_jax",
    "climb_then_anneal_parsimony",
]
