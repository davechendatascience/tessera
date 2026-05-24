"""JAX jit-compilation of tessera Expr trees (Tier 2 GPU port).

For a tree, build a pure JAX function that takes a tuple of input arrays
(one per Var in the order given by `var_names`) and returns the tree's
output array. Wrap with `jax.jit` so repeated evaluation drops from
ms-per-call to μs-per-call after first compile.

Tier 2 scope
------------
- **Pure-pointwise trees**: trees that contain no FunctionalOp /
  FunctionalOp2D. Covered fully by jit.
- **Mixed trees**: trees that contain FunctionalOp need to call out to
  `Measure.apply`, which has Python-level dispatch + a numpy
  kernel-materialise step. These do NOT jit-compile cleanly. Caller
  falls back to `evaluate()` (Tier 1 JAX path; slower but correct).

Caching
-------
Module-level cache keyed by `(str(tree), tuple(var_names))`. The same
tree compiled twice reuses the cached jit. Tree-equality determines
hits; semantically-equal trees with different `str()` representations
(e.g., differently-ordered comm operations) miss. Run trees through
`simplify_canonical` for cache-friendly normalization.

Use `clear_jit_cache()` to drop entries (e.g., for benchmarking the
cold path or for memory management).
"""
from __future__ import annotations

from typing import Callable, Sequence

from .tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    BIN_OP_FNS, UN_OP_FNS,
)


_CACHE: dict = {}


def is_pure_pointwise(node: Node) -> bool:
    """True if `node` contains no FunctionalOp / FunctionalOp2D anywhere.

    Pure-pointwise trees are the Tier 2 jit-compile candidates."""
    if isinstance(node, (Var, Const)):
        return True
    if isinstance(node, (FunctionalOp, FunctionalOp2D)):
        return False
    if isinstance(node, BinOp):
        return is_pure_pointwise(node.a) and is_pure_pointwise(node.b)
    if isinstance(node, UnOp):
        return is_pure_pointwise(node.a)
    return False


def _build_fn(tree: Node, var_idx: dict[str, int]) -> Callable:
    """Recursively build a pure Python callable for the tree.

    The result `f(args)` takes a tuple `args` of arrays (in `var_idx`'s
    indexing) and returns the tree's output. When traced by `jax.jit`
    with JAX inputs, the array_module dispatch in the op tables routes
    all ops to `jnp`, producing pure XLA HLO.
    """
    if isinstance(tree, Var):
        i = var_idx[tree.name]
        return lambda args: args[i]
    if isinstance(tree, Const):
        c = float(tree.value)
        # Capture c by value so the closure doesn't reference tree
        return lambda args, _c=c: _c
    if isinstance(tree, BinOp):
        fa = _build_fn(tree.a, var_idx)
        fb = _build_fn(tree.b, var_idx)
        op_fn = BIN_OP_FNS[tree.op]
        return lambda args: op_fn(fa(args), fb(args))
    if isinstance(tree, UnOp):
        fa = _build_fn(tree.a, var_idx)
        op_fn = UN_OP_FNS[tree.op]
        return lambda args: op_fn(fa(args))
    raise TypeError(f"unsupported node type {type(tree).__name__}")


def compile_tree(tree: Node, var_names: Sequence[str]) -> Callable:
    """Compile a pure-pointwise tree to a jax-jitted function.

    Parameters
    ----------
    tree : Node
        Pure-pointwise expression tree.
    var_names : Sequence[str]
        Ordered list of variable names. The returned callable takes a
        tuple `args` of arrays in this order.

    Returns
    -------
    Callable
        `f(args: tuple) -> jax_array`. First call compiles (slow ~100ms);
        subsequent calls run from XLA (μs).

    Raises
    ------
    ValueError
        If the tree contains FunctionalOp / FunctionalOp2D. Use
        `evaluate()` instead for mixed trees.
    ImportError
        If JAX is not installed.
    """
    if not is_pure_pointwise(tree):
        raise ValueError(
            "compile_tree: only pure-pointwise trees can be jit-compiled "
            "in Tier 2. Trees containing FunctionalOp / FunctionalOp2D "
            "must go through evaluate() (slower but correct)."
        )

    import jax   # noqa: F401  - raises ImportError clearly if missing

    var_names = tuple(var_names)
    key = (str(tree), var_names)
    if key in _CACHE:
        return _CACHE[key]

    import jax.numpy as jnp

    var_idx = {v: i for i, v in enumerate(var_names)}
    raw_fn = _build_fn(tree, var_idx)

    # Wrap so the output is always broadcast to the shape of the first
    # input (= [N]). Handles constant-only trees (output is a Python
    # float) and reduce_*-wrapped trees (output is a scalar array) by
    # broadcasting them up to a length-N vector. For trees that already
    # produce [N], the broadcast_to is a no-op.
    def _wrapped(args):
        out = raw_fn(args)
        return jnp.broadcast_to(out, args[0].shape)

    jitted = jax.jit(_wrapped)
    _CACHE[key] = jitted
    return jitted


def evaluate_jit(tree: Node, env: dict):
    """Convenience: compile + call.

    Pure-pointwise trees only. The variable order is `sorted(env.keys())`
    for determinism, so the same env dict gives the same cache key.

    For mixed trees, raises ValueError -- use `evaluate()` instead.
    """
    var_names = sorted(env.keys())
    fn = compile_tree(tree, var_names)
    args = tuple(env[v] for v in var_names)
    return fn(args)


_IMG_PREDICTOR_CACHE: dict = {}


def compile_image_predictor(tree: Node, batch_var: str = "image",
                             reduce: str = "mean") -> Callable:
    """Compile a tree → jit+vmap'd per-sample predictor.

    Use case: the tree operates on ONE sample (e.g. a 2D image of shape
    [H, W]), and you want to evaluate it efficiently on a batch of N
    samples (shape [N, H, W]). Returns a callable
    `f(image_batch) -> [N] jax_array` that:
      1. vmaps the tree-evaluation over the leading (sample) axis
      2. reduces each sample's per-pixel output to a scalar
         (NaN-safe mean / max / sum)
      3. jits the whole thing into one XLA program

    Cached: calling with the same (tree, batch_var, reduce) returns the
    same compiled function.

    Parameters
    ----------
    tree : Node
        Any tree (can include FunctionalOp, FunctionalOp2D — these go
        through the Tier 1 JAX path inside the jit trace).
    batch_var : str
        Name of the var that holds the per-sample input (default "image").
        Other Vars in the tree are unsupported in this initial version.
    reduce : {"mean", "max", "sum", "none"}
        How to collapse the per-sample output to a scalar. "mean" is
        NaN-safe (zero-mask + count). "none" returns the raw per-sample
        output unchanged.

    Returns
    -------
    Callable
        `f(batch: jax_array of shape [N, ...]) -> [N] jax_array`.
        First call compiles (slow); later calls run from XLA (μs).
    """
    import jax
    import jax.numpy as jnp

    # Top-level evaluate is imported lazily to avoid an import cycle.
    from .tree import evaluate as _evaluate_tree

    key = (str(tree), batch_var, reduce)
    if key in _IMG_PREDICTOR_CACHE:
        return _IMG_PREDICTOR_CACHE[key]

    def _per_sample(sample):
        # `sample` is a single example shape [H, W] (or [H,] for 1-D, etc.)
        env = {batch_var: sample}
        out = _evaluate_tree(tree, env)
        out = jnp.asarray(out)
        if reduce == "none":
            return out
        if out.ndim == 0:
            return out
        if reduce == "mean":
            mask = jnp.isfinite(out)
            n_valid = mask.sum()
            safe = jnp.where(mask, out, jnp.zeros_like(out))
            return jnp.where(n_valid > 0,
                             safe.sum() / jnp.maximum(n_valid, 1),
                             jnp.nan)
        if reduce == "max":
            mask = jnp.isfinite(out)
            safe = jnp.where(mask, out, -jnp.inf)
            val = safe.max()
            return jnp.where(mask.any(), val, jnp.nan)
        if reduce == "sum":
            mask = jnp.isfinite(out)
            safe = jnp.where(mask, out, jnp.zeros_like(out))
            return jnp.where(mask.any(), safe.sum(), jnp.nan)
        raise ValueError(f"unknown reduce={reduce!r}")

    batched = jax.vmap(_per_sample)
    jitted = jax.jit(batched)
    _IMG_PREDICTOR_CACHE[key] = jitted
    return jitted


def clear_image_predictor_cache() -> None:
    """Drop all compiled image-predictor entries from the module-level cache."""
    _IMG_PREDICTOR_CACHE.clear()


def image_predictor_cache_size() -> int:
    return len(_IMG_PREDICTOR_CACHE)


def clear_jit_cache() -> None:
    """Drop all compiled-tree entries from the module-level cache.

    Useful for benchmarking cold-path performance or controlling memory
    usage (each entry holds a compiled XLA program).
    """
    _CACHE.clear()


def jit_cache_size() -> int:
    """Number of compiled trees currently cached."""
    return len(_CACHE)


__all__ = [
    "compile_tree", "evaluate_jit",
    "compile_image_predictor",
    "is_pure_pointwise",
    "clear_jit_cache", "jit_cache_size",
    "clear_image_predictor_cache", "image_predictor_cache_size",
]
