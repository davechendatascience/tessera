"""Tier 3 GPU port: batched-population evaluation via jax.vmap.

The GP population of K trees usually contains clusters of trees sharing
the same TOPOLOGY (op structure + Var positions) but differing in their
constant values. For each such cluster of size K_t, we can:

1. Parameterize the topology by a `[K_t, M]` constants tensor (M = number
   of Const leaves in the topology)
2. Compile a function `f(args, consts) -> output_array` once
3. `jax.vmap` over the leading axis of `consts` so that one jit-compiled
   GPU kernel evaluates all K_t trees in parallel

This is the standard approach used by recent JAX-based SR efforts. The
key claim: a GP population evaluated this way runs in ~constant wall-
clock as K grows, until GPU memory or population-cluster diversity
becomes the bottleneck.

Tier 3 covers pure-pointwise trees only. Mixed trees (containing
FunctionalOp / FunctionalOp2D) fall back to the per-tree Tier 2 path
or the eager Tier 1 path.

Public API:
    topology_key(tree) -> str
        Identifies a tree's structural topology, ignoring Const values.

    extract_constants(tree) -> list[float]
        Returns Const values in pre-order tree-walk.

    compile_topology(template, var_names) -> callable
        Returns a jit+vmap function f(args, consts_batch) where
        consts_batch is shape [K, M]. Cached by topology + var_names.

    evaluate_population(trees, env) -> list[jax_array]
        Group trees by topology, vmap each group, return outputs in
        input order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

from .tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    BIN_OP_FNS, UN_OP_FNS,
)
from .jit import is_pure_pointwise


_TOPO_CACHE: dict = {}


# ---------------- Topology fingerprint + constants extraction ----------------

def topology_key(node: Node) -> str:
    """String identifier of a tree's topology with Const VALUES erased.

    All Const leaves become "C" — so trees that differ only in their
    constant values share a topology_key. Var names ARE preserved; trees
    with different Var positions get different topologies (which is
    correct — they'd evaluate differently).
    """
    if isinstance(node, Const):
        return "C"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, BinOp):
        return f"{node.op}({topology_key(node.a)},{topology_key(node.b)})"
    if isinstance(node, UnOp):
        return f"{node.op}({topology_key(node.a)})"
    raise TypeError(f"topology_key: unsupported node {type(node).__name__}")


def extract_constants(node: Node) -> list[float]:
    """Return Const values in pre-order tree-walk order.

    Consistent walk-order matters: the topology-compiled function reads
    constants by INDEX from a [K, M] array, so the i-th Const in the
    walk must always map to the same array position.
    """
    if isinstance(node, Const):
        return [float(node.value)]
    if isinstance(node, Var):
        return []
    if isinstance(node, BinOp):
        return extract_constants(node.a) + extract_constants(node.b)
    if isinstance(node, UnOp):
        return extract_constants(node.a)
    raise TypeError(f"extract_constants: unsupported node {type(node).__name__}")


def n_constants(node: Node) -> int:
    """Number of Const leaves in the tree (== len(extract_constants))."""
    if isinstance(node, Const):
        return 1
    if isinstance(node, Var):
        return 0
    if isinstance(node, BinOp):
        return n_constants(node.a) + n_constants(node.b)
    if isinstance(node, UnOp):
        return n_constants(node.a)
    return 0


# ---------------- Parametric-tree builder ----------------

def _build_parametric_fn(node: Node, var_idx: dict[str, int],
                          const_counter: list[int]) -> Callable:
    """Recursively build f(args, consts) where consts is a 1-D vector
    indexed by pre-order Const position.

    Const leaves capture their index in const_counter (mutable list used
    as a counter) and read from `consts[i]` at call time. Var leaves
    capture their position in var_idx and read from `args[j]`.
    """
    if isinstance(node, Const):
        i = const_counter[0]
        const_counter[0] += 1
        return lambda args, consts, _i=i: consts[_i]
    if isinstance(node, Var):
        j = var_idx[node.name]
        return lambda args, consts, _j=j: args[_j]
    if isinstance(node, BinOp):
        fa = _build_parametric_fn(node.a, var_idx, const_counter)
        fb = _build_parametric_fn(node.b, var_idx, const_counter)
        op_fn = BIN_OP_FNS[node.op]
        return lambda args, consts: op_fn(fa(args, consts), fb(args, consts))
    if isinstance(node, UnOp):
        fa = _build_parametric_fn(node.a, var_idx, const_counter)
        op_fn = UN_OP_FNS[node.op]
        return lambda args, consts: op_fn(fa(args, consts))
    raise TypeError(f"_build_parametric_fn: unsupported node {type(node).__name__}")


# ---------------- Topology compile (jit + vmap) ----------------

def compile_topology(template: Node, var_names: Sequence[str]) -> Callable:
    """Compile a topology to a jit+vmap function.

    Returns a callable f(args, consts_batch) where:
      args         : tuple of arrays in var_names order
      consts_batch : shape [K, M] array of constants (K=batch size,
                     M=number of Consts in template); pass shape [K, 0]
                     if template has no Consts

    Output shape: [K, N] where N is the array length of any var.

    Cached by (topology_key(template), tuple(var_names)). Two trees
    with identical topology + matching var_names share the cache entry
    even if their constants differ.

    Raises ValueError if template has FunctionalOp.
    """
    if not is_pure_pointwise(template):
        raise ValueError(
            "compile_topology: only pure-pointwise topologies. Trees with "
            "FunctionalOp must use evaluate() (slower but correct)."
        )

    import jax

    var_names = tuple(var_names)
    key = (topology_key(template), var_names)
    if key in _TOPO_CACHE:
        return _TOPO_CACHE[key]

    var_idx = {v: i for i, v in enumerate(var_names)}
    counter = [0]
    raw_fn = _build_parametric_fn(template, var_idx, counter)

    # vmap over the leading axis of consts; args broadcast (in_axes=None)
    vmapped = jax.vmap(raw_fn, in_axes=(None, 0))
    jitted = jax.jit(vmapped)
    _TOPO_CACHE[key] = jitted
    return jitted


# ---------------- Population evaluation ----------------

def evaluate_population(
    trees: Sequence[Node],
    env: dict,
    var_names: Sequence[str] | None = None,
):
    """Evaluate a list of trees on `env` using topology-batched JAX.

    Groups trees by topology; for each topology, builds a [K_t, M_t]
    constants tensor and runs one vmapped jit call. Outputs are returned
    in the same order as the input `trees`.

    Parameters
    ----------
    trees : Sequence[Node]
        Pure-pointwise trees. Mixed trees raise ValueError.
    env : dict[name -> jax_array]
        Variable arrays. JAX arrays expected.
    var_names : Sequence[str] | None
        Variable order. If None, uses sorted(env.keys()).

    Returns
    -------
    list[jax_array]
        Outputs in tree-input order. Each output has shape [N].

    Performance
    -----------
    A population of K trees clustering into G topology groups (avg group
    size K/G) evaluates in roughly G separate vmapped jit calls. The
    larger the groups, the better the throughput. Late-GP populations
    typically have heavy topology clustering due to convergence; early-
    GP populations are more diverse (smaller groups).
    """
    import jax.numpy as jnp

    if var_names is None:
        var_names = sorted(env.keys())
    var_names = tuple(var_names)
    args = tuple(env[v] for v in var_names)

    # Group trees by topology, preserving original indices
    groups: dict[str, list[tuple[int, Node]]] = defaultdict(list)
    for i, tree in enumerate(trees):
        if not is_pure_pointwise(tree):
            raise ValueError(
                f"evaluate_population: tree at index {i} has FunctionalOp; "
                "Tier 3 only supports pure-pointwise trees."
            )
        groups[topology_key(tree)].append((i, tree))

    results: list = [None] * len(trees)

    for topo, items in groups.items():
        template = items[0][1]   # any tree of this topology works
        K = len(items)
        M = n_constants(template)

        consts_lists = [extract_constants(tree) for _, tree in items]
        if M == 0:
            # No constants — compute once, broadcast K times
            consts_batch = jnp.zeros((K, 0))
        else:
            consts_batch = jnp.asarray(consts_lists)   # [K, M]

        fn = compile_topology(template, var_names)
        y_batch = fn(args, consts_batch)               # [K, N]

        for k, (orig_i, _) in enumerate(items):
            results[orig_i] = y_batch[k]

    return results


def evaluate_population_stacked(
    trees: Sequence[Node],
    env: dict,
    var_names: Sequence[str] | None = None,
):
    """Same as evaluate_population, but returns a single [K, N] tensor in
    input order. One block_until_ready instead of K.

    This is the **correct API for the GP loop**: downstream code wants
    per-candidate losses, which compute naturally as
        loss[k] = mean(axis=-1)((y_pred[k] - y_true)**2)
    over a [K, N] tensor — one JAX op, one kernel, one sync.

    The per-tree-list API of `evaluate_population` is ergonomic but
    forces K separate Python-level slices + K block_until_ready calls,
    which adds ~100 ms of fixed Python overhead at K=200. Avoid it in
    hot paths.

    Returns
    -------
    jax_array of shape [K, N], rows in the original tree-input order.
    """
    import jax.numpy as jnp
    import numpy as np

    if var_names is None:
        var_names = sorted(env.keys())
    var_names = tuple(var_names)
    args = tuple(env[v] for v in var_names)
    K = len(trees)

    # Group trees by topology; remember each row's destination index
    groups: dict[str, list[tuple[int, Node]]] = defaultdict(list)
    for i, tree in enumerate(trees):
        if not is_pure_pointwise(tree):
            raise ValueError(
                f"evaluate_population_stacked: tree at index {i} has "
                "FunctionalOp; Tier 3 only supports pure-pointwise trees."
            )
        groups[topology_key(tree)].append((i, tree))

    # Build the inverse permutation (orig_idx -> concat-order position)
    concat_to_orig: list[int] = []
    for items in groups.values():
        concat_to_orig.extend(i for i, _ in items)
    inverse = np.zeros(K, dtype=np.int32)
    for concat_pos, orig_i in enumerate(concat_to_orig):
        inverse[orig_i] = concat_pos
    inverse_jax = jnp.asarray(inverse)

    # Evaluate each topology, collect [K_t, N] blocks
    blocks = []
    for items in groups.values():
        template = items[0][1]
        K_t = len(items)
        M = n_constants(template)
        if M == 0:
            consts_batch = jnp.zeros((K_t, 0))
        else:
            consts_batch = jnp.asarray([extract_constants(t) for _, t in items])
        fn = compile_topology(template, var_names)
        blocks.append(fn(args, consts_batch))   # [K_t, N]

    concat = jnp.concatenate(blocks, axis=0)    # [K, N] in concat order
    return concat[inverse_jax]                  # [K, N] in input order


class PopulationEvaluator:
    """Pre-compiled evaluator for a fixed population.

    Use this when you'll evaluate the same population on multiple `env`s
    (e.g., across const-opt iterations, walk-forward windows, or
    repeated benchmark timings). It does the topology grouping +
    constants packing + jit compilation ONCE in __init__; each __call__
    just dispatches the precomputed jitted functions on the new env.

    Usage:
        pe = PopulationEvaluator(trees, var_names=["x"])
        y_stack = pe(env_jax)          # [K, N] tensor in tree-input order

    Lifecycle: a `PopulationEvaluator` is invalidated as soon as any
    tree in the population is mutated (a new tree with a different
    topology would not be in the cache). The GP loop should rebuild it
    once per generation, or you can use `evaluate_population_stacked`
    which rebuilds the cache every call (cheap thanks to
    `_TOPO_CACHE`).
    """

    def __init__(self, trees: Sequence[Node], var_names: Sequence[str]):
        import jax.numpy as jnp
        import numpy as np

        var_names = tuple(var_names)
        self.var_names = var_names
        self.K = len(trees)

        # Group by topology
        groups: dict[str, list[tuple[int, Node]]] = defaultdict(list)
        for i, tree in enumerate(trees):
            if not is_pure_pointwise(tree):
                raise ValueError(
                    f"PopulationEvaluator: tree {i} has FunctionalOp; "
                    "Tier 3 only supports pure-pointwise trees."
                )
            groups[topology_key(tree)].append((i, tree))

        # Inverse permutation once
        concat_to_orig: list[int] = []
        for items in groups.values():
            concat_to_orig.extend(i for i, _ in items)
        inverse = np.zeros(self.K, dtype=np.int32)
        for concat_pos, orig_i in enumerate(concat_to_orig):
            inverse[orig_i] = concat_pos
        self._inverse = jnp.asarray(inverse)

        # Pre-allocate per-topology (consts_batch, compiled_fn)
        self._groups: list[tuple[object, Callable]] = []
        for items in groups.values():
            template = items[0][1]
            K_t = len(items)
            M = n_constants(template)
            if M == 0:
                consts_batch = jnp.zeros((K_t, 0))
            else:
                consts_batch = jnp.asarray(
                    [extract_constants(t) for _, t in items]
                )
            fn = compile_topology(template, var_names)
            self._groups.append((consts_batch, fn))

    def __call__(self, env: dict):
        """Evaluate the population on env. Returns [K, N] tensor in
        input order."""
        import jax.numpy as jnp
        args = tuple(env[v] for v in self.var_names)
        blocks = [fn(args, consts) for consts, fn in self._groups]
        concat = jnp.concatenate(blocks, axis=0)
        return concat[self._inverse]


def clear_topo_cache() -> None:
    """Drop all compiled topology entries."""
    _TOPO_CACHE.clear()


def topo_cache_size() -> int:
    return len(_TOPO_CACHE)


__all__ = [
    "topology_key", "extract_constants", "n_constants",
    "compile_topology", "evaluate_population", "evaluate_population_stacked",
    "PopulationEvaluator",
    "clear_topo_cache", "topo_cache_size",
]
