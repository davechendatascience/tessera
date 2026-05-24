"""Cross-tree subexpression materialization.

The bridge between `tessera.expression.cache` (per-Measure caching) and
`tessera.expression.batched` (population vmap on pure-pointwise trees).

The problem
-----------
A GP population of K trees often contains the SAME expensive subtree
(e.g., `FunctionalOp2D(Laplacian, image)`) in many candidates. The
Tier-3 batched JAX path evaluates each tree independently — so that
shared subtree is recomputed K times per generation. With K=60 and
~3 unique kernels in the population, we waste ~20× the FunctionalOp
compute.

The chess analog (per `docs/research/fit_as_perfect_info_game.md` §12):
this is the transposition-table lever for SR. A perfect-info engine
doesn't re-evaluate positions it has seen — neither should an SR
engine re-evaluate subtrees it has seen.

The solution
------------
`materialize_shared_subtrees(trees, env)`:

1. Walk the population, count subtree occurrences (by `canonical_key`)
2. For subtrees appearing >= threshold trees, pre-evaluate them once
   on the env via the standard `evaluate()` path
3. Bind each result to a synthetic `Var(_cached_k)`; rewrite trees
   to use the synthetic Var instead of recomputing
4. Return the (rewritten_trees, augmented_env)

After this, the rewritten trees may become pure-pointwise (the cached
subtree replaced by a Var), making them eligible for Tier-3 batched
JAX evaluation that wasn't accessible before.

Axiomatic design
----------------
This module adds no new evaluation primitives. It uses `evaluate()`
as a black-box and AST manipulation (same as `mutation.replace_at`).
The synthetic-Var rewrite preserves tree-shape contracts; the
augmented env contains pre-evaluated arrays of the same type the
original env held.

Extension points (anticipating future upgrades):
- `is_cacheable`: callable(Node) -> bool. Default: FunctionalOp /
  FunctionalOp2D nodes. Extension: compound pointwise expressions, or
  any user-supplied predicate.
- `canonical_key`: callable(Node) -> hashable. Default: `str(node)`.
  Extension: e-graph orbit-ID for true semantic equivalence (loose
  coupling to the future equality-saturation work).
- `cache`: optional `SubtreeCache` dict for cross-call persistence
  (e.g., across GP generations). Default: per-call only.

No circular dependencies: this module depends only on `tree` and
`cache` (depth 1 and 2 in the layering audit).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

from .tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    evaluate,
)


# ---------------- Defaults for extension points ----------------

def default_is_cacheable(node: Node) -> bool:
    """Default predicate: FunctionalOp / FunctionalOp2D are cacheable.

    These are the expensive evaluations (O(N · kernel_size)). Pointwise
    BinOps / UnOps are cheap enough that caching overhead may exceed
    the savings.
    """
    return isinstance(node, (FunctionalOp, FunctionalOp2D))


def default_canonical_key(node: Node) -> str:
    """Default canonical key: `str(node)`. Relies on tessera's
    deterministic tree-stringification.

    Catches syntactic identity but NOT semantic equivalence (a+b vs
    b+a get different keys). Pair with `simplify_canonical` before
    calling for AC-normalised matching. Future: replace with e-graph
    orbit-ID once equality saturation ships.
    """
    return str(node)


# ---------------- Subtree iteration ----------------

def iter_subtrees(node: Node) -> Iterable[Node]:
    """Yield every node in the tree (pre-order)."""
    yield node
    if isinstance(node, BinOp):
        yield from iter_subtrees(node.a)
        yield from iter_subtrees(node.b)
    elif isinstance(node, UnOp):
        yield from iter_subtrees(node.a)
    elif isinstance(node, FunctionalOp):
        for a in node.args:
            yield from iter_subtrees(a)
    elif isinstance(node, FunctionalOp2D):
        yield from iter_subtrees(node.arg)
    # Var / Const: leaves, no children


# ---------------- Rewriting ----------------

def _rewrite_tree(
    node: Node,
    replacements: dict[str, Var],
    canonical_key: Callable[[Node], str],
) -> Node:
    """Recursively rewrite `node`: any subtree whose canonical_key
    matches a replacement is replaced with that Var.

    Replacements applied at the highest-matching level (caller's choice
    of canonical_key controls match granularity)."""
    key = canonical_key(node)
    if key in replacements:
        return replacements[key]
    if isinstance(node, (Var, Const)):
        return node
    if isinstance(node, BinOp):
        return BinOp(node.op,
                     _rewrite_tree(node.a, replacements, canonical_key),
                     _rewrite_tree(node.b, replacements, canonical_key))
    if isinstance(node, UnOp):
        return UnOp(node.op,
                    _rewrite_tree(node.a, replacements, canonical_key))
    if isinstance(node, FunctionalOp):
        new_args = tuple(
            _rewrite_tree(a, replacements, canonical_key)
            for a in node.args
        )
        return FunctionalOp(node.functional, new_args)
    if isinstance(node, FunctionalOp2D):
        return FunctionalOp2D(
            node.measure_2d,
            _rewrite_tree(node.arg, replacements, canonical_key),
        )
    return node


# ---------------- Public API ----------------

def materialize_shared_subtrees(
    trees: Sequence[Node],
    env: dict,
    *,
    threshold: int = 2,
    is_cacheable: Callable[[Node], bool] | None = None,
    canonical_key: Callable[[Node], str] | None = None,
    cache: dict | None = None,
    var_prefix: str = "_cached_",
    fill_warmup: float = 0.0,
) -> tuple[list[Node], dict, dict]:
    """Pre-evaluate subtrees shared across the population.

    Walks `trees`, finds subtrees appearing in >= `threshold` trees
    (counted by `canonical_key`, default `str(node)`), pre-evaluates
    each on `env` once, and rewrites the trees to reference synthetic
    Vars instead of recomputing.

    Parameters
    ----------
    trees : Sequence[Node]
        The candidate population.
    env : dict[name -> array]
        Variable arrays.
    threshold : int
        Minimum number of trees a subtree must appear in to be
        materialized. Default 2: cache anything shared by two or
        more trees.
    is_cacheable : callable(Node) -> bool, optional
        Predicate selecting which subtrees are worth caching. Default
        `default_is_cacheable` (FunctionalOp / FunctionalOp2D).
    canonical_key : callable(Node) -> hashable, optional
        Maps a subtree to its identity. Default `default_canonical_key`
        (= `str(node)`). Use e-graph identity for semantic matching.
    cache : dict, optional
        External cache (`canonical_key -> array`). If provided, the
        function looks up here first and writes new materializations
        back. Enables cross-call reuse (e.g. across GP generations).
        Default None: per-call only.
    var_prefix : str
        Prefix for synthetic Var names (default `_cached_`). Choose so
        it does not clash with user vars.
    fill_warmup : float
        Forwarded to `evaluate(...)` for the subtree evaluation.

    Returns
    -------
    rewritten_trees : list[Node]
        The population with shared subtrees replaced by synthetic Vars.
    augmented_env : dict
        Original `env` plus the materialized subtree results, keyed by
        synthetic Var name.
    stats : dict
        Diagnostic info: keys "n_materialized" (how many subtrees got
        cached), "n_replacements" (total subtree-substitutions across
        population), "cache_hits" (subtrees served from external cache),
        "subtrees_per_key" (Counter of usage counts).
    """
    if is_cacheable is None:
        is_cacheable = default_is_cacheable
    if canonical_key is None:
        canonical_key = default_canonical_key

    # 1. Count subtree occurrences across the population.
    #    We count occurrences ACROSS DISTINCT TREES, not within one tree,
    #    so a tree with `f(x) + f(x)` counts `f(x)` once, not twice.
    counts: Counter = Counter()
    for tree in trees:
        seen_in_tree: set = set()
        for sub in iter_subtrees(tree):
            if not is_cacheable(sub):
                continue
            k = canonical_key(sub)
            if k in seen_in_tree:
                continue
            seen_in_tree.add(k)
            counts[k] += 1

    # 2. Pick which subtrees to materialize.
    to_cache = {k for k, c in counts.items() if c >= threshold}

    # 3. Find a representative subtree object for each cached key.
    representative: dict[str, Node] = {}
    for tree in trees:
        for sub in iter_subtrees(tree):
            if not is_cacheable(sub):
                continue
            k = canonical_key(sub)
            if k in to_cache and k not in representative:
                representative[k] = sub

    # 4. Materialize each, using the external cache where possible.
    augmented_env = dict(env)
    replacements: dict[str, Var] = {}
    n_cache_hits = 0
    for i, (k, sub) in enumerate(representative.items()):
        if cache is not None and k in cache:
            result = cache[k]
            n_cache_hits += 1
        else:
            result = evaluate(sub, env, fill_warmup=fill_warmup)
            if cache is not None:
                cache[k] = result
        var_name = f"{var_prefix}{i}"
        augmented_env[var_name] = result
        replacements[k] = Var(var_name)

    # 5. Rewrite each tree.
    if replacements:
        rewritten = [
            _rewrite_tree(t, replacements, canonical_key) for t in trees
        ]
    else:
        rewritten = list(trees)

    # 6. Count total substitutions across population for diagnostics.
    n_replacements = 0
    for tree in trees:
        for sub in iter_subtrees(tree):
            if canonical_key(sub) in to_cache:
                n_replacements += 1

    stats = {
        "n_materialized": len(replacements),
        "n_replacements": n_replacements,
        "cache_hits": n_cache_hits,
        "subtrees_per_key": dict(counts),
    }
    return rewritten, augmented_env, stats


__all__ = [
    "materialize_shared_subtrees",
    "iter_subtrees",
    "default_is_cacheable",
    "default_canonical_key",
]
