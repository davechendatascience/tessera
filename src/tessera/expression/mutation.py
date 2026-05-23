"""Random tree generation and mutation operators for the GP search.

Mutation operators
------------------
Six pointwise-tree ops + one functional-specific:

  - subtree_swap       : pick a random subtree, replace with a fresh random subtree
  - subtree_crossover  : splice a subtree from parent B into parent A
  - constant_jitter    : multiply every Const in the tree by exp(N(0, σ))
  - term_insert        : wrap root in `root +/- random_subtree`
  - term_delete        : if a subtree is Add/Sub, drop one operand
  - op_swap            : swap a pointwise operator for a related one
                         (add↔sub, mul↔div, min↔max, tanh↔sign, abs↔neg)
  - measure_mutate     : (new for tessera) inside a FunctionalOp, replace
                         the measure with a perturbed/different one

Each operator returns a new tree (Node is immutable). The top-level
`mutate()` dispatcher picks one operator at random (weighted by
OP_WEIGHTS), applies it, and validates the result. On failure it retries.

Random-tree generation
----------------------
`random_tree(rng, feature_names, max_depth, ...)` builds a syntactically
valid tree from scratch. Used to seed the GP population and to provide
fresh material for `subtree_swap` and `term_insert`.

Validators
----------
`validate_tree(node, feature_names)` returns None if valid, else an
error string. Checks: depth ≤ MAX_DEPTH, complexity ≤ MAX_COMPLEXITY,
all Var leaves reference known features, no constant blow-ups.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence

from .measure import (
    Measure, Atom,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
    DENSITY_FAMILIES,
)
from .measure_2d import (
    Measure2D, Atom2D,
    measure_2d_atomic, measure_2d_separable,
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
)
from .functional import (
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
)
from .tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D, Node,
    BIN_OPS, UN_OPS, complexity, depth, used_features, iter_subtrees, replace_at,
)


# ---------------- Constraints ----------------

MAX_DEPTH = 8
MAX_COMPLEXITY = 40
MAX_CONST_MAGNITUDE = 1e4


def validate_tree(node: Node, feature_names: set[str]) -> Optional[str]:
    """Return None if valid, else a short error string."""
    if depth(node) > MAX_DEPTH:
        return f"depth {depth(node)} > MAX_DEPTH {MAX_DEPTH}"
    if complexity(node) > MAX_COMPLEXITY:
        return f"complexity {complexity(node)} > MAX_COMPLEXITY {MAX_COMPLEXITY}"
    unknown = used_features(node) - feature_names
    if unknown:
        return f"unknown features: {sorted(unknown)}"
    for sub in iter_subtrees(node):
        if isinstance(sub, Const):
            if abs(sub.value) > MAX_CONST_MAGNITUDE or not math.isfinite(sub.value):
                return f"constant magnitude blow-up: {sub.value}"
    return None


# ---------------- Random leaf / measure constructors ----------------

def _random_const(rng: random.Random) -> Const:
    """A small float in a useful range — log-uniform on (0.01, 10), random sign."""
    sign = rng.choice([-1.0, 1.0])
    mag = math.exp(rng.uniform(math.log(0.01), math.log(10.0)))
    return Const(round(sign * mag, 6))


def _random_var(rng: random.Random, feature_names: list[str]) -> Var:
    return Var(rng.choice(feature_names))


# Candidate halflives and lag scales for the GP to choose from
_HALFLIFE_CANDIDATES = (3.0, 6.0, 12.0, 24.0, 48.0, 168.0, 336.0, 720.0)
_LAG_CANDIDATES = (1, 2, 3, 6, 12, 24, 48, 72, 168, 336)
_WINDOW_CANDIDATES = (3, 6, 12, 24, 48, 168, 336)
_POWER_LAW_ALPHA_CANDIDATES = (1.2, 1.5, 2.0, 2.5, 3.0)


def random_measure(rng: random.Random) -> Measure:
    """Sample a random measure: lag / diff / ema / roll_mean / power_law /
    signed_sum. Mixture roughly favors the most-useful families."""
    r = rng.random()
    if r < 0.20:
        return measure_lag(rng.choice(_LAG_CANDIDATES))
    if r < 0.35:
        return measure_diff(rng.choice(_LAG_CANDIDATES))
    if r < 0.65:
        return measure_ema(rng.choice(_HALFLIFE_CANDIDATES))
    if r < 0.80:
        return measure_roll_mean(rng.choice(_WINDOW_CANDIDATES))
    if r < 0.90:
        return measure_power_law(
            scale=rng.choice(_HALFLIFE_CANDIDATES),
            alpha=rng.choice(_POWER_LAW_ALPHA_CANDIDATES),
        )
    # Otherwise a small signed_sum
    n_atoms = rng.randint(2, 4)
    weights_and_lags = [
        (round(rng.uniform(-1.5, 1.5), 4), rng.choice(_LAG_CANDIDATES))
        for _ in range(n_atoms)
    ]
    return measure_signed_sum(weights_and_lags)


def random_measure_2d(rng: random.Random) -> Measure2D:
    """Sample a random 2-D measure for use in FunctionalOp2D nodes.

    The mixture is biased toward PDE-discovery primitives — Laplacian,
    grad_x, diff_t, and small custom atomic stencils — but also includes
    separable-density combinations.
    """
    r = rng.random()
    if r < 0.30:
        return measure_2d_laplacian_5pt()
    if r < 0.45:
        return measure_2d_grad_x()
    if r < 0.60:
        return measure_2d_diff_t(lag_t=rng.choice([1, 2, 3, 6, 24]))
    if r < 0.80:
        # Separable density: 1-D measure along time × 1-D along space
        return measure_2d_separable(
            measure_t=random_measure(rng),
            measure_x=random_measure(rng),
        )
    # Small custom atomic stencil — 2-5 atoms at small lags
    n_atoms = rng.randint(2, 5)
    atoms = []
    for _ in range(n_atoms):
        w = round(rng.uniform(-1.5, 1.5), 4)
        lt = rng.choice([0, 0, 0, 1, 2])     # time mostly 0 for spatial ops
        lx = rng.choice([-2, -1, 0, 0, +1, +2])
        atoms.append((w, lt, lx))
    return measure_2d_atomic(atoms)


def random_functional(rng: random.Random) -> Functional:
    """Sample a random Functional: 50% Linear, 25% Bilinear, 25% Volterra2."""
    r = rng.random()
    if r < 0.5:
        return LinearFunctional(measure=random_measure(rng))
    if r < 0.75:
        return SeparableBilinear(
            measure_a=random_measure(rng),
            measure_b=random_measure(rng),
        )
    return Volterra2(
        measure_a=random_measure(rng),
        measure_b=random_measure(rng),
    )


# ---------------- Random tree builder ----------------

def random_tree(
    rng: random.Random,
    feature_names: list[str],
    *,
    max_depth: int = 4,
    leaf_prob: float = 0.10,
    unop_prob: float = 0.20,
    binop_prob: float = 0.45,
    funcop_prob: float = 0.25,
    enable_2d: bool = False,
    pointwise_only: bool = False,
) -> Node:
    """Generate a random tree with bounded depth.

    At each non-terminal level, the probabilities (leaf_prob, unop_prob,
    binop_prob, funcop_prob) must sum to 1 (we normalise if not).
    At depth = max_depth, only leaves are generated.

    Parameters
    ----------
    enable_2d : bool
        If True, the funcop_prob branch chooses between FunctionalOp (1-D)
        and FunctionalOp2D (2-D) with 50/50 odds. The caller is responsible
        for ensuring the env passed to evaluate() is 2-D-compatible.
    pointwise_only : bool
        If True, no FunctionalOp / FunctionalOp2D nodes are generated; the
        funcop_prob mass is redistributed to binop_prob. Use for pure
        ODE rediscovery where the target is a closed-form expression in
        the input variables (e.g., Lorenz-63: dx/dt = 10(y-x)).
    """
    if pointwise_only:
        # Redistribute funcop weight into binop, keep proportions otherwise
        binop_prob = binop_prob + funcop_prob
        funcop_prob = 0.0

    if max_depth <= 1:
        if rng.random() < 0.80:
            return _random_var(rng, feature_names)
        return _random_const(rng)

    total = leaf_prob + unop_prob + binop_prob + funcop_prob
    leaf_p = leaf_prob / total
    unop_p = unop_prob / total
    binop_p = binop_prob / total

    r = rng.random()
    if r < leaf_p:
        if rng.random() < 0.80:
            return _random_var(rng, feature_names)
        return _random_const(rng)

    if r < leaf_p + unop_p:
        op = rng.choice(UN_OPS)
        return UnOp(op, random_tree(rng, feature_names, max_depth=max_depth - 1,
                                    enable_2d=enable_2d, pointwise_only=pointwise_only))

    if r < leaf_p + unop_p + binop_p:
        op = rng.choice(BIN_OPS)
        return BinOp(
            op,
            random_tree(rng, feature_names, max_depth=max_depth - 1,
                        enable_2d=enable_2d, pointwise_only=pointwise_only),
            random_tree(rng, feature_names, max_depth=max_depth - 1,
                        enable_2d=enable_2d, pointwise_only=pointwise_only),
        )

    # FunctionalOp branch (only reachable when pointwise_only=False) —
    # choose 1-D or 2-D.
    if enable_2d and rng.random() < 0.5:
        m2d = random_measure_2d(rng)
        arg = random_tree(rng, feature_names, max_depth=max_depth - 1,
                          enable_2d=enable_2d, pointwise_only=pointwise_only)
        return FunctionalOp2D(m2d, arg)

    fn = random_functional(rng)
    args = tuple(
        random_tree(rng, feature_names, max_depth=max_depth - 1,
                    enable_2d=enable_2d, pointwise_only=pointwise_only)
        for _ in range(fn.n_inputs)
    )
    return FunctionalOp(fn, args)


# ---------------- Mutation operators ----------------

def subtree_swap(
    tree: Node,
    rng: random.Random,
    feature_names: list[str],
    *,
    subtree_max_depth: int = 3,
    pointwise_only: bool = False,
    enable_2d: bool = False,
) -> Node:
    """Replace a random subtree with a fresh random subtree."""
    n_subs = sum(1 for _ in iter_subtrees(tree))
    idx = rng.randrange(n_subs)
    fresh = random_tree(
        rng, feature_names, max_depth=subtree_max_depth,
        pointwise_only=pointwise_only, enable_2d=enable_2d,
    )
    return replace_at(tree, idx, fresh)


def subtree_crossover(
    parent_a: Node,
    parent_b: Node,
    rng: random.Random,
) -> Node:
    """Cut a random subtree from B and splice into A at a random point."""
    b_subs = list(iter_subtrees(parent_b))
    donor = rng.choice(b_subs)
    n_subs_a = sum(1 for _ in iter_subtrees(parent_a))
    idx = rng.randrange(n_subs_a)
    return replace_at(parent_a, idx, donor)


def constant_jitter(
    tree: Node,
    rng: random.Random,
    *,
    sigma: float = 0.1,
) -> Node:
    """Multiply every Const value by exp(N(0, sigma))."""
    def visit(n: Node) -> Node:
        if isinstance(n, Const):
            mult = math.exp(rng.gauss(0.0, sigma))
            return Const(round(n.value * mult, 6))
        if isinstance(n, (Var,)):
            return n
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.a))
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.a), visit(n.b))
        if isinstance(n, FunctionalOp):
            return FunctionalOp(n.functional, tuple(visit(a) for a in n.args))
        raise TypeError(type(n))
    return visit(tree)


def term_insert(
    tree: Node,
    rng: random.Random,
    feature_names: list[str],
    *,
    pointwise_only: bool = False,
    enable_2d: bool = False,
) -> Node:
    """Wrap root in `root +/- random_subtree`. Grows complexity by ~3-5."""
    op = rng.choice(["add", "sub"])
    addend = random_tree(
        rng, feature_names, max_depth=2,
        pointwise_only=pointwise_only, enable_2d=enable_2d,
    )
    return BinOp(op, tree, addend)


def term_delete(
    tree: Node,
    rng: random.Random,
) -> Node:
    """If a random subtree is Add/Sub, drop one of its operands."""
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, BinOp) and s.op in ("add", "sub")
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    assert isinstance(sub, BinOp)
    replacement = sub.a if rng.random() < 0.5 else sub.b
    return replace_at(tree, idx, replacement)


_OP_SWAP_GROUPS: list[set[str]] = [
    {"add", "sub"},
    {"mul", "div"},
    {"min", "max"},
    {"tanh", "sign"},
    {"abs", "neg"},
]


def op_swap(tree: Node, rng: random.Random) -> Node:
    """Swap a pointwise operator with a related one (add↔sub etc.)."""
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, (BinOp, UnOp))
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    group = next((g for g in _OP_SWAP_GROUPS if sub.op in g), None)
    if group is None:
        return tree
    others = [o for o in group if o != sub.op]
    if not others:
        return tree
    new_op = rng.choice(others)
    if isinstance(sub, BinOp):
        return replace_at(tree, idx, BinOp(new_op, sub.a, sub.b))
    return replace_at(tree, idx, UnOp(new_op, sub.a))


def measure_2d_mutate(tree: Node, rng: random.Random) -> Node:
    """Inside a FunctionalOp2D, replace the wrapped Measure2D with a new
    random one. Companion to measure_mutate, for PDE-discovery trees."""
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, FunctionalOp2D)
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    assert isinstance(sub, FunctionalOp2D)
    new_m2d = random_measure_2d(rng)
    return replace_at(tree, idx, FunctionalOp2D(new_m2d, sub.arg))


def measure_mutate(tree: Node, rng: random.Random) -> Node:
    """Inside a FunctionalOp, replace the wrapped measure(s) with new
    random ones.

    This is the tessera-specific operator. The GP can mutate a
    measure's structure (halflife, lag positions, atom weights) without
    rebuilding the entire tree topology.
    """
    candidates = [
        (i, s) for i, s in enumerate(iter_subtrees(tree))
        if isinstance(s, FunctionalOp)
    ]
    if not candidates:
        return tree
    idx, sub = rng.choice(candidates)
    assert isinstance(sub, FunctionalOp)
    fn = sub.functional

    # Replace one of the measures inside the functional
    if isinstance(fn, LinearFunctional):
        new_fn = LinearFunctional(measure=random_measure(rng))
    elif isinstance(fn, SeparableBilinear):
        # Mutate one of the two measures (50/50)
        if rng.random() < 0.5:
            new_fn = SeparableBilinear(
                measure_a=random_measure(rng), measure_b=fn.measure_b,
            )
        else:
            new_fn = SeparableBilinear(
                measure_a=fn.measure_a, measure_b=random_measure(rng),
            )
    elif isinstance(fn, Volterra2):
        if rng.random() < 0.5:
            new_fn = Volterra2(
                measure_a=random_measure(rng), measure_b=fn.measure_b,
            )
        else:
            new_fn = Volterra2(
                measure_a=fn.measure_a, measure_b=random_measure(rng),
            )
    else:
        return tree   # unknown functional class, skip

    return replace_at(tree, idx, FunctionalOp(new_fn, sub.args))


# ---------------- Top-level dispatcher ----------------

def collapse_functional_chain(tree: Node, rng: random.Random) -> Node:
    """Collapse a nested LinearFunctional chain via measure convolution.

    Pattern matched:
        FunctionalOp(LinearFunctional(μ), (FunctionalOp(LinearFunctional(ν), (x,)),))

    Replaced with:
        FunctionalOp(LinearFunctional(μ ∗ ν), (x,))

    This is the measure-algebra identity `L_μ(L_ν(x)) ≡ L_{μ∗ν}(x)`
    from docs/research_notes/measure_theory_and_perfect_info.md §3.3.
    Strictly reduces tree node count (typically by ~3 nodes), and the
    composed measure is automatically canonicalised by
    `Measure.__post_init__`.

    Raises ValueError if the tree contains no such pattern (the
    `mutate()` dispatcher catches and retries with a different op).
    """
    # Collect pre-order indices of nodes that match the pattern
    candidates: list[int] = []
    for idx, sub in enumerate(iter_subtrees(tree)):
        if (
            isinstance(sub, FunctionalOp)
            and isinstance(sub.functional, LinearFunctional)
            and len(sub.args) == 1
            and isinstance(sub.args[0], FunctionalOp)
            and isinstance(sub.args[0].functional, LinearFunctional)
        ):
            candidates.append(idx)
    if not candidates:
        raise ValueError("no LinearFunctional chain to collapse")
    chosen_idx = rng.choice(candidates)

    # Find the actual node at chosen_idx
    chosen_node: Node | None = None
    for idx, sub in enumerate(iter_subtrees(tree)):
        if idx == chosen_idx:
            chosen_node = sub
            break
    assert chosen_node is not None
    assert isinstance(chosen_node, FunctionalOp)
    inner = chosen_node.args[0]
    assert isinstance(inner, FunctionalOp)

    # Build the collapsed node: L_{μ ∗ ν}(x) where x = inner.args[0]
    composed_measure = chosen_node.functional.measure.compose(
        inner.functional.measure
    )
    new_functional = LinearFunctional(measure=composed_measure)
    new_node = FunctionalOp(new_functional, inner.args)
    return replace_at(tree, chosen_idx, new_node)


OP_WEIGHTS = {
    "subtree_swap":              0.20,
    "subtree_crossover":         0.20,
    "constant_jitter":           0.15,
    "term_insert":                0.10,
    "term_delete":                0.10,
    "op_swap":                    0.10,
    "measure_mutate":            0.07,
    "measure_2d_mutate":         0.03,
    "collapse_functional_chain": 0.05,
}


def mutate(
    parents: list[Node],
    rng: random.Random,
    feature_names: list[str],
    *,
    max_attempts: int = 5,
    pointwise_only: bool = False,
    enable_2d: bool = False,
) -> Optional[Node]:
    """Produce one valid offspring from a list of parents.

    Returns the new tree, or None if all attempts violated constraints.

    When `pointwise_only=True`, the dispatcher skips measure_mutate and
    measure_2d_mutate (no functional ops to mutate) and passes the flag
    to subtree_swap / term_insert (which call random_tree).
    """
    if not parents:
        raise ValueError("mutate needs at least one parent")
    feat_set = set(feature_names)
    keys = list(OP_WEIGHTS.keys())
    weights = list(OP_WEIGHTS.values())

    for _ in range(max_attempts):
        op_name = rng.choices(keys, weights=weights, k=1)[0]
        if pointwise_only and op_name in (
            "measure_mutate", "measure_2d_mutate", "collapse_functional_chain",
        ):
            # No FunctionalOp / FunctionalOp2D in pointwise-only trees.
            continue
        try:
            if op_name == "subtree_swap":
                child = subtree_swap(
                    parents[0], rng, feature_names,
                    pointwise_only=pointwise_only, enable_2d=enable_2d,
                )
            elif op_name == "subtree_crossover":
                if len(parents) < 2:
                    continue
                child = subtree_crossover(parents[0], parents[1], rng)
            elif op_name == "constant_jitter":
                child = constant_jitter(parents[0], rng)
            elif op_name == "term_insert":
                child = term_insert(
                    parents[0], rng, feature_names,
                    pointwise_only=pointwise_only, enable_2d=enable_2d,
                )
            elif op_name == "term_delete":
                child = term_delete(parents[0], rng)
            elif op_name == "op_swap":
                child = op_swap(parents[0], rng)
            elif op_name == "measure_mutate":
                child = measure_mutate(parents[0], rng)
            elif op_name == "measure_2d_mutate":
                child = measure_2d_mutate(parents[0], rng)
            elif op_name == "collapse_functional_chain":
                child = collapse_functional_chain(parents[0], rng)
            else:
                continue
        except Exception:
            continue
        if validate_tree(child, feat_set) is None:
            return child
    return None


__all__ = [
    "MAX_DEPTH", "MAX_COMPLEXITY", "MAX_CONST_MAGNITUDE",
    "validate_tree",
    "random_measure", "random_measure_2d", "random_functional", "random_tree",
    "subtree_swap", "subtree_crossover", "constant_jitter",
    "term_insert", "term_delete", "op_swap",
    "measure_mutate", "measure_2d_mutate", "collapse_functional_chain",
    "OP_WEIGHTS", "mutate",
]
