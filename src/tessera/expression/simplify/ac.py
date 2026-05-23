"""AC (Associative-Commutative) normalisation for tessera Expr trees.

Sorts the children of commutative ops and flattens chains of
associative ops into a canonical left-leaning shape. Result: `a + b`
and `b + a` produce the same tree; `(a + b) + c`, `a + (b + c)`, and
`(c + b) + a` all canonicalise to the same form.

Why this matters
----------------
Without AC normalisation, parsimony regularisation is "distorted by
arbitrary syntactic differences" (per the perplexity research note):
two semantically-identical expressions get DIFFERENT complexity
scores because their tree shape differs. Cache hit rate also suffers
(`a + b` and `b + a` miss each other in the FunctionalCache).

After normalisation:
- Pareto front shows each equivalence class once (no `a+b` AND `b+a`)
- Cache hits improve on commutative subtrees
- Constant folding becomes more aggressive (constants cluster after
  sort, so `2 + x + 3` after AC norm has shape `(x + 2) + 3` or
  similar where the consts are adjacent and a subsequent
  `simplify()` collapses them to `x + 5`)

AC ops
------
Commutative AND associative under tessera's evaluator semantics:
  - `add`     (a + b = b + a, (a+b)+c = a+(b+c))
  - `mul`     (a * b = b * a, (a*b)*c = a*(b*c))
  - `min`     (np.minimum is comm+assoc)
  - `max`     (np.maximum is comm+assoc)

NOT in this set:
  - `sub`, `div` — not commutative (a-b ≠ b-a)
  - `gt`, `lt`, `ge`, `le` — not commutative
"""
from __future__ import annotations

from ..tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    complexity,
)


# Operators that are commutative AND associative under tessera's semantics
_AC_OPS = frozenset({"add", "mul", "min", "max"})


def _flatten_assoc(node: Node, op: str) -> list[Node]:
    """Recursively flatten a chain of `op` BinOps into a list of leaf-like
    children. `(a op b) op c` → `[a, b, c]`. Non-matching nodes pass through
    untouched (treated as single leaves)."""
    if isinstance(node, BinOp) and node.op == op:
        return _flatten_assoc(node.a, op) + _flatten_assoc(node.b, op)
    return [node]


def _build_left_leaning(children: list[Node], op: str) -> Node:
    """Rebuild a left-leaning binary tree from a list of children.
    `[a, b, c, d]` → `((a op b) op c) op d`."""
    if not children:
        raise ValueError("can't build a tree from zero children")
    if len(children) == 1:
        return children[0]
    result = children[0]
    for child in children[1:]:
        result = BinOp(op, result, child)
    return result


def _node_sort_key(node: Node) -> tuple:
    """Total order on Nodes for canonical-form sorting.

    Primary key: complexity (smallest leaves first).
    Tie-breaker: string representation (lexicographic).

    This puts `Var` and `Const` (cx=1) before larger subtrees, giving
    readable forms like `x + 2*y + cos(z)` rather than
    `cos(z) + 2*y + x`.
    """
    return (complexity(node), str(node))


def simplify_ac(node: Node) -> Node:
    """Bottom-up AC normalisation.

    For each commutative-associative BinOp, the subtree is flattened,
    its children are sorted by `_node_sort_key`, and a canonical
    left-leaning tree is rebuilt.

    Non-AC nodes (sub, div, indicators, UnOp, FunctionalOp,
    FunctionalOp2D) are recursed into but their structure is
    preserved as-is.

    Idempotent: `simplify_ac(simplify_ac(t)) == simplify_ac(t)` for
    all trees t.

    Note on composition with the rule-based simplifier
    --------------------------------------------------
    The convention in tessera is `simplify_canonical = simplify ∘
    simplify_ac` — AC normalisation FIRST, then rule-based folds.
    Running them in this order lets constants cluster after the AC
    sort (since Const(x) and Const(y) become adjacent in `(... +
    Const(x) + Const(y))`, and the rule-based simplifier then folds
    them via the BinOp(op, Const, Const) → Const(op(a,b)) rule.
    """
    if isinstance(node, (Var, Const)):
        return node

    if isinstance(node, UnOp):
        return UnOp(node.op, simplify_ac(node.a))

    if isinstance(node, BinOp):
        a = simplify_ac(node.a)
        b = simplify_ac(node.b)
        op = node.op
        if op in _AC_OPS:
            children = _flatten_assoc(BinOp(op, a, b), op)
            children.sort(key=_node_sort_key)
            return _build_left_leaning(children, op)
        return BinOp(op, a, b)

    if isinstance(node, FunctionalOp):
        new_args = tuple(simplify_ac(arg) for arg in node.args)
        return FunctionalOp(node.functional, new_args)

    if isinstance(node, FunctionalOp2D):
        return FunctionalOp2D(node.measure_2d, simplify_ac(node.arg))

    raise TypeError(type(node))
