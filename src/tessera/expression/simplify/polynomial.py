"""Polynomial canonicalisation — additive like-term collection.

Target shape: sums of monomials `Σ c_i · (x_{a_i})^{d_{a_i}} · (x_{b_i})^{d_{b_i}} · ...`

What this canonicaliser does
----------------------------
- Flatten add/sub chains.
- For each summand, attempt to match as `coef · product(var_k^degree_k)`.
- Collect coefficients by exponent-tuple key. Sum like terms.
- Drop zero-coefficient terms.
- Re-emit canonical sum tree: monomials sorted by total degree then
  lexicographic on exponent tuple; opaque (non-monomial) summands
  appended at the end in sorted-by-str order.

What it does NOT do (out of scope; future ships may extend)
-----------------------------------------------------------
- Factoring: `x^2 - 1` → `(x+1)(x-1)`
- Expansion: `(x+1)(x-1)` → `x^2 - 1`
- Trigonometric identities: `sin²+cos² → 1`
- Series recognition: `x - x³/6 + x⁵/120` → `sin(x)`
- General algebraic CAS operations

The boundary is deliberate: the canonicaliser ONLY handles the shape
produced by `tessera.search.sufficient_stats.polish_tree_with_polynomial_term`,
which appends `Σ c_k · monomial_k` onto an existing tree. Within that
shape, it collapses redundancy completely. Outside the shape, summands
pass through as opaque, preserving semantics.

Cost-of-extension
-----------------
Adding a new monomial-flavoured pattern (cross-terms with constants,
ratios `c/x`, etc.) is a single new branch in `_match_monomial`. The
emit-side machinery stays unchanged. Each extension is well-bounded;
the module's reasoning surface stays small.

Composition
-----------
`simplify_polynomial` is idempotent and monotone-in-complexity (each
output's cx ≤ input's cx, equality on already-canonical inputs).
Standalone idiom: `simplify_polynomial(simplify_canonical(tree))`.
Convenience wrapper `simplify_full(tree)` in the package __init__.
"""
from __future__ import annotations

from ..tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
)


# ----------------------------------------------------------------------
# Sum flattening
# ----------------------------------------------------------------------

def _flatten_sum(node: Node) -> list[tuple[int, Node]]:
    """Walk add/sub chains; return list of (sign, leaf_node) pairs.

    `(a + b) - c` becomes `[(+1, a), (+1, b), (-1, c)]`. Subtraction
    on the RIGHT side of sub flips the sign of THAT subterm only.
    """
    out: list[tuple[int, Node]] = []

    def go(n: Node, sign: int) -> None:
        if isinstance(n, BinOp) and n.op == "add":
            go(n.a, sign)
            go(n.b, sign)
        elif isinstance(n, BinOp) and n.op == "sub":
            go(n.a, sign)
            go(n.b, -sign)
        else:
            out.append((sign, n))

    go(node, +1)
    return out


# ----------------------------------------------------------------------
# Monomial matcher
# ----------------------------------------------------------------------

ExponentTuple = tuple[tuple[str, int], ...]
"""Sorted tuple of (var_name, degree) pairs. ((), ) is the constant
monomial (degree 0)."""


def _match_monomial(node: Node) -> tuple[float, ExponentTuple] | None:
    """Try to interpret `node` as `coef · product(var_k^degree_k)`.

    Returns (coef, exponents) where exponents is sorted by var name,
    or None if `node` is not a monomial-shaped subtree.

    Handles:
      Const(c)                      → (c, ())
      Var(x)                        → (1.0, (('x', 1),))
      UnOp(neg, m)  if m is monomial → (-coef, exponents)
      BinOp(mul, a, b)  if both are monomials → product

    Anything else (add, sub, div, pow, sin, sqrt, FunctionalOp, ...)
    returns None — the summand is "opaque" and will be preserved
    verbatim in the canonical output.
    """
    if isinstance(node, Const):
        return (float(node.value), ())
    if isinstance(node, Var):
        return (1.0, ((node.name, 1),))
    if isinstance(node, UnOp) and node.op == "neg":
        sub = _match_monomial(node.a)
        if sub is None:
            return None
        coef, exps = sub
        return (-coef, exps)
    if isinstance(node, BinOp) and node.op == "mul":
        left = _match_monomial(node.a)
        right = _match_monomial(node.b)
        if left is None or right is None:
            return None
        coef = left[0] * right[0]
        bag: dict[str, int] = {}
        for v, d in left[1]:
            bag[v] = bag.get(v, 0) + d
        for v, d in right[1]:
            bag[v] = bag.get(v, 0) + d
        # Drop degree-0 entries (shouldn't appear, but guard for it)
        exps = tuple(sorted((v, d) for v, d in bag.items() if d != 0))
        return (coef, exps)
    return None


# ----------------------------------------------------------------------
# Monomial emission
# ----------------------------------------------------------------------

def _emit_monomial(coef: float, exponents: ExponentTuple) -> Node:
    """Build a canonical tree representing `coef · product(var^degree)`.

    Conventions chosen for idempotency — `_match_monomial(_emit_monomial(c, e))`
    must recover `(c, e)` exactly:

      coef=0           → Const(0.0)   (caller should usually filter first)
      degree 0 (const) → Const(coef)
      coef = +1        → bare product (no leading Const(1)·)
      coef = -1        → UnOp(neg, product)
      otherwise        → BinOp(mul, Const(coef), product)

    Variable product is a left-folded multiplication chain. For
    (x, 2, y, 1): `((x * x) * y)`. Order follows `exponents`'
    sort order (alphabetical by var name).
    """
    if not exponents:
        return Const(float(coef))
    parts: list[Node] = []
    for var, deg in exponents:
        for _ in range(deg):
            parts.append(Var(var))
    if not parts:
        return Const(float(coef))
    product: Node = parts[0]
    for p in parts[1:]:
        product = BinOp("mul", product, p)
    if abs(coef - 1.0) < 1e-12:
        return product
    if abs(coef + 1.0) < 1e-12:
        return UnOp("neg", product)
    return BinOp("mul", Const(float(coef)), product)


# ----------------------------------------------------------------------
# Sum canonicalisation
# ----------------------------------------------------------------------

def _canonicalize_sum(node: Node) -> Node:
    """Flatten the sum tree, collect like monomials, re-emit canonical."""
    summands = _flatten_sum(node)
    # Recurse into each summand BEFORE matching — a summand might
    # itself contain a polynomial sub-sum (e.g., `mul(c, add(x, x))`
    # won't reach here, but `unop(neg, add(x, x))` will).
    summands = [(sign, simplify_polynomial(n)) for sign, n in summands]

    bag: dict[ExponentTuple, float] = {}
    opaque: list[Node] = []

    for sign, s in summands:
        m = _match_monomial(s)
        if m is None:
            opaque.append(s if sign > 0 else UnOp("neg", s))
            continue
        coef, exps = m
        coef = sign * coef
        bag[exps] = bag.get(exps, 0.0) + coef

    # Sort key: degree-total first, then exponent tuple lex.
    def _sort_key(exps: ExponentTuple) -> tuple:
        total_degree = sum(d for _, d in exps)
        return (total_degree, exps)

    terms: list[Node] = []
    for exps in sorted(bag.keys(), key=_sort_key):
        coef = bag[exps]
        if abs(coef) < 1e-12:
            continue
        terms.append(_emit_monomial(coef, exps))

    # Opaque terms after polynomial terms; sort by str for stability.
    opaque.sort(key=str)
    terms.extend(opaque)

    if not terms:
        return Const(0.0)
    if len(terms) == 1:
        return terms[0]
    result = terms[0]
    for t in terms[1:]:
        result = BinOp("add", result, t)
    return result


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def simplify_polynomial(node: Node) -> Node:
    """Canonicalise additive polynomial subtrees throughout the tree.

    Bottom-up: recurses into all subtrees and rewrites add/sub chains
    in canonical sum-of-monomials form. Non-additive subtrees pass
    through unchanged.

    Idempotent. Monotone in complexity (`complexity(out) ≤
    complexity(in)`, with equality only when `in` is already
    canonical).

    See module docstring for the precise scope and out-of-scope items.
    """
    if isinstance(node, (Var, Const)):
        return node
    if isinstance(node, UnOp):
        return UnOp(node.op, simplify_polynomial(node.a))
    if isinstance(node, BinOp):
        if node.op in ("add", "sub"):
            return _canonicalize_sum(node)
        return BinOp(
            node.op,
            simplify_polynomial(node.a),
            simplify_polynomial(node.b),
        )
    if isinstance(node, FunctionalOp):
        new_args = tuple(simplify_polynomial(arg) for arg in node.args)
        return FunctionalOp(node.functional, new_args)
    if isinstance(node, FunctionalOp2D):
        return FunctionalOp2D(node.measure_2d, simplify_polynomial(node.arg))
    raise TypeError(type(node))


__all__ = ["simplify_polynomial"]
