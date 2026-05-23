"""Rule-based simplifier — the original tree.simplify() function, moved
into the simplify subpackage so AC normalisation / polynomial /
e-graph simplifiers can live as siblings.

Operates bottom-up on tessera Expr trees and folds:
  - Constant arithmetic (BinOp(op, Const, Const) → Const(op(a,b)))
  - X − X → 0
  - X + 0, X − 0, X * 1, X / 1 → X
  - X * 0, X / 0, 0 / X → 0    (matches BIN_OP_FNS["div"] safe-divide)
  - 0 − X → neg(X)
  - neg(neg(X)) → X
  - abs(neg(X)) → abs(X)
  - min(X, X), max(X, X) → X
  - gt(X, X), lt(X, X) → 0
  - ge(X, X), le(X, X) → 1
  - Unary constant folding (neg, abs, tanh, sign, step on Const)

Recurses into FunctionalOp / FunctionalOp2D arguments but never folds
across a measure — applying a measure to a constant is not a free
identity in general.
"""
from __future__ import annotations

from ..tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    BIN_OP_FNS, UN_OP_FNS,
)


def _is_const_value(node: Node, value: float, tol: float = 1e-12) -> bool:
    return isinstance(node, Const) and abs(node.value - value) <= tol


def simplify(node: Node) -> Node:
    """Bottom-up algebraic simplification.

    See module docstring for the full list of folds. Semantic-preserving
    under tessera's evaluator (with safe-divide convention for X/0).
    """
    if isinstance(node, (Var, Const)):
        return node

    if isinstance(node, UnOp):
        a = simplify(node.a)
        if isinstance(a, Const):
            try:
                return Const(float(UN_OP_FNS[node.op](a.value)))
            except (ValueError, OverflowError):
                pass
        if node.op == "neg" and isinstance(a, UnOp) and a.op == "neg":
            return a.a
        if node.op == "abs" and isinstance(a, UnOp) and a.op == "neg":
            return UnOp("abs", a.a)
        return UnOp(node.op, a)

    if isinstance(node, BinOp):
        a = simplify(node.a)
        b = simplify(node.b)
        op = node.op

        # Constant folding when both sides are constants
        if isinstance(a, Const) and isinstance(b, Const):
            try:
                v = BIN_OP_FNS[op](a.value, b.value)
                return Const(float(v))
            except (ValueError, OverflowError, ZeroDivisionError):
                pass

        # Algebraic identities
        if op == "sub" and a == b:
            return Const(0.0)
        if op == "add":
            if _is_const_value(a, 0.0):
                return b
            if _is_const_value(b, 0.0):
                return a
        if op == "sub":
            if _is_const_value(b, 0.0):
                return a
            if _is_const_value(a, 0.0):
                return simplify(UnOp("neg", b))
        if op == "mul":
            if _is_const_value(a, 0.0) or _is_const_value(b, 0.0):
                return Const(0.0)
            if _is_const_value(a, 1.0):
                return b
            if _is_const_value(b, 1.0):
                return a
        if op == "div":
            # Safe-divide: anything / 0 → 0
            if _is_const_value(b, 0.0):
                return Const(0.0)
            # 0 / anything → 0
            if _is_const_value(a, 0.0):
                return Const(0.0)
            if _is_const_value(b, 1.0):
                return a
        # min/max with equal args
        if op in ("min", "max") and a == b:
            return a

        # Indicator identities on equal arguments
        if op in ("gt", "lt") and a == b:
            return Const(0.0)
        if op in ("ge", "le") and a == b:
            return Const(1.0)

        return BinOp(op, a, b)

    if isinstance(node, FunctionalOp):
        new_args = tuple(simplify(arg) for arg in node.args)
        return FunctionalOp(node.functional, new_args)

    if isinstance(node, FunctionalOp2D):
        return FunctionalOp2D(node.measure_2d, simplify(node.arg))

    raise TypeError(type(node))
