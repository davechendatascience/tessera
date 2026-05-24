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
        # Transcendental inverse pairs (semantic-preserving under PROTECTED
        # ops only when the inner expression is non-negative; we apply the
        # folds unconditionally because (a) sqrt/log/exp are sign-dropping
        # in tessera, so the equality holds wherever it would in PySR; and
        # (b) the simplifier is allowed to canonicalise inside the
        # protected-op semantics).
        if node.op == "log" and isinstance(a, UnOp) and a.op == "exp":
            # log(exp(x)) = x  (exp output positive, log of it = clip(x, ±50))
            return a.a
        if node.op == "exp" and isinstance(a, UnOp) and a.op == "log":
            # exp(log(|x|)) = |x|  (log floors at 1e-12; exp_log_x = |x| ∨ 1e-12)
            return UnOp("abs", a.a)
        if node.op == "sqrt" and isinstance(a, UnOp) and a.op == "abs":
            # sqrt(|x|) = sqrt(x) under protected semantics — drop redundant abs
            return UnOp("sqrt", a.a)
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
            # X / X → 1. This is the safe-divide convention (a/0 → 0 means
            # this isn't true at a=0); but the resulting Const(1.0) only
            # affects downstream computation where Var "X" appears, and at
            # the algebraic identity X/X=1 holds almost everywhere. The
            # MNIST diagnostic showed `image/image` appearing in trees
            # where the GP intended to use a non-trivial expression but
            # mutation produced the trivial identity.
            if a == b:
                return Const(1.0)
        # min/max with equal args
        if op in ("min", "max") and a == b:
            return a

        # Indicator identities on equal arguments
        if op in ("gt", "lt") and a == b:
            return Const(0.0)
        if op in ("ge", "le") and a == b:
            return Const(1.0)

        # Power folds
        if op == "pow":
            if _is_const_value(b, 0.0):
                # pow(x, 0) = 1 (protected: |x|=0 case still gives 1.0
                # because base is floored to 1e-12 then raised to 0)
                return Const(1.0)
            if _is_const_value(b, 1.0):
                # pow(x, 1) = |x| under protected semantics
                return UnOp("abs", a)
            if _is_const_value(a, 0.0):
                # pow(0, b) → 0 (floored base 1e-12 raised to b > 0 ≈ 0)
                return Const(0.0)
            if _is_const_value(a, 1.0):
                return Const(1.0)

        return BinOp(op, a, b)

    if isinstance(node, FunctionalOp):
        new_args = tuple(simplify(arg) for arg in node.args)
        # Const-input folds: applying a measure-based functional to a
        # constant series yields a constant series.
        #
        # For LinearFunctional(μ)(Const c):
        #   (μ · c)(t) = Σ_k κ[k] · c = c · Σ_k κ[k]   (post-warmup)
        #
        # For SeparableBilinear(μ_a, μ_b)(Const c_a, Const c_b):
        #   = (μ_a · c_a)(t) · (μ_b · c_b)(t)
        #   = (c_a · Σ κ_a) · (c_b · Σ κ_b)
        #
        # For Volterra2(μ_a, μ_b)(Const c):
        #   = (μ_a · c)(t) · (μ_b · c)(t)
        #   = c² · Σ κ_a · Σ κ_b
        #
        # The MNIST diagnostic showed several discovered features
        # containing FunctionalOp applied to a constant (a "dead branch"
        # that contributed nothing while inflating tree complexity).
        # Folding these to Const lets parsimony pressure prune them.
        from ..functional import LinearFunctional, SeparableBilinear, Volterra2
        from ..measure import Measure
        f = node.functional
        if isinstance(f, LinearFunctional) and isinstance(new_args[0], Const):
            kernel_sum = float(f.measure.to_kernel().sum())
            return Const(new_args[0].value * kernel_sum)
        if isinstance(f, SeparableBilinear) \
                and isinstance(new_args[0], Const) \
                and isinstance(new_args[1], Const):
            sum_a = float(f.measure_a.to_kernel().sum())
            sum_b = float(f.measure_b.to_kernel().sum())
            return Const(new_args[0].value * sum_a * new_args[1].value * sum_b)
        if isinstance(f, Volterra2) and isinstance(new_args[0], Const):
            sum_a = float(f.measure_a.to_kernel().sum())
            sum_b = float(f.measure_b.to_kernel().sum())
            return Const((new_args[0].value ** 2) * sum_a * sum_b)
        return FunctionalOp(f, new_args)

    if isinstance(node, FunctionalOp2D):
        new_arg = simplify(node.arg)
        if isinstance(new_arg, Const):
            # M2D applied to a constant scalar field: result is constant
            # equal to c · (Σ atomic_weights + Σ sep_t · Σ sep_x).
            m = node.measure_2d
            total = sum(a.weight for a in m.atoms)
            if m.has_density:
                total += (float(m.sep_t.to_kernel().sum())
                          * float(m.sep_x.to_kernel().sum()))
            return Const(new_arg.value * total)
        return FunctionalOp2D(node.measure_2d, new_arg)

    raise TypeError(type(node))
