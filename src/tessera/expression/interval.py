"""Interval-arithmetic evaluation of tessera Expr trees.

Computes elementwise lower and upper bounds on `evaluate(tree, env)`
given lower and upper bounds on each Var in `env`. Used by the search
submodule to derive cheap loss lower bounds and prune candidates that
provably can't beat the incumbent Pareto front.

The idea
--------
For each row of the dataset, the candidate tree's output is bounded by
[lo, hi] derived from the input variables' empirical [min, max] (or
any user-supplied bounds). For MSE-style losses, the LOSS lower bound
is computable in O(N) from the prediction interval bounds — and is
typically achieved by the OPTIMAL prediction inside the interval.

If `loss_lower_bound + parsimony * cx > current_incumbent_loss`, the
candidate is provably suboptimal at its complexity and can be pruned
WITHOUT full evaluation. This is branch-and-bound exploiting the fact
that SR has full information from the dataset (it's not adversarial).

What's covered
--------------
- Pointwise ops: add, sub, mul, div (safe-divide), min, max,
  gt, lt, ge, le → all have closed-form interval semantics
- Unary: neg, abs, tanh, sign, step → all monotone-or-bounded
- Const, Var → exact (for Var, from env_intervals)

What's NOT covered (returns conservative [-inf, +inf])
------------------------------------------------------
- FunctionalOp (LinearFunctional, SeparableBilinear, Volterra2) — could
  be tightened later with measure-norm bounds, deferred for now
- FunctionalOp2D — same

This means trees containing functionals get a loose bound and pruning
is less effective. Pure pointwise trees see the full benefit. Future
work: tighten functional bounds using the measure's L1 norm and the
input bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
)


@dataclass(frozen=True)
class Interval:
    """Closed interval [lo, hi]. Both endpoints are float; lo <= hi
    enforced at construction except for the special infeasibility
    sentinel Interval(inf, -inf).
    """
    lo: float
    hi: float

    def __post_init__(self):
        # Allow inf as either bound; allow the empty sentinel.
        if not (math.isnan(self.lo) or math.isnan(self.hi)):
            if self.lo > self.hi and not (math.isinf(self.lo)
                                          and math.isinf(self.hi)):
                raise ValueError(f"degenerate interval [{self.lo}, {self.hi}]")

    @classmethod
    def from_array(cls, x: np.ndarray) -> "Interval":
        """Construct the tightest interval covering all finite values in x."""
        mask = np.isfinite(x)
        if not mask.any():
            return cls(-math.inf, math.inf)
        return cls(float(np.min(x[mask])), float(np.max(x[mask])))

    @classmethod
    def point(cls, value: float) -> "Interval":
        return cls(value, value)

    @classmethod
    def unbounded(cls) -> "Interval":
        return cls(-math.inf, math.inf)


# ---------------- Binary interval ops ----------------

def _ival_add(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo + b.lo, a.hi + b.hi)


def _ival_sub(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo - b.hi, a.hi - b.lo)


def _ival_mul(a: Interval, b: Interval) -> Interval:
    """Multiplication of intervals: lo = min of 4 corner products,
    hi = max of the same. Handles all sign combinations correctly."""
    corners = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    # Filter NaN from corners (e.g. 0 * inf = NaN); treat as ±inf bounds
    finite = [c for c in corners if not math.isnan(c)]
    if not finite:
        return Interval.unbounded()
    return Interval(min(finite), max(finite))


def _ival_div(a: Interval, b: Interval) -> Interval:
    """Safe-divide interval semantics: matches BIN_OP_FNS["div"].

    If 0 ∈ [b.lo, b.hi], the result includes 0 (from safe-divide branch
    where b=0) AND extends to ±inf (the limit as b → 0 from non-zero
    side). Conservative: return [-inf, +inf].
    """
    if b.lo <= 0.0 <= b.hi:
        # 0 in denominator interval — conservative
        return Interval.unbounded()
    # 0 not in denominator: standard interval division via reciprocal
    recip = Interval(1.0 / b.hi, 1.0 / b.lo)
    return _ival_mul(a, recip)


def _ival_min(a: Interval, b: Interval) -> Interval:
    return Interval(min(a.lo, b.lo), min(a.hi, b.hi))


def _ival_max(a: Interval, b: Interval) -> Interval:
    return Interval(max(a.lo, b.lo), max(a.hi, b.hi))


def _ival_compare(a: Interval, b: Interval, op: str) -> Interval:
    """gt/lt/ge/le all return {0.0, 1.0}. Tight bounds:
      gt(a,b) is guaranteed 1.0 iff a.lo > b.hi (always strictly greater)
      gt(a,b) is guaranteed 0.0 iff a.hi <= b.lo (never strictly greater)
      otherwise the interval is [0, 1].
    """
    if op == "gt":
        if a.lo > b.hi:
            return Interval.point(1.0)
        if a.hi <= b.lo:
            return Interval.point(0.0)
    elif op == "lt":
        if a.hi < b.lo:
            return Interval.point(1.0)
        if a.lo >= b.hi:
            return Interval.point(0.0)
    elif op == "ge":
        if a.lo >= b.hi:
            return Interval.point(1.0)
        if a.hi < b.lo:
            return Interval.point(0.0)
    elif op == "le":
        if a.hi <= b.lo:
            return Interval.point(1.0)
        if a.lo > b.hi:
            return Interval.point(0.0)
    return Interval(0.0, 1.0)


_BIN_IVAL_FNS = {
    "add": _ival_add, "sub": _ival_sub, "mul": _ival_mul, "div": _ival_div,
    "min": _ival_min, "max": _ival_max,
}


# ---------------- Unary interval ops ----------------

def _ival_neg(a: Interval) -> Interval:
    return Interval(-a.hi, -a.lo)


def _ival_abs(a: Interval) -> Interval:
    """|·| has min = 0 if 0 ∈ [lo, hi] else min of |lo|, |hi|;
    max = max of |lo|, |hi|."""
    abs_lo, abs_hi = abs(a.lo), abs(a.hi)
    new_hi = max(abs_lo, abs_hi)
    if a.lo <= 0.0 <= a.hi:
        new_lo = 0.0
    else:
        new_lo = min(abs_lo, abs_hi)
    return Interval(new_lo, new_hi)


def _ival_tanh(a: Interval) -> Interval:
    """tanh is monotone, so the interval is exact."""
    return Interval(math.tanh(a.lo), math.tanh(a.hi))


def _ival_sign(a: Interval) -> Interval:
    """sign returns -1, 0, or +1. Tight bound by zero-containment:
      a.hi < 0: result is exactly -1
      a.lo > 0: result is exactly +1
      a.lo == a.hi == 0: result is exactly 0
      otherwise: [-1, 1]
    """
    if a.hi < 0:
        return Interval.point(-1.0)
    if a.lo > 0:
        return Interval.point(1.0)
    if a.lo == 0.0 and a.hi == 0.0:
        return Interval.point(0.0)
    return Interval(-1.0, 1.0)


def _ival_step(a: Interval) -> Interval:
    """step(x) = 1 if x > 0 else 0. Tight bounds:
      a.lo > 0: result is exactly 1
      a.hi <= 0: result is exactly 0
      otherwise: [0, 1]
    """
    if a.lo > 0:
        return Interval.point(1.0)
    if a.hi <= 0:
        return Interval.point(0.0)
    return Interval(0.0, 1.0)


_UN_IVAL_FNS = {
    "neg": _ival_neg, "abs": _ival_abs, "tanh": _ival_tanh,
    "sign": _ival_sign, "step": _ival_step,
}


# ---------------- The interval evaluator ----------------

def interval_evaluate(
    node: Node,
    env_intervals: dict[str, Interval],
) -> Interval:
    """Evaluate a tree on interval-valued inputs to produce an interval-
    valued output.

    Parameters
    ----------
    node : Node
        Root of the tree.
    env_intervals : dict[str, Interval]
        Maps Var names to bounds on the variable's value. Use
        `Interval.from_array(data)` to derive bounds from training data.

    Returns
    -------
    Interval — bounds on `evaluate(node, env)` for ANY values in
    env compatible with the per-variable bounds. The bound is
    rigorous (sound): the actual output is guaranteed to lie in
    [lo, hi]. It may not be tight (e.g. for `x - x`, naive interval
    arithmetic gives [a.lo - a.hi, a.hi - a.lo] when the actual
    answer is 0; the dependency problem is a known interval-arithmetic
    limitation that AC normalisation / simplify_canonical helps with).
    """
    if isinstance(node, Const):
        return Interval.point(node.value)
    if isinstance(node, Var):
        return env_intervals.get(node.name, Interval.unbounded())

    if isinstance(node, UnOp):
        a = interval_evaluate(node.a, env_intervals)
        fn = _UN_IVAL_FNS.get(node.op)
        if fn is None:
            return Interval.unbounded()
        return fn(a)

    if isinstance(node, BinOp):
        a = interval_evaluate(node.a, env_intervals)
        b = interval_evaluate(node.b, env_intervals)
        if node.op in ("gt", "lt", "ge", "le"):
            return _ival_compare(a, b, node.op)
        fn = _BIN_IVAL_FNS.get(node.op)
        if fn is None:
            return Interval.unbounded()
        return fn(a, b)

    if isinstance(node, (FunctionalOp, FunctionalOp2D)):
        # TODO: tighten using the measure's L1-norm bound times the
        # input interval's max-abs. For now, conservative.
        return Interval.unbounded()

    raise TypeError(type(node))


def env_intervals_from_arrays(
    env: dict[str, np.ndarray],
) -> dict[str, Interval]:
    """Convenience: derive per-Var intervals from a training-data env.
    Uses `Interval.from_array(arr)` for each entry."""
    return {name: Interval.from_array(arr) for name, arr in env.items()}
