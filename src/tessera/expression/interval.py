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

What's covered for measure-theoretic operators
-----------------------------------------------
- LinearFunctional(μ)(x): |L(x)| ≤ ||μ||_1 · max(|x.lo|, |x.hi|)
  where ||μ||_1 = ∑_k |κ[k]| is the L1 norm of the discrete kernel.
- SeparableBilinear(μ_a, μ_b)(x, y): elementwise L_μ_a(x) * L_μ_b(y).
  Interval-multiplied via the LinearFunctional bound for each side.
- Volterra2(μ_a, μ_b)(x): same as SeparableBilinear with y = x.

What's NOT covered (returns conservative [-inf, +inf])
------------------------------------------------------
- FunctionalOp2D — 2-D measures on space-time fields. Bound derivable
  via 2-D L1 norm; deferred for the first pass.

Pure pointwise + 1-D functional trees see the full benefit. Future
work: 2-D measure L1 norms, AC-norm propagation through functionals
to defeat the dependency problem (e.g., L_μ(x) - L_μ(x) ≡ 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
)
from .functional import LinearFunctional, SeparableBilinear, Volterra2
from .measure import Measure


def measure_l1_norm(m: Measure) -> float:
    """L1 norm = total variation of the discrete kernel.

    For a measure with atomic + density parts:
        ||μ||_1 = ∑_k |κ[k]|   where κ = m.to_kernel()

    Exact (not an upper bound) under tessera's discrete-time semantics.
    Used by interval evaluation of LinearFunctional / SeparableBilinear
    / Volterra2.
    """
    kernel = m.to_kernel()
    return float(np.sum(np.abs(kernel)))


def measure_2d_l1_norm(m_2d) -> float:
    """L1 norm of a 2D measure (Measure2D).

    Decomposition:
        ||μ||_1 = Σ_i |w_i|             (atomic part)
                + ||κ_t||_1 · ||κ_x||_1  (separable density part)

    The product structure for the density comes from Fubini's
    theorem: for a separable kernel κ(s_t, s_x) = κ_t(s_t) · κ_x(s_x),
    the double sum factors. (Reznikov §3.7, Theorem 143.)
    """
    atomic_l1 = sum(abs(a.weight) for a in m_2d.atoms)
    if m_2d.has_density:
        density_l1 = measure_l1_norm(m_2d.sep_t) * measure_l1_norm(m_2d.sep_x)
    else:
        density_l1 = 0.0
    return float(atomic_l1 + density_l1)


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


def _ival_pow(a: Interval, b: Interval) -> Interval:
    """Protected pow: pow(|a|, clip(b, ±8)). Output is non-negative.

    Bound logic: base ∈ [max(0, ε), max(|a.lo|, |a.hi|)]; exp clipped to
    [-8, +8]. Use monotonicity in exp when base is fixed >=1 or <=1 to
    derive endpoint bounds; conservative fallback to [0, +inf]."""
    # Output is non-negative (sign-dropped).
    abs_a = _ival_abs(a)
    base_lo = max(abs_a.lo, 1e-12)
    base_hi = abs_a.hi
    if not math.isfinite(base_hi):
        return Interval(0.0, math.inf)
    exp_lo = max(b.lo, -8.0)
    exp_hi = min(b.hi, 8.0)
    if exp_lo > exp_hi:
        # exponent interval entirely outside clip range → degenerate; conservative
        return Interval(0.0, math.inf)
    # Evaluate at the four corners (base_lo/hi, exp_lo/hi).
    corners = []
    for base in (base_lo, base_hi):
        for exp in (exp_lo, exp_hi):
            try:
                v = base ** exp
            except (OverflowError, ValueError):
                v = math.inf
            if math.isnan(v):
                v = math.inf
            corners.append(v)
    return Interval(max(0.0, min(corners)), max(corners))


def _ival_atan2(y: Interval, x: Interval) -> Interval:
    """atan2: output always in [-π, π]. Tighter bounds possible based on
    quadrant of the input interval; deferred for now (matches conservative
    sin/cos approach)."""
    return Interval(-math.pi, math.pi)


_BIN_IVAL_FNS = {
    "add": _ival_add, "sub": _ival_sub, "mul": _ival_mul, "div": _ival_div,
    "min": _ival_min, "max": _ival_max,
    "pow": _ival_pow,
    "atan2": _ival_atan2,
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


def _ival_reduce_minmax(a: Interval) -> Interval:
    """reduce_mean / reduce_max / reduce_min / reduce_std: the result is
    a scalar inside the input's interval. Mean and max stay in [lo, hi].
    """
    return Interval(a.lo, a.hi)


def _ival_reduce_sum(a: Interval) -> Interval:
    """reduce_sum: bound depends on the array size which the interval
    evaluator doesn't know. Conservative: [-inf, +inf] when input
    spans zero; same-sign-bound otherwise."""
    if a.lo >= 0:
        return Interval(0.0, math.inf)
    if a.hi <= 0:
        return Interval(-math.inf, 0.0)
    return Interval.unbounded()


def _ival_reduce_std(a: Interval) -> Interval:
    """reduce_std: std ∈ [0, (hi - lo) / 2] for finite samples; can be
    larger in pathological cases. Conservative: [0, hi - lo]."""
    if math.isfinite(a.lo) and math.isfinite(a.hi):
        spread = a.hi - a.lo
        return Interval(0.0, spread)
    return Interval(0.0, math.inf)


def _ival_sqrt(a: Interval) -> Interval:
    """Protected sqrt: sqrt(|x|). Output is non-negative; monotone in |x|."""
    abs_a = _ival_abs(a)
    lo = math.sqrt(abs_a.lo) if math.isfinite(abs_a.lo) else math.inf
    hi = math.sqrt(abs_a.hi) if math.isfinite(abs_a.hi) else math.inf
    return Interval(lo, hi)


def _ival_log(a: Interval) -> Interval:
    """Protected log: log(max(|x|, 1e-12)). Monotone in |x|."""
    abs_a = _ival_abs(a)
    lo_x = max(abs_a.lo, 1e-12)
    hi_x = max(abs_a.hi, 1e-12)
    lo = math.log(lo_x)
    hi = math.log(hi_x) if math.isfinite(hi_x) else math.inf
    return Interval(lo, hi)


def _ival_exp(a: Interval) -> Interval:
    """Protected exp: exp(clip(x, ±50)). Monotone; tight bounds via clip."""
    lo_x = max(a.lo, -50.0)
    hi_x = min(a.hi, 50.0)
    if lo_x > hi_x:
        # input interval entirely above +50 or below -50 → clip collapses
        clipped = 50.0 if a.lo > 50.0 else -50.0
        v = math.exp(clipped)
        return Interval(v, v)
    return Interval(math.exp(lo_x), math.exp(hi_x))


def _ival_sin(a: Interval) -> Interval:
    """sin: bounded [-1, 1] everywhere. Conservative interval — does
    NOT exploit monotonicity within ±π/2 intervals. Tightness deferred."""
    # Tight bound: if the interval is short enough to lie within a
    # monotone region, we could compute sin(lo), sin(hi) and order. But
    # detecting "no π/2+kπ in the interval" is fiddly and the conservative
    # bound is sound. Pruning correctness preserved; tightness left
    # for a future refinement (see docs/research/fit_as_perfect_info_game.md
    # §12 on tightening B&B bounds).
    return Interval(-1.0, 1.0)


def _ival_cos(a: Interval) -> Interval:
    """cos: same as sin; bounded [-1, 1]."""
    return Interval(-1.0, 1.0)


def _ival_acos(a: Interval) -> Interval:
    """acos: input clipped to [-1, 1]; output always in [0, π]."""
    return Interval(0.0, math.pi)


def _ival_asin(a: Interval) -> Interval:
    """asin: input clipped to [-1, 1]; output in [-π/2, π/2]."""
    return Interval(-math.pi / 2, math.pi / 2)


_UN_IVAL_FNS = {
    "neg": _ival_neg, "abs": _ival_abs, "tanh": _ival_tanh,
    "sign": _ival_sign, "step": _ival_step,
    "sqrt": _ival_sqrt, "log": _ival_log, "exp": _ival_exp,
    "sin": _ival_sin, "cos": _ival_cos,
    "acos": _ival_acos, "asin": _ival_asin,
    "reduce_mean": _ival_reduce_minmax,
    "reduce_max":  _ival_reduce_minmax,
    "reduce_sum":  _ival_reduce_sum,
    "reduce_std":  _ival_reduce_std,
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

    if isinstance(node, FunctionalOp):
        return _interval_functional(node, env_intervals)

    if isinstance(node, FunctionalOp2D):
        arg_iv = interval_evaluate(node.arg, env_intervals)
        if not (math.isfinite(arg_iv.lo) and math.isfinite(arg_iv.hi)):
            return Interval.unbounded()
        L1 = measure_2d_l1_norm(node.measure_2d)
        M = max(abs(arg_iv.lo), abs(arg_iv.hi))
        return Interval(-L1 * M, L1 * M)

    raise TypeError(type(node))


def _interval_functional(
    node: FunctionalOp,
    env_intervals: dict[str, Interval],
) -> Interval:
    """L1-norm-based interval bound for tessera's 1-D Functional types.

    For LinearFunctional(μ) applied to x with x[t] ∈ [lo, hi]:
        |L_μ(x)(t)| = |∑_k κ_μ[k] · x[t-k]|
                   ≤ ∑_k |κ_μ[k]| · max(|lo|, |hi|)
                   = ||μ||_1 · max(|lo|, |hi|)
    So  L_μ(x) ∈ [-||μ||_1 · M, +||μ||_1 · M]  where M = max(|lo|, |hi|).

    For SeparableBilinear(μ_a, μ_b)(x, y) = L_μ_a(x) · L_μ_b(y):
        bound is the interval product of the two LinearFunctional bounds.

    For Volterra2(μ_a, μ_b)(x): same as SeparableBilinear with y=x.
    """
    f = node.functional
    if isinstance(f, LinearFunctional):
        x_iv = interval_evaluate(node.args[0], env_intervals)
        if not (math.isfinite(x_iv.lo) and math.isfinite(x_iv.hi)):
            return Interval.unbounded()
        L1 = measure_l1_norm(f.measure)
        M = max(abs(x_iv.lo), abs(x_iv.hi))
        return Interval(-L1 * M, L1 * M)

    if isinstance(f, SeparableBilinear):
        x_iv = interval_evaluate(node.args[0], env_intervals)
        y_iv = interval_evaluate(node.args[1], env_intervals)
        if not all(math.isfinite(v) for v in
                   (x_iv.lo, x_iv.hi, y_iv.lo, y_iv.hi)):
            return Interval.unbounded()
        L1_a = measure_l1_norm(f.measure_a)
        L1_b = measure_l1_norm(f.measure_b)
        M_x = max(abs(x_iv.lo), abs(x_iv.hi))
        M_y = max(abs(y_iv.lo), abs(y_iv.hi))
        # |L_a(x) * L_b(y)| ≤ L1_a*M_x * L1_b*M_y
        prod_bound = L1_a * M_x * L1_b * M_y
        return Interval(-prod_bound, prod_bound)

    if isinstance(f, Volterra2):
        x_iv = interval_evaluate(node.args[0], env_intervals)
        if not (math.isfinite(x_iv.lo) and math.isfinite(x_iv.hi)):
            return Interval.unbounded()
        L1_a = measure_l1_norm(f.measure_a)
        L1_b = measure_l1_norm(f.measure_b)
        M = max(abs(x_iv.lo), abs(x_iv.hi))
        prod_bound = L1_a * L1_b * M * M
        return Interval(-prod_bound, prod_bound)

    # Unknown Functional subtype — conservative
    return Interval.unbounded()


def env_intervals_from_arrays(
    env: dict[str, np.ndarray],
) -> dict[str, Interval]:
    """Convenience: derive per-Var intervals from a training-data env.
    Uses `Interval.from_array(arr)` for each entry."""
    return {name: Interval.from_array(arr) for name, arr in env.items()}
