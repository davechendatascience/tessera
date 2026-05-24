"""Expr tree for the symbolic search.

Every search candidate is a `Node`, a frozen tagged-union dataclass tree:

    Var(name)               — reference to a named series in the env
    Const(value)            — float literal
    BinOp(op, a, b)         — pointwise binary op (add/sub/mul/div/min/max)
    UnOp(op, a)             — pointwise unary op (tanh/abs/sign/neg)
    FunctionalOp(f, args)   — wrap a Functional (Linear / Bilinear / Volterra2)
                              applied to one or two child nodes

The whole tree is immutable + hashable. Equal-by-value trees compare equal,
so the GP search can use trees as dict keys and the cache can identify
equivalent subexpressions without explicit structural sharing.

Pointwise vs functional semantics
---------------------------------
- Pointwise nodes (BinOp/UnOp/Const) return either a numpy array of length N
  or a scalar; numpy broadcasting handles the mix.
- FunctionalOp nodes ALWAYS return an array (the measure-theoretic
  convolution of their inputs). Scalar inputs are broadcast to an array
  of the appropriate length before applying the Functional.
- FunctionalCache is wired in at FunctionalOp evaluation: each functional
  argument's subtree gets a deterministic `var_id` derived from `str(node)`,
  so two trees that contain identical subexpressions share the cache slot.

Operator alphabets
------------------
Pointwise op names match the standard symbolic-regression vocabulary.
Functional ops are NOT enumerated here — they're carried *inside* the
FunctionalOp node as a Functional instance (with its own measures and
parameters). The GP mutation operators (see `mutation.py`) work with
both: pointwise op-swap mutates BinOp/UnOp.op strings, while measure
mutation rebuilds the FunctionalOp.functional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Union

import numpy as np

from .functional import Functional, LinearFunctional, SeparableBilinear, Volterra2
from .measure_2d import Measure2D
from .cache import FunctionalCache


# ---------------- Pointwise operator tables ----------------

# Binary
# Protected `pow`: pow(|a|, b) with clipped exponent to keep the GP search
# numerically safe. PySR uses the same convention (`safe_pow`). The sign of
# `a` is dropped; if you need signed powers, build them explicitly via
# sign(a) * pow(|a|, b).
_POW_EXP_CLIP = 8.0   # |b| capped at 8 to avoid overflow
_POW_BASE_FLOOR = 1e-12  # |a| floored to avoid 0**negative


def _safe_pow(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    base = np.maximum(np.abs(a_arr), _POW_BASE_FLOOR)
    exp = np.clip(b_arr, -_POW_EXP_CLIP, _POW_EXP_CLIP)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.power(base, exp)
    return np.where(np.isfinite(out), out, 0.0)


BIN_OP_FNS: dict[str, Callable] = {
    "add":  lambda a, b: a + b,
    "sub":  lambda a, b: a - b,
    "mul":  lambda a, b: a * b,
    "div":  lambda a, b: np.where(b == 0, 0.0, a / np.where(b == 0, 1.0, b)),
    "min":  np.minimum,
    "max":  np.maximum,
    # Threshold indicators — return float64 {0.0, 1.0} per element.
    # Bridge to EML-style binary-window primitives without needing a
    # dedicated WeightedIndicatorSum operator (see docs/roadmap.md §3).
    "gt":   lambda a, b: (np.asarray(a) > np.asarray(b)).astype(np.float64),
    "lt":   lambda a, b: (np.asarray(a) < np.asarray(b)).astype(np.float64),
    "ge":   lambda a, b: (np.asarray(a) >= np.asarray(b)).astype(np.float64),
    "le":   lambda a, b: (np.asarray(a) <= np.asarray(b)).astype(np.float64),
    # Protected power: pow(|a|, clip(b, ±8)). Drops sign of base; use
    # sign(a)*pow(|a|, b) for signed forms. Matches PySR's safe_pow.
    "pow":  _safe_pow,
}
BIN_OPS = tuple(BIN_OP_FNS.keys())

# Unary
def _reduce_mean(x):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not mask.any():
        return float("nan")
    return float(x[mask].mean())


def _reduce_max(x):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not mask.any():
        return float("nan")
    return float(x[mask].max())


def _reduce_sum(x):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not mask.any():
        return float("nan")
    return float(x[mask].sum())


def _reduce_std(x):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if not mask.any():
        return float("nan")
    if int(mask.sum()) < 2:
        return 0.0
    return float(x[mask].std())


# Protected transcendentals. PySR conventions (`safe_sqrt`, `safe_log`,
# `safe_exp`) — kept identical so trees translate cleanly across tools.
_EXP_CLIP = 50.0  # exp(50) ≈ 5.18e21, exp(-50) ≈ 1.9e-22 — well inside float64
_LOG_FLOOR = 1e-12


def _safe_sqrt(x):
    return np.sqrt(np.abs(np.asarray(x, dtype=np.float64)))


def _safe_log(x):
    return np.log(np.maximum(np.abs(np.asarray(x, dtype=np.float64)), _LOG_FLOOR))


def _safe_exp(x):
    return np.exp(np.clip(np.asarray(x, dtype=np.float64), -_EXP_CLIP, _EXP_CLIP))


UN_OP_FNS: dict[str, Callable] = {
    "tanh": np.tanh,
    "abs":  np.abs,
    "sign": np.sign,
    "neg":  np.negative,
    # Heaviside step: 1.0 if x > 0 else 0.0. Equivalent to gt(x, 0) but
    # cheaper (no broadcast on the threshold side) and easier for the
    # GP to discover.
    "step": lambda x: (np.asarray(x) > 0.0).astype(np.float64),
    # Protected transcendentals (PySR convention).
    # sqrt: sqrt(|x|). Sign-dropping; pair with sign(x) if signed roots needed.
    # log:  log(max(|x|, 1e-12)). Drops sign; well-defined for any input.
    # exp:  exp(clip(x, ±50)). Bounded to avoid overflow.
    "sqrt": _safe_sqrt,
    "log":  _safe_log,
    "exp":  _safe_exp,
    # Reductions: array → scalar. Always reduce ALL axes. Used to convert
    # a 2-D feature map into a translation-invariant scalar prediction
    # (see docs/research_notes/invariance_in_sr.md). The GP can place
    # these anywhere; downstream BinOps will numpy-broadcast the scalar.
    "reduce_mean": _reduce_mean,
    "reduce_max":  _reduce_max,
    "reduce_sum":  _reduce_sum,
    "reduce_std":  _reduce_std,
}
UN_OPS = tuple(UN_OP_FNS.keys())


# ---------------- Node types ----------------

@dataclass(frozen=True)
class Var:
    """Reference to a named series in the evaluation environment."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    """Float literal."""
    value: float

    def __str__(self) -> str:
        # Round for stable identifiers (cache keys, mutation hashing)
        return f"{self.value:.6g}"


@dataclass(frozen=True)
class BinOp:
    """Pointwise binary operator."""
    op: str
    a: "Node"
    b: "Node"

    def __post_init__(self):
        if self.op not in BIN_OP_FNS:
            raise ValueError(f"unknown binary op {self.op!r}; valid: {BIN_OPS}")

    def __str__(self) -> str:
        # Infix for arithmetic + comparison; prefix for min/max
        if self.op in ("add", "sub", "mul", "div", "gt", "lt", "ge", "le"):
            sym = {
                "add": "+", "sub": "-", "mul": "*", "div": "/",
                "gt": ">", "lt": "<", "ge": ">=", "le": "<=",
            }[self.op]
            return f"({self.a} {sym} {self.b})"
        return f"{self.op}({self.a}, {self.b})"


@dataclass(frozen=True)
class UnOp:
    """Pointwise unary operator."""
    op: str
    a: "Node"

    def __post_init__(self):
        if self.op not in UN_OP_FNS:
            raise ValueError(f"unknown unary op {self.op!r}; valid: {UN_OPS}")

    def __str__(self) -> str:
        return f"{self.op}({self.a})"


@dataclass(frozen=True)
class FunctionalOp:
    """Apply a 1-D Functional (Linear / Bilinear / Volterra2) to argument node(s).

    The functional's `n_inputs` must match `len(args)`. Constructors check
    this immediately (so malformed trees fail fast).
    """
    functional: Functional
    args: tuple["Node", ...]

    def __post_init__(self):
        expected = self.functional.n_inputs
        if len(self.args) != expected:
            raise ValueError(
                f"FunctionalOp arity mismatch: {self.functional.__class__.__name__} "
                f"expects {expected} args, got {len(self.args)}"
            )

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.functional}({args_str})"


@dataclass(frozen=True)
class FunctionalOp2D:
    """Apply a Measure2D to a single 2-D argument.

    Used in PDE-discovery trees where the input env contains 2-D fields
    U(t, x) instead of (or alongside) 1-D series. Pointwise ops
    (BinOp/UnOp) broadcast naturally across the 2-D arrays.

    The single-argument constraint reflects that 2-D measures are
    structurally similar to 1-D linear functionals — bilinear /
    multilinear 2-D ops can be expressed by composition with BinOp/mul.
    """
    measure_2d: Measure2D
    arg: "Node"

    def __str__(self) -> str:
        return f"M2D[{self.measure_2d}]({self.arg})"


# Tagged-union alias
Node = Union[Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D]


# ---------------- Structural helpers ----------------

def complexity(node: Node) -> int:
    """Total number of nodes in the tree (the SR-standard complexity measure)."""
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, UnOp):
        return 1 + complexity(node.a)
    if isinstance(node, BinOp):
        return 1 + complexity(node.a) + complexity(node.b)
    if isinstance(node, FunctionalOp):
        return 1 + sum(complexity(a) for a in node.args)
    if isinstance(node, FunctionalOp2D):
        return 1 + complexity(node.arg)
    raise TypeError(type(node))


def depth(node: Node) -> int:
    """Maximum tree depth (root = 1)."""
    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, UnOp):
        return 1 + depth(node.a)
    if isinstance(node, BinOp):
        return 1 + max(depth(node.a), depth(node.b))
    if isinstance(node, FunctionalOp):
        return 1 + max(depth(a) for a in node.args)
    if isinstance(node, FunctionalOp2D):
        return 1 + depth(node.arg)
    raise TypeError(type(node))


def used_features(node: Node) -> set[str]:
    """Names of all `Var` leaves in the tree."""
    out: set[str] = set()

    def visit(n: Node) -> None:
        if isinstance(n, Var):
            out.add(n.name)
        elif isinstance(n, UnOp):
            visit(n.a)
        elif isinstance(n, BinOp):
            visit(n.a); visit(n.b)
        elif isinstance(n, FunctionalOp):
            for a in n.args:
                visit(a)
        elif isinstance(n, FunctionalOp2D):
            visit(n.arg)

    visit(node)
    return out


def iter_subtrees(node: Node) -> Iterator[Node]:
    """Pre-order traversal yielding every subtree (root first)."""
    yield node
    if isinstance(node, UnOp):
        yield from iter_subtrees(node.a)
    elif isinstance(node, BinOp):
        yield from iter_subtrees(node.a)
        yield from iter_subtrees(node.b)
    elif isinstance(node, FunctionalOp):
        for a in node.args:
            yield from iter_subtrees(a)
    elif isinstance(node, FunctionalOp2D):
        yield from iter_subtrees(node.arg)


def replace_at(root: Node, target_index: int, new_node: Node) -> Node:
    """Replace the subtree at the given pre-order index with `new_node`.

    Returns a new tree (the original is unchanged — Node is immutable).
    """
    counter = [0]

    def visit(n: Node) -> Node:
        idx = counter[0]
        counter[0] += 1
        if idx == target_index:
            return new_node
        if isinstance(n, (Var, Const)):
            return n
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.a))
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.a), visit(n.b))
        if isinstance(n, FunctionalOp):
            new_args = tuple(visit(a) for a in n.args)
            return FunctionalOp(n.functional, new_args)
        if isinstance(n, FunctionalOp2D):
            return FunctionalOp2D(n.measure_2d, visit(n.arg))
        raise TypeError(type(n))

    return visit(root)


# ---------------- Constant introspection ----------------

def collect_const_values(node: Node) -> list[float]:
    """Return the value of each Const leaf in pre-order traversal.

    Used by GP's constant-optimisation polish step: extract the current
    numerical constants, hand them to scipy.optimize.minimize, then
    splat the optimised values back in via `set_const_values`.
    """
    vals: list[float] = []

    def visit(n: Node) -> None:
        if isinstance(n, Const):
            vals.append(n.value)
        elif isinstance(n, Var):
            return
        elif isinstance(n, UnOp):
            visit(n.a)
        elif isinstance(n, BinOp):
            visit(n.a); visit(n.b)
        elif isinstance(n, FunctionalOp):
            for a in n.args:
                visit(a)
        elif isinstance(n, FunctionalOp2D):
            visit(n.arg)

    visit(node)
    return vals


def set_const_values(node: Node, new_values: list[float]) -> Node:
    """Return a new tree with each Const leaf replaced by the next value
    from `new_values` (consumed in pre-order).

    Length mismatch behaviour:
      - if `new_values` has fewer entries than Const leaves, the extras
        keep their original values.
      - if it has more, the surplus is silently ignored.
    """
    vals_iter = iter(new_values)

    def visit(n: Node) -> Node:
        if isinstance(n, Const):
            try:
                return Const(float(next(vals_iter)))
            except StopIteration:
                return n
        if isinstance(n, Var):
            return n
        if isinstance(n, UnOp):
            return UnOp(n.op, visit(n.a))
        if isinstance(n, BinOp):
            return BinOp(n.op, visit(n.a), visit(n.b))
        if isinstance(n, FunctionalOp):
            return FunctionalOp(n.functional, tuple(visit(a) for a in n.args))
        if isinstance(n, FunctionalOp2D):
            return FunctionalOp2D(n.measure_2d, visit(n.arg))
        raise TypeError(type(n))

    return visit(node)


# NOTE: `simplify` and its helpers moved to `tessera.expression.simplify`
# (a subpackage). Import from there directly:
#     from tessera.expression.simplify import simplify
# The top-level `from tessera.expression import simplify` re-export
# in `__init__.py` is unchanged.


# ---------------- Evaluation ----------------

def _maybe_broadcast(v, n: int) -> np.ndarray:
    """If v is a scalar, broadcast to a length-n array (1-D). Else return as-is."""
    if np.isscalar(v):
        return np.full(n, float(v), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def evaluate(
    node: Node,
    env: dict[str, np.ndarray],
    cache: FunctionalCache | None = None,
    *,
    fill_warmup: float | None = np.nan,
) -> np.ndarray:
    """Evaluate the tree on the given environment.

    Parameters
    ----------
    node : Node
        Root of the tree.
    env : dict[name -> np.ndarray]
        Maps Var names to their data arrays. All arrays must share length.
    cache : FunctionalCache | None
        If provided, FunctionalOp evaluations are memoized via this cache.
        The `var_id` for each functional argument is `str(subtree)`.
    fill_warmup : float | None
        Forwarded to measure applies (default NaN).

        **Gotcha on composition**: the recursive EMA fast path propagates
        NaN forever once any input is NaN (the y[t] = α·x[t] + (1−α)·y[t−1]
        recursion). So if you compose `ema(diff(x, k), h)`, the warmup
        NaN from the inner `diff` poisons the outer EMA. For nested
        FunctionalOp evaluation, pass `fill_warmup=0.0` to substitute
        zeros for warmup rows. Plain numerical convention; documented at
        the call site.

    Returns
    -------
    np.ndarray
        The evaluated series. Pointwise operations may yield a scalar in
        principle (constant tree), but for any tree with a Var or
        FunctionalOp, the result is a length-N array.
    """
    if isinstance(node, Var):
        if node.name not in env:
            raise KeyError(f"variable {node.name!r} not in env (have {list(env)})")
        return np.asarray(env[node.name], dtype=np.float64)

    if isinstance(node, Const):
        return np.float64(node.value)   # scalar; broadcasts on use

    if isinstance(node, BinOp):
        a = evaluate(node.a, env, cache, fill_warmup=fill_warmup)
        b = evaluate(node.b, env, cache, fill_warmup=fill_warmup)
        return BIN_OP_FNS[node.op](a, b)

    if isinstance(node, UnOp):
        a = evaluate(node.a, env, cache, fill_warmup=fill_warmup)
        return UN_OP_FNS[node.op](a)

    if isinstance(node, FunctionalOp):
        # Resolve arg arrays (broadcasting scalars). Need a length, so peek
        # the env or compute one arg first.
        any_var_name = next(iter(env))
        any_val = env[any_var_name]
        n = any_val.shape[-1] if any_val.ndim == 1 else any_val.shape[0]

        arg_arrays: list[np.ndarray] = []
        arg_ids: list[str] = []
        for arg_node in node.args:
            val = evaluate(arg_node, env, cache, fill_warmup=fill_warmup)
            arg_arrays.append(_maybe_broadcast(val, n))
            arg_ids.append(str(arg_node))

        if cache is not None:
            from .functional import apply_with_cache
            return apply_with_cache(
                node.functional, cache,
                var_ids=tuple(arg_ids),
                xs=tuple(arg_arrays),
                fill_warmup=fill_warmup,
            )
        else:
            return node.functional.apply(*arg_arrays, fill_warmup=fill_warmup)

    if isinstance(node, FunctionalOp2D):
        # The argument must evaluate to a 2-D array (T, X). The Measure2D
        # apply runs the atomic shift-and-accumulate + the separable
        # density along each axis.
        val = evaluate(node.arg, env, cache, fill_warmup=fill_warmup)
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 0:
            # Scalar broadcast — get the env shape from any 2-D Var
            for v in env.values():
                if np.ndim(v) == 2:
                    arr = np.full(v.shape, float(arr), dtype=np.float64)
                    break
            else:
                raise ValueError(
                    "FunctionalOp2D: arg is scalar and no 2-D field in env to broadcast to"
                )
        if arr.ndim != 2:
            raise ValueError(
                f"FunctionalOp2D: arg must evaluate to a 2-D array, got shape {arr.shape}"
            )
        return node.measure_2d.apply(arr, fill_warmup=fill_warmup)

    raise TypeError(f"unknown node type {type(node)}")


__all__ = [
    "Var", "Const", "BinOp", "UnOp", "FunctionalOp", "FunctionalOp2D", "Node",
    "BIN_OPS", "UN_OPS",
    "BIN_OP_FNS", "UN_OP_FNS",
    "complexity", "depth", "used_features", "iter_subtrees", "replace_at",
    "evaluate",
]
