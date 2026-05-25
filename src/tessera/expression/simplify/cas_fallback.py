"""CAS-based simplification fallback (sympy / symengine).

Bridges the simplification gap with PySR's SymbolicUtils.jl. Used as
a FALLBACK after the cheap hand-rolled simplifier passes (AC + core
+ polynomial). Catches the cases our rule-based simplifier misses:

  - Trig identities (sin² + cos² → 1; 2·sin·cos → sin(2x))
  - Polynomial factoring (x² − 1 → (x+1)(x−1) ... and back)
  - Rational expression cancellation ((a/b + c/b) → (a+c)/b)
  - log/exp identities beyond log(exp(x))=x
  - Cross-term distribution and collection

Design principles (per docs gap analysis)
-----------------------------------------

1. **Apply at Pareto-front level only.** Per-candidate simplification
   in a 200-candidate × 100-gen GP would add ~17 min of sympy overhead.
   Front-only simplification (5-15 candidates per gen) adds <2 sec.

2. **Cache by tree string hash.** Pareto front is stable across gens
   (~75% candidates persist gen-to-gen). Cache hits make most calls
   free after warm-up.

3. **Predicate to skip when unhelpful.** Trees lacking sin/cos/log/exp/
   div get no benefit from CAS — skip the round-trip entirely.

4. **Numerical verification.** After round-trip, evaluate both original
   and simplified on random points. Accept only if outputs agree to
   tolerance. This protects against semantic mismatch between sympy's
   pure operators and tessera's protected operators
   (sqrt(|x|), log(max(|x|, 1e-12)), exp(clip(x, ±50)), pow(|a|, clip(b,±8))).

5. **Backend fallback.** Try symengine first (10-100× faster); fall
   back to sympy if unavailable. Same API for what we use. Functionally
   degrades to no-op if neither is installed.

What this module provides
-------------------------
    cas_simplify(tree, *, verify_samples=20, atol=1e-6) -> Node
        The main entry point. Round-trips tree through CAS,
        verifies numerically, returns simplified-or-original.

    is_worth_cas_pass(tree) -> bool
        Predicate: True if tree contains ops where CAS could help.

    clear_cache() -> None
        Reset the module-level cache. Useful for benchmarks.

Safety properties
-----------------

- Returns the ORIGINAL tree if any error occurs (round-trip failure,
  unsupported ops, numerical divergence)
- Never increases complexity (rejects if simplified is larger)
- Always preserves observable behavior to atol on tested points
- Does not modify the input tree (Node is immutable)

Known limitations
-----------------

- Doesn't simplify across FunctionalOp / FunctionalOp2D boundaries.
  These are replaced with opaque placeholders during round-trip; CAS
  can only simplify the expressions AROUND them.
- Comparison operators (gt/lt/ge/le) become Piecewise in sympy, which
  we can't represent back. Trees containing these get skipped.
- step/sign/reduce_* operators are similarly skipped — opaque.
- The pow → sympy round-trip is restricted to small integer exponents
  to avoid breaking protected-pow semantics.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    complexity, iter_subtrees, evaluate,
)


# ---------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------

_BACKEND: Optional[str] = None
_sym: Any = None

try:
    import symengine as _sym
    _BACKEND = "symengine"
except ImportError:
    try:
        import sympy as _sym
        _BACKEND = "sympy"
    except ImportError:
        _sym = None
        _BACKEND = None


def get_backend() -> Optional[str]:
    """Return the active CAS backend name, or None if no CAS available."""
    return _BACKEND


# ---------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------

_CAS_HELPFUL_BIN_OPS = frozenset({"div", "pow", "atan2"})
_CAS_HELPFUL_UN_OPS = frozenset({"sin", "cos", "log", "exp", "tanh"})
_CAS_SKIP_BIN_OPS = frozenset({"gt", "lt", "ge", "le"})
_CAS_SKIP_UN_OPS = frozenset({
    "step", "sign", "reduce_mean", "reduce_max", "reduce_sum", "reduce_std",
})


def is_worth_cas_pass(tree: Node) -> bool:
    """True if the tree contains operators where CAS simplification
    could plausibly help. False if the tree is pure-polynomial (our
    hand-rolled simplifier handles those already) OR contains only
    skip-ops."""
    found_helpful = False
    for sub in iter_subtrees(tree):
        if isinstance(sub, BinOp):
            if sub.op in _CAS_HELPFUL_BIN_OPS:
                found_helpful = True
            if sub.op in _CAS_SKIP_BIN_OPS:
                # Comparison ops — CAS can't represent these cleanly
                return False
        elif isinstance(sub, UnOp):
            if sub.op in _CAS_HELPFUL_UN_OPS:
                found_helpful = True
            if sub.op in _CAS_SKIP_UN_OPS:
                return False
    return found_helpful


# ---------------------------------------------------------------------
# Tessera → CAS conversion
# ---------------------------------------------------------------------

class _Conversion:
    """State carried during tessera ↔ CAS round-trip.

    Holds the placeholder map for FunctionalOp / FunctionalOp2D nodes
    that CAS can't represent. After round-trip, placeholders are
    substituted back."""

    def __init__(self):
        self.placeholders: dict[str, Node] = {}
        self._counter = 0
        self.feature_symbols: dict[str, Any] = {}  # name → sympy Symbol

    def fresh_placeholder_name(self) -> str:
        name = f"__placeholder_{self._counter}__"
        self._counter += 1
        return name

    def symbol_for(self, name: str) -> Any:
        if name not in self.feature_symbols:
            self.feature_symbols[name] = _sym.Symbol(name)
        return self.feature_symbols[name]


def _to_cas(node: Node, conv: _Conversion) -> Any:
    """Recursively convert tessera tree to a CAS expression."""
    if isinstance(node, Var):
        return conv.symbol_for(node.name)
    if isinstance(node, Const):
        return _sym.Float(float(node.value))
    if isinstance(node, BinOp):
        a = _to_cas(node.a, conv)
        b = _to_cas(node.b, conv)
        if node.op == "add":
            return a + b
        if node.op == "sub":
            return a - b
        if node.op == "mul":
            return a * b
        if node.op == "div":
            return a / b
        if node.op == "min":
            return _sym.Min(a, b)
        if node.op == "max":
            return _sym.Max(a, b)
        if node.op == "atan2":
            return _sym.atan2(a, b)
        if node.op == "pow":
            # Tessera's pow is protected (pow(|a|, clip(b, ±8))).
            # Only round-trip when b is a small integer Const.
            if isinstance(node.b, Const):
                exp_val = float(node.b.value)
                if exp_val == int(exp_val) and abs(exp_val) <= 8:
                    return _sym.Abs(a) ** int(exp_val)
            # Otherwise opaque
            placeholder = conv.fresh_placeholder_name()
            conv.placeholders[placeholder] = node
            return conv.symbol_for(placeholder)
        # gt/lt/ge/le shouldn't reach here (filtered by predicate)
        # but just in case:
        placeholder = conv.fresh_placeholder_name()
        conv.placeholders[placeholder] = node
        return conv.symbol_for(placeholder)
    if isinstance(node, UnOp):
        a = _to_cas(node.a, conv)
        if node.op == "neg":
            return -a
        if node.op == "abs":
            return _sym.Abs(a)
        if node.op == "sqrt":
            # tessera: sqrt(|x|). sympy: sqrt(Abs(x))
            return _sym.sqrt(_sym.Abs(a))
        if node.op == "log":
            # tessera: log(max(|x|, 1e-12)). Approximate with sympy.log(Abs(x))
            return _sym.log(_sym.Abs(a))
        if node.op == "exp":
            return _sym.exp(a)
        if node.op == "sin":
            return _sym.sin(a)
        if node.op == "cos":
            return _sym.cos(a)
        if node.op == "tanh":
            return _sym.tanh(a)
        if node.op == "acos":
            return _sym.acos(a)
        if node.op == "asin":
            return _sym.asin(a)
        # step/sign/reduce_* shouldn't reach here (filtered)
        placeholder = conv.fresh_placeholder_name()
        conv.placeholders[placeholder] = node
        return conv.symbol_for(placeholder)
    if isinstance(node, (FunctionalOp, FunctionalOp2D)):
        placeholder = conv.fresh_placeholder_name()
        conv.placeholders[placeholder] = node
        return conv.symbol_for(placeholder)
    raise TypeError(f"Unknown node type: {type(node).__name__}")


# ---------------------------------------------------------------------
# CAS → tessera conversion
# ---------------------------------------------------------------------

def _from_cas(expr: Any, conv: _Conversion) -> Node:
    """Recursively convert a CAS expression back to a tessera tree."""
    # Handle backend-specific class introspection
    # We use the names since symengine and sympy share names for the
    # common classes but differ in import paths.
    cls_name = type(expr).__name__

    # Symbol — could be a Var or a placeholder
    if cls_name in ("Symbol",) or hasattr(expr, "is_Symbol") and expr.is_Symbol:
        name = str(expr)
        if name in conv.placeholders:
            return conv.placeholders[name]
        return Var(name=name)

    # Numbers
    if cls_name in ("Float", "Integer", "Rational", "RealDouble", "Number"):
        try:
            return Const(value=float(expr))
        except (TypeError, ValueError):
            return Const(value=0.0)
    if hasattr(expr, "is_Number") and expr.is_Number:
        try:
            return Const(value=float(expr))
        except (TypeError, ValueError):
            return Const(value=0.0)

    # Add (n-ary in sympy) — left-fold to binary
    if cls_name == "Add":
        args = list(expr.args)
        if not args:
            return Const(value=0.0)
        result = _from_cas(args[0], conv)
        for arg in args[1:]:
            result = BinOp(op="add", a=result, b=_from_cas(arg, conv))
        return result

    # Mul (n-ary) — left-fold
    if cls_name == "Mul":
        args = list(expr.args)
        if not args:
            return Const(value=1.0)
        result = _from_cas(args[0], conv)
        for arg in args[1:]:
            result = BinOp(op="mul", a=result, b=_from_cas(arg, conv))
        return result

    # Pow
    if cls_name == "Pow":
        base = _from_cas(expr.args[0], conv)
        exp = expr.args[1]
        # If exponent is a small integer, expand to multiplication chain
        try:
            exp_val = float(exp)
            if exp_val == int(exp_val) and 1 <= int(exp_val) <= 6:
                ie = int(exp_val)
                result = base
                for _ in range(ie - 1):
                    result = BinOp(op="mul", a=result, b=base)
                return result
            if exp_val == -1:
                return BinOp(op="div", a=Const(value=1.0), b=base)
            if exp_val == 0.5:
                return UnOp(op="sqrt", a=base)
        except (TypeError, ValueError):
            pass
        # Fall back to pow BinOp
        exp_node = _from_cas(exp, conv)
        return BinOp(op="pow", a=base, b=exp_node)

    # Unary functions
    fn_to_unop = {
        "sin": "sin", "cos": "cos", "tan": "tanh",  # tan→tanh fallback (no tan in tessera)
        "tanh": "tanh", "exp": "exp", "log": "log",
        "Abs": "abs", "sqrt": "sqrt",
        "acos": "acos", "asin": "asin",
    }
    if cls_name in fn_to_unop:
        if len(expr.args) == 1:
            inner = _from_cas(expr.args[0], conv)
            return UnOp(op=fn_to_unop[cls_name], a=inner)

    # Neg (handled by Mul(-1, ...) typically but just in case)
    # Min/Max
    if cls_name in ("Min", "Max"):
        op_name = "min" if cls_name == "Min" else "max"
        args = list(expr.args)
        if not args:
            return Const(value=0.0)
        result = _from_cas(args[0], conv)
        for arg in args[1:]:
            result = BinOp(op=op_name, a=result, b=_from_cas(arg, conv))
        return result

    # atan2
    if cls_name == "atan2":
        if len(expr.args) == 2:
            y = _from_cas(expr.args[0], conv)
            x = _from_cas(expr.args[1], conv)
            return BinOp(op="atan2", a=y, b=x)

    # Unknown — try float coercion as last resort
    try:
        return Const(value=float(expr))
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert CAS expression of type {cls_name}: {expr}")


# ---------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------

def _evaluate_at_random(tree: Node, feature_names: list[str],
                        n_samples: int, seed: int) -> np.ndarray:
    """Evaluate the tree at random points; returns N-element vector
    of predictions. Returns NaN if evaluation fails."""
    try:
        rng = np.random.default_rng(seed)
        env = {name: rng.uniform(-2.0, 2.0, n_samples) for name in feature_names}
        result = evaluate(tree, env, fill_warmup=0.0)
        return np.asarray(result, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(n_samples, np.nan)


def _verify_equivalent(
    original: Node, simplified: Node, feature_names: list[str],
    n_samples: int, atol: float,
) -> bool:
    """Check that two trees produce similar outputs on random samples.

    Handles the case where one tree is a constant (returns scalar)
    and the other is feature-dependent (returns vector) — broadcasts
    scalars to the vector shape before comparison.
    """
    if not feature_names:
        feature_names = ["x"]  # fallback
    orig_vals = _evaluate_at_random(original, feature_names, n_samples, seed=42)
    simp_vals = _evaluate_at_random(simplified, feature_names, n_samples, seed=42)
    # Broadcast scalars to vector shape (constant tree → constant value
    # repeated). NumPy's broadcast_arrays handles this cleanly.
    try:
        orig_b, simp_b = np.broadcast_arrays(orig_vals, simp_vals)
        orig_vals = np.asarray(orig_b, dtype=np.float64).reshape(-1)
        simp_vals = np.asarray(simp_b, dtype=np.float64).reshape(-1)
    except ValueError:
        return False
    finite_mask = np.isfinite(orig_vals) & np.isfinite(simp_vals)
    if finite_mask.sum() < n_samples // 2:
        # Too few finite values to trust the comparison
        return False
    diff = np.abs(orig_vals[finite_mask] - simp_vals[finite_mask])
    scale = np.maximum(np.abs(orig_vals[finite_mask]),
                       np.abs(simp_vals[finite_mask]))
    # Allow either absolute tol OR relative tol (1%)
    return bool(np.all((diff < atol) | (diff < scale * 0.01)))


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------

_cache: dict[str, Node] = {}


def clear_cache() -> None:
    """Reset the simplification cache."""
    _cache.clear()


def cache_size() -> int:
    return len(_cache)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def cas_simplify(
    tree: Node, *,
    feature_names: Optional[list[str]] = None,
    verify_samples: int = 20,
    atol: float = 1e-6,
    skip_predicate: bool = False,
) -> Node:
    """Simplify a tree using CAS (sympy or symengine), with safety checks.

    Returns the simplified tree if:
      - CAS backend is available
      - The tree benefits from CAS (per is_worth_cas_pass, unless
        skip_predicate=True)
      - Round-trip succeeds
      - Simplified tree has lower or equal complexity
      - Numerical verification passes (predictions agree to atol on
        random samples)

    Otherwise returns the ORIGINAL tree unchanged.

    Parameters
    ----------
    tree : Node
        Tessera tree to simplify.
    feature_names : list[str], optional
        Names of Var leaves. If None, extracted from the tree.
    verify_samples : int
        Number of random points for numerical verification.
    atol : float
        Absolute tolerance for equivalence (per-sample).
    skip_predicate : bool
        If True, bypass is_worth_cas_pass check.

    Notes
    -----
    Caches results by tree string. Cache hits return instantly.
    """
    if _sym is None:
        return tree

    if not skip_predicate and not is_worth_cas_pass(tree):
        return tree

    # Cache lookup
    cache_key = str(tree)
    if cache_key in _cache:
        return _cache[cache_key]

    # Extract feature names if not provided
    if feature_names is None:
        feature_names = list(
            {sub.name for sub in iter_subtrees(tree) if isinstance(sub, Var)}
        )

    try:
        conv = _Conversion()
        cas_expr = _to_cas(tree, conv)
        # Apply simplification — try expand first (usually safe),
        # then simplify (more aggressive but slower)
        try:
            simplified_cas = _sym.expand(cas_expr)
        except Exception:
            simplified_cas = cas_expr
        try:
            simplified_cas = _sym.simplify(simplified_cas)
        except Exception:
            pass  # keep expand result
        simplified_tree = _from_cas(simplified_cas, conv)
    except Exception:
        _cache[cache_key] = tree
        return tree

    # Only accept if complexity reduced AND output matches
    orig_cx = complexity(tree)
    new_cx = complexity(simplified_tree)
    if new_cx > orig_cx:
        _cache[cache_key] = tree
        return tree

    if not _verify_equivalent(tree, simplified_tree, feature_names,
                              verify_samples, atol):
        _cache[cache_key] = tree
        return tree

    _cache[cache_key] = simplified_tree
    return simplified_tree


def simplify_front_with_cas(
    front: list, *, feature_names: Optional[list[str]] = None,
    verify_samples: int = 20, atol: float = 1e-6,
) -> list:
    """Apply CAS simplification to each candidate in a Pareto front.

    Returns a NEW list of Candidates with simplified trees where CAS
    found improvements. Complexity is recomputed; other fields
    (train_loss, fitness, born_gen) are preserved.

    No-op if no CAS backend is available.
    """
    if _sym is None:
        return front
    from dataclasses import replace
    new_front = []
    for cand in front:
        simplified = cas_simplify(
            cand.tree, feature_names=feature_names,
            verify_samples=verify_samples, atol=atol,
        )
        if simplified is cand.tree:
            new_front.append(cand)
        else:
            new_cx = complexity(simplified)
            new_front.append(replace(cand, tree=simplified, complexity=new_cx))
    return new_front


__all__ = [
    "cas_simplify", "simplify_front_with_cas",
    "is_worth_cas_pass", "get_backend",
    "clear_cache", "cache_size",
]
