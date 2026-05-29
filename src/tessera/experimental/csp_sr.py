"""CSP-enumerated symbolic regression over tessera's vocabulary (D1).

Provenance: docs/research/differentiable_eml_jax.md — "Method D", graduated
from the diff_sr exploration to use tessera's real operator vocabulary and
emit real tessera `Expr` trees.

Idea (gradient-free, no iterative refinement)
---------------------------------------------
The structure search is a constraint-satisfaction problem (Knuth TAOCP
Vol 4 Fascicle 7 §7.2.2.3). We enumerate CONST-FREE tessera expression
trees — the "dictionary" — by increasing size, with:
  - symmetry-breaking (commutative children ordered, canonical-key dedup),
  - numerical dedup (drop columns equal to an earlier one, e.g. neg(neg)),
  - connectivity automatic (every tree node feeds the root).
Constants then enter ONLY as LINEAR coefficients: y ≈ c0 + Σ cₖ·φₖ(x),
fit by orthogonal matching pursuit (closed-form least squares per step).

Why no differentiability: once the structure is fixed, linear-in-parameter
constants are solved EXACTLY by least squares — strictly better than
gradient descent. The linear basis already captures affine offsets, phase
(A·sin + B·cos = C·sin(x+φ)) and amplitude. The only thing it cannot reach
is a constant buried *inside* a nonlinearity (e.g. sin(c·x), non-integer
c) — the known limit; a cheap 1-step Gauss-Newton refine on the selected
feature is the extension (not implemented here).

This is the SINDy idea with a CSP-generated dictionary over tessera's ops.
Output is a tessera `Expr`, so it composes with simplify / complexity /
the benchmark verdict classifiers.

Status: untested at module-add time.
Graduation: recovers a meaningful fraction of Feynman + dynamical-systems
targets gradient-free, competitively with GP, much faster.
Removal: enumeration explodes before reaching useful structures and adds
nothing over the existing GP path.
Initial commit: 2026-05-30   Last evaluation: never
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from tessera.expression.tree import (
    Node, Var, Const, BinOp, UnOp,
    BIN_OP_FNS, UN_OP_FNS, complexity,
)


# Sensible SR defaults (subset of tessera's full vocabulary). Pass the full
# key lists of BIN_OP_FNS / UN_OP_FNS to use the "original op vocab".
DEFAULT_UNARY = ["neg", "sqrt", "log", "exp", "sin", "cos", "tanh", "abs"]
DEFAULT_BINARY = ["add", "sub", "mul", "div"]
COMMUTATIVE = {"add", "mul", "min", "max"}


@dataclass
class CSPSRConfig:
    unary: List[str] = field(default_factory=lambda: list(DEFAULT_UNARY))
    binary: List[str] = field(default_factory=lambda: list(DEFAULT_BINARY))
    max_size: int = 4            # max feature-tree size (operator nodes)
    max_terms: int = 4           # OMP terms (sparsity of the linear fit)
    max_features: int = 60000    # cap on the enumerated dictionary (logged)
    parsimony: float = 1e-4      # tie-break toward smaller features in OMP
    recover_thresh: float = 0.9999


@dataclass
class CSPSRResult:
    expr: Node                   # tessera Expr (the fitted symbolic form)
    r2: float
    complexity: int
    n_terms: int
    n_features: int
    terms: list = field(default_factory=list)   # [(coef, feature_str), ...]
    intercept: float = 0.0


# --------------------------------------------------------------------
# Canonical key (symmetry-breaking + dedup) and size
# --------------------------------------------------------------------

def _key(n: Node) -> str:
    if isinstance(n, Var):
        return f"v:{n.name}"
    if isinstance(n, Const):
        return f"k:{n.value}"
    if isinstance(n, UnOp):
        return f"{n.op}({_key(n.a)})"
    return f"{n.op}({_key(n.a)},{_key(n.b)})"


def _size(n: Node) -> int:
    if isinstance(n, (Var, Const)):
        return 0
    if isinstance(n, UnOp):
        return 1 + _size(n.a)
    return 1 + _size(n.a) + _size(n.b)


# --------------------------------------------------------------------
# Const-free tree enumeration (the CSP dictionary)
# --------------------------------------------------------------------

def _gen_trees(size, feature_names, cfg, memo):
    if size in memo:
        return memo[size]
    out, seen = [], set()

    def add(t):
        k = _key(t)
        if k not in seen:
            seen.add(k); out.append(t)

    if size == 0:
        for name in feature_names:
            add(Var(name))
        memo[size] = out
        return out
    for op in cfg.unary:
        for c in _gen_trees(size - 1, feature_names, cfg, memo):
            add(UnOp(op, c))
    for op in cfg.binary:
        comm = op in COMMUTATIVE
        for ls in range(size):
            L = _gen_trees(ls, feature_names, cfg, memo)
            R = _gen_trees(size - 1 - ls, feature_names, cfg, memo)
            for cl in L:
                kl = _key(cl)
                for cr in R:
                    if comm and kl > _key(cr):
                        continue
                    add(BinOp(op, cl, cr))
    memo[size] = out
    return out


# --------------------------------------------------------------------
# Column evaluation with shared-subexpression cache (tessera op fns)
# --------------------------------------------------------------------

def _make_evaluator(env: Dict[str, np.ndarray]):
    col_cache: Dict[str, np.ndarray] = {}

    def ev(n: Node) -> np.ndarray:
        k = _key(n)
        c = col_cache.get(k)
        if c is not None:
            return c
        if isinstance(n, Var):
            v = np.asarray(env[n.name], dtype=np.float64)
        elif isinstance(n, Const):
            v = np.full_like(next(iter(env.values())), float(n.value), dtype=np.float64)
        elif isinstance(n, UnOp):
            v = np.asarray(UN_OP_FNS[n.op](ev(n.a)), dtype=np.float64)
        else:
            v = np.asarray(BIN_OP_FNS[n.op](ev(n.a), ev(n.b)), dtype=np.float64)
        col_cache[k] = v
        return v

    return ev


# --------------------------------------------------------------------
# Orthogonal matching pursuit (closed-form, gradient-free)
# --------------------------------------------------------------------

def _omp(Phi, y, sizes, max_terms, recover_thresh, parsimony):
    N, F = Phi.shape
    ones = np.ones((N, 1))
    ybar = float(y.mean())
    ss_tot = float(np.sum((y - ybar) ** 2)) + 1e-30
    Phic = Phi - Phi.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Phic, axis=0) + 1e-12
    sizes = np.asarray(sizes, dtype=np.float64)
    selected, coef, intercept, r, best_r2 = [], np.zeros(0), ybar, y - ybar, 0.0
    for _ in range(max_terms):
        rn = np.linalg.norm(r) + 1e-12
        rho = np.abs(Phic.T @ r) / (norms * rn)       # normalized corr [0,1]
        scores = rho - parsimony * sizes              # parsimony tie-break
        if selected:
            scores[selected] = -1e9
        k = int(np.argmax(scores))
        trial = selected + [k]
        A = np.hstack([ones, Phi[:, trial]])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r_new = y - A @ c
        r2 = 1.0 - float(np.sum(r_new ** 2)) / ss_tot
        if r2 <= best_r2 + 1e-9 and selected:
            break
        selected, coef, intercept, r, best_r2 = trial, c[1:], float(c[0]), r_new, r2
        if best_r2 > recover_thresh:
            break
    return selected, coef, intercept, best_r2


# --------------------------------------------------------------------
# Assemble the fitted tessera Expr
# --------------------------------------------------------------------

def _build_expr(intercept, coefs, feats) -> Node:
    terms: List[Node] = []
    for c, f in zip(coefs, feats):
        if abs(c) < 1e-9:
            continue
        terms.append(f if abs(c - 1.0) < 1e-9 else BinOp("mul", Const(float(c)), f))
    if abs(intercept) > 1e-9 or not terms:
        terms.append(Const(float(intercept)))
    expr = terms[0]
    for t in terms[1:]:
        expr = BinOp("add", expr, t)
    return expr


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------

def discover(env: Dict[str, np.ndarray], y: np.ndarray,
             cfg: Optional[CSPSRConfig] = None,
             log: Callable[[str], None] = lambda s: None) -> CSPSRResult:
    """Discover a symbolic form for `y` from features in `env`.

    `env`: {feature_name: array(N,)}.  `y`: array(N,).
    Returns a CSPSRResult whose `.expr` is a tessera Node."""
    cfg = cfg or CSPSRConfig()
    y = np.asarray(y, dtype=np.float64)
    feature_names = list(env.keys())
    ev = _make_evaluator(env)

    memo = {}
    feats, cols, sizes, seen = [], [], [], set()
    capped = False
    for size in range(0, cfg.max_size + 1):
        for t in _gen_trees(size, feature_names, cfg, memo):
            try:
                col = ev(t)
            except Exception:
                continue
            if col.shape != y.shape or not np.all(np.isfinite(col)) \
                    or float(np.std(col)) < 1e-9:
                continue
            h = hash(np.round(col, 6).tobytes())
            if h in seen:
                continue
            seen.add(h)
            feats.append(t); cols.append(col); sizes.append(_size(t))
            if len(feats) >= cfg.max_features:
                capped = True
                break
        if capped:
            log(f"feature cap {cfg.max_features} hit at size {size}")
            break
    if not feats:
        return CSPSRResult(expr=Const(float(y.mean())), r2=0.0,
                           complexity=1, n_terms=0, n_features=0,
                           intercept=float(y.mean()))
    Phi = np.stack(cols, axis=1)
    log(f"dictionary: {Phi.shape[1]} features (sizes 0..{cfg.max_size})")

    selected, coef, intercept, r2 = _omp(
        Phi, y, sizes, cfg.max_terms, cfg.recover_thresh, cfg.parsimony)
    sel_feats = [feats[k] for k in selected]
    expr = _build_expr(intercept, coef, sel_feats)
    return CSPSRResult(
        expr=expr, r2=float(r2), complexity=complexity(expr),
        n_terms=len(selected), n_features=Phi.shape[1],
        terms=[(float(c), _key(f)) for c, f in zip(coef, sel_feats)],
        intercept=float(intercept),
    )


def expr_to_str(n: Node) -> str:
    """Readable infix-ish string for an Expr (independent of tessera's
    own printer, which we don't depend on here)."""
    if isinstance(n, Var):
        return n.name
    if isinstance(n, Const):
        return f"{n.value:.4g}"
    if isinstance(n, UnOp):
        return f"{n.op}({expr_to_str(n.a)})"
    return f"{n.op}({expr_to_str(n.a)}, {expr_to_str(n.b)})"


__all__ = [
    "CSPSRConfig", "CSPSRResult", "discover", "expr_to_str",
    "DEFAULT_UNARY", "DEFAULT_BINARY",
]
