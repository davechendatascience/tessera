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
    max_terms: int = 4           # max terms in the sparse linear fit
    max_features: int = 60000    # cap on the enumerated dictionary (logged)
    parsimony: float = 1e-4      # tie-break toward smaller features
    beam_width: int = 10         # beam search width (1 = greedy OMP)
    topk: int = 24               # candidates expanded per beam entry per step
    recover_thresh: float = 0.9999
    # Sparse-polynomial (SINDy) mode: if set, use a degree-bounded MONOMIAL
    # library + STLSQ (joint least-squares + iterative thresholding) instead
    # of free-form enumeration + forward selection. STLSQ recovers jointly-
    # predictive-but-marginally-uncorrelated terms that forward selection
    # (OMP/beam) is blind to. Requires the library to be small (F < N).
    poly_degree: Optional[int] = None
    stlsq_threshold: float = 0.05
    # Canonical constant leaves (e.g. [1.0, 0.5, 2.0]). The const-free
    # enumeration cannot place a constant INSIDE a nonlinearity (the `1`
    # in sqrt(1−v²/c²), the `½` in exp(−x²/2)) — those are neither
    # variables nor out-front linear coefficients. Adding a few canonical
    # constants as LEAVES fixes the common cases with no optimization
    # (they evaluate to fixed columns, still a linear fit).
    const_leaves: List[float] = field(default_factory=list)
    # Nonlinear-constant refine (for ARBITRARY embedded constants that
    # canonical leaves can't cover, e.g. sin(2.3·x)). Bounded fallback:
    # enumerate small templates with ONE free-constant placeholder and fit
    # it by least-squares. Reintroduces per-template nonlinear optimization,
    # so it's gated (runs only when the linear fit is not already exact).
    nonlinear_const: bool = False
    nl_max_size: int = 3
    nl_max_templates: int = 1200


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

    cap = cfg.max_features                 # bound per-size to avoid blow-up
    if size == 0:
        for name in feature_names:
            add(Var(name))
        for c in cfg.const_leaves:         # canonical embedded constants
            add(Const(float(c)))
        memo[size] = out
        return out
    for op in cfg.unary:
        for c in _gen_trees(size - 1, feature_names, cfg, memo):
            add(UnOp(op, c))
        if len(out) >= cap:
            memo[size] = out
            return out
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
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break
        if len(out) >= cap:
            break
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

def _beam_search(Phi, y, sizes, max_terms, recover_thresh, parsimony,
                 beam_width, topk):
    """Beam search over feature SUBSETS — the less-myopic selector.

    Greedy OMP commits to its first feature and can't backtrack; if the
    true terms aren't individually the most-correlated (they usually
    aren't for multi-term targets), greedy never assembles them. Beam
    search keeps the top-`beam_width` partial subsets so a true term
    survives even when it isn't the single best next pick, and the subsets
    are ranked with a parsimony tie-break (fewer terms, then smaller
    features) — a lightweight Pareto preference among equal-accuracy fits.
    """
    N, F = Phi.shape
    ones = np.ones((N, 1))
    ybar = float(y.mean())
    ss_tot = float(np.sum((y - ybar) ** 2)) + 1e-30
    Phic = Phi - Phi.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Phic, axis=0) + 1e-12
    sizes = np.asarray(sizes, dtype=np.float64)

    # beam entry: (selected_tuple, coef, intercept, residual, r2)
    beam = [((), np.zeros(0), ybar, y - ybar, 0.0)]
    best = beam[0]
    for _ in range(max_terms):
        scored, seen = [], set()
        for sel, _c, _b, r, _r2 in beam:
            rn = np.linalg.norm(r) + 1e-12
            score = np.abs(Phic.T @ r) / (norms * rn) - parsimony * sizes
            for k in sel:
                score[k] = -1e9
            for k in np.argsort(-score)[:topk]:
                trial = tuple(sorted(sel + (int(k),)))
                if trial in seen:
                    continue
                seen.add(trial)
                A = np.hstack([ones, Phi[:, list(trial)]])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                r_new = y - A @ c
                r2 = 1.0 - float(np.sum(r_new ** 2)) / ss_tot
                scored.append((r2, trial, c, r_new))
        if not scored:
            break
        scored.sort(key=lambda s: (-s[0], len(s[1]),
                                   float(sizes[list(s[1])].sum())))
        beam = [(tr, c[1:], float(c[0]), rnew, r2)
                for (r2, tr, c, rnew) in scored[:beam_width]]
        if beam[0][4] > best[4]:
            best = beam[0]
        if best[4] > recover_thresh:
            break
    sel, coef, intercept, _r, r2 = best
    return list(sel), coef, intercept, r2


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


def _monomial_features(feature_names, degree):
    """Degree-bounded monomial library (products of variables) as tessera
    trees — the SINDy-style dictionary for sparse polynomial dynamics."""
    from itertools import combinations_with_replacement as cwr
    feats = []
    for deg in range(1, degree + 1):
        for combo in cwr(feature_names, deg):
            node: Node = Var(combo[0])
            for nm in combo[1:]:
                node = BinOp("mul", node, Var(nm))
            feats.append(node)
    return feats


def _stlsq(Phi, y, threshold, max_iter=12):
    """Sequential thresholded least squares (Brunton et al. 2016, SINDy).
    Joint least-squares fit over ALL features, then iteratively zero
    coefficients below `threshold` and refit on the survivors. Sees
    jointly-predictive terms regardless of marginal correlation."""
    N = Phi.shape[0]
    A = np.hstack([np.ones((N, 1)), Phi])
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    support = np.ones(A.shape[1], dtype=bool)
    for _ in range(max_iter):
        small = np.abs(c) < threshold
        small[0] = False                       # never threshold the intercept
        new_support = ~small
        if np.array_equal(new_support, support) or not new_support[1:].any():
            break
        support = new_support
        c = np.zeros(A.shape[1])
        c[support], *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
    r2 = 1.0 - float(np.sum((y - A @ c) ** 2)) / ss_tot
    return c[1:], float(c[0]), r2


# --------------------------------------------------------------------
# Nonlinear-constant refine (bounded fallback for embedded constants)
# --------------------------------------------------------------------

_FREE_CONST = "__c__"


def _uses_free(t):
    if isinstance(t, Var):
        return t.name == _FREE_CONST
    if isinstance(t, Const):
        return False
    if isinstance(t, UnOp):
        return _uses_free(t.a)
    return _uses_free(t.a) or _uses_free(t.b)


def _eval_raw(t, envf, theta, N):
    if isinstance(t, Var):
        return np.full(N, theta) if t.name == _FREE_CONST else envf[t.name]
    if isinstance(t, Const):
        return np.full(N, float(t.value))
    if isinstance(t, UnOp):
        return np.asarray(UN_OP_FNS[t.op](_eval_raw(t.a, envf, theta, N)), float)
    return np.asarray(BIN_OP_FNS[t.op](_eval_raw(t.a, envf, theta, N),
                                       _eval_raw(t.b, envf, theta, N)), float)


def _subst_free(t, theta):
    if isinstance(t, Var):
        return Const(float(theta)) if t.name == _FREE_CONST else t
    if isinstance(t, Const):
        return t
    if isinstance(t, UnOp):
        return UnOp(t.op, _subst_free(t.a, theta))
    return BinOp(t.op, _subst_free(t.a, theta), _subst_free(t.b, theta))


def _nonlinear_refine(env, y, cfg, log=lambda s: None):
    """Fit a free embedded constant in a small template by least squares.
    Returns (rel, tree_with_free_const, (theta, a, b)) for the best
    template found, or (inf, None, None)."""
    from scipy.optimize import least_squares
    feature_names = list(env.keys())
    N = len(y)
    envf = {k: np.asarray(env[k], dtype=np.float64) for k in feature_names}
    names = feature_names + [_FREE_CONST]
    memo, templates = {}, []
    for size in range(1, cfg.nl_max_size + 1):
        for t in _gen_trees(size, names, cfg, memo):
            if _uses_free(t):
                templates.append(t)
        if len(templates) >= cfg.nl_max_templates:
            templates = templates[:cfg.nl_max_templates]
            break
    log(f"nonlinear refine: {len(templates)} free-const templates")
    ss = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    best = (float("inf"), None, None)
    for t in templates:
        def resid(p):
            theta, a, b = p
            v = _eval_raw(t, envf, theta, N)
            v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            return a * v + b - y
        for theta0 in (0.5, 1.0, 2.0):
            try:
                r = least_squares(resid, [theta0, 1.0, 0.0], max_nfev=50)
            except Exception:
                continue
            rel = float(np.sum(r.fun ** 2) / ss)
            if rel < best[0]:
                best = (rel, t, tuple(r.x))
        if best[0] < 1e-10:
            break
    return best


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

    # ---- candidate feature trees: monomial library or free enumeration ----
    if cfg.poly_degree is not None:
        candidate_trees = _monomial_features(feature_names, cfg.poly_degree)
    else:
        memo = {}
        candidate_trees = [t for size in range(cfg.max_size + 1)
                           for t in _gen_trees(size, feature_names, cfg, memo)]

    # ---- build the dictionary (evaluate, dedup, filter) ----
    feats, cols, sizes, seen = [], [], [], set()
    for t in candidate_trees:
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
            log(f"feature cap {cfg.max_features} hit")
            break
    if not feats:
        return CSPSRResult(expr=Const(float(y.mean())), r2=0.0,
                           complexity=1, n_terms=0, n_features=0,
                           intercept=float(y.mean()))
    Phi = np.stack(cols, axis=1)
    log(f"dictionary: {Phi.shape[1]} features")

    # ---- fit ----
    if cfg.poly_degree is not None:                  # SINDy: joint fit + threshold
        coefs_all, intercept, r2 = _stlsq(Phi, y, cfg.stlsq_threshold)
        selected = [i for i in range(len(feats)) if abs(coefs_all[i]) > 1e-9]
        coef = [coefs_all[i] for i in selected]
    elif cfg.beam_width <= 1:                         # greedy OMP
        selected, coef, intercept, r2 = _omp(
            Phi, y, sizes, cfg.max_terms, cfg.recover_thresh, cfg.parsimony)
    else:                                            # beam search (forward)
        selected, coef, intercept, r2 = _beam_search(
            Phi, y, sizes, cfg.max_terms, cfg.recover_thresh, cfg.parsimony,
            cfg.beam_width, cfg.topk)
    sel_feats = [feats[k] for k in selected]
    expr = _build_expr(intercept, coef, sel_feats)

    # Nonlinear-constant refine (gated fallback): only if the linear fit
    # is not already exact, and only when enabled. Recovers ARBITRARY
    # embedded constants the const-free dictionary cannot reach.
    if cfg.nonlinear_const and (1.0 - r2) > 1e-8:
        rel_nl, t_nl, params = _nonlinear_refine(env, y, cfg, log)
        if t_nl is not None and rel_nl < (1.0 - r2):
            theta, a, b = params
            sub = _subst_free(t_nl, theta)
            expr_nl = _build_expr(b, [a], [sub])
            return CSPSRResult(
                expr=expr_nl, r2=float(1.0 - rel_nl), complexity=complexity(expr_nl),
                n_terms=1, n_features=Phi.shape[1],
                terms=[(float(a), _key(sub))], intercept=float(b))

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
