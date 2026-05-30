"""Decomposition-directed csp_sr — break the deep-structure wall TOP-DOWN.

Provenance: docs/research/deep_symbolic_csp.md. Bottom-up stacking
(`discover_deep`) overfits and never beats single-layer enumeration; deep
COMPOSITIONAL structure is reached by DECOMPOSITION (AI-Feynman's core
insight: pre-search structure detection, not better search) wrapped around
csp_sr's exact shallow solver.

`discover_decompose(env, y, cfg)`:
  1. BASE — try `discover()` directly; if it clears the threshold, done.
  2. OUTER-OP PEEL — for phi in {sqrt, square, log, exp, inverse}, transform
     y to the inner target, recurse, wrap, and VERIFY against the original
     y. Reuses `coordinate_discovery.TARGET_TRANSFORMS` / `INVERSE_TRANSFORMS`.
     Example: y = sqrt(inner) -> peel "square" (inner = y^2) -> recurse ->
     wrap sqrt. Recovers sqrt(1 - v^2/c^2) as sqrt(1 + c*v^2) (the constant
     becomes the linear intercept after peeling).
  3. SEPARABILITY — group interacting variables (a cheap interaction graph),
     fit each group against the target (independent sampling => a group-only
     fit recovers that group's additive component up to a constant), jointly
     refit, and VERIFY. Multiplicative separability is reached via the
     "log_abs" peel (log turns a product into a sum) and the power-law
     detector fast-path.

Gradient-free. Crucially, every leaf is a clean-target shallow `discover()`
call (never a fit on a high-variance fitted-intermediate), so the
`discover_deep` conditioning blow-up cannot occur here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from tessera.expression.tree import (
    Node, Var, Const, BinOp, UnOp, evaluate, complexity,
)
from tessera.sr.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.experimental.coordinate_discovery import (
    TARGET_TRANSFORMS, INVERSE_TRANSFORMS,
)
from tessera.search.decompose import detect_power_law, build_power_law_tree


@dataclass
class DecomposeResult:
    expr: Node
    r2: float
    method: str                 # human-readable trace of how it was built
    complexity: int = 0


# --------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------

def _r2(pred, y) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return -np.inf
    ss = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    return float(1.0 - np.sum((y - pred) ** 2) / ss)


def _eval(expr: Node, env: Dict[str, np.ndarray]):
    try:
        v = np.asarray(evaluate(expr, env), dtype=np.float64)
    except Exception:
        return None
    return v


def _linfit(cols: List[np.ndarray], y: np.ndarray):
    """Least-squares y ~ sum c_i * col_i + b. Returns (coefs, b, r2)."""
    A = np.column_stack(list(cols) + [np.ones(len(y))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c[:-1], float(c[-1]), _r2(A @ c, y)


def _combo_expr(coefs, exprs, intercept) -> Node:
    """Balanced add-tree of sum_i coef_i * expr_i + intercept."""
    terms: List[Node] = []
    for c, e in zip(coefs, exprs):
        if abs(c) < 1e-12:
            continue
        terms.append(e if abs(c - 1.0) < 1e-9
                     else BinOp("mul", Const(float(c)), e))
    if abs(intercept) > 1e-9 or not terms:
        terms.append(Const(float(intercept)))
    while len(terms) > 1:
        terms = [BinOp("add", terms[i], terms[i + 1]) if i + 1 < len(terms)
                 else terms[i] for i in range(0, len(terms), 2)]
    return terms[0]


# --------------------------------------------------------------------
# additive-separability grouping (interaction graph)
# --------------------------------------------------------------------

def _gam_residual(env: Dict[str, np.ndarray], y: np.ndarray, degree: int = 3):
    """Residual after a single-variable additive (GAM-ish) polynomial fit.
    What remains is driven by cross-variable INTERACTIONS."""
    cols = []
    for name in env:
        x = env[name]
        for d in range(1, degree + 1):
            cols.append(x ** d)
    if not cols:
        return y - y.mean()
    A = np.column_stack(cols + [np.ones(len(y))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ c


def _interaction_groups(env: Dict[str, np.ndarray], y: np.ndarray,
                        tol: float = 0.04,
                        log: Callable[[str], None] = lambda s: None):
    """Partition variables into groups of mutually-interacting variables.

    Build a graph with an edge (i,j) when some cross-feature of x_i,x_j
    correlates with the additive-model residual (so x_i,x_j genuinely
    interact). Connected components are the groups. Returns the groups, or
    None if there is <2 groups (no useful split) or <4 variables."""
    names = list(env.keys())
    m = len(names)
    if m < 4:
        return None
    rho = _gam_residual(env, y)
    rho = rho - rho.mean()
    rn = float(np.linalg.norm(rho)) + 1e-12
    if rn < 1e-9 * (abs(float(y.mean())) + 1.0):
        return None                          # already additive in single vars

    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    def corr(f):
        f = f - f.mean()
        return abs(float(f @ rho) / ((np.linalg.norm(f) + 1e-12) * rn))

    for i in range(m):
        xi = env[names[i]]
        for j in range(i + 1, m):
            xj = env[names[j]]
            p = xi * xj
            # odd (x_i x_j) AND even ((x_i x_j)^2) cross-features: symmetric
            # data hides interactions in the odd term, so test both.
            if max(corr(p), corr(p * p)) > tol:
                union(i, j)

    comp: Dict[int, List[str]] = {}
    for i in range(m):
        comp.setdefault(find(i), []).append(names[i])
    groups = list(comp.values())
    if len(groups) < 2:
        return None
    # hard structural constraint: reject a DEGENERATE split (one giant blob +
    # a few leftover singletons) — what an unstructured feature set (e.g.
    # image pixels) produces. A real factorization has balanced groups; a
    # blob is just the base fit wearing an "additive" hat.
    if max(len(g) for g in groups) > 0.7 * m:
        return None
    return groups


# --------------------------------------------------------------------
# leaf solver: polynomial (STLSQ) first, then free-form
# --------------------------------------------------------------------

def _poly_nfeat(n: int, d: int) -> int:
    """# monomials of degree 1..d over n variables (combinations w/ repl)."""
    return sum(math.comb(n + k - 1, k) for k in range(1, d + 1))


def _solve_leaf(env: Dict[str, np.ndarray], y: np.ndarray,
                cfg: CSPSRConfig,
                log: Callable[[str], None] = lambda s: None):
    """Solve one node. Peeling an outer nonlinearity usually leaves a
    POLYNOMIAL inner target, and STLSQ recovers polynomials EXACTLY — so try
    polynomial mode at increasing degree (while the monomial library stays
    well-posed, F < 0.9 N) before the free-form enumeration. Returns the
    best CSPSRResult."""
    n, N = len(env), len(y)
    best = None
    if cfg.poly_degree is None:          # only auto-poly if caller didn't pin a mode
        d = 2
        while d <= 8 and _poly_nfeat(n, d) < 0.9 * N:
            r = discover(env, y, replace(cfg, poly_degree=d), log)
            if best is None or r.r2 > best.r2:
                best = r
            if best.r2 >= 1.0 - 1e-9:        # only stop on MACHINE-precision
                return best                  # (a loose stop accepts Class-B
            d += 1                           # poly approximations of smooth y)
    r = discover(env, y, cfg, log)       # free-form fallback
    if best is None or r.r2 > best.r2:
        best = r
    return best


# --------------------------------------------------------------------
# the recursive decomposition driver
# --------------------------------------------------------------------

_PEELS = ("square", "sqrt_abs", "log_abs", "inverse")


def discover_decompose(env: Dict[str, np.ndarray], y: np.ndarray,
                       cfg: Optional[CSPSRConfig] = None,
                       max_depth: int = 3, thresh: float = 1.0 - 1e-9,
                       val_frac: float = 0.2,
                       log: Callable[[str], None] = lambda s: None
                       ) -> DecomposeResult:
    """Recover a symbolic form for y by top-down decomposition around the
    shallow csp_sr solver. Gradient-free, and NOT boosting: each node is a
    single clean-target fit, so there is no ensemble to inflate. Capacity is
    controlled three ways instead of by an L0/shrinkage penalty on a growing
    ensemble:
      - shallow, clean-target leaves (bounded `max_size` / STLSQ degree);
      - a MACHINE-PRECISION short-circuit (`thresh`≈1−1e-9), so a smooth
        target is taken in its true form, never as a Class-B approximation;
      - VALIDATION-GATED acceptance: a peel / separability branch is taken
        only if it beats `base` on a held-out slice (`val_frac`), so the
        structure must GENERALIZE, not merely lower training error. Combined
        with a hard structural constraint that rejects degenerate (blob)
        groupings, this removes the only place the recursion could behave
        like an uncontrolled additive ensemble.
    `.r2` of the result is the held-out score."""
    cfg = cfg or CSPSRConfig()
    y = np.asarray(y, dtype=np.float64)
    env = {k: np.asarray(v, dtype=np.float64) for k, v in env.items()}
    N = len(y)
    if val_frac and N >= 40:
        ntr = N - max(1, int(round(val_frac * N)))
        envf = {k: v[:ntr] for k, v in env.items()}
        envv = {k: v[ntr:] for k, v in env.items()}
        yf, yv = y[:ntr], y[ntr:]
    else:                                    # tiny sample: no held-out gate
        envf, yf, envv, yv = env, y, env, y
    return _decompose(envf, yf, envv, yv, cfg, max_depth, thresh, log, 0)


def _decompose(envf, yf, envv, yv, cfg, max_depth, thresh, log, depth):
    """Inner recursion. `*f` = fit slice, `*v` = held-out validation slice.
    Branches are scored on val; the machine-precision short-circuit uses the
    fit score (an exact structure is exact on both)."""
    ind = "  " * depth

    def vscore(expr):
        return _r2(_eval(expr, envv), yv)

    # 1. base — polynomial STLSQ first, then free-form
    r = _solve_leaf(envf, yf, cfg, log)
    best = (r.expr, r.r2, vscore(r.expr), "base")   # expr, fit_r2, val_r2, method
    log(f"{ind}base fit={r.r2:.4f} val={best[2]:.4f}")
    if r.r2 >= thresh or max_depth <= 0:
        return DecomposeResult(best[0], best[2], best[3], complexity(best[0]))

    # 2. outer-op peel (validation-gated)
    for name in _PEELS:
        innf, innv = TARGET_TRANSFORMS[name](yf), TARGET_TRANSFORMS[name](yv)
        if not np.all(np.isfinite(innf)) or float(np.std(innf)) < 1e-12:
            continue
        sub = _decompose(envf, innf, envv, innv, cfg, max_depth - 1, thresh,
                         log, depth + 1)
        cand = INVERSE_TRANSFORMS[name](sub.expr)
        vf, vv = _r2(_eval(cand, envf), yf), vscore(cand)
        log(f"{ind}peel[{name}] fit={vf:.4f} val={vv:.4f}")
        if vv > best[2] + 1e-6:                       # must beat base on VAL
            best = (cand, vf, vv, f"peel[{name}]->{sub.method}")
        if vf >= thresh and vv >= thresh:
            return DecomposeResult(cand, vv, best[3], complexity(cand))

    # 3. additive separability (degenerate groupings already rejected; the
    #    split is accepted only if it beats base on validation)
    groups = _interaction_groups(envf, yf, log=log)
    if groups:
        gexprs, gcolsf, ok = [], [], True
        for g in groups:
            gsub = _decompose({k: envf[k] for k in g}, yf,
                              {k: envv[k] for k in g}, yv,
                              cfg, max_depth - 1, thresh, log, depth + 1)
            cf = _eval(gsub.expr, envf)
            if cf is None:
                ok = False
                break
            gexprs.append(gsub.expr)
            gcolsf.append(cf)
        if ok:
            coefs, intercept, _ = _linfit(gcolsf, yf)
            expr = _combo_expr(coefs, gexprs, intercept)
            vf, vv = _r2(_eval(expr, envf), yf), vscore(expr)
            log(f"{ind}additive fit={vf:.4f} val={vv:.4f}")
            if vv > best[2] + 1e-6:
                best = (expr, vf, vv, "additive[" + "|".join(
                    "".join(g) for g in groups) + "]")
            if vf >= thresh and vv >= thresh:
                return DecomposeResult(best[0], vv, best[3], complexity(best[0]))

    # 4. multiplicative power-law fast-path (validation-gated)
    fit = detect_power_law(envf, yf, r2_threshold=0.0)
    if fit is not None:
        tree = build_power_law_tree(fit)
        if tree is not None and _eval(tree, envv) is not None:
            vv = vscore(tree)
            if vv > best[2] + 1e-6:
                best = (tree, _r2(_eval(tree, envf), yf), vv, "powerlaw")

    return DecomposeResult(best[0], best[2], best[3], complexity(best[0]))


__all__ = ["DecomposeResult", "discover_decompose"]
