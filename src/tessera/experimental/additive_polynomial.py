"""Additive polynomial structure detector (Conjecture C8).

Provenance: C8 from discussion 2026-05-29. Complements decompose v2
(which catches MULTIPLICATIVE structure C·∏xᵢ^{aᵢ}) by catching
ADDITIVE structure: sums of monomials with total degree ≤ D.

Theoretical pre-analysis: `docs/research/c8_additive_polynomial.md`.

Status: **UNTESTED** at module-add time. Initial A/B target: Feynman
        benchmark (incremental on top of decompose v2). Pre-analysis
        predicts +2-4 exact transitions on additively-structured
        equations (I.11.19 dot product, possibly I.24.6 with D=4,
        possibly I.18.4 numerator).

The C8 conjecture
-----------------

A meaningful fraction of physical relationships are SUMS of
multiplicative terms — dot products (Σ xᵢ·yᵢ), polynomials in mixed
powers (½·k·(ω²+ω₀²)·x²), additive Lagrangian-style forms. The
existing power-law detector treats these as failures because the
TOTAL log|y| isn't linear in log|xᵢ|; it's a log of a sum.

C8 detects them by fitting y directly via polynomial basis in raw
feature space. For n variables and total-degree cap D:
  - Basis monomials: x₁, x₂, ..., x₁², x₁x₂, ..., up to total degree D
  - Solve OLS: y ≈ c₀ + Σ_α c_α · monomial_α(x)
  - If R² ≥ threshold, build seed tree from top-N coefficients

The seed tree is `Σ c_α · ∏ xᵢ^{eᵢα}` — a tree of sums of products.

Architectural pattern
---------------------

Same as decompose v2 — detect-then-seed. C8 is the multiplicative-
to-additive complement:

| Layer | Catches | Module |
|---|---|---|
| Power-law products | C·∏xᵢ^{aᵢ} | decompose.power_law (production) |
| Exp wrappers | ±exp(C·∏xᵢ^{aᵢ}) | decompose.exp_wrapper (production) |
| Additive polynomial | Σ c_α·monomial_α | **additive_polynomial (this module)** |

Together these cover the most common algebraic structural classes.

Graduation criterion
--------------------
On Feynman, the A/B test runs WITH decompose v2 ENABLED on both arms;
C8 is added to the ON arm via `precomputed_seed_trees`. Graduation
requires:
- At least +2 NEW exact transitions beyond decompose v2 alone
- 0 regressions on currently-exact equations
- At least 1 of the new exacts is a known additive form (I.11.19,
  I.24.6, I.18.4, etc.)

Removal criterion
-----------------
0 new exact transitions OR ≥ 1 regression that doesn't trace to
selection-layer noise.

Initial commit: 2026-05-29
Last evaluation: never

What this module provides
-------------------------

    AdditivePolynomialFit
        Dataclass: feature_names, monomial_labels, coefficients,
        exponents_per_monomial, r2, intercept.

    enumerate_monomials(feature_names, max_degree)
        Build monomial basis spec. Returns list of dicts
        {label, exponents} for each monomial of total degree 1..D.

    detect_additive_polynomial(env, y, max_degree=3, r2_threshold=0.99,
                                top_n=8, coef_threshold=1e-6)
        Fit polynomial via OLS, return fit if R² above threshold.

    build_additive_polynomial_tree(fit)
        Build a Node tree representing the seed expression.

    additive_polynomial_seed(env, y, ...)
        Orchestrator: returns (seed_tree, fit) or (None, None).

Design notes
------------

Why we fit y in raw space (not log y): additive structure can't
generally be recovered from log y because log(a + b) ≠ log a + log b.
The whole point is that the additive layer is what's missing from
log-log regression.

Basis size: for n variables and total-degree D, the number of
monomials is C(n+D, D) - 1 (excluding constant — that's the
intercept term):

  n=2, D=3:  9 monomials
  n=4, D=3:  34 monomials
  n=4, D=4:  69 monomials
  n=9, D=3:  219 monomials
  n=9, D=4:  714 monomials

The basis can get big fast. We cap at max_basis_size and skip
detection if exceeded (signals "data too high-dim for polynomial
detector at this degree"). High-arity Feynman targets (I.9.18 9
variables) won't fit cleanly at D=3 anyway because they're rational
not polynomial — silent skip is the right behavior.

Top-N filtering: after the OLS fit, we keep only the top-N
coefficients by |c|. Without this, the seed tree would have every
monomial (35+ terms for n=4, D=3), making cx huge. Top-N keeps the
seed structurally meaningful while letting the GP refine the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Optional

import numpy as np

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, Node,
)


# ---------------- Monomial enumeration ----------------

def enumerate_monomials(
    feature_names: tuple[str, ...] | list[str],
    max_degree: int,
) -> list[dict]:
    """Enumerate all monomials of total degree 1..max_degree.

    Returns list of dicts:
        {
          'label': str,                     # human-readable e.g. "x*y^2"
          'exponents': dict[name, int],     # {'x':1, 'y':2}
          'total_degree': int,
        }

    Excludes the constant term (degree 0); the intercept is handled
    separately in the OLS fit.
    """
    feature_names = tuple(feature_names)
    n = len(feature_names)
    monomials: list[dict] = []
    for d in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            exp: dict[str, int] = {}
            for i in combo:
                name = feature_names[i]
                exp[name] = exp.get(name, 0) + 1
            parts = []
            for nm in sorted(exp.keys()):
                e = exp[nm]
                if e == 1:
                    parts.append(nm)
                else:
                    parts.append(f"{nm}^{e}")
            monomials.append({
                "label": " * ".join(parts),
                "exponents": exp,
                "total_degree": d,
            })
    return monomials


# ---------------- Result type ----------------

@dataclass
class AdditivePolynomialFit:
    """Result of an additive-polynomial regression fit."""
    feature_names: tuple[str, ...]
    intercept: float
    # Parallel arrays for kept (top-N) terms:
    coefficients: tuple[float, ...]
    monomial_exponents: tuple[dict, ...]    # each = {name: int}
    monomial_labels: tuple[str, ...]
    r2: float
    n_basis_total: int                       # how many monomials were fit
    n_basis_kept: int                        # how many survived top-N + threshold

    def __str__(self) -> str:
        terms = []
        if abs(self.intercept) > 1e-8:
            terms.append(f"{self.intercept:.4g}")
        for c, lbl in zip(self.coefficients, self.monomial_labels):
            sign = "+ " if c >= 0 else "- "
            terms.append(f"{sign}{abs(c):.4g}·{lbl}")
        body = " ".join(terms) if terms else "0"
        return f"AddPoly({body}, R²={self.r2:.4f})"


# ---------------- Detector ----------------

def detect_additive_polynomial(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    max_degree: int = 3,
    r2_threshold: float = 0.99,
    top_n: int = 8,
    coef_threshold: float = 1e-6,
    max_basis_size: int = 500,
    min_valid_frac: float = 0.95,
) -> Optional[AdditivePolynomialFit]:
    """Fit y via polynomial OLS in monomials up to total degree D.

    Returns an AdditivePolynomialFit if R² ≥ r2_threshold; None otherwise.

    Caveats and silent skips
    ------------------------
    - If the monomial basis size exceeds max_basis_size, skip (data
      too high-dim for the chosen D; widening D would explode further).
    - If too many samples have NaN/inf in features or target, skip.
    - If OLS is singular (rank-deficient basis), skip.
    - Top-N filtering keeps the seed tree manageable; the suppressed
      monomials contribute to fit quality but not to the seed.
    """
    feature_names = tuple(env.keys())
    monomials = enumerate_monomials(feature_names, max_degree)
    if len(monomials) > max_basis_size:
        return None

    # Build design matrix.
    n_samples = len(y)
    if n_samples < len(monomials) + 5:
        # Underdetermined / overfitted by construction.
        return None

    # Validity mask.
    valid = np.isfinite(y).copy()
    for name in feature_names:
        valid &= np.isfinite(env[name])
    if valid.mean() < min_valid_frac:
        return None

    yv = np.asarray(y[valid], dtype=np.float64)
    # Variance check: avoid degenerate constant targets.
    if np.var(yv) < 1e-15:
        return None

    cols = []
    for m in monomials:
        col = np.ones(int(valid.sum()), dtype=np.float64)
        for name, e in m["exponents"].items():
            x = np.asarray(env[name][valid], dtype=np.float64)
            col = col * (x ** e)
        cols.append(col)
    X = np.column_stack(cols)
    # Add intercept column.
    A = np.column_stack([X, np.ones(X.shape[0])])

    if not np.all(np.isfinite(X)):
        return None

    try:
        coefs, *_ = np.linalg.lstsq(A, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None

    intercept = float(coefs[-1])
    raw_coefs = coefs[:-1]

    predicted = A @ coefs
    residuals = yv - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    if ss_tot < 1e-12:
        return None
    r2 = 1.0 - ss_res / ss_tot
    if r2 < r2_threshold:
        return None

    # Top-N filter by magnitude.
    abs_coefs = np.abs(raw_coefs)
    order = np.argsort(abs_coefs)[::-1]
    kept_idx: list[int] = []
    for idx in order:
        if abs_coefs[idx] < coef_threshold:
            break
        kept_idx.append(int(idx))
        if len(kept_idx) >= top_n:
            break

    if not kept_idx:
        return None

    kept_coefs = tuple(float(raw_coefs[i]) for i in kept_idx)
    kept_exps = tuple(monomials[i]["exponents"] for i in kept_idx)
    kept_labels = tuple(monomials[i]["label"] for i in kept_idx)

    return AdditivePolynomialFit(
        feature_names=feature_names,
        intercept=intercept,
        coefficients=kept_coefs,
        monomial_exponents=kept_exps,
        monomial_labels=kept_labels,
        r2=float(r2),
        n_basis_total=len(monomials),
        n_basis_kept=len(kept_idx),
    )


# ---------------- Tree builder ----------------

def _build_monomial_tree(exponents: dict[str, int]) -> Node:
    """Build a Node tree for a single monomial ∏ xᵢ^{eᵢ}.

    Conventions:
      e=1: bare Var
      e=2: Var*Var (smaller cx than pow(Var, 2))
      e=3: Var*Var*Var
      e≥4: pow(Var, Const(e))

    Multi-variable monomials are left-folded multiplications.
    """
    factors: list[Node] = []
    for name in sorted(exponents.keys()):
        e = exponents[name]
        if e <= 0:
            continue
        var = Var(name)
        if e == 1:
            factors.append(var)
        elif e == 2:
            factors.append(BinOp("mul", var, var))
        elif e == 3:
            factors.append(BinOp("mul", BinOp("mul", var, var), var))
        else:
            factors.append(BinOp("pow", var, Const(float(e))))
    if not factors:
        return Const(1.0)
    product: Node = factors[0]
    for f in factors[1:]:
        product = BinOp("mul", product, f)
    return product


def build_additive_polynomial_tree(
    fit: AdditivePolynomialFit,
    *,
    intercept_threshold: float = 1e-8,
) -> Optional[Node]:
    """Build the seed tree: Σ c_α · ∏ xᵢ^{eᵢα} + intercept.

    Returns None if all coefficients are below threshold (fit is
    essentially constant — not a useful seed).
    """
    terms: list[Node] = []
    if abs(fit.intercept) > intercept_threshold:
        terms.append(Const(float(fit.intercept)))
    for c, exps in zip(fit.coefficients, fit.monomial_exponents):
        if abs(c) < 1e-12:
            continue
        mono_tree = _build_monomial_tree(exps)
        # If coefficient is essentially 1, omit the multiplier
        if abs(c - 1.0) < 1e-8:
            term = mono_tree
        elif abs(c + 1.0) < 1e-8:
            term = UnOp("neg", mono_tree)
        else:
            term = BinOp("mul", Const(float(c)), mono_tree)
        terms.append(term)
    if not terms:
        return None
    out: Node = terms[0]
    for t in terms[1:]:
        out = BinOp("add", out, t)
    return out


# ---------------- Top-level orchestrator ----------------

def additive_polynomial_seed(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    max_degree: int = 3,
    r2_threshold: float = 0.99,
    top_n: int = 8,
) -> tuple[Optional[Node], Optional[AdditivePolynomialFit]]:
    """Detect additive polynomial structure; build a seed tree if found.

    Returns (tree, fit) suitable for injection into the GP initial
    population (via `precomputed_seed_trees` in the C8 A/B runner,
    per the experimental discipline that production does not import
    from experimental).

    Both are None if no fit at the given threshold.
    """
    fit = detect_additive_polynomial(
        env, y,
        max_degree=max_degree,
        r2_threshold=r2_threshold,
        top_n=top_n,
    )
    if fit is None:
        return None, None
    tree = build_additive_polynomial_tree(fit)
    return tree, fit


__all__ = [
    "AdditivePolynomialFit",
    "enumerate_monomials",
    "detect_additive_polynomial",
    "build_additive_polynomial_tree",
    "additive_polynomial_seed",
]
