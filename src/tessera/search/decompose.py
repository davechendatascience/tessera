"""Decomposition pre-pass — detect structural symmetries before search.

Step 3 of the Feynman improvement plan (this session). AI Feynman's
core advantage is not better search — it's pre-search structure
detection: dimensional analysis, separability tests, translation
symmetries, etc. Each detected symmetry shrinks the effective search
space, often turning a 5-variable problem into a 2-variable one.

MVP scope (this module)
-----------------------
Multiplicative separability via log-log linear regression. A function
f(x_1, ..., x_n) is multiplicatively separable in all its variables
iff:

    log|f| = a_0 + Σ a_i · log|x_i|

I.e., f = C · ∏ x_i^{a_i} (a power-law product). This form covers a
meaningful fraction of Feynman targets:

    I.12.1   mu·Nn                   exponents = (1, 1)
    I.12.5   q1·q2/r^2               exponents = (1, 1, -2)
    I.14.3   m·g·z                   exponents = (1, 1, 1)
    I.29.4   omega/c                 exponents = (1, -1)
    I.34.8   q·v·B/p                 exponents = (1, 1, 1, -1)
    I.39.22  n·k·T/V                 exponents = (1, 1, 1, -1)
    I.43.31  k·T/(6·pi·eta·r)        exponents = (1, 1, -1, -1)  C=1/(6π)
    I.43.43  kappa·v^2/(n·sigma)     exponents = (1, 2, -1, -1)

Non-power-law targets (Gaussians, Lorentz boosts, distance formulas,
trigonometric) won't pass the R² threshold and will be silently
skipped by the pre-pass.

Output: a seed tree to inject into the GP initial population. If the
seed is correct, the GP will pick it up via const-opt polish; if the
seed is misleading, normal selection pressure will discard it.

Independent justification (§6.1)
-------------------------------
Power laws are a universal mathematical class; log-log linear
regression has been used to identify them since the 19th century
(Pareto, Zipf, Newton). The test does not reference the Feynman
benchmark. The seed-injection mechanism preserves GP semantics — it
adds one candidate to the initial population without changing search,
scoring, or selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, Node,
)


# ---------------- Result type ----------------

@dataclass
class PowerLawFit:
    """Result of a power-law regression fit."""
    feature_names: tuple[str, ...]
    exponents: tuple[float, ...]
    constant: float
    r2: float
    log_residual_std: float

    def __str__(self) -> str:
        terms = [f"{name}^{e:.3f}" for name, e in
                 zip(self.feature_names, self.exponents) if abs(e) > 0.01]
        return f"{self.constant:.4g} * {' * '.join(terms)}  (R²={self.r2:.4f})"


@dataclass
class ExpWrapperFit:
    """Result of an exp-wrapper fit: f = exp(sign · inner)."""
    inner: PowerLawFit  # the power-law fit to |log|y||
    sign: int           # +1 if log|y| ≥ 0 (exp(+inner)); -1 if log|y| ≤ 0

    def __str__(self) -> str:
        sign_str = "" if self.sign > 0 else "-"
        return f"exp({sign_str}{self.inner})"


# ---------------- Detector ----------------

def detect_power_law(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    r2_threshold: float = 0.99,
    min_positive_frac: float = 0.95,
    log_floor: float = 1e-300,
) -> Optional[PowerLawFit]:
    """Test if log|y| ≈ a_0 + Σ a_i log|x_i| holds with R² ≥ threshold.

    Returns a PowerLawFit if the test passes; None otherwise. The test
    is per-equation (one fit total, not per-variable).

    Caveats and silent skips
    ------------------------
    - If y has too many non-positive values (< min_positive_frac of
      |y| > 1e-12), magnitude regression isn't meaningful → skip.
    - If any x_i has too many non-positive values, skip.
    - If log values aren't all finite (overflow/zero), skip.
    - Caller is responsible for choosing whether to act on the result.

    The function is read-only on env and y.
    """
    feature_names = tuple(env.keys())
    n = len(y)

    # y-side check: log|y| must be defined for enough samples.
    abs_y = np.abs(y)
    pos_y_mask = abs_y > 1e-12
    if pos_y_mask.mean() < min_positive_frac:
        return None
    if not np.all(np.isfinite(y)):
        return None

    # x-side check: same per-variable.
    cols: list[np.ndarray] = []
    for name in feature_names:
        x = np.asarray(env[name])
        if x.shape[0] != n:
            return None
        abs_x = np.abs(x)
        if (abs_x > 1e-12).mean() < min_positive_frac:
            return None
        cols.append(np.log(np.maximum(abs_x, log_floor)))
    if not cols:
        return None

    log_y = np.log(np.maximum(abs_y, log_floor))
    if not np.all(np.isfinite(log_y)):
        return None

    # Mask to samples where all log values are finite.
    valid = np.isfinite(log_y).copy()
    for c in cols:
        valid &= np.isfinite(c)
    if valid.sum() < max(50, n // 4):
        return None

    X = np.column_stack(cols)[valid]
    yv = log_y[valid]
    # Design matrix with intercept column.
    A = np.column_stack([X, np.ones(len(X))])
    try:
        coefs, *_ = np.linalg.lstsq(A, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None

    exponents = tuple(float(e) for e in coefs[:-1])
    log_constant = float(coefs[-1])

    predicted = A @ coefs
    residuals = yv - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    if ss_tot < 1e-12:
        # y is essentially constant → degenerate.
        return None
    r2 = 1.0 - ss_res / ss_tot
    log_residual_std = float(np.std(residuals))

    if r2 < r2_threshold:
        return None

    return PowerLawFit(
        feature_names=feature_names,
        exponents=exponents,
        constant=float(np.exp(log_constant)),
        r2=float(r2),
        log_residual_std=log_residual_std,
    )


# ---------------- exp-wrapper detector ----------------

def detect_exp_wrapper(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    r2_threshold: float = 0.99,
    min_positive_frac: float = 0.95,
    min_sign_consistency: float = 0.95,
    log_floor: float = 1e-300,
) -> Optional[ExpWrapperFit]:
    """Test if y = ±exp(C · ∏ x_i^{a_i}) by applying power-law detection to
    log|y|.

    Returns an ExpWrapperFit if the test passes; None otherwise. The
    detector requires:
      1. log|y| has consistent sign on ≥ `min_sign_consistency` of samples
         (mixed-sign would mean the inner power-law product crosses zero,
         which a positive-only ∏x^a product can't do).
      2. The absolute value |log|y|| is well-fit by a power-law in the
         variables at R² ≥ r2_threshold.

    Caveats and silent skips
    ------------------------
    - Pure-power-law forms (f itself a power-law product) will ALSO pass
      this test (log|f| = a_0 + Σ a_i log|x_i| is itself a sum, which
      after taking |...| and power-law-fitting may give a high R²). The
      orchestrator (`power_law_seed`) handles disambiguation by trying
      base power-law first.
    - Forms with additive offsets inside the exp (e.g., I.40.1: n_0 ·
      exp(-mgx/(kT)) has log|f| = log(n_0) - mgx/(kT)) are rejected
      because the constant log(n_0) prevents the inner power-law from
      fitting cleanly.
    """
    abs_y = np.abs(y)
    if (abs_y > 1e-12).mean() < min_positive_frac:
        return None
    if not np.all(np.isfinite(y)):
        return None

    log_abs_y = np.log(np.maximum(abs_y, log_floor))
    if not np.all(np.isfinite(log_abs_y)):
        return None

    # Sign consistency check.
    n_pos = int((log_abs_y > 0).sum())
    n_neg = int((log_abs_y < 0).sum())
    n_total = len(log_abs_y)
    if n_pos / n_total >= min_sign_consistency:
        sign = +1
    elif n_neg / n_total >= min_sign_consistency:
        sign = -1
    else:
        return None  # mixed sign → not a pure exp wrapper

    # Build a new target: |log|y||. Skip samples where it's ~0 (y ≈ 1)
    # because log(0) would be -inf and the power-law regression can't
    # handle it; equivalently, those samples don't constrain the exponents.
    target = np.abs(log_abs_y)
    keep_mask = target > 1e-9
    if keep_mask.sum() < max(50, n_total // 4):
        return None

    # Construct a filtered env and pass to the power-law detector.
    filtered_env = {name: np.asarray(arr)[keep_mask]
                    for name, arr in env.items()}
    filtered_target = target[keep_mask]

    inner = detect_power_law(
        filtered_env, filtered_target,
        r2_threshold=r2_threshold,
        min_positive_frac=min_positive_frac,
        log_floor=log_floor,
    )
    if inner is None:
        return None

    return ExpWrapperFit(inner=inner, sign=sign)


# ---------------- Tree builder ----------------

def _maybe_round_exponent(a: float, snap_tol: float = 0.02) -> float:
    """Round exponent to nearest integer or simple rational if close.

    Returns the rounded value or `a` unchanged. Snap targets:
      integers, halves (k/2), thirds (k/3) for |k| ≤ 8.
    """
    candidates: list[float] = []
    for k in range(-8, 9):
        candidates.append(float(k))
    for k in range(-8, 9):
        if k != 0:
            candidates.extend([k / 2.0, k / 3.0])

    best = a
    best_err = float("inf")
    for c in candidates:
        err = abs(a - c)
        if err < snap_tol and err < best_err:
            best = c
            best_err = err
    return best


def build_power_law_tree(
    fit: PowerLawFit,
    *,
    round_exponents: bool = True,
    snap_tol: float = 0.02,
    exponent_skip_threshold: float = 0.01,
) -> Optional[Node]:
    """Build a Node tree representing `C · ∏ x_i^{a_i}` from a PowerLawFit.

    Returns None if the fit reduces to a constant (no variable
    contributes). Skips terms whose exponent has magnitude below
    `exponent_skip_threshold` (treats them as 1).
    """
    factors: list[Node] = []
    for name, e in zip(fit.feature_names, fit.exponents):
        e_round = _maybe_round_exponent(e, snap_tol=snap_tol) if round_exponents else e
        if abs(e_round) < exponent_skip_threshold:
            continue
        if abs(e_round - 1.0) < 1e-6:
            factors.append(Var(name))
        elif abs(e_round + 1.0) < 1e-6:
            factors.append(BinOp("div", Const(1.0), Var(name)))
        elif abs(e_round - 2.0) < 1e-6:
            # Prefer x*x over pow(x, 2) — smaller cx and the safe-pow
            # base/exp clipping is sidestepped on this common form.
            factors.append(BinOp("mul", Var(name), Var(name)))
        elif abs(e_round + 2.0) < 1e-6:
            factors.append(BinOp("div", Const(1.0),
                                 BinOp("mul", Var(name), Var(name))))
        elif abs(e_round - 0.5) < 1e-6:
            factors.append(UnOp("sqrt", Var(name)))
        elif abs(e_round + 0.5) < 1e-6:
            factors.append(BinOp("div", Const(1.0), UnOp("sqrt", Var(name))))
        else:
            factors.append(BinOp("pow", Var(name), Const(e_round)))

    if not factors:
        return None

    # Multiply factors left-associatively.
    product: Node = factors[0]
    for f in factors[1:]:
        product = BinOp("mul", product, f)

    # Prepend the multiplicative constant unless it's ≈ 1.
    if abs(fit.constant - 1.0) > 1e-3:
        product = BinOp("mul", Const(float(fit.constant)), product)

    return product


def build_exp_wrapper_tree(
    fit: ExpWrapperFit,
    *,
    round_exponents: bool = True,
    snap_tol: float = 0.02,
    exponent_skip_threshold: float = 0.01,
) -> Optional[Node]:
    """Build a Node tree representing `exp(sign · C · ∏ x_i^{a_i})` from
    an ExpWrapperFit.

    The inner power-law product is built via `build_power_law_tree`;
    then the sign and `exp(...)` are wrapped around it.
    """
    inner_tree = build_power_law_tree(
        fit.inner,
        round_exponents=round_exponents,
        snap_tol=snap_tol,
        exponent_skip_threshold=exponent_skip_threshold,
    )
    if inner_tree is None:
        return None
    if fit.sign < 0:
        inner_tree = UnOp("neg", inner_tree)
    return UnOp("exp", inner_tree)


# ---------------- Top-level entry ----------------

def power_law_seed(
    env: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    r2_threshold: float = 0.99,
    round_exponents: bool = True,
    try_exp_wrapper: bool = True,
) -> tuple[Optional[Node], Optional[object]]:
    """Detect a structural seed and build a tree for the GP initial pop.

    Strategy (orchestrator pattern):
      1. Try base power-law `f = C · ∏ x_i^{a_i}` first. Higher-precedence
         because it produces simpler seeds.
      2. If power-law rejects (R² < threshold), try exp-wrapper
         `f = ±exp(C · ∏ x_i^{a_i})`.
      3. Return the first hit, or (None, None) if both reject.

    The GP's normal selection pressure decides if the seed survives.
    Returns (tree, fit) where fit is a PowerLawFit or ExpWrapperFit.
    """
    # 1. Base power-law.
    fit_pl = detect_power_law(env, y, r2_threshold=r2_threshold)
    if fit_pl is not None:
        tree = build_power_law_tree(fit_pl, round_exponents=round_exponents)
        return tree, fit_pl

    # 2. Exp-wrapper fallback.
    if try_exp_wrapper:
        fit_exp = detect_exp_wrapper(env, y, r2_threshold=r2_threshold)
        if fit_exp is not None:
            tree = build_exp_wrapper_tree(fit_exp, round_exponents=round_exponents)
            return tree, fit_exp

    return None, None


__all__ = [
    "PowerLawFit",
    "ExpWrapperFit",
    "detect_power_law",
    "detect_exp_wrapper",
    "build_power_law_tree",
    "build_exp_wrapper_tree",
    "power_law_seed",
]
