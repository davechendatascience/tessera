"""Sufficient-statistic precomputation for analytical Δloss (Regime B).

PHASE 2 (this revision): adds `mutate_add_polynomial_term` — builds the
optimal additive polynomial term tree from a PolynomialMoments instance,
to be used as a GP polish step.



For MSE loss `L(f) = (1/N) Σ (f(x_i) - y_i)²`, an additive mutation
`f' = f + δ` has Δloss decomposing as:

    Δloss = (2/N) Σ residual_i · δ(x_i) + (1/N) Σ δ(x_i)²

where `residual_i = f(x_i) - y_i` (prediction minus target).

If δ lives in a fixed basis δ(x) = Σ_k c_k · φ_k(x) — polynomial monomials,
RBF kernels, any precomputed feature bank — then both terms collapse to
sums over the basis index:

    Δloss = (2/N) c · R + (1/N) cᵀ G c

where `G_kj = Σ φ_k(x_i) φ_j(x_i)` is the basis-Gram matrix and
`R_k = Σ residual_i · φ_k(x_i)` is the residual-basis projection.
Both are PRECOMPUTED ONCE in O(N · K) and O(N · K²); each subsequent
mutation `Δloss(c)` evaluates in **O(K²) — independent of N**.

This is the Knuth-shaped answer to "calculus of loss impact" raised in
`docs/research/analytical_delta_loss.md`. The FMM analog: O(N²) → O(N)
by exploiting algebraic structure of the kernel; here the kernel is
the basis. Pattern reusable for any linear-in-parameters mutation class.

What's here
-----------
    PolynomialMoments — precomputes G, R for a user-specified basis.
                        Exposes delta_loss(c), optimal_coefficients(),
                        optimal_delta_loss().

    monomial_basis    — helper that builds the standard polynomial
                        basis up to a max total degree across selected
                        features.

What's NOT here yet (deferred to §4.4 / Phase 2)
------------------------------------------------
    GP integration. The `mutate_add_polynomial_term` operator and the
    GPConfig.use_sufficient_stats flag are Phase 2 of the ship plan in
    `docs/planned/roadmap.md` §2.3.

    Multi-output basis. Current API assumes scalar targets. Vector
    targets would need stacked R per output dim; not needed for
    Feynman.

    Non-MSE losses. The decomposition assumes squared error; PnL /
    classification losses need their own analytical form (or fall
    back to Regime A re-eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


BasisFn = Callable[[np.ndarray], np.ndarray]
"""A basis function: maps X of shape (N, D) to phi(X) of shape (N,)."""


@dataclass
class PolynomialMoments:
    """Sufficient statistics for analytical Δloss in a fixed basis.

    Construct once per (X, y, current_predictions) snapshot; reuse for
    O(K²) Δloss queries until the current best candidate changes (then
    rebuild with the new predictions).

    Attributes
    ----------
    G : np.ndarray, shape (K, K)
        Basis-Gram matrix `G_kj = Σ φ_k(x_i) · φ_j(x_i)` (un-normalised).
    R : np.ndarray, shape (K,)
        Residual-basis projection `R_k = Σ residual_i · φ_k(x_i)`.
    N : int
        Number of samples (for the 1/N and 2/N factors in delta_loss).
    K : int
        Number of basis functions (for shape checks on c).
    """

    G: np.ndarray
    R: np.ndarray
    N: int
    K: int = field(init=False)

    def __post_init__(self) -> None:
        if self.G.shape != (self.R.shape[0], self.R.shape[0]):
            raise ValueError(
                f"shape mismatch: G is {self.G.shape}, R is {self.R.shape}"
            )
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        self.K = self.R.shape[0]

    @classmethod
    def from_basis(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        basis: Sequence[BasisFn],
    ) -> "PolynomialMoments":
        """Build moments from explicit basis-function callables.

        Parameters
        ----------
        X : (N, D) ndarray
            Feature matrix.
        y : (N,) ndarray
            Targets.
        predictions : (N,) ndarray
            Current candidate's predictions f(x_i). Residual is
            computed internally as `predictions - y`.
        basis : sequence of K callables
            Each maps X -> (N,). The basis δ(x) = Σ c_k φ_k(x) lives
            in this span.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N = X.shape[0]
        if y.shape[0] != N or predictions.shape[0] != N:
            raise ValueError(
                f"size mismatch: X has N={N}, y has {y.shape[0]}, "
                f"predictions has {predictions.shape[0]}"
            )
        K = len(basis)
        if K == 0:
            raise ValueError("basis must be non-empty")

        Phi = np.empty((K, N), dtype=np.float64)
        for k, phi in enumerate(basis):
            row = np.asarray(phi(X), dtype=np.float64).reshape(-1)
            if row.shape[0] != N:
                raise ValueError(
                    f"basis function {k} returned shape {row.shape}, "
                    f"expected ({N},)"
                )
            Phi[k] = row

        residual = predictions - y
        G = Phi @ Phi.T
        R = Phi @ residual
        return cls(G=G, R=R, N=N)

    def delta_loss(self, c: np.ndarray) -> float:
        """Evaluate Δloss for δ(x) = Σ c_k φ_k(x).

        Cost: O(K²). INDEPENDENT OF N. This is the O(1)-in-N call the
        whole module exists for.

        Parameters
        ----------
        c : (K,) ndarray
            Coefficient vector in the basis.

        Returns
        -------
        float
            Δloss = loss(f + δ) - loss(f).
        """
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.K:
            raise ValueError(
                f"coefficient shape mismatch: c has {c.shape[0]}, "
                f"basis has K={self.K}"
            )
        linear = 2.0 * np.dot(c, self.R) / self.N
        quadratic = float(c @ self.G @ c) / self.N
        return float(linear + quadratic)

    def optimal_coefficients(self, ridge: float = 1e-10) -> np.ndarray:
        """Closed-form optimal c* minimising Δloss.

        c* = -G⁻¹ R (ridge-regularised for numerical stability).
        O(K³) one-time cost (matrix solve).
        """
        G_reg = self.G + ridge * np.eye(self.K)
        return -np.linalg.solve(G_reg, self.R)

    def optimal_delta_loss(self, ridge: float = 1e-10) -> float:
        """Closed-form best achievable Δloss in this basis.

        Always ≤ 0 (Cauchy-Schwarz). Equals -(1/N) Rᵀ G⁻¹ R for the
        unregularised case.
        """
        return self.delta_loss(self.optimal_coefficients(ridge=ridge))


def monomial_basis(
    feature_indices: Sequence[int],
    max_degree: int,
    include_constant: bool = False,
) -> list[BasisFn]:
    """Build the standard univariate-monomial basis.

    For each feature index d in `feature_indices` and each degree k in
    1..max_degree, produces a basis function φ(X) = X[:, d]^k.

    The cross-terms (e.g., x_0 · x_1) are NOT included here — that's
    a multivariate-monomial helper for later. This helper supports
    "add a power of one feature" mutations, which covers most of the
    Feynman polynomial-target slice.

    Parameters
    ----------
    feature_indices : sequence of int
        Which columns of X to build monomials over.
    max_degree : int
        Largest exponent (≥ 1).
    include_constant : bool
        If True, prepend a constant basis function φ(X) = 1.

    Returns
    -------
    list of basis-function callables, suitable for
    `PolynomialMoments.from_basis(..., basis=...)`.
    """
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")

    basis: list[BasisFn] = []
    if include_constant:
        basis.append(lambda X: np.ones(X.shape[0], dtype=np.float64))

    for d in feature_indices:
        for k in range(1, max_degree + 1):
            basis.append(_make_monomial(d, k))
    return basis


def _make_monomial(feature_idx: int, degree: int) -> BasisFn:
    """Closure for X -> X[:, feature_idx]**degree. Standalone to avoid
    the lambda-loop-capture pitfall."""
    def phi(X: np.ndarray) -> np.ndarray:
        return X[:, feature_idx] ** degree
    return phi


__all__ = [
    "PolynomialMoments", "monomial_basis", "BasisFn",
    "build_polynomial_term_tree", "polish_tree_with_polynomial_term",
]


# ----------------------------------------------------------------------
# Phase 2 — tree construction for the GP polish step
# ----------------------------------------------------------------------

def build_polynomial_term_tree(
    feature_names: list[str],
    feature_indices: list[int],
    max_degree: int,
    coefficients: np.ndarray,
    top_n: int | None = None,
    coef_threshold: float = 1e-6,
    include_constant: bool = False,
):
    """Construct an Expr tree representing Σ c_k · φ_k(x).

    The basis-function ordering must match `monomial_basis(feature_indices,
    max_degree, include_constant=include_constant)`. Imports tessera tree
    types lazily so this module stays import-cheap and decoupled from
    `tessera.expression.*` until actually used.

    Parameters
    ----------
    feature_names : list of str
        Full name list (env.keys() ordering).
    feature_indices : list of int
        Which features the basis covers (index into feature_names).
    max_degree : int
        Max exponent used in `monomial_basis`.
    coefficients : (K,) ndarray
        Coefficient vector in the basis. Typically the output of
        `PolynomialMoments.optimal_coefficients()`.
    top_n : int or None
        Keep only the top-N coefficients by absolute magnitude. None =
        keep all. Limits tree complexity blowup.
    coef_threshold : float
        Coefficients with |c| < threshold are skipped entirely (after
        top-N filter). Avoids adding effectively-zero terms.
    include_constant : bool
        Must match the value used when constructing the basis.

    Returns
    -------
    Node | None
        A tessera Expr tree representing the sum, or None if every
        coefficient was below threshold (degenerate "add nothing"
        case).
    """
    # Lazy imports — sufficient_stats stays as a pure-numerics module
    # at import time; tree construction is opt-in.
    from tessera.expression.tree import Var, Const, BinOp

    c = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    expected_K = (
        (1 if include_constant else 0)
        + len(feature_indices) * max_degree
    )
    if c.shape[0] != expected_K:
        raise ValueError(
            f"coefficient vector has K={c.shape[0]}, expected "
            f"{expected_K} from (include_constant={include_constant}, "
            f"{len(feature_indices)} features × {max_degree} degrees)"
        )

    # Build (basis_idx, exponent_tuple, name_or_None) list.
    # exponent_tuple = (feature_idx, exponent); name = constant marker.
    basis_meta = []
    if include_constant:
        basis_meta.append((None, None))
    for d in feature_indices:
        for k in range(1, max_degree + 1):
            basis_meta.append((d, k))

    # Pick which indices to include.
    order = np.argsort(-np.abs(c))  # descending magnitude
    if top_n is not None:
        order = order[:top_n]
    keep = [int(i) for i in order if abs(c[int(i)]) >= coef_threshold]
    if not keep:
        return None

    # Sort by basis index so the resulting tree is canonical
    # (deterministic across orderings of identical coefficients).
    keep.sort()

    def _term_node(coef: float, meta):
        feat_idx, exponent = meta
        coef_node = Const(value=float(coef))
        if feat_idx is None:
            # Constant term
            return coef_node
        var = Var(name=feature_names[feat_idx])
        # Use multiplication chain (x*x*...*x) rather than BinOp("pow"),
        # because the tessera "pow" op is PROTECTED — pow(|x|, k) — which
        # strips the sign of the base and breaks odd-degree polynomials
        # (e.g., x^3 at x=-1 would return +1, not -1). Multiplication is
        # cheaper in complexity too.
        power_node = var
        for _ in range(exponent - 1):
            power_node = BinOp(op="mul", a=power_node, b=var)
        return BinOp(op="mul", a=coef_node, b=power_node)

    nodes = [_term_node(float(c[i]), basis_meta[i]) for i in keep]
    # Left-fold into a sum tree
    result = nodes[0]
    for n in nodes[1:]:
        result = BinOp(op="add", a=result, b=n)
    return result


def polish_tree_with_polynomial_term(
    tree,
    predictions: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_indices: list[int],
    max_degree: int,
    *,
    top_n: int | None = 3,
    coef_threshold: float = 1e-6,
    include_constant: bool = False,
    ridge: float = 1e-8,
):
    """One-shot GP polish: build moments, find optimal polynomial
    addition, splice it onto the tree.

    Returns
    -------
    (new_tree, expected_delta_loss, kept_terms) :
        new_tree : Node or original tree if no term added.
        expected_delta_loss : float — analytical prediction from
                              PolynomialMoments.delta_loss for the
                              top-N truncated coefficient vector.
                              Useful as a sanity check vs full re-eval.
        kept_terms : int — number of basis functions added (0 if no
                    change).
    """
    from tessera.expression.tree import BinOp

    basis = monomial_basis(
        feature_indices, max_degree, include_constant=include_constant
    )
    moments = PolynomialMoments.from_basis(X, y, predictions, basis)
    c_opt = moments.optimal_coefficients(ridge=ridge)

    # Filter to top-N magnitude + threshold
    abs_c = np.abs(c_opt)
    order = np.argsort(-abs_c)
    if top_n is not None:
        order = order[:top_n]
    keep_mask = np.zeros_like(c_opt)
    for i in order:
        if abs_c[int(i)] >= coef_threshold:
            keep_mask[int(i)] = 1.0
    c_truncated = c_opt * keep_mask
    kept = int(keep_mask.sum())
    if kept == 0:
        return tree, 0.0, 0

    expected_dl = moments.delta_loss(c_truncated)

    addition = build_polynomial_term_tree(
        feature_names=feature_names,
        feature_indices=feature_indices,
        max_degree=max_degree,
        coefficients=c_truncated,
        top_n=None,  # already truncated above
        coef_threshold=coef_threshold,
        include_constant=include_constant,
    )
    if addition is None:
        return tree, 0.0, 0
    new_tree = BinOp(op="add", a=tree, b=addition)
    return new_tree, float(expected_dl), kept

