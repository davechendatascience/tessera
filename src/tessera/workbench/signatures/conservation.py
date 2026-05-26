"""Conservation-law signature — finds approximately-conserved polynomials.

For a trajectory x(t), a conserved quantity Q(x) satisfies:
    Q(x(t)) ≈ const for all t

We search over low-degree polynomial bases of the state and find the
polynomial that minimizes its variance along the trajectory relative
to its overall variance over the trajectory's range.

Returns the best Q at each degree d ∈ {1, 2, 3} so that consumers
can decide which form is most likely a real conservation law.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_conservation(
    traj: Trajectory,
    *,
    degrees: Iterable[int] = (1, 2, 3),
    variance_ratio_threshold: float = 0.05,
) -> SignatureValue:
    """Search for low-degree polynomial invariants Q(x).

    For each degree d, build the basis of all polynomial terms of total
    degree exactly d, then find unit-norm coefficient vector c such that
    Q(x) = c @ phi(x) has minimum variance along the trajectory subject
    to Q having some variance over a randomized null. The variance ratio
    Var(Q on trajectory) / Var(Q on shuffled trajectory) < threshold
    indicates a strong conservation.

    Returns
    -------
    value : dict[int, dict]
        Per-degree result: {d: {'variance_ratio': float, 'coefs': ndarray}}.
        Lower variance_ratio = stronger conservation law.
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n, d_state = obs.shape
    if n < 100 or d_state < 1:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes="too few samples for conservation search",
        )

    # Reduce dimensionality if state is very high (PDE)
    if d_state > 4:
        # Use PCA top-4 components for tractability
        Xc = obs - obs.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:4].T
        d_state = 4
        notes_prefix = f"reduced state dim from {obs.shape[1]} to 4 via PCA; "
    else:
        X = obs.astype(np.float64)
        notes_prefix = ""

    results = {}
    rng = np.random.default_rng(0)
    for d in degrees:
        phi = _polynomial_basis(X, d)
        if phi.shape[1] == 0:
            continue
        # Center features
        phi_c = phi - phi.mean(axis=0)
        # The direction of minimum variance is the smallest-singular-value
        # right singular vector of phi_c.
        try:
            _, s, Vt = np.linalg.svd(phi_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if s[-1] / (s[0] + 1e-12) > 0.99:
            # All directions are roughly equal variance — no clear invariant
            continue
        coefs = Vt[-1]  # unit-norm direction of minimum variance
        Q = phi @ coefs
        Q_var = float(Q.var())
        # Randomized null: shuffle each column independently to break the
        # trajectory's temporal coherence, then recompute Q
        phi_shuffled = np.column_stack([
            phi[:, j][rng.permutation(n)] for j in range(phi.shape[1])
        ])
        Q_null = phi_shuffled @ coefs
        null_var = float(Q_null.var())
        ratio = Q_var / (null_var + 1e-12)
        results[d] = {
            "variance_ratio": ratio,
            "coefs_l2": float(np.linalg.norm(coefs)),
            "is_conserved": ratio < variance_ratio_threshold,
        }

    return SignatureValue(
        value=results,
        confidence=min(1.0, n / 500.0),
        n_samples_used=n,
        notes=notes_prefix + f"tested degrees={list(degrees)}",
    )


def _polynomial_basis(X: np.ndarray, degree: int) -> np.ndarray:
    """All monomials of total degree exactly = `degree`.

    For X shape (n, d_state), returns (n, n_monomials).
    """
    n, d = X.shape
    if degree == 0:
        return np.ones((n, 1))
    monomials = []
    for combo in combinations_with_replacement(range(d), degree):
        col = np.ones(n)
        for idx in combo:
            col *= X[:, idx]
        monomials.append(col)
    return np.column_stack(monomials)
