"""Effective-dimensionality signature.

For an ODE state trajectory, the data lives on (or near) a lower-dimensional
manifold than the ambient embedding dimension. The effective dimension is
useful for distinguishing systems and detecting when the observable
contains redundant components.

For Stage 2 we implement two complementary estimates:
  - PCA participation ratio: cheap, gives the linear-projection dim
  - Correlation dimension (Grassberger-Procaccia): nonlinear; only for
    longer trajectories

For PDEs the effective dim is the spatial-mode effective rank.
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_effective_dimensionality(
    traj: Trajectory,
    *,
    max_samples_for_corrdim: int = 5000,
) -> SignatureValue:
    """Estimate effective dimensionality of the state manifold.

    Method
    ------
    1. Compute PCA participation ratio:
         d_PR = (sum lambda_i)^2 / sum (lambda_i^2)
       where lambda_i are the singular values squared of the centered
       state matrix. Robust, cheap, gives linear-effective dim.
    2. If n_samples >= 500 and ambient dim small, also report
       Grassberger-Procaccia correlation dimension estimate as
       additional context.

    Returns
    -------
    value : float  — participation-ratio dim (primary)
    notes : may include 'corrdim' estimate
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < 20:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
        )

    X = obs.reshape(n, -1).astype(np.float64)
    Xc = X - X.mean(axis=0)
    # Singular values; eigenvalues of cov are s^2 / (n-1)
    s = np.linalg.svd(Xc, compute_uv=False)
    lams = s ** 2
    total = float(lams.sum())
    if total < 1e-12:
        return SignatureValue(
            value=0.0, confidence=0.0, n_samples_used=n,
            notes="zero-variance state",
        )
    d_pr = (float(lams.sum()) ** 2) / float((lams ** 2).sum())

    notes = f"participation_ratio dim={d_pr:.3f}"

    # Optional: correlation dim (Grassberger-Procaccia) for small ambient dim
    ambient = X.shape[1]
    if ambient <= 6 and n >= 500:
        try:
            corr_dim = _correlation_dimension(X[:min(max_samples_for_corrdim, n)])
            notes += f"; corrdim={corr_dim:.3f}"
        except Exception:
            pass

    return SignatureValue(
        value=float(d_pr),
        confidence=min(1.0, n / 200.0),
        n_samples_used=n,
        notes=notes,
    )


def _correlation_dimension(X: np.ndarray) -> float:
    """Grassberger-Procaccia correlation dim — log-log slope of C(eps) vs eps."""
    n = X.shape[0]
    # Pairwise distances, subsampled
    idx = np.random.default_rng(0).choice(n, size=min(n, 1500), replace=False)
    Xs = X[idx]
    # Compute all pairwise distances
    d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    d = d[np.triu_indices_from(d, k=1)]
    if d.size == 0:
        return float("nan")
    dmin, dmax = float(np.percentile(d, 1)), float(np.percentile(d, 50))
    if dmin <= 0 or dmax <= dmin:
        return float("nan")
    eps = np.geomspace(dmin, dmax, num=10)
    C = np.array([float((d < e).sum()) / d.size for e in eps])
    valid = C > 0
    if valid.sum() < 4:
        return float("nan")
    slope, _ = np.polyfit(np.log(eps[valid]), np.log(C[valid]), 1)
    return float(slope)
