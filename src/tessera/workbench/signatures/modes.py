"""Mode-count signature via Gaussian mixture model + BIC selection.

For a trajectory whose state samples cluster around K attractors /
regimes, GMM with K components should fit better than K-1 or K+1
on a BIC scoring. We sweep candidate K and return the BIC-minimizing
component count.

For PDE trajectories (high-dimensional state), we project to the top
PCA components before GMM to make the fit tractable.

LIMITATION (documented for Stage 3 refinement)
----------------------------------------------
GMM-BIC measures *statistical density clusters*, not topological mode
counts. For systems where the trajectory traces a limit cycle (Van der
Pol, FHN oscillatory regime, harmonic motion, periodic pendulum), the
samples DENSELY populate the cycle and GMM finds multiple clusters
along it — the reported mode_count will exceed the true topological
mode count (= 1). This is a known issue with applying GMM to time-
series data without proper recurrence analysis.

For chaotic attractors with truly disconnected lobes (Lorenz-63), the
estimate is reliable. For single-orbit systems, treat the reading as
"phase-space density clusters" rather than "true mode count."

Stage 3 will calibrate by introducing a recurrence-network alternative;
for now Stage 2 ships the GMM/BIC approach with this caveat documented.
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_mode_count(
    traj: Trajectory,
    *,
    max_k: int = 5,
    pca_components: int = 2,
    min_samples: int = 100,
) -> SignatureValue:
    """Estimate the number of modes (attractors / regimes).

    Method
    ------
    1. Reshape observable to (n_samples, n_features); subsample if long.
    2. If n_features > pca_components, project via PCA.
    3. Sweep K in [1, max_k], fit GMM, record BIC.
    4. Return K minimizing BIC.

    Returns
    -------
    value : int  (1..max_k)
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < min_samples:
        return SignatureValue(
            value=1, confidence=0.0, n_samples_used=n,
            notes=f"too few samples (<{min_samples}); defaulting to 1",
        )

    X = obs.reshape(n, -1).astype(np.float64)
    if X.shape[1] > pca_components:
        # Simple PCA via SVD on centered data
        Xc = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:pca_components].T

    # Subsample to break trajectory autocorrelation — GMM/BIC is biased
    # toward higher K on dense-autocorrelated time series because nearby
    # points naturally cluster. Take every k-th point so the subsample
    # is closer to iid in phase space.
    target_subsample = 500
    if n > target_subsample:
        stride = max(1, n // target_subsample)
        X = X[::stride]

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return SignatureValue(
            value=1, confidence=0.0, n_samples_used=n,
            notes="sklearn not available; defaulting to 1",
        )

    bic_scores = {}
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=0, max_iter=100, n_init=1)
            gmm.fit(X)
            bic_scores[k] = float(gmm.bic(X))
        except Exception:
            continue
    if not bic_scores:
        return SignatureValue(
            value=1, confidence=0.0, n_samples_used=n,
            notes="GMM fit failed for all k",
        )
    best_k = min(bic_scores, key=bic_scores.get)
    return SignatureValue(
        value=int(best_k),
        confidence=min(1.0, n / 500.0),
        n_samples_used=n,
        notes=f"BIC scores: {bic_scores}",
    )
