"""Lyapunov-exponent estimate via the Rosenstein algorithm.

For chaotic systems, nearby trajectories diverge exponentially:
    |delta(t)| ≈ |delta(0)| * exp(lambda_max * t)

Rosenstein's algorithm: in time-delay embedding space, for each point,
find its nearest neighbor (with temporal separation > Theiler window),
then track how the distance between their trajectories grows. The
slope of log(distance) vs time gives lambda_max.

Returns
-------
value : float
    Estimated max Lyapunov exponent (per unit time).
    > 0 indicates chaos. Near 0 means neutral; negative means contracting.
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_lyapunov(
    traj: Trajectory,
    *,
    embed_dim: int = 3,
    delay: int = 1,
    theiler: int = 10,
    n_evolve: int = 30,
) -> SignatureValue:
    """Rosenstein algorithm estimate of max Lyapunov exponent.

    Returns
    -------
    value : float
        Lambda_max estimate; positive = chaotic, near-zero = regular.
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    min_needed = embed_dim * delay + n_evolve + 2 * theiler + 10
    if n < min_needed:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes=f"need >= {min_needed} samples; got {n}",
        )

    t = traj.t
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    if dt <= 0:
        dt = 1.0

    # Single-channel: PCA-1 if multi-component
    X = obs.reshape(n, -1).astype(np.float64)
    if X.shape[1] > 1:
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        x = Xc @ Vt[0]
    else:
        x = X[:, 0]
    if x.var() < 1e-12:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="signal has zero variance",
        )

    # Build embedding
    emb_len = n - (embed_dim - 1) * delay
    E = np.column_stack([x[(embed_dim - 1 - i) * delay:
                           (embed_dim - 1 - i) * delay + emb_len]
                         for i in range(embed_dim)])

    # For each point, find nearest neighbor with index distance > theiler
    n_pts = E.shape[0]
    pairs = []
    for i in range(n_pts - n_evolve):
        d = np.linalg.norm(E - E[i], axis=1)
        d[max(0, i - theiler):min(n_pts, i + theiler + 1)] = np.inf
        # Limit search to points that have n_evolve room
        d[n_pts - n_evolve:] = np.inf
        j = int(np.argmin(d))
        if d[j] == np.inf:
            continue
        pairs.append((i, j))

    if len(pairs) < 50:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="too few valid neighbor pairs",
        )

    # Track log distance for k = 0 .. n_evolve
    log_d = np.zeros(n_evolve)
    count = np.zeros(n_evolve)
    for (i, j) in pairs:
        for k in range(n_evolve):
            if i + k >= n_pts or j + k >= n_pts:
                continue
            d_k = np.linalg.norm(E[i + k] - E[j + k])
            if d_k > 0:
                log_d[k] += np.log(d_k)
                count[k] += 1

    valid = count > 0
    if valid.sum() < 5:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="insufficient valid evolution steps",
        )
    log_d[valid] /= count[valid]

    # Linear fit over the early growth region
    ks = np.arange(n_evolve)[valid]
    fit_len = min(15, valid.sum())
    slope = float(np.polyfit(ks[:fit_len] * dt, log_d[valid][:fit_len], 1)[0])

    return SignatureValue(
        value=slope,
        confidence=min(1.0, n / 1000.0),
        n_samples_used=n,
        notes=f"valid pairs={len(pairs)}, fit_len={fit_len}",
    )
