"""Smoothness signature — Hölder-regularity estimate from data.

For a trajectory y(t), structure-function scaling gives:
    S_2(tau) := <(y(t+tau) - y(t))^2> ~ tau^(2*alpha)
where alpha is the Hölder exponent (1 = smooth/analytic, 0.5 = Brownian,
0 = white noise). We estimate alpha from a log-log fit of S_2 vs tau
over small-tau scales.

For multi-component state, we average alpha across components weighted
by their variance.
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_smoothness(
    traj: Trajectory,
    *,
    n_lags: int = 8,
    min_lag: int = 1,
) -> SignatureValue:
    """Estimate Hölder regularity exponent of the trajectory.

    Returns
    -------
    value : float
        Estimated alpha. Roughly:
          - 1.0  → analytic / smooth (typical for ODE)
          - 0.5  → Brownian-like
          - 0.0  → uncorrelated noise
        Algebraic-of-iid trajectories return near 0 because there's no
        temporal smoothness.
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < 4 * n_lags:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="too few samples for structure-function fit",
        )

    lags = np.unique(np.round(
        np.geomspace(min_lag, n // 4, num=n_lags)
    ).astype(int))
    log_lags = np.log(lags.astype(float))

    # Weight components by variance
    component_vars = obs.var(axis=0)
    if float(component_vars.sum()) < 1e-12:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="trajectory has near-zero variance",
        )
    weights = component_vars / component_vars.sum()

    alphas = []
    for j in range(obs.shape[1]):
        x = obs[:, j].astype(np.float64)
        if x.var() < 1e-12:
            continue
        # S_2(tau) = mean of squared diffs
        s2 = np.array([float(np.mean((x[lag:] - x[:-lag]) ** 2)) for lag in lags])
        # Avoid log(0)
        s2 = np.maximum(s2, 1e-30)
        log_s2 = np.log(s2)
        # Linear fit: log_s2 = 2*alpha * log_lag + const
        slope, _ = np.polyfit(log_lags, log_s2, 1)
        alphas.append(slope / 2.0)

    if not alphas:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="all components had zero variance",
        )

    alpha_est = float(np.average(alphas, weights=weights[:len(alphas)]))
    # Clip to plausible range
    alpha_est = float(np.clip(alpha_est, 0.0, 2.0))

    return SignatureValue(
        value=alpha_est,
        confidence=min(1.0, n / 200.0),
        n_samples_used=n,
        notes=f"per-component alphas={[round(a,3) for a in alphas]}",
    )
