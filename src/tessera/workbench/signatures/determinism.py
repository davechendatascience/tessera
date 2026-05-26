"""Determinism vs stochasticity signature.

For a trajectory, test whether the next value is predictable from past
values (deterministic) or has irreducible randomness (stochastic).

Method: k-nearest-neighbor forecasting in time-delay embedding space.
We embed the trajectory at a fixed delay tau and dimension m, then
for each test point find its k nearest neighbors in the training set
and predict by averaging their next-step values. Compare prediction
error to surrogate-data null (phase-randomized version of the same
signal).

Returns
-------
value : float (z-score of predictability vs null)
    Large positive z = deterministic. Near zero = stochastic.
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_determinism(
    traj: Trajectory,
    *,
    embed_dim: int = 3,
    delay: int = 1,
    k: int = 5,
    n_surrogates: int = 5,
    n_test: int = 200,
) -> SignatureValue:
    """k-NN forecasting test for determinism.

    Returns
    -------
    value : float
        z-score = (null_err - actual_err) / null_std.
        z > 2: clearly deterministic.
        z < 1: likely stochastic.
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    min_needed = embed_dim * delay + n_test + 10
    if n < min_needed:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes=f"need >= {min_needed} samples; got {n}",
        )

    # Use first principal component (or single component if 1D)
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

    actual_err = _knn_one_step_error(x, embed_dim, delay, k, n_test)

    # Surrogate null: phase-randomize signal, repeat
    rng = np.random.default_rng(0)
    null_errs = []
    for _ in range(n_surrogates):
        x_surr = _phase_randomize(x, rng)
        null_errs.append(_knn_one_step_error(x_surr, embed_dim, delay, k, n_test))
    null_errs = np.array(null_errs)
    null_mean = float(null_errs.mean())
    null_std = float(null_errs.std()) + 1e-9

    z = (null_mean - actual_err) / null_std

    return SignatureValue(
        value=float(z),
        confidence=min(1.0, n / 500.0),
        n_samples_used=n,
        notes=f"actual_err={actual_err:.4f}, null_mean={null_mean:.4f}",
    )


def _knn_one_step_error(x: np.ndarray, m: int, tau: int, k: int, n_test: int) -> float:
    """Embed signal, predict next value via k-NN, return mean abs error
    normalized by signal std."""
    n = x.size
    emb_len = n - (m - 1) * tau - 1  # need one more for the target
    if emb_len < n_test + 2 * k:
        return float("nan")
    # Build embedding
    E = np.column_stack([x[(m - 1 - i) * tau:(m - 1 - i) * tau + emb_len]
                         for i in range(m)])
    targets = x[(m - 1) * tau + 1:(m - 1) * tau + 1 + emb_len]
    # Split: test = last n_test, train = the rest minus a buffer
    n_train = emb_len - n_test - k
    if n_train < 2 * k:
        return float("nan")
    E_train, t_train = E[:n_train], targets[:n_train]
    E_test, t_test = E[n_train:n_train + n_test], targets[n_train:n_train + n_test]
    # For each test point find k nearest in train and average their targets
    preds = np.zeros(n_test)
    for i in range(n_test):
        d = np.linalg.norm(E_train - E_test[i], axis=1)
        idx = np.argpartition(d, k)[:k]
        preds[i] = t_train[idx].mean()
    err = float(np.mean(np.abs(preds - t_test)))
    return err / (float(x.std()) + 1e-9)


def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize a signal while preserving its power spectrum."""
    n = x.size
    X = np.fft.rfft(x)
    # Randomize phases of non-DC, non-Nyquist bins
    phases = rng.uniform(0, 2 * np.pi, size=X.size)
    phases[0] = 0.0
    if n % 2 == 0 and X.size > 0:
        phases[-1] = 0.0
    X_rand = np.abs(X) * np.exp(1j * phases)
    return np.fft.irfft(X_rand, n=n)
