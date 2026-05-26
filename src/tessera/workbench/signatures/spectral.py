"""Spectral content signature — power spectrum summary.

Distinguishes periodic (sharp peaks), quasi-periodic (a few peaks),
chaotic (broadband), and noise-like (flat) trajectories.

Method: Welch periodogram on the first state component (or its principal
PCA component for high-dim state), then summarize:
  - peak_height: max power / mean power (concentrated at one freq?)
  - spectral_flatness: geometric mean / arithmetic mean of spectrum
                       (1.0 = white noise, low = structured)
  - dominant_freq: argmax frequency
"""
from __future__ import annotations

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_spectral_content(
    traj: Trajectory,
    *,
    nperseg: int = 256,
) -> SignatureValue:
    """Welch power spectrum summary.

    Returns
    -------
    value : dict
        {
          'peak_height': float (max_power / mean_power, larger = more peaked),
          'spectral_flatness': float in [0, 1] (1 = white, 0 = pure tone),
          'dominant_freq': float in [0, 0.5] (cycles per sample),
        }
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < nperseg:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes=f"need >= {nperseg} samples; got {n}",
        )

    # Choose one signal: largest-variance component (or PCA-1 for high-dim)
    X = obs.reshape(n, -1).astype(np.float64)
    if X.shape[1] > 4:
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        x = Xc @ Vt[0]
    else:
        x = X[:, int(np.argmax(X.var(axis=0)))]

    if x.var() < 1e-12:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes="signal has zero variance",
        )

    # Welch via FFT on overlapping segments
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2
    starts = range(0, n - nperseg + 1, nperseg - noverlap)
    win = np.hanning(nperseg)
    psds = []
    for s in starts:
        seg = x[s:s + nperseg] * win
        f = np.fft.rfft(seg)
        psds.append(np.abs(f) ** 2)
    if not psds:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes="no segments",
        )
    psd = np.mean(psds, axis=0)
    # Skip DC bin
    psd = psd[1:]
    if psd.size == 0 or psd.sum() < 1e-12:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes="no spectral content",
        )

    psd_norm = psd / psd.sum()
    peak_height = float(psd.max() / (psd.mean() + 1e-12))
    # Spectral flatness: geometric mean / arithmetic mean
    log_psd = np.log(psd + 1e-30)
    gmean = float(np.exp(log_psd.mean()))
    amean = float(psd.mean())
    spectral_flatness = gmean / (amean + 1e-12)
    dominant_freq = float(np.argmax(psd) + 1) / nperseg  # +1 because we skipped DC

    return SignatureValue(
        value={
            "peak_height": peak_height,
            "spectral_flatness": float(spectral_flatness),
            "dominant_freq": dominant_freq,
        },
        confidence=min(1.0, n / 500.0),
        n_samples_used=n,
    )
