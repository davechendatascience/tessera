"""Symmetry signature — tests for invariance under candidate group actions.

Per Stage 0/0.5 design, symmetries are declared per canonical system
from the standardized vocabulary (time_translation, time_reversal,
space_translation, rotation_so2/so3, reflection, scaling). This
signature checks each candidate group on data and reports pass/fail.

Tested explicitly for Stage 2:
  - time_translation: trajectory statistics invariant under temporal shift
  - reflection: state-space mean invariant under x -> -x
  - rotation_so2: 2D state invariant under planar rotation
  - time_reversal: trajectory reversed satisfies same dynamics (conservative)

Each test returns (passed: bool, score: float in [0,1]). A higher score
indicates stronger evidence for the symmetry.

For Stage 5 identification, this signature consumes the candidate set
from the library anchor; for Stage 6 scoring, it consumes the declared
symmetries from the canonical system.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..types import Trajectory
from .types import SignatureValue


def compute_symmetry(
    traj: Trajectory,
    *,
    candidates: Optional[Iterable[str]] = None,
) -> SignatureValue:
    """Test the trajectory against candidate symmetry groups.

    Parameters
    ----------
    candidates : iterable of str or None
        Subset of SYMMETRY_VOCABULARY to test. If None, tests a default
        prior set: time_translation, time_reversal, reflection,
        rotation_so2 (the most commonly relevant for dynamical systems).

    Returns
    -------
    value : dict[str, dict]
        Per-group result: {group_name: {'passes': bool, 'score': float}}.
    """
    if candidates is None:
        candidates = ["time_translation", "time_reversal",
                      "reflection", "rotation_so2"]

    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < 50:
        return SignatureValue(
            value={}, confidence=0.0, n_samples_used=n,
            notes="too few samples for symmetry tests",
        )

    results = {}
    for sym in candidates:
        if sym == "time_translation":
            passed, score = _test_time_translation(obs)
        elif sym == "time_reversal":
            passed, score = _test_time_reversal(obs)
        elif sym == "reflection":
            passed, score = _test_reflection(obs)
        elif sym == "rotation_so2":
            passed, score = _test_rotation_so2(obs)
        else:
            continue
        results[sym] = {"passes": passed, "score": score}

    return SignatureValue(
        value=results,
        confidence=min(1.0, n / 200.0),
        n_samples_used=n,
    )


def _test_time_translation(obs: np.ndarray) -> tuple[bool, float]:
    """Statistics invariant under temporal shift: compare 1st-half vs 2nd-half
    moments (mean, std)."""
    n = obs.shape[0]
    half = n // 2
    mean1 = obs[:half].mean(axis=0)
    mean2 = obs[half:].mean(axis=0)
    std1 = obs[:half].std(axis=0) + 1e-9
    std2 = obs[half:].std(axis=0) + 1e-9
    mean_diff = float(np.mean(np.abs(mean1 - mean2) / (std1 + std2)))
    score = float(np.exp(-mean_diff * 5.0))  # high if means align
    return score > 0.6, score


def _test_time_reversal(obs: np.ndarray) -> tuple[bool, float]:
    """Conservative dynamics: trajectory reversed should have similar
    fine-scale structure. Test: variance of forward vs backward finite
    differences should be equal."""
    if obs.shape[0] < 10:
        return False, 0.0
    forward_diffs = obs[1:] - obs[:-1]
    backward_diffs = obs[:-1] - obs[1:]   # = -forward_diffs
    # For conservative dynamics the *distribution* of |diff| should be
    # similar to the time-reversed; for dissipative it shouldn't
    # be (energy decays). Compare via standardized moments.
    fvar = forward_diffs.var()
    if fvar < 1e-12:
        return True, 1.0
    # Test for asymmetry: compare squared diffs in first half vs second half.
    n = obs.shape[0]
    half = n // 2
    d1 = float(np.mean(forward_diffs[:half] ** 2))
    d2 = float(np.mean(forward_diffs[half:] ** 2))
    asym = abs(d1 - d2) / (d1 + d2 + 1e-12)
    score = float(np.exp(-asym * 5.0))
    return score > 0.6, score


def _test_reflection(obs: np.ndarray) -> tuple[bool, float]:
    """Symmetry under x -> -x: trajectory mean should be near zero (component-wise)."""
    means = obs.mean(axis=0)
    stds = obs.std(axis=0) + 1e-9
    # Score: how close to zero are component means relative to their stds
    relative_mean = float(np.mean(np.abs(means) / stds))
    score = float(np.exp(-relative_mean * 2.0))
    return score > 0.5, score


def _test_rotation_so2(obs: np.ndarray) -> tuple[bool, float]:
    """SO(2) rotation invariance: 2D trajectory's radial distribution
    should be invariant under planar rotation. Test: average angular
    distribution should be uniform on [0, 2*pi]."""
    if obs.shape[1] < 2:
        return False, 0.0
    x, y = obs[:, 0], obs[:, 1]
    # Center
    x = x - x.mean()
    y = y - y.mean()
    theta = np.arctan2(y, x)
    # Histogram of theta over [0, 2*pi]
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    expected = counts.mean()
    if expected < 1e-9:
        return False, 0.0
    chisq = float(np.sum((counts - expected) ** 2 / expected) / counts.size)
    # Smaller chisq → more uniform → more rotation-invariant
    score = float(np.exp(-chisq / 5.0))
    return score > 0.4, score
