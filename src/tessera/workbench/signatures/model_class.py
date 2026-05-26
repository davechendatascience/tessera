"""Tier A signatures — model class discriminators.

These three signatures are cheap (O(N) or O(N log N)) and run before
any within-class signature extraction. Together they classify the
inferred ModelClass of unknown data.

Per Stage 0.5 §6 + Stage 2.5 §5:

  1. permutation_invariance — algebraic-vs-dynamical:
     pure functions of iid inputs are permutation-invariant in row
     order; dynamical-system data is not.

  2. autocorrelation_structure — algebraic / ODE / PDE:
     algebraic-of-iid has ACF(τ ≠ 0) ≈ 0; ODE has strong time ACF
     only; PDE has both time AND space ACF.

  3. stencil_locality — PDE-specific:
     u(t+1, x) predictable from u(t, x-k:x+k) for some finite k;
     algebraic and ODE fail this test.

The classify_model_class() driver applies these in sequence and
returns the most likely ModelClass.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import ModelClass, Trajectory
from .types import SignatureValue


# ----------------------------------------------------------------------
# Permutation invariance
# ----------------------------------------------------------------------

def compute_permutation_invariance(
    traj: Trajectory,
    *,
    n_shuffles: int = 8,
    seed: int = 0,
) -> SignatureValue:
    """Test whether the trajectory's row order matters.

    For algebraic data: shuffling rows preserves the joint distribution
    over (inputs, output), so per-row statistics are unchanged. For
    dynamical data: shuffling destroys the temporal/spatial coherence
    of state evolution, so summary statistics that depend on order
    (e.g., autocorrelation at lag 1) change drastically.

    Method
    ------
    Compare lag-1 autocorrelation of the observable's first column on
    the original trajectory vs `n_shuffles` random permutations. If
    permuting changes the lag-1 ACF by a large factor, the data has
    order structure (dynamical). If permuting leaves it ~unchanged
    (within noise), the data is order-invariant (algebraic).

    Returns
    -------
    value : float in [0, 1]
        1.0 = perfectly permutation-invariant (algebraic-like)
        0.0 = strongly order-dependent (dynamical-like)
    """
    obs = traj.observable
    if obs.ndim == 1:
        obs = obs[:, None]
    n = obs.shape[0]
    if n < 20:
        return SignatureValue(
            value=float("nan"), confidence=0.0, n_samples_used=n,
            notes="too few samples for permutation invariance test",
        )

    col = obs[:, 0]
    orig_acf = _lag1_acf(col)

    rng = np.random.default_rng(seed)
    shuffled_acfs = []
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        shuffled_acfs.append(_lag1_acf(col[perm]))
    shuffled_mean = float(np.mean(np.abs(shuffled_acfs)))

    # Invariance score: how close orig_acf is to the shuffled-null distribution
    # Use ratio: 1 - |orig|/(|orig|+|shuffled_mean|+eps), so 1.0 = invariant
    denom = abs(orig_acf) + shuffled_mean + 1e-9
    score = 1.0 - abs(orig_acf) / denom

    return SignatureValue(
        value=float(score),
        confidence=min(1.0, n / 100.0),
        n_samples_used=n,
        notes=f"orig_lag1_acf={orig_acf:.4f}; shuffled_mean={shuffled_mean:.4f}",
    )


def _lag1_acf(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = x.var()
    if var < 1e-12:
        return 0.0
    return float(np.mean(x[:-1] * x[1:]) / var)


# ----------------------------------------------------------------------
# Autocorrelation structure
# ----------------------------------------------------------------------

def compute_autocorrelation_structure(
    traj: Trajectory,
    *,
    max_lag_time: int = 20,
    max_lag_space: int = 10,
) -> SignatureValue:
    """Compute ACF along time (and space if PDE-shaped).

    For a (n_t, ...) trajectory:
      - Compute ACF along time axis for each spatial / state index
      - If state has > 1 spatial dimension (PDE shape (n_t, n_x)),
        also compute spatial ACF at fixed time

    The returned dict gives 'time_acf_max' (max non-trivial-lag ACF)
    and optionally 'space_acf_max'. High time ACF + low space ACF =
    ODE. High both = PDE. Low both = algebraic.

    Returns
    -------
    value : dict[str, float]
        {'time_acf_max': float, 'space_acf_max': Optional[float]}
    """
    obs = traj.observable
    n = obs.shape[0]
    if n < max_lag_time + 5:
        return SignatureValue(
            value={"time_acf_max": float("nan"), "space_acf_max": None},
            confidence=0.0, n_samples_used=n,
            notes="too few time samples",
        )

    # Reduce to (n_t, n_features) for time-ACF
    if obs.ndim == 1:
        col_data = obs[:, None]
    else:
        col_data = obs.reshape(n, -1)
    n_features = col_data.shape[1]

    # Time ACF: max |ACF(lag)| over lags 1..max_lag_time, averaged over features
    time_acfs = []
    for j in range(n_features):
        x = col_data[:, j]
        x = x - x.mean()
        var = x.var()
        if var < 1e-12:
            continue
        acfs = [float(np.mean(x[:-lag] * x[lag:]) / var)
                for lag in range(1, min(max_lag_time, n // 2))]
        time_acfs.append(max(abs(a) for a in acfs) if acfs else 0.0)
    time_acf_max = float(np.mean(time_acfs)) if time_acfs else 0.0

    # Space ACF: if PDE-shaped (n_t, n_x) with n_x > 4
    space_acf_max = None
    if obs.ndim == 2 and obs.shape[1] >= max_lag_space + 5:
        n_x = obs.shape[1]
        space_acfs = []
        sample_times = np.linspace(0, n - 1, num=min(20, n), dtype=int)
        for ti in sample_times:
            row = obs[ti].astype(np.float64)
            row_c = row - row.mean()
            var = row_c.var()
            if var < 1e-12:
                continue
            acfs = [float(np.mean(row_c[:-lag] * row_c[lag:]) / var)
                    for lag in range(1, min(max_lag_space, n_x // 2))]
            space_acfs.append(max(abs(a) for a in acfs) if acfs else 0.0)
        if space_acfs:
            space_acf_max = float(np.mean(space_acfs))

    return SignatureValue(
        value={"time_acf_max": time_acf_max, "space_acf_max": space_acf_max},
        confidence=min(1.0, n / 100.0),
        n_samples_used=n,
    )


# ----------------------------------------------------------------------
# Stencil locality (PDE detection)
# ----------------------------------------------------------------------

def compute_stencil_locality(
    traj: Trajectory,
    *,
    max_stencil_half_width: int = 5,
) -> SignatureValue:
    """Test if u(t+1, x) is predictable from u(t, x-k:x+k) for finite k.

    PDE data has a finite stencil width (the simulator's spatial
    differencing pattern). Algebraic and ODE data don't have spatial
    structure to exploit. This signature returns the smallest stencil
    half-width for which linear regression of u(t+1, x) on the stencil
    neighbors explains > 0.99 of variance.

    Only meaningful for trajectories with shape (n_t, n_x); returns
    confidence=0 for ODE-shaped trajectories.

    Returns
    -------
    value : Optional[int]
        Stencil half-width (1 = 3-point stencil, 2 = 5-point, ...);
        None if not PDE-shaped or no clean stencil found.
    """
    obs = traj.observable
    if obs.ndim != 2 or obs.shape[1] < 2 * max_stencil_half_width + 5:
        return SignatureValue(
            value=None, confidence=0.0,
            n_samples_used=obs.shape[0],
            notes="not PDE-shaped (need 2D (n_t, n_x) with sufficient n_x)",
        )

    n_t, n_x = obs.shape
    if n_t < 10:
        return SignatureValue(
            value=None, confidence=0.0, n_samples_used=n_t,
            notes="too few time steps",
        )
    if not np.all(np.isfinite(obs)):
        return SignatureValue(
            value=None, confidence=0.0, n_samples_used=n_t,
            notes="trajectory contains NaN/inf",
        )

    # For each candidate stencil half-width k, fit linear model:
    #   u(t+1, x) - u(t, x) = sum_j c_j * u(t, x+j) for j in [-k, k]
    # If R^2 > 0.99 at small k, declare that k as the stencil width.
    interior_lo = max_stencil_half_width
    interior_hi = n_x - max_stencil_half_width
    if interior_hi <= interior_lo + 5:
        return SignatureValue(
            value=None, confidence=0.0, n_samples_used=n_t,
            notes="not enough interior columns",
        )

    targets = (obs[1:, interior_lo:interior_hi] - obs[:-1, interior_lo:interior_hi]).ravel()
    var_targets = targets.var()
    if var_targets < 1e-12:
        return SignatureValue(
            value=None, confidence=0.0, n_samples_used=n_t,
            notes="target has near-zero variance (trivial dynamics)",
        )

    best_k = None
    best_r2 = -np.inf
    for k in range(1, max_stencil_half_width + 1):
        # Build feature matrix: stack u(t, x+j) for j in [-k, k] over (t, x)
        cols = []
        for j in range(-k, k + 1):
            col = obs[:-1, interior_lo + j:interior_hi + j].ravel()
            cols.append(col)
        X = np.stack(cols, axis=1)
        # Least squares
        coefs, *_ = np.linalg.lstsq(X, targets, rcond=None)
        residuals = targets - X @ coefs
        r2 = 1.0 - residuals.var() / var_targets
        if r2 > best_r2:
            best_r2 = r2
            if r2 > 0.99:
                best_k = k
                break  # earliest k that explains > 99% variance

    return SignatureValue(
        value=best_k,
        confidence=min(1.0, n_t / 50.0) if best_k is not None else 0.5,
        n_samples_used=n_t,
        notes=f"best_r2={best_r2:.4f}",
    )


# ----------------------------------------------------------------------
# classify_model_class — Tier A driver
# ----------------------------------------------------------------------

def classify_model_class(
    traj: Trajectory,
    *,
    permutation_threshold: float = 0.8,
    time_acf_threshold: float = 0.3,
    space_acf_threshold: float = 0.3,
) -> tuple[ModelClass, dict[str, SignatureValue]]:
    """Apply the three Tier A discriminators and infer the model class.

    Decision logic
    --------------
      1. If trajectory has spatial structure (state has ≥ 2D shape
         (n_t, n_x) with n_x ≥ ~10) AND spatial autocorrelation is
         strong (s_acf >= space_acf_threshold):
           → PDE (regardless of stencil-linearity test; nonlinear
            PDEs like Burgers fail the linear stencil test but still
            have strong spatial ACF)
      2. Else if permutation_invariance >= permutation_threshold AND
         time_acf < time_acf_threshold:
           → ALGEBRAIC (no order structure)
      3. Else if time_acf >= time_acf_threshold:
           → ODE (time autocorrelation but no spatial structure)
      4. Otherwise: ALGEBRAIC fallback (lowest-commitment default)

    Returns
    -------
    inferred : ModelClass
    diagnostics : dict[str, SignatureValue]
        The three Tier A signature values for inspection.
    """
    perm = compute_permutation_invariance(traj)
    acf = compute_autocorrelation_structure(traj)
    stencil = compute_stencil_locality(traj)
    diagnostics = {
        "permutation_invariance": perm,
        "autocorrelation_structure": acf,
        "stencil_locality": stencil,
    }

    # Robust read with NaN handling
    perm_v = perm.value if isinstance(perm.value, float) and not np.isnan(perm.value) else 0.0
    acf_dict = acf.value if isinstance(acf.value, dict) else {"time_acf_max": 0.0, "space_acf_max": None}
    t_acf = acf_dict.get("time_acf_max", 0.0)
    if isinstance(t_acf, float) and np.isnan(t_acf):
        t_acf = 0.0
    s_acf = acf_dict.get("space_acf_max")

    # Rule 1: PDE — has spatial autocorrelation. Stencil-locality test
    # is a bonus confirmation but isn't required (nonlinear PDEs like
    # Burgers fail the linear stencil test but ARE still PDEs).
    if s_acf is not None and s_acf >= space_acf_threshold:
        return ModelClass.PDE, diagnostics

    # Rule 2: algebraic (permutation-invariant and no time autocorrelation)
    if perm_v >= permutation_threshold and t_acf < time_acf_threshold:
        return ModelClass.ALGEBRAIC, diagnostics

    # Rule 3: ODE (time autocorrelation, no spatial structure)
    if t_acf >= time_acf_threshold:
        return ModelClass.ODE, diagnostics

    # Rule 4: fallback
    return ModelClass.ALGEBRAIC, diagnostics
