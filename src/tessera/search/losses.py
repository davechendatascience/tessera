"""Loss functions + validity precheck used across all searchers.

The contract for a loss function in tessera.search:
    loss_fn(y_pred: np.ndarray, y_true: np.ndarray) -> float

It MUST handle:
  - shape broadcasting (scalar y_pred, fewer dims, etc.)
  - the residual NaN mask (some entries may be NaN even after the
    precheck rejected majority-NaN predictions)

It should NOT re-implement the NaN-fraction threshold check — that
lives in `_prediction_is_valid` and is called by the scoring path
BEFORE the loss_fn, so every loss function gets the same robust
handling.
"""
from __future__ import annotations
import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error.

    The NaN-fraction validity check happens before this is called (see
    `_prediction_is_valid`), so this function only handles shape
    broadcasting and the residual NaN mask. Returns inf only on
    irrecoverable shape mismatch.
    """
    if np.isscalar(y_pred):
        y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_true.shape:
        try:
            y_pred = np.broadcast_to(y_pred, y_true.shape)
        except ValueError:
            return float("inf")
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return float("inf")
    err = y_pred[mask] - y_true[mask]
    return float(np.mean(err ** 2))


def _prediction_is_valid(
    y_pred,
    y_true: np.ndarray,
    min_valid_frac: float,
) -> bool:
    """Centralised NaN-fraction precheck called before any loss_fn.

    Returns False if:
      - y_pred is a non-finite scalar
      - shape can't broadcast to y_true
      - fewer than `min_valid_frac` of entries are finite
    """
    if np.isscalar(y_pred):
        return bool(np.isfinite(float(y_pred)))
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_true.shape:
        try:
            np.broadcast_to(y_pred, y_true.shape)
        except ValueError:
            return False
        # broadcastable but not equal — let the loss handle it
        return True
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    n_valid = int(mask.sum())
    return n_valid >= max(2, int(min_valid_frac * len(y_true)))
