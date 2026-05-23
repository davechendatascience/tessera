"""Trading-flavoured losses for SR.

Two losses with the same physics: take a real-valued symbolic regression
output `y_pred`, treat it as a *position signal*, score it against a
forward return `y_true` minus transaction-fee penalty for sign flips.

  - `pnl_loss_hard`   — uses discrete `sign(y_pred)` for the position.
                         Non-smooth in y_pred: gradient-based const-opt
                         (BFGS) sees a piecewise-constant landscape and
                         stalls. Use Nelder-Mead / Powell.

  - `pnl_loss_smooth` — uses `tanh(beta * y_pred)` for the position
                         (Hamiltonian relaxation). Smooth + differentiable
                         in y_pred for finite beta; recovers the hard
                         loss exactly as beta → ∞. BFGS / L-BFGS-B work
                         in the inner const-opt loop.

Both functions are TOP-LEVEL so they can be sent across worker
processes via `functools.partial`:

    from functools import partial
    from tessera.search.losses_trading import pnl_loss_smooth

    loss_fn = partial(pnl_loss_smooth, beta=10.0, lambda_flip=0.1)
    GP(cfg, loss_fn=loss_fn).run(env, y, features)

Why "Hamiltonian"
-----------------
The smooth position function `tanh(β·p)` and an optional double-well
potential `(position² − 1)²` together form an Ising-like energy:

    H(p) = − ⟨tanh(β·p) · fwd⟩   ← PnL term (drives position toward +1
                                    where fwd>0, toward −1 where fwd<0)
         + λ_pos · ⟨(tanh(β·p)² − 1)²⟩   ← double-well, pulls |position|→1
         + λ_flip · ⟨(Δ tanh(β·p))²⟩      ← smooth flip penalty

As β → ∞ all three terms recover their discrete counterparts
(sign-based PnL, exact ±1 enforcement, L1-on-sign flip rate). At
finite β the loss is C² in y_pred, which is what gradient methods need.

This is the "phase-transition" framing in physics: continuous order
parameter `tanh(β·p)` becomes the discrete order parameter `sign(p)`
in the high-β limit.
"""
from __future__ import annotations
import numpy as np


def pnl_loss_hard(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    lambda_fee: float = 0.003,
) -> float:
    """Discrete PnL+flip loss (non-smooth in y_pred).

    Equivalent to PySR's run_pysr_eml_1h.py loss with the default
    lambda_fee=0.003 (≈ 30 bp / flip at retail fees).

      L = − mean(tanh(100·y_pred) · y_true)
          + lambda_fee · sum(|diff(sign(y_pred))|) / (2 · N)

    The first term uses tanh(100·y_pred) — already a smooth-ish proxy
    for sign — but the flip-rate term is `diff(sign())` which is
    discontinuous. So this loss is non-smooth in the y_pred values
    where the prediction passes through zero.

    Use with Nelder-Mead / Powell / SA for inner-loop optimisation.
    Don't use with BFGS — it'll stall.
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
    yp = np.where(mask, y_pred, 0.0)
    yt = np.where(mask, y_true, 0.0)

    pos = np.tanh(yp * 100.0)
    pnl_mean = float(np.mean(pos * yt))
    sig = np.sign(yp)
    flip_rate = float(np.sum(np.abs(np.diff(sig))) / (2.0 * len(yp)))
    return -pnl_mean + lambda_fee * flip_rate


def pnl_loss_smooth(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    beta: float = 10.0,
    lambda_pos: float = 0.0,
    lambda_flip: float = 0.1,
) -> float:
    """Hamiltonian-smooth PnL loss.

    Replaces the discontinuous `sign()` flip-rate with a smooth
    L2-on-diff(position) term, where position = `tanh(β·y_pred)` is
    itself smooth.

      L = − mean(tanh(β·y_pred) · y_true)                  PnL term
          + lambda_pos · mean((tanh(β·y_pred)² − 1)²)       double-well (off by default)
          + lambda_flip · mean(diff(tanh(β·y_pred))²)        smooth flip penalty

    Parameters
    ----------
    beta : float
        "Inverse temperature" of the position function. Larger β ⇒
        sharper sign-like behaviour AND less smooth gradient. Defaults
        to 10 which is smooth enough for BFGS but already squashes
        |y_pred| > 0.3 to ~±1.
    lambda_pos : float
        Weight on the double-well potential pulling |position| toward
        1. Off (0.0) by default. Turn on (e.g. 0.01) to discourage
        no-position equilibria; can also destabilise const-opt if too
        high.
    lambda_flip : float
        Weight on the L2 flip penalty. Larger ⇒ smoother positions, fewer
        flips. Default 0.1 ≈ same order as the discrete penalty at
        lambda_fee=0.003 once you account for the unit-variance factor.

    Returns
    -------
    float — finite for any finite y_pred, NaN for shape mismatch.

    Behaviour at large beta
    -----------------------
    For β = 50-100: pos → sign(y_pred); the loss approaches the hard
    version. For β = 10 (default): pos is genuinely smooth and BFGS
    works in the const-opt inner loop.
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
    yp = np.where(mask, y_pred, 0.0)
    yt = np.where(mask, y_true, 0.0)

    pos = np.tanh(beta * yp)
    pnl_mean = float(np.mean(pos * yt))

    loss = -pnl_mean
    if lambda_pos > 0:
        # double-well: position should sit near ±1 (or 0 if you set it
        # the other way — here it pulls toward ±1).
        well = (pos * pos - 1.0) ** 2
        loss += lambda_pos * float(np.mean(well))
    if lambda_flip > 0:
        # L2 flip penalty on the smooth position.
        d_pos = np.diff(pos)
        loss += lambda_flip * float(np.mean(d_pos * d_pos))
    return loss
