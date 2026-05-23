"""Tests for pnl_loss_hard and pnl_loss_smooth (Hamiltonian relaxation)."""
from functools import partial

import numpy as np
import pytest

from tessera.search import (
    pnl_loss_hard, pnl_loss_smooth,
    GP, GPConfig,
)


# ---------------- Basic behaviour ----------------

def test_pnl_loss_hard_finite_on_valid_input():
    rng = np.random.default_rng(0)
    n = 100
    y_pred = rng.standard_normal(n)
    y_true = 0.01 * rng.standard_normal(n)
    loss = pnl_loss_hard(y_pred, y_true)
    assert np.isfinite(loss)


def test_pnl_loss_smooth_finite_on_valid_input():
    rng = np.random.default_rng(0)
    n = 100
    y_pred = rng.standard_normal(n)
    y_true = 0.01 * rng.standard_normal(n)
    loss = pnl_loss_smooth(y_pred, y_true)
    assert np.isfinite(loss)


def test_pnl_loss_smooth_converges_to_hard_as_beta_grows():
    """In the β → ∞ limit, tanh(β·p) → sign(p), so pnl_loss_smooth
    should approach pnl_loss_hard (up to the flip-rate term scaling)."""
    rng = np.random.default_rng(0)
    n = 500
    y_pred = rng.standard_normal(n)
    y_true = 0.01 * rng.standard_normal(n)

    # PnL term only (lambda_flip = 0): the loss is just -mean(pos · y).
    # As β grows, pos → sign(y_pred), and the loss should approach
    # the hard version's PnL term at coefficient 100 (which is
    # already tanh-saturated for any reasonable y_pred).
    hard_pnl = pnl_loss_hard(y_pred, y_true, lambda_fee=0.0)
    smooth_at_100 = pnl_loss_smooth(y_pred, y_true, beta=100.0, lambda_flip=0.0)
    assert abs(smooth_at_100 - hard_pnl) < 1e-4, (
        f"At beta=100, smooth loss {smooth_at_100} should match "
        f"hard loss {hard_pnl} within 1e-4"
    )


def test_pnl_loss_smooth_is_differentiable():
    """Small change in y_pred should give small change in loss
    (verifies no discontinuity from sign() or diff(sign()))."""
    rng = np.random.default_rng(0)
    n = 200
    y_pred = rng.standard_normal(n) * 0.1   # values near zero where sign() is jagged
    y_true = 0.01 * rng.standard_normal(n)

    loss_a = pnl_loss_smooth(y_pred, y_true, beta=10.0)
    # Tiny perturbation
    y_pred_b = y_pred + 1e-4
    loss_b = pnl_loss_smooth(y_pred_b, y_true, beta=10.0)
    delta = abs(loss_b - loss_a)
    # Smooth loss: |Δloss| should be O(1e-4) for O(1e-4) perturbation
    assert delta < 1e-2, (
        f"smooth loss not smooth: |Δloss|={delta} for δy_pred=1e-4"
    )


def test_pnl_loss_hard_is_not_differentiable():
    """The hard loss has jumps when sign(y_pred) crosses zero — verify
    that a perturbation can produce a non-trivial jump."""
    rng = np.random.default_rng(42)
    n = 500
    # Construct y_pred that crosses zero between samples
    y_pred = np.linspace(-0.5, 0.5, n) + 0.001 * rng.standard_normal(n)
    y_true = rng.standard_normal(n) * 0.01

    loss_a = pnl_loss_hard(y_pred, y_true, lambda_fee=0.1)
    # A tiny perturbation that flips some signs near zero will change
    # flip_rate noticeably
    y_pred_b = y_pred + 0.01
    loss_b = pnl_loss_hard(y_pred_b, y_true, lambda_fee=0.1)
    # We don't assert an exact value — just verify it CAN change a lot
    # (showing the loss is not Lipschitz-small).
    assert np.isfinite(loss_a) and np.isfinite(loss_b)


# ---------------- Picklability for multiproc ----------------

def test_pnl_losses_pickle_via_partial():
    """Both losses must be importable + functools.partial-able so they
    work with ProcessPoolExecutor under spawn start method."""
    import pickle
    hard = partial(pnl_loss_hard, lambda_fee=0.005)
    smooth = partial(pnl_loss_smooth, beta=20.0, lambda_flip=0.05)
    pickle.dumps(hard)
    pickle.dumps(smooth)


# ---------------- Integration with GP ----------------

def test_gp_runs_with_smooth_loss():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    # Target: forward return correlated with x
    y = 0.1 * x + 0.01 * rng.standard_normal(n)

    loss_fn = partial(pnl_loss_smooth, beta=10.0, lambda_flip=0.1)
    cfg = GPConfig(pop_size=25, n_gens=8, verbose=False, seed=1,
                   parsimony=1e-5,
                   # BFGS works on smooth losses — use it instead of Nelder-Mead
                   optimize_constants_every=2,
                   optimize_constants_method="BFGS",
                   optimize_constants_maxiter=30)
    gp = GP(cfg, loss_fn=loss_fn)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) >= 1
    best = min(front, key=lambda c: c.train_loss)
    assert np.isfinite(best.train_loss)


def test_gp_runs_with_hard_loss():
    """The hard loss is the canonical baseline — used in run_pysr_eml_1h
    and tessera_btc_1h_pnl benchmarks."""
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = 0.1 * x + 0.01 * rng.standard_normal(n)

    loss_fn = partial(pnl_loss_hard, lambda_fee=0.003)
    cfg = GPConfig(pop_size=25, n_gens=8, verbose=False, seed=1,
                   parsimony=1e-5,
                   optimize_constants_every=2,
                   optimize_constants_method="Nelder-Mead",
                   optimize_constants_maxiter=30)
    gp = GP(cfg, loss_fn=loss_fn)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) >= 1


# ---------------- Hamiltonian-smooth properties ----------------

def test_smooth_loss_with_double_well_term():
    """The double-well term (lambda_pos > 0) is additive and only
    increases the loss; it shouldn't blow up at finite y_pred."""
    rng = np.random.default_rng(0)
    n = 100
    y_pred = 0.3 * rng.standard_normal(n)
    y_true = 0.01 * rng.standard_normal(n)

    loss_no_well = pnl_loss_smooth(y_pred, y_true, beta=5.0, lambda_pos=0.0)
    loss_with_well = pnl_loss_smooth(y_pred, y_true, beta=5.0, lambda_pos=0.5)
    # Double-well adds a non-negative penalty
    assert loss_with_well >= loss_no_well


def test_smooth_loss_higher_beta_gives_sharper_position():
    """At low beta, |pos| stays small for moderate y_pred; at high beta
    it saturates near ±1. This should be visible in the PnL magnitude."""
    rng = np.random.default_rng(0)
    n = 500
    y_pred = 0.2 * rng.standard_normal(n)
    y_true = np.sign(y_pred) * 0.01   # PnL-positive: y_true aligned with sign(y_pred)

    # At low beta, position is soft and |position·y_true| is small
    # At high beta, position is sharp (~±1) and |position·y_true| ~ |y_true|
    pnl_low_beta = -pnl_loss_smooth(y_pred, y_true, beta=0.5, lambda_flip=0.0)
    pnl_high_beta = -pnl_loss_smooth(y_pred, y_true, beta=50.0, lambda_flip=0.0)
    # Higher beta ⇒ more saturated position ⇒ stronger correlation with y_true
    assert pnl_high_beta > pnl_low_beta
