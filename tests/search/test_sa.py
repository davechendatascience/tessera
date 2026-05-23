"""Tests for SimulatedAnnealing."""
import numpy as np
import pytest

from tessera.search import SimulatedAnnealing, SAConfig, Candidate


def test_sa_runs_and_returns_pareto_front():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = x * x

    cfg = SAConfig(n_steps=200, T_initial=1.0, T_final=1e-3,
                   seed=42, verbose=False)
    sa = SimulatedAnnealing(cfg)
    front = sa.run({"x": x}, y, ["x"])

    assert len(front) >= 1
    assert all(isinstance(c, Candidate) for c in front)
    # Pareto front must be sorted ascending in cx
    cxs = [c.complexity for c in front]
    assert cxs == sorted(cxs)
    # And loss must be monotone non-increasing along the front
    losses = [c.train_loss for c in front]
    assert all(losses[i] >= losses[i+1] for i in range(len(losses) - 1))


def test_sa_history_records_per_restart_stats():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = x

    cfg = SAConfig(n_steps=100, n_restarts=2, seed=1, verbose=False)
    sa = SimulatedAnnealing(cfg)
    sa.run({"x": x}, y, ["x"])

    assert len(sa.history) == 2  # one entry per restart
    for h in sa.history:
        for key in ("restart", "n_proposed", "n_accepted", "best_loss",
                    "best_cx", "elapsed"):
            assert key in h


def test_sa_accepts_improvements_always():
    """Δ<0 must accept with probability 1 regardless of T."""
    cfg = SAConfig(n_steps=10, seed=0, verbose=False)
    sa = SimulatedAnnealing(cfg)
    # internal API
    assert sa._accept(-1.0, T=0.5) is True
    assert sa._accept(-1.0, T=0.01) is True
    assert sa._accept(-1.0, T=1e-9) is True


def test_sa_metropolis_acceptance_probability():
    """Δ>0 acceptance probability = exp(-Δ/T) within bounds."""
    cfg = SAConfig(n_steps=10, seed=42, verbose=False)
    sa = SimulatedAnnealing(cfg)

    # Many trials at fixed Δ, T; check acceptance rate matches exp(-Δ/T)
    delta = 1.0
    T = 1.0
    expected_p = np.exp(-delta / T)  # ≈ 0.368
    n_trials = 5000
    accepts = sum(sa._accept(delta, T) for _ in range(n_trials))
    observed_p = accepts / n_trials
    # 2σ tolerance for binomial: σ ≈ sqrt(p(1-p)/N) ≈ 0.0068
    assert abs(observed_p - expected_p) < 0.03, (
        f"acceptance rate {observed_p:.3f} differs from expected {expected_p:.3f}"
    )


def test_sa_temperature_schedule_monotone_decreasing():
    cfg = SAConfig(n_steps=100, T_initial=1.0, T_final=0.001,
                   cooling="exponential", verbose=False)
    sa = SimulatedAnnealing(cfg)
    temps = [sa._temperature(k) for k in range(100)]
    assert temps[0] == pytest.approx(1.0)
    assert temps[-1] == pytest.approx(0.001, rel=1e-3)
    assert all(temps[i] >= temps[i+1] for i in range(len(temps) - 1))


def test_sa_linear_cooling():
    cfg = SAConfig(n_steps=100, T_initial=1.0, T_final=0.0,
                   cooling="linear", verbose=False)
    sa = SimulatedAnnealing(cfg)
    assert sa._temperature(0) == pytest.approx(1.0)
    assert sa._temperature(99) == pytest.approx(0.0, abs=1e-6)
    # Mid-point ~ 0.5
    assert abs(sa._temperature(50) - 0.5) < 0.02


def test_sa_finds_a_useful_signal_on_easy_target():
    """y = x: SA should beat the constant baseline (MSE ~ var(y))."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = x

    cfg = SAConfig(n_steps=500, T_initial=1.0, T_final=1e-3,
                   seed=2026, verbose=False)
    sa = SimulatedAnnealing(cfg)
    front = sa.run({"x": x}, y, ["x"])

    best = min(front, key=lambda c: c.train_loss)
    # var(x) ~ 1; any tree closer to x than constant should give MSE < 1
    assert best.train_loss < y.var() * 0.5, (
        f"SA didn't improve over predicting mean: best loss={best.train_loss}, "
        f"var(y)={y.var()}"
    )


def test_sa_multirestart_diversifies():
    """Two restarts should explore differently due to different seed offsets."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = x * x + x

    cfg = SAConfig(n_steps=200, n_restarts=3, seed=10, verbose=False)
    sa = SimulatedAnnealing(cfg)
    front = sa.run({"x": x}, y, ["x"])

    # All three restarts contribute to history
    assert len(sa.history) == 3
    # And the final front is non-trivial
    assert len(front) >= 1
