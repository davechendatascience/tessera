"""Tests for tessera.koopman.LatentKoopman.

Tiny synthetic data generators are inlined here so the test suite stays
self-contained (no dependency on a benchmarks framework).
"""
import numpy as np
import pytest

from tessera.koopman import LatentKoopman


# ---------------- inline synthetic dynamical systems ----------------

def linear_system_trajectory(T: int, d: int = 3, rng=None) -> np.ndarray:
    """Damped rotation y_{t+1} = A y_t (mostly deterministic).

    A is block-diagonal with a 0.99-damped rotation block and a 0.95-damped
    scalar. Output is 3-D. Tiny noise floor so test/train differ slightly.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    theta = 0.07
    A = np.array([
        [0.99 * np.cos(theta), -0.99 * np.sin(theta), 0.0],
        [0.99 * np.sin(theta),  0.99 * np.cos(theta), 0.0],
        [0.0,                   0.0,                  0.95],
    ])
    y = np.zeros((T, d))
    y[0] = rng.standard_normal(d)
    for t in range(T - 1):
        y[t + 1] = A @ y[t] + 0.001 * rng.standard_normal(d)
    return y


def van_der_pol_trajectory(T: int, mu: float = 1.0, dt: float = 0.05,
                             burn_in: int = 500, rng=None) -> np.ndarray:
    """Van der Pol oscillator: x'' - μ(1 - x²)x' + x = 0. RK4 integration."""
    if rng is None:
        rng = np.random.default_rng(0)
    def rhs(state):
        x, v = state
        return np.array([v, mu * (1 - x * x) * v - x])
    state = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
    # burn-in to land on limit cycle
    for _ in range(burn_in):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    out = np.zeros((T, 2))
    out[0] = state
    for t in range(T - 1):
        k1 = rhs(out[t])
        k2 = rhs(out[t] + 0.5 * dt * k1)
        k3 = rhs(out[t] + 0.5 * dt * k2)
        k4 = rhs(out[t] + dt * k3)
        out[t + 1] = out[t] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# --------------- core model tests ---------------

def test_init_validates_params():
    with pytest.raises(ValueError):
        LatentKoopman(p=0)
    with pytest.raises(ValueError):
        LatentKoopman(p=2, k=0)


def test_target_mode_validation():
    with pytest.raises(ValueError):
        LatentKoopman(p=4, k=2, target_mode="invalid")


def test_fit_returns_self():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2, lambda_pred=1e-3, mu=1e-3)
    result = m.fit(X)
    assert result is m
    assert m.E_.shape == (2, 4 * 3)
    assert m.K_.shape == (2, 2)
    assert m.D_.shape == (3, 2)


def test_predict_one_step_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2).fit(X)
    pred = m.predict_one_step(X[10])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_predict_horizon_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2).fit(X)
    out = m.predict_horizon(X[10], h=5)
    assert out.shape == (6, 3)
    assert np.allclose(out[0], X[10])


def test_p_larger_than_data_raises():
    X = np.random.randn(10, 2)
    m = LatentKoopman(p=20, k=2)
    with pytest.raises(ValueError):
        m.fit(X)


def test_eigenmodes_returns_pair():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2).fit(X)
    eigs, vecs = m.eigenmodes()
    assert eigs.shape == (2,)
    assert vecs.shape == (2, 2)


def test_singular_values_exposed():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2).fit(X)
    assert m.sing_values_ is not None
    assert m.sing_values_.ndim == 1
    assert (m.sing_values_ >= 0).all()
    assert np.all(np.diff(m.sing_values_) <= 1e-10)


def test_n_params_counts_E_K_D():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    m = LatentKoopman(p=4, k=2).fit(X)
    expected = (2 * 12) + (2 * 2) + (3 * 2)  # E + K + D = 24 + 4 + 6 = 34
    assert m.n_params() == expected


def test_on_linear_system_beats_mean_baseline():
    """LatentKoopman on a smooth linear system should beat predicting the mean."""
    train = linear_system_trajectory(T=500, d=3, rng=np.random.default_rng(0))
    test = linear_system_trajectory(T=200, d=3, rng=np.random.default_rng(1))
    m = LatentKoopman(p=4, k=2, lambda_pred=1e-6, mu=1e-6).fit(train)
    mean_err = np.mean(np.sum((test[1:] - test[:-1].mean(axis=0)) ** 2, axis=1))
    model_err = []
    for t in range(4, len(test) - 1):
        history = test[t - 3 : t + 1]
        pred = m.predict_one_step(test[t], history=history)
        model_err.append(np.sum((pred - test[t + 1]) ** 2))
    assert np.mean(model_err) < mean_err * 0.8, (
        f"latent koopman ({np.mean(model_err):.3e}) didn't beat mean ({mean_err:.3e})"
    )


def test_delta_mode_runs_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)).cumsum(axis=0)   # random walk (non-stationary)
    m = LatentKoopman(p=4, k=2, target_mode="delta", lambda_pred=1e-3, mu=1e-3).fit(X)
    pred = m.predict_one_step(X[100], history=X[97:101])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_delta_mode_beats_level_on_trending_multistep():
    """On a series with a strong trend, multi-step rollout in level mode degrades
    badly because the model is anchored to train_mean. Delta mode integrates the
    trend slope and stays accurate over long horizons."""
    rng = np.random.default_rng(0)
    T = 800
    trend = np.linspace(0, 200, T)[:, None] * np.array([[1.0, 0.5, -0.3]])
    noise = 0.5 * rng.standard_normal((T, 3))
    X = trend + noise

    train, test = X[:400], X[400:]

    m_level = LatentKoopman(p=8, k=2, target_mode="level", lambda_pred=1e-3, mu=1e-3).fit(train)
    m_delta = LatentKoopman(p=8, k=2, target_mode="delta", lambda_pred=1e-3, mu=1e-3).fit(train)

    # h=20 step rollout, compare predicted vs true at h=20
    h = 20
    err_level = []; err_delta = []
    for t in range(8, len(test) - h):
        history = test[t - 7 : t + 1]
        p_level = m_level.predict_horizon(test[t], h=h, history=history)
        p_delta = m_delta.predict_horizon(test[t], h=h, history=history)
        err_level.append(np.sum((p_level[-1] - test[t + h]) ** 2))
        err_delta.append(np.sum((p_delta[-1] - test[t + h]) ** 2))
    assert np.mean(err_delta) < 0.33 * np.mean(err_level), (
        f"delta-mode ({np.mean(err_delta):.3e}) didn't beat level-mode "
        f"({np.mean(err_level):.3e}) on multi-step trending rollout"
    )


def test_beats_naive_dmd_on_van_der_pol():
    """Time-delay embedding gives latent Koopman an edge on Van der Pol.

    Naive DMD baseline: y_{t+1} = A y_t  (no lag embedding), fit by OLS.
    Latent Koopman with p=8 lag-embedding should beat it.
    """
    train = van_der_pol_trajectory(T=2000, mu=1.0, rng=np.random.default_rng(0))
    test = van_der_pol_trajectory(T=500, mu=1.0, rng=np.random.default_rng(1))

    # Naive DMD: y_{t+1} ≈ A y_t
    Y_curr = train[:-1].T   # (d, T-1)
    Y_next = train[1:].T    # (d, T-1)
    A_dmd = np.linalg.solve(
        (Y_curr @ Y_curr.T) + 1e-6 * np.eye(2),
        (Y_next @ Y_curr.T).T,
    ).T

    lk = LatentKoopman(p=8, k=6, lambda_pred=1e-6, mu=1e-6).fit(train)

    err_dmd = []; err_lk = []
    for t in range(8, len(test) - 1):
        history = test[t - 7 : t + 1]
        err_dmd.append(np.sum((A_dmd @ test[t] - test[t + 1]) ** 2))
        pred_lk = lk.predict_one_step(test[t], history=history)
        err_lk.append(np.sum((pred_lk - test[t + 1]) ** 2))
    assert np.mean(err_lk) < np.mean(err_dmd), (
        f"latent ({np.mean(err_lk):.3e}) didn't beat naive DMD "
        f"({np.mean(err_dmd):.3e}) on Van der Pol"
    )
