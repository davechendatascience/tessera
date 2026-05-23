# `tessera.koopman`

Explicit-latent Koopman with time-delay embedding — closed-form
identification, single-matmul forecast. See [`docs/koopman.md`](../../../docs/koopman.md)
for the full design notes.

## Quick start

```python
import numpy as np
from tessera.koopman import LatentKoopman

# Trajectory: (T, d) array of observations
X = np.random.standard_normal((1000, 3))

# Fit
m = LatentKoopman(
    p=8,             # past lag (time-delay embedding depth)
    k=4,             # latent dimension (bottleneck)
    lambda_pred=1e-4,
    mu=1e-4,
).fit(X)

# One-step forecast
y_next = m.predict_one_step(X[-1])

# Multi-step rollout
trajectory = m.predict_horizon(X[-1], h=20)  # shape (21, 3)

# Inspect: singular values of beta, eigenvalues of K
print(m.sing_values_)            # how concentrated is forward predictability?
eigs, vecs = m.eigenmodes()      # Koopman spectrum
```

## When to use

- The full state isn't observed; time-delay embedding lifts the
  past into a richer latent.
- Multi-feature panels where different latent dimensions can
  specialise (no shared `C` constraint, unlike N4SID).
- You want a closed-form fit (no EM, no MCMC).
- You want forecast as a single matmul on the past window (no
  Kalman loop at test time).

## When NOT to use

- High-dim chaos where the latent must be large — EDMD with
  polynomial lifts may compress better.
- Long-horizon chaotic forecasting — $K^h$ compounds errors faster
  than lifted-dimension EDMD.
- Pure autocorrelation-free signals — Wold ceiling applies; SVD
  truncation cannot create information.

## API

```python
class LatentKoopman:
    def __init__(p=8, k=4, lambda_pred=1e-4, mu=1e-4, target_mode="level"):
        ...

    def fit(X) -> LatentKoopman: ...
    def predict_one_step(x_t, history=None) -> np.ndarray: ...
    def predict_horizon(x_0, h, history=None) -> np.ndarray: ...
    def eigenmodes() -> tuple[np.ndarray, np.ndarray]: ...
    def n_params() -> int: ...

    # After .fit:
    E_           # (k, p*d) encoder
    K_           # (k, k) Koopman operator on latent
    D_           # (d, k) decoder
    beta_        # (d, p*d) full prediction operator (before SVD truncation)
    sing_values_ # singular values of beta_ (diagnostic)
```

`target_mode="delta"` is provided for trending / non-stationary
series — fits to predict the increment $y_{t+1} - y_t$ instead of
$y_{t+1}$, with a per-coordinate mean-delta correction so constant
trend slope is representable. See `docs/koopman.md#delta-mode`.

`history` is the optional context of $p$ preceding observations. If
omitted, the model falls back to the training tail with the last bar
replaced by the queried state — fine for in-distribution use, but for
out-of-distribution test trajectories you should pass `history`
explicitly.

## Tests

```bash
pytest tests/koopman/        # 13 tests
```

Covered:
- Constructor validation
- Fit shape contracts (E, K, D dimensions)
- One-step + horizon forecast shape & finiteness
- `p > T` raises
- Eigenmode + singular value access
- Beats mean baseline on a damped-rotation linear system
- Delta mode runs on a random walk
- Delta mode beats level mode on a trending series at h=20
- Beats naive DMD on Van der Pol (time-delay embedding pays off)
