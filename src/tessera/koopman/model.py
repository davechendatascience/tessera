"""LatentKoopman — closed-form implementation per docs/koopman.md.

Reduced-rank one-step prediction operator → SVD → separate encoder/decoder.
"""
from __future__ import annotations
import numpy as np


def _ridge_solve(A: np.ndarray, B: np.ndarray, reg: float) -> np.ndarray:
    """Solve A·X = B with ridge: X = (A + reg·I)^{-1} B."""
    n = A.shape[0]
    return np.linalg.solve(A + reg * np.eye(n), B)


class LatentKoopman:
    """Explicit-latent Koopman with time-delay embedding.

    Identification (closed-form, 5 steps):
      1. Build Y_past (pd × N), Y_next (d × N), N = T - p
      2. β = Y_next · Y_past^T · (Y_past Y_past^T + λI)^{-1}    (d × pd)
         SVD β = U Σ V^T, truncate to rank k
      3. E = V_k^T                                              (k × pd)
      4. K = X_+ X_-^T · (X_- X_-^T + μI)^{-1}, X = E Y_past   (k × k)
      5. D = Y_next X_-^T · (X_- X_-^T + μI)^{-1}              (d × k)

    Forecast: ŷ_{t+h} = D K^h E ỹ_t  (no filtering, no iteration).

    Parameters
    ----------
    p : int
        past lag (time-delay embedding depth).
    k : int
        latent dimension (bottleneck).
    lambda_pred : float
        ridge for the prediction operator β.
    mu : float
        ridge for the latent OLS (K and D fits).

    Attributes (after .fit)
    ----------
    E_ : (k, pd) encoder
    K_ : (k, k) Koopman operator
    D_ : (d, k) decoder
    sing_values_ : (rank,) singular values of β (for diagnostics)
    p, k, d : dimensions
    """

    def __init__(self, p: int = 8, k: int = 4,
                  lambda_pred: float = 1e-4, mu: float = 1e-4,
                  target_mode: str = "level"):
        """
        target_mode :
            'level' — fit β to predict y_{t+1} (default; matches doc spec)
            'delta' — fit β to predict (y_{t+1} − y_t); convert back at predict time.
                       Useful when the series is non-stationary (e.g. trending price).
                       The latent encodes the dynamics of the *increment* process.
        """
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if target_mode not in ("level", "delta"):
            raise ValueError(f"target_mode must be 'level' or 'delta', got {target_mode!r}")
        self.p = p
        self.k = k
        self.lambda_pred = lambda_pred
        self.mu = mu
        self.target_mode = target_mode
        # filled in fit
        self.E_: np.ndarray | None = None
        self.K_: np.ndarray | None = None
        self.D_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.sing_values_: np.ndarray | None = None
        self.d: int | None = None
        # cache of last training tail (for predicting from a single x_t)
        self._tail: np.ndarray | None = None
        self._train_mean: np.ndarray | None = None

    @staticmethod
    def _past_stack(Y: np.ndarray, p: int):
        """Build past-stack matrix Y_past (pd × N) and next-step Y_next (d × N).

        Y_past[:, t] = [y_{t+p-1}, y_{t+p-2}, ..., y_t]  (most recent first)
        Y_next[:, t] = y_{t+p}

        So for t = 0,…,N-1 we have N = T - p valid columns.
        """
        T, d = Y.shape
        N = T - p
        if N < 2:
            raise ValueError(f"Need T > p+1; T={T} p={p}")
        Y_past = np.zeros((p * d, N))
        Y_next = np.zeros((d, N))
        for t in range(N):
            stack = Y[t : t + p][::-1]
            Y_past[:, t] = stack.flatten()
            Y_next[:, t] = Y[t + p]
        return Y_past, Y_next

    def fit(self, X: np.ndarray) -> "LatentKoopman":
        """Fit on a single trajectory of shape (T, d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, d = X.shape
        self.d = d
        # Center (improves conditioning); model predicts centered, we re-add at the end.
        # In delta mode the mean shift cancels out in differences, but we still center
        # to keep E and K's input scale comparable to level mode.
        self._train_mean = X.mean(axis=0)
        Y = X - self._train_mean

        p = self.p
        Y_past, Y_next = self._past_stack(Y, p)
        # Y_past shape (pd, N), Y_next shape (d, N)

        # In delta mode, the prediction target is the increment y_{t+p} - y_{t+p-1}.
        # We additionally subtract the MEAN delta (the underlying trend slope) before
        # fitting D, and add it back at predict time. Without this offset, a pure
        # trend (constant slope + noise) is unfittable: the latent has no bias term,
        # so D would forcibly predict 0 instead of the true slope.
        self._delta_mean = np.zeros(d)
        if self.target_mode == "delta":
            Y_last_obs = Y_past[:d, :]
            Y_next = Y_next - Y_last_obs               # raw delta
            self._delta_mean = Y_next.mean(axis=1)
            Y_next = Y_next - self._delta_mean[:, None]   # centered delta for D fit

        # Step 2: β = Y_next · Y_past^T · (Y_past Y_past^T + λI)^{-1}
        YpYp = Y_past @ Y_past.T                       # (pd, pd)
        YnYp = Y_next @ Y_past.T                       # (d, pd)
        # β = YnYp · (YpYp + λI)^{-1}, computed via solving (YpYp + λI)^T x = YnYp^T
        beta = np.linalg.solve(YpYp.T + self.lambda_pred * np.eye(p * d),
                                YnYp.T).T
        self.beta_ = beta

        # SVD of β, rank-k truncation
        U, S, Vt = np.linalg.svd(beta, full_matrices=False)
        k = min(self.k, len(S))
        self.k = k
        self.sing_values_ = S
        U_k = U[:, :k]; S_k = S[:k]; V_k = Vt[:k, :]   # V_k is (k, pd)

        # Step 3: E = V_k (k × pd)
        self.E_ = V_k.copy()

        # Step 4: K from latent trajectory
        # X_- = E · Y_past (k × N).  X_+ shifted by 1 in the SAME past-stack.
        # Equivalent: X_+ uses Y_past_{t+1}. Need an additional column from Y[t+p]…
        # Easiest: build a second past-stack with shift +1 ON THE SAME TIME RANGE
        # i.e., for t = 0..N-2, X_+_t = E · [y_{t+p}, y_{t+p-1}, …, y_{t+1}]
        N = Y_past.shape[1]
        X_minus = self.E_ @ Y_past                     # (k, N)
        # Build shifted past stack of length N-1
        Y_past_shift = np.zeros((p * d, N - 1))
        for t in range(N - 1):
            stack = Y[t + 1 : t + p + 1][::-1]
            Y_past_shift[:, t] = stack.flatten()
        X_plus = self.E_ @ Y_past_shift                # (k, N-1)
        X_minus_for_K = X_minus[:, :-1]                # match shape (k, N-1)

        XmXmT = X_minus_for_K @ X_minus_for_K.T
        XpXmT = X_plus @ X_minus_for_K.T
        # K = XpXmT · (XmXmT + μI)^{-1}
        self.K_ = np.linalg.solve(XmXmT.T + self.mu * np.eye(k), XpXmT.T).T

        # Step 5: D = Y_next · X_-^T · (X_- X_-^T + μI)^{-1}    (d × k)
        XX = X_minus @ X_minus.T
        YX = Y_next @ X_minus.T
        self.D_ = np.linalg.solve(XX.T + self.mu * np.eye(k), YX.T).T

        # Cache last p training bars (centered) as default context for predicting from x_t
        self._tail = Y[-p:].copy()
        return self

    # ---------------- prediction ----------------

    def _make_past_stack(self, history: np.ndarray) -> np.ndarray:
        """Build pd-vector past-stack from history. Accepts (p, d) exactly, or
        any (≥p, d) where we use the LAST p bars."""
        h = np.asarray(history, dtype=float)
        if h.shape[0] < self.p:
            raise ValueError(f"history must have at least {self.p} rows, got {h.shape[0]}")
        if h.shape[1] != self.d:
            raise ValueError(f"history must have {self.d} columns, got {h.shape[1]}")
        if h.shape[0] > self.p:
            h = h[-self.p:]
        return h[::-1].flatten()

    def predict_one_step(self, x_t: np.ndarray, history: np.ndarray | None = None) -> np.ndarray:
        """Predict y_{t+1} given current state x_t.

        Per spec: D maps current latent → NEXT observation directly (D was fit
        as D · X_- ≈ Y_next). One-step prediction is D · E · ỹ_t — no K applied.

        In delta mode, D · E · ỹ_t is the predicted *delta*, and we add the current
        observation x_t back to get y_{t+1}.

        history : optional (p, d) array of preceding p observations to use as past-stack.
                  If None, uses _tail (last p training bars) and substitutes x_t for the last.
        """
        x_t_arr = np.asarray(x_t, dtype=float).ravel()
        if history is None:
            x_t_centered = x_t_arr - self._train_mean
            tail = self._tail.copy()
            tail[-1] = x_t_centered
            past = self._make_past_stack(tail)
        else:
            history_centered = np.asarray(history, dtype=float) - self._train_mean
            past = self._make_past_stack(history_centered)
        z = self.E_ @ past
        d_out = self.D_ @ z
        if self.target_mode == "delta":
            # d_out is the *deviation from mean delta*; add back delta_mean for full delta,
            # then add x_t (raw) for the next observation y_{t+1} = x_t + delta.
            return x_t_arr + d_out + self._delta_mean
        else:
            return d_out + self._train_mean

    def predict_horizon(self, x_0: np.ndarray, h: int, history: np.ndarray | None = None) -> np.ndarray:
        """Roll forward h steps from x_0.

        Level mode: ŷ_{t+h} = D K^{h-1} E ỹ_t   (advance latent h-1 times before D)
        Delta mode: ŷ_{t+h} = ŷ_{t+h-1} + D K^{h-1} E ỹ_t  (D outputs delta; integrate)

        Returns shape (h+1, d), first row is x_0.
        """
        x_0_arr = np.asarray(x_0, dtype=float).ravel()
        if history is None:
            tail = self._tail.copy()
            tail[-1] = x_0_arr - self._train_mean
        else:
            hist_arr = np.asarray(history, dtype=float)
            if hist_arr.shape[0] > self.p:
                hist_arr = hist_arr[-self.p:]
            tail = hist_arr - self._train_mean
        past = self._make_past_stack(tail)
        z = self.E_ @ past              # x_t

        out = np.zeros((h + 1, self.d))
        out[0] = x_0_arr

        z_advanced = z.copy()
        if self.target_mode == "delta":
            # D outputs (delta - mean_delta); add mean_delta back, integrate from x_0
            running = x_0_arr.copy()
            for i in range(1, h + 1):
                delta = (self.D_ @ z_advanced) + self._delta_mean
                running = running + delta
                out[i] = running
                z_advanced = self.K_ @ z_advanced
        else:
            for i in range(1, h + 1):
                out[i] = (self.D_ @ z_advanced) + self._train_mean
                z_advanced = self.K_ @ z_advanced
        return out

    def eigenmodes(self):
        """Return (eigvals, eigvecs) of K. None if not fitted."""
        if self.K_ is None:
            return None
        return np.linalg.eig(self.K_)

    def n_params(self) -> int:
        if self.E_ is None: return 0
        return int(self.E_.size + self.K_.size + self.D_.size)
