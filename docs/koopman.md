# Latent Koopman — design notes

**Status: shipping in `tessera.koopman` (v0.1.1).**

- Public API: `from tessera.koopman import LatentKoopman`
- Tests: `tests/koopman/test_koopman.py` (13 tests, all passing)

## Motivation

A standard linear state-space model (N4SID family) identifies a $k$-dim
latent state by SVD of the joint past-future projection. Two structural
points motivate an alternative:

1. **The identification objective is reconstruction-flavoured.** N4SID
   SVDs $\Pi = Y_f \cdot \text{pinv}(Y_p)$ — modes that fit the joint
   covariance, not modes that maximise one-step forecast directly.

2. **A single matrix `C` plays two roles.** It decodes the latent to
   the current observation $y_t = C x_t$ AND, via $y_\text{next} = C A x_t$,
   to the next. Each latent dimension must simultaneously serve
   reconstruction and propagation.

The Koopman framing offers an alternative: an explicit latent factor
with its own dynamics matrix $K$, and SEPARATE encoder $E$ (past →
latent) and decoder $D$ (latent → next observation). The objective is
one-step forecast directly. This doc specifies the
linear-deterministic version (no neural nets, no MCMC).

## Architecture

Observation $y_t \in \mathbb{R}^d$. Past-stacked observable (time-delay
embedding, the Koopman "lift"):

$$\tilde y_t = \begin{bmatrix} y_t \\ y_{t-1} \\ \vdots \\ y_{t-p+1} \end{bmatrix} \in \mathbb{R}^{pd}$$

Three linear maps and the latent state:

| Symbol | Shape | Role |
|---|---|---|
| $E$ | $k \times pd$ | encoder: past-stack → latent. $x_t = E \tilde y_t$ |
| $K$ | $k \times k$ | Koopman dynamics on latent. $x_{t+1} = K x_t$ |
| $D$ | $d \times k$ | decoder: latent → next observation. $y_{t+1} = D x_t + \varepsilon$ |

For h-step forecast: $\hat y_{t+h} = D K^{h-1} E \tilde y_t$.

The latent $x_t$ is explicit (low-dim, $k$ free hyperparameter), the
dynamics live on it, and the encoder/decoder are SEPARATE matrices —
none tied to the others by reuse.

## Identification procedure

Closed-form, no iteration required (no EM, no MCMC). Five steps.

### Step 1 — Build data matrices

From training data with $T$ observations:

$$\mathbf{Y}_\text{past} \in \mathbb{R}^{pd \times N}, \quad \mathbf{Y}_\text{next} \in \mathbb{R}^{d \times N}, \quad N = T - p$$

Column $t$ of $\mathbf{Y}_\text{past}$ is $\tilde y_t$; column $t$ of
$\mathbf{Y}_\text{next}$ is $y_{t+1}$.

### Step 2 — Reduced-rank one-step prediction operator

Ridge OLS:

$$\beta = \mathbf{Y}_\text{next} \cdot \mathbf{Y}_\text{past}^\top \cdot \left( \mathbf{Y}_\text{past}\mathbf{Y}_\text{past}^\top + \lambda I \right)^{-1} \in \mathbb{R}^{d \times pd}$$

This is the BEST linear forecast of $y_{t+1}$ from $\tilde y_t$,
ridge-regularized.

SVD: $\beta = U \Sigma V^\top$. Truncate to rank $k$:

$$\beta_k = U_k \Sigma_k V_k^\top$$

Discards $r-k$ singular components that contribute least to one-step
prediction.

### Step 3 — Encoder

$$\boxed{E = V_k^\top \in \mathbb{R}^{k \times pd}}$$

The encoder is the top-$k$ right singular vectors of the prediction
operator. These are the principal directions in past-stack space that
carry forward predictive information.

### Step 4 — Identify $K$ on the latent trajectory

Compute the latent trajectory from training data:

$$X_- = E \cdot \mathbf{Y}_\text{past} \in \mathbb{R}^{k \times N}$$

The "advanced" latent uses $\tilde y_{t+1}$:

$$X_+ = E \cdot \mathbf{Y}_\text{past}^{(+)} \in \mathbb{R}^{k \times (N-1)}$$

OLS the Koopman operator:

$$\boxed{K = X_+ X_-^\top \left( X_- X_-^\top + \mu I \right)^{-1} \in \mathbb{R}^{k \times k}}$$

Its eigenvalues have Koopman-mode interpretation
(growth/decay/oscillation rates of latent components).

### Step 5 — Identify decoder $D$

OLS from current latent to next observation:

$$\boxed{D = \mathbf{Y}_\text{next} X_-^\top \left( X_- X_-^\top + \mu I \right)^{-1} \in \mathbb{R}^{d \times k}}$$

## Forecast / use at test time

Given a test past-stack $\tilde y_t$:

$$\hat y_{t+1} = D \cdot E \cdot \tilde y_t \quad \text{(one-step)}$$
$$\hat y_{t+h} = D \cdot K^{h-1} \cdot E \cdot \tilde y_t \quad \text{(h-step)}$$

There is NO filtering loop at test time. The forecast is a single
matrix multiply on the past-stack window. This is a structural
property distinct from the Kalman filter, which incrementally updates
a posterior given the current observation.

**Subtle but critical**: one-step is $D \cdot E \tilde y_t$ — **no $K$
applied**. The reason: in Step 5, $D$ is fit such that
$D \cdot X_- \approx Y_\text{next}$, i.e., $D$ maps the **current**
latent directly to the **next** observation. Applying $K$ would
predict $y_{t+2}$ instead.

## Delta mode (`target_mode="delta"`)

For non-stationary series (trending price, accumulating count), the
implementation supports a delta-target mode: fit $\beta$ to predict
the increment $y_{t+1} - y_t$ rather than $y_{t+1}$ directly. At
predict time, the model outputs $\hat y_{t+1} = y_t + \hat\Delta$.
Additionally, the mean training-delta is subtracted before fitting
$D$ and added back at predict time, so a constant-slope trend is
representable (the latent has no bias term).

In practice: a random-walk-with-drift or a linear-trend series
where level mode degrades over multi-step rollouts because the
model is anchored to `train_mean`, delta mode integrates the slope
and stays accurate. Tested at `test_delta_mode_beats_level_on_trending_multistep`.

## Comparison with N4SID linear SSM

| Aspect | N4SID-style SSM | Latent Koopman |
|---|---|---|
| Latent state | Yes ($k$-dim) | Yes ($k$-dim) |
| Past-encoding | Implicit via $C$ (reuse) | Explicit $E$ matrix |
| Forward-decoding | Same $C$ used | Separate $D$ matrix |
| Identification | Hankel projection: $\| Y_f - \Gamma_f x \|^2$ | Reduced-rank one-step: $\| Y_\text{next} - DKE Y_\text{past} \|^2$ |
| Distinct linear maps | 2 ($A$, $C$) | 3 ($E$, $K$, $D$) |
| Closed-form | Yes (SVD of $\Pi$) | Yes (SVD of $\beta$, then two OLS) |
| Use at test | Recursive Kalman filter | Single matmul on past window |
| Noise model | Native Gaussian dynamics + obs | Optional, deterministic by default |

## Expected behaviour and limitations

The Wold decomposition still applies. For data with near-zero
autocorrelation in raw $y$, the prediction operator $\beta$ has
near-zero singular values; SVD truncation cannot create information.
**Latent Koopman will not exceed the Wold ceiling on
uncorrelated returns alone.**

Where it should beat N4SID, if anywhere:

- **Multi-feature panels**. Latent dimensions can specialise to
  different sources of forward signal without each having to also
  reconstruct $y_t$ well.
- **High-dimensional input, low-dimensional target.** When $d$ is
  large but we predict a scalar/small target, $E$ has freedom that
  N4SID's $C^\top \approx E$ tying does not allow.

## Implementation considerations

**Numerical**:
- Ridge regularisation $\lambda$ in Step 2 is essential — $pd$ can
  exceed $N$. Set $\lambda \approx 10^{-3} \cdot \text{tr}(Y_\text{past}Y_\text{past}^\top) / pd$.
- Ridge in Step 4/5 ($\mu$) similar.
- All steps are $O((pd)^3)$ for matrix solves and $O(N \cdot (pd)^2)$
  for regressions.

**Hyperparameters**:
- $p$ (past lag): 8–120 typical. Bigger = richer time-delay
  embedding, more parameters to fit.
- $k$ (latent dim): 3–100. Same caution as N4SID's $k$.
- $\lambda, \mu$: ridge strengths; small (1e-4 default). Use train
  cross-val if tuning matters.

**Centring**: the implementation centres training data by its mean
before fitting; predictions are de-centred before return. Improves
numerical conditioning on systems with non-zero-mean attractors.

**Indexing convention**: past-stack is "most recent first":
$\tilde y_t = [y_t, y_{t-1}, \ldots, y_{t-p+1}]^\top$. The first $d$
entries of $\tilde y_t$ are the current observation $y_t$.

## Diagnostics

For each fit, report:

1. **Singular value spectrum of $\beta$** — steep decay = signal
   concentrated in few modes. Gradual decay = mostly noise.
2. **Eigenvalues of $K$** — on/inside/outside unit circle? Real vs
   complex? Few dominant modes vs spread spectrum?
3. **TRAIN vs TEST one-step corr** — gap quantifies overfitting /
   noise modelling.
4. **DKE consistency check** — compare $DKE$ in Steps 4–5 to
   $\beta_k$ in Step 2. Large discrepancy signals poor Step-4
   conditioning.

## What's NOT yet implemented from this doc

1. **Three-way joint with target head** — an SVD-based joint
   reconstruction-prediction-target objective that stacks
   $\beta_\text{cur}$, $\beta_\text{next}$, $\beta_\text{target}$ and
   takes a joint truncated SVD. The current `LatentKoopman` does only
   Steps 1–5.
2. **Optional Gaussian-noise wrapping** for Bayesian uncertainty
   propagation ($Q_K$ from $X_+ - K X_-$ residuals, $R_D$ from
   $Y_\text{next} - D X_-$ residuals).
3. **Forward-streaming filter** — a stateful version that maintains
   the latent across consecutive calls (Kalman-like update) for live
   signals.

Each is straightforward and separately scoped — implement when needed.

## References

- Koopman, B. O. (1931). Hamiltonian Systems and Transformation in Hilbert Space.
- Schmid, P. J. (2010). Dynamic Mode Decomposition. *J. Fluid Mech.*
- Kutz, Brunton, Brunton, Proctor (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM.
- Williams, Rowley, Kevrekidis (2015). A kernel-based method for data-driven Koopman spectral analysis. (EDMD baseline)
- Tu et al. (2014). On dynamic mode decomposition. (DMD baseline)
- Reinsel & Velu (1998). *Multivariate Reduced-Rank Regression*. (Step 2 is reduced-rank ridge regression)
- Otto, S. E., Rowley, C. W. (2019). Linearly-Recurrent Autoencoder Networks for Learning Dynamics. *SIAM J Appl Dyn Sys.* (Closest match to the explicit-latent E/K/D factorisation; sec. 3 gives the closed-form linear case.)
- Brunton et al. (2017). Chaos as an intermittently forced linear system (HAVOK). (Time-delay embedding rationale)
- Lusch, Kutz, Brunton (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nat Commun.* (Nonlinear lifting; linear Koopman in latent — same architecture.)
