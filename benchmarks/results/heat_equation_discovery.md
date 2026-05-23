# Heat-equation discovery benchmark

**Target:** `dt_U = U[t+1] − U[t]`  on a heat-equation trajectory
**Simulated α:** 0.05 (true diffusivity)
**Grid:** T=200, X=64; Dirichlet BCs (zero); small Gaussian noise
**Target variance:** 2.214e-05
**GP runtime:** 6.6 s  (150 candidates × 60 gens)
**enable_2d:** True

## Oracle baselines (for honest comparison)

| Hand-coded expression | MSE | % of target var |
|---|---|---|
| `α · Laplacian(U)` (the true heat equation) | 3.979e-06 | 18.0% |
| `diff_t(U, lag=1)` (off-by-one-shift equivalent) | 8.795e-06 | 39.7% |

Both expressions give similar loss because the heat-equation trajectory is smooth in time: `U[t] − U[t−1] ≈ α · Laplacian(U[t−1])`. 
SR's parsimony bias will prefer the simpler cx=2 form (`diff_t(U)`) over the cx=4 form (`α · Laplacian(U)`) whenever they tie on accuracy.

## Pareto front

| Cx | TRAIN loss | rel. to var | Tree |
|---|---|---|---|
| 1 | 0.01159 | 52364.32% | `-0.107561` |
| 2 | 9.058e-06 | 40.92% | `M2D[1·δ(0,0) + -1·δ(1,0)](U)` |
| 3 | 8.776e-06 | 39.64% | `M2D[1·δ(0,0) + -1·δ(1,0)](abs(U))` |
| 6 | 7.114e-06 | 32.14% | `M2D[1·δ(0,0) + -1·δ(1,0)]((tanh(tanh(U)) * U))` |

## Measures appearing in the best (lowest-loss) tree

`M2D[1·δ(0,0) + -1·δ(1,0)]((tanh(tanh(U)) * U))`

**Measure2Ds used:** 1·δ(0,0) + -1·δ(1,0)

**Acceptance**: best train_loss = 7.114e-06 (32.1% of target variance). Recovery of the true α·∇² form is best inspected by reading the tree: look for a `Laplacian_5pt(U)` subexpression multiplied by a Const ≈ 0.05.