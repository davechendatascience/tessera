# Shipped: gradient-free symbolic regression (`tessera.search.csp`)

Status: **shipped** (graduated from `experimental/` 2026-05-30). A
gradient-free, enumerative symbolic-regression engine and a top-down
decomposition driver, in tessera's operator vocabulary.

## What it is

`tessera.search.csp` is a different paradigm from the population-based searchers
(GP, SA, Random):

1. **`discover`** — enumerate a CSP-generated **const-free expression
   dictionary** by increasing size, with symmetry-breaking (commutative-child
   ordering, canonical-key dedup) and numerical dedup, then fit a **sparse
   LINEAR combination** of it (beam search / STLSQ / OMP). Constants enter only
   as closed-form least-squares coefficients — **no gradient descent**. This is
   the SINDy idea generalised to a CSP-enumerated dictionary (and the FFX idea
   with tessera's vocabulary). Output is a real tessera `Expr`.
2. **`discover_decompose`** — a top-down **decomposition** driver around
   `discover`: try the shallow solver, else **peel an outer op**
   (`sqrt/square/log/exp/inverse`, reusing `coordinate_discovery`), recurse,
   and **verify**; or detect **separability** (interaction-graph grouping,
   validation-gated) and fit each group. A **polynomial-STLSQ leaf** recovers
   the common case where peeling leaves a polynomial. This reaches deep
   compositional laws a single enumeration can't, while every leaf stays a
   clean-target shallow fit (so it cannot suffer the boosting blow-up below).

## API

```python
from tessera.search.csp import (
    discover, discover_decompose, CSPSRConfig, expr_to_str,
)
res = discover_decompose(env, y, CSPSRConfig(max_size=3))   # env: {name: (N,)}, y: (N,)
print(expr_to_str(res.expr), res.r2)                        # tessera Expr + held-out R^2
```

Backward-compat shims remain at `tessera.experimental.csp_sr` /
`tessera.experimental.csp_decompose`.

## Results

- **Feynman (held-out, `rel<1e-8` machine-precision symbolic match).**
  `discover` alone: 8/30 exact. `discover_decompose`: **24/30 exact, zero
  losses** — including the size-6 relativistic forms (`1/√(1−v²/c²)`) that a
  single enumeration cannot reach. `benchmarks/run_feynman_decompose.py`,
  `benchmarks/results/feynman_decompose.md`.
- **Dynamical systems.** Lorenz + Rössler right-hand sides recovered exactly
  (6/6), gradient-free. `benchmarks/run_dynamical_csp.py`.
- **Head-to-head vs gplearn** (same subset, same `rel<1e-8` metric):
  **24/30 exact vs gplearn 4/30** — gplearn (basic GP, no constant
  optimisation) recovers only the const-free product forms. PySR (GP +
  const-opt) and AI-Feynman (NN + dimensional analysis) recover more on the
  full benchmark but are far heavier — a different design point.
  `benchmarks/run_feynman_vs_baselines.py`,
  `benchmarks/results/feynman_vs_baselines.md`.

## Honest scope (negative results recorded)

- **Deep stacking is a dead end.** `discover_deep` / `discover_boosted`
  (gradient-free symbolic gradient boosting) overfits and never robustly beats
  a single deeper enumeration — see `docs/research/deep_symbolic_csp.md`.
  Decomposition (top-down), not stacking (bottom-up), is what breaks the deep
  wall.
- **Symbolic structure does not beat a CNN on perception** at matched
  parameters — confirmed by our matched-param experiments and the
  optimization-regimes analysis. SR's value is **discovery + interpretability**,
  not vision accuracy. See `docs/research/optimization_regimes.md`.

## Provenance

Research journey: `docs/research/deep_symbolic_csp.md` (decomposition vs
stacking), `docs/research/optimization_regimes.md` (csp vs backprop vs
diff_eml), `docs/research/differentiable_eml_jax.md` (the diff-SR arc this
graduated from).
