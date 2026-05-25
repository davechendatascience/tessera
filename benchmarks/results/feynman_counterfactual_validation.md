# Cross-benchmark validation: C5 counterfactual ranking on Feynman

Tests whether the C5 finding (CF ranking reliably identifies
mechanism-capturing candidates) generalizes from heat equation to
Feynman benchmarks.

**Result: PARTIAL validation — C5 doesn't hurt on Feynman but doesn't help either.** The mechanism that makes C5 work on heat equation (catching Class-B-style natural-overfit) requires the benchmark to HAVE that failure mode. Feynman targets are noiseless and have no natural-overfit candidates on the Pareto front, so CF ranking has nothing to discriminate.

## Headline

| Statistic | Value |
|---|---|
| Total (target, seed) pairs | 24 |
| CF picked different candidate than train-loss | **0/24** |
| CF strictly better on TEST | 0/24 |
| CF strictly worse on TEST | 0/24 |
| CF matches train selection (every case) | 24/24 |

CF ranking and train-loss selection ALWAYS agreed on which candidate to pick across 8 Feynman targets × 3 seeds.

## Why C5 didn't differentiate on Feynman

The C5 finding on heat equation worked because the Pareto front sometimes contained **Class B** candidates: trees of shape `Laplacian(U) / reduce_max(U)` that fit TRAIN at ~1× oracle by exploiting trajectory-specific scalars. On held-out trajectories (the CF set), reduce_max returns DIFFERENT values, so Class B's predictions diverge catastrophically. CF ranking penalized Class B and preferred Class C.

On Feynman:

| Feature | Heat equation | Feynman |
|---|---|---|
| Noise present? | Yes (σ=0.002) | No (deterministic targets) |
| TRAIN-specific tricks available? | Yes (reduce_max over trajectory) | No (no trajectory structure to summarize) |
| Class B emerges on Pareto front? | Sometimes (~1/5 baseline) | Never observed |
| Class A is structurally generic? | Yes (`diff_t` works for any smooth U) | Yes (most fits are honest approximations) |
| CF ranking has discriminating info? | Yes — penalizes Class B | No — all candidates are honest fits |

Without natural-overfit candidates in the front, CF ranking falls back to "rank by aggregate goodness" which is equivalent to train-loss ranking. There's no information to add.

## Per-target detail

| Target | Formula | seed | BT test_rel | BC test_rel | Pick |
|---|---|---|---|---|---|
| I.6.20a | `exp(-θ²/2)` | 2026 | 0.0264 | 0.0264 | same |
| I.6.20a | | 2027 | 0.0099 | 0.0099 | same |
| I.6.20a | | 2028 | 0.0004 | 0.0004 | same |
| I.8.14 | `sqrt(distance²)` | 2026 | 0.6356 | 0.6356 | same |
| I.8.14 | | 2027 | 0.2235 | 0.2235 | same |
| I.8.14 | | 2028 | 0.3143 | 0.3143 | same |
| I.12.1 | `μ·N` | 2026-2028 | 0.0000 | 0.0000 | same (oracle) |
| I.12.5 | `q1·q2/r²` | 2026 | 0.1090 | 0.1090 | same |
| I.12.5 | | 2027 | 0.0000 | 0.0000 | same (oracle) |
| I.12.5 | | 2028 | 0.2230 | 0.2230 | same |
| I.14.3 | `m·g·z` | 2026-2028 | 0/0.11/0 | same | same |
| I.15.3t | Lorentz | 2026-2028 | 0.003/0.002/0.13 | same | same |
| I.27.6 | `d1·d2/(d1+d2)` | 2026-2028 | ~0.04/0.09/0.03 | same | same |
| I.43.31 | Stokes-Einstein | 2026-2028 | 0.38/0.31/0.43 | same | same |

In every case, the CF-best candidate is the same object as the train-loss-best candidate.

## What this means for deploying C5 as a default tessera tool

The earlier plan was: validate C5 on Feynman, then apply as default tessera tool. The validation result refines this:

**Verdict on default deployment:**

- **Safe to add as default**: 0/24 cases where CF picks worse than train-loss. No risk of harming Feynman-like benchmarks.
- **Helpful only when the failure mode exists**: CF ranking adds value on benchmarks with natural-overfit candidates (heat equation, PDE discovery, time-series with TRAIN-specific structure). For closed-form analytical SR (Feynman, IK), it's a no-op.
- **Should be documented as conditional**: "If you suspect your Pareto front contains TRAIN-specific overfit candidates, CF ranking helps select against them. Otherwise it's neutral."

**Recommendation:** ship C5 as a tessera tool but NOT as the default best-by-train replacement. Make it explicit opt-in (e.g., `tessera.search.counterfactual_rank(front, perturbations)`) with documentation about when it helps.

This is more honest than "default-on diagnostic". The mechanism is conditional on the benchmark having Class-B-style failure modes; for benchmarks without them, ad-hoc parsimony + train MSE selection is already optimal.

## The deeper insight (from cross-benchmark)

The basket pattern from heat-equation experiments was:

> Scoring-layer interventions don't help; data-level interventions and post-hoc selection do.

The Feynman cross-benchmark refines this to:

> Scoring-layer interventions don't help on ANY benchmark we tested.  
> Post-hoc selection (C5) helps when the benchmark has natural-overfit failure modes; is neutral otherwise.  
> Data-level interventions (reduce_* downweight, multi-trajectory) help on dynamical-system benchmarks where TRAIN structure can be exploited.

The conditional-helpfulness pattern is itself useful information: tessera's tools have benchmark-class-dependent payoff. There's no universal scoring/selection improvement.

## What this experiment did NOT do

- Did not test C5 with INTERVENTIONAL counterfactuals (changing input ranges, modifying constants). These might discriminate on Feynman where resampling didn't.
- Did not test other basket conjectures on Feynman (C1, C3, C4, C6). Per the pattern they likely fail similarly.
- Did not test on noisy Feynman (with added measurement noise). With noise, Class-B-style overfit might appear, and CF ranking might help.

The clean result for the original question: C5 generalization is conditional. It helps on benchmark classes that have its target failure mode; it's safely neutral on classes that don't.

## Reproducing

```
python benchmarks/run_feynman_counterfactual_validation.py --seeds 3
```

Wall-clock ~22 seconds.
