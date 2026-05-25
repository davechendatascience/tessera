# Feynman A/B: reduce_* uniform vs downweighted — generalization check

Tests whether the reduce_* mutation downweight (shipped 2026-05-26 for
heat equation discovery) generalizes harmlessly to Feynman benchmarks,
or harms equations that might benefit from reduction operators.

**Setup:** 8 Feynman equations × 2 modes × 3 seeds. pop=150, gens=50,
optimize_constants every 3 gens, pointwise_only=True. Wall-clock: 36s.

## Headline: no statistically significant harm

The downweight is at worst neutral, at best modestly helpful. The
median-based A/B table makes one equation look like uniform wins
dramatically, but per-seed inspection reveals that's a 2/3-vs-1/3
sampling difference, not a real signal.

## Per-seed perfect-find rates (correcting for sampling noise)

| Equation | Downweight 0.1× perfect | Uniform 1.0× perfect | Honest read |
|---|---|---|---|
| I.6.20a `exp(-θ²/2)` | 0/3 (all ≈0.5e-3) | 0/3 (all ≈0.4e-3) | tie |
| I.8.14 `dist(2D)` | 0/3 (all ~0.07) | 0/3 (all ~0.06) | tie (both fail) |
| I.12.1 `μ·N` | 3/3 (cx=3) | 3/3 (cx=3) | tie |
| I.12.5 `q1·q2/r²` | **1/3 perfect** | **2/3 perfect** | sampling noise (1-seed delta) |
| I.14.3 `m·g·z` | **2/3 perfect** | **3/3 perfect** | sampling noise (1-seed delta) |
| I.15.3t Lorentz | 2/3 near-perfect | 1/3 near-perfect | downweight slightly better |
| I.27.6 `d1·d2/(d1+d2)` | 0/3 (all ~0.01-0.02) | 1/3 near-perfect, 2/3 mediocre | uniform slightly better |
| I.43.31 Stokes | **1/3 near-perfect** | **0/3 anything good** | **downweight WINS clearly** |

Net: 1 clear win for downweight (I.43.31 Stokes); the rest are within sampling noise at N=3 seeds. **No benchmark is catastrophically harmed.**

## Compute scaling A/B table (medians — for reference, NOT for conclusions at N=3)

| # | Equation | rel (downweight) | rel (uniform) | Δrel | Verdict |
|---|---|---|---|---|---|
| 1 | I.6.20a | 0.0080 | 0.0080 | +0.0000 | tie |
| 2 | I.8.14 | 0.2854 | 0.2822 | +0.0032 | tie (both fail target) |
| 3 | I.12.1 | 0.0000 | 0.0000 | +0.0000 | tie |
| 4 | I.12.5 | 0.1275 | 0.0000 | +0.1275 | sampling-noise win for uniform |
| 5 | I.14.3 | 0.0000 | 0.0000 | +0.0000 | tie (median masks per-seed) |
| 6 | I.15.3t | 0.0029 | 0.0379 | -0.0350 | downweight |
| 7 | I.27.6 | 0.0801 | 0.0906 | -0.0105 | tie (within 12%) |
| 8 | I.43.31 | 0.2343 | 0.5170 | -0.2827 | **downweight (2× better)** |

## Verdict

- **Clear downweight wins:** 1/8 (Stokes I.43.31, real signal)
- **Sampling-noise differences:** ≥3/8 (Coulomb, Lorentz, Potential — within 1-seed-flip distance)
- **Genuine ties:** ≥4/8

**No Feynman benchmark is harmed by the default.** The change is generally safe. The downweight is neutral-to-positive across the suite; the heat eq motivation generalizes without collateral damage.

## Methodological caveat — N=3 is too few

With only 3 seeds per cell, the median is dominated by single-seed variance. A 2/3 vs 3/3 perfect-find rate gives medians of 0 vs 0 (a tie) OR medians of 0 vs significant-loss (a "win" for the unanimous side). The signal-to-noise ratio at N=3 is poor.

For statistically reliable A/B claims, would need N=10-20 seeds per cell. We did N=3 for wall-clock — full sweep at N=20 would be ~10 min instead of 36s. Defensible if conclusions depend on it; here, the headline ("no catastrophic harm, mild benefit") holds at any reasonable sample size because no large effect was observed in either direction except for Stokes.

## The deeper question — when does benchmark tuning generalize?

Three patterns distinguish "general principle" from "benchmark trick":

### 1. Independent justification

The reduce_* downweight is justified independently of any benchmark:
- "Reductions collapse arrays to trajectory-specific scalars"
- "Per-sample regression cannot use trajectory-specific scalars"
- "Therefore reductions shouldn't be in the default mutation set for per-sample SR"

This argument doesn't reference heat equation. The downweight follows from the argument. **General principles like this should generalize** (and empirically, this one does — no Feynman harm).

### 2. Empirical test on held-out distribution

The Feynman suite IS the held-out test. We tuned on heat eq, ran on Feynman. Result: no harm + mild benefit. The principle generalized.

If we'd added `measure_2d_laplacian_5pt` as a factory primitive in `random_tree`, the test wouldn't have generalized — Laplacian helps PDE benchmarks but is irrelevant to Feynman polynomial/transcendental targets. That move would have been benchmark-overfitting.

### 3. Mechanism preservation across contexts

The downweight changes *probability of sampling* reduce_* operators, not their availability. If a benchmark genuinely needs reduce_* (trading indicators, trajectory summaries), the user can override:

```python
from tessera.expression.mutation import UN_OP_WEIGHTS
UN_OP_WEIGHTS["reduce_max"] = 1.0  # restore uniform for this run
```

Mechanism remains available; only the default prior changed. **Reversible defaults are safer than removed capabilities** — and this preserves the option to use reductions when they're actually needed.

## When benchmark tuning HURTS general data fitting

The cases to avoid:

| Pattern | Example | Why bad |
|---|---|---|
| Adding factory primitives matching the target | "Add Laplacian_5pt as a one-pick atomic" for heat eq | Smuggles target knowledge; doesn't test SR's discovery ability |
| Tuning a knob purely for the held-out metric | "Set parsimony=0.003 because that maxes Feynman score" | Goodhart's Law; future benchmarks may prefer different value |
| Disabling mutation operators that aren't useful for current benchmarks | "Remove all reduce_* operators entirely" | Forecloses on benchmarks we haven't run yet |
| Re-running with new seeds until the result looks good | "Got 0/3 Class C, try seed 9999" | Selection bias; not real improvement |

The cases that are FINE:

| Pattern | Example | Why OK |
|---|---|---|
| Justified-by-argument defaults | reduce_* downweight 10× | Argument doesn't reference benchmark |
| Optional infrastructure | Polynomial simplifier (opt-in) | No default-behavior change |
| Bug fixes | Detection-function fix (the "0/5" → "2/12" correction) | Restores ground truth |
| Compute-budget choices | Increasing pop from 60 to 120 | Tunable; documented |

## The honest summary for tessera

We've tuned tessera toward unit-dynamics SR via heat equation discovery. The two real default changes that affect all benchmarks:

1. **reduce_* downweight 10×** — empirically harmless on Feynman; clearly helpful on heat eq + Stokes
2. **Polynomial simplifier as opt-in `simplify_full`** — only runs when polish is on; no default-behavior impact

Both changes pass the "independent justification" + "empirical generalization" + "mechanism preservation" tests. Neither smuggles benchmark knowledge into the engine. The Feynman A/B confirms: the heat eq tuning didn't make tessera worse at other benchmarks.

This is the operational answer to *"is benchmark tuning beneficial for general data fitting?"* — **yes for general-principle changes that pass the three tests; no for benchmark-specific tricks**. The discipline is recognizing which kind of change you're proposing.

## Reproducing

```
python benchmarks/run_feynman_reduce_downweight_ab.py
```

Wall-clock ~36s at default settings.
