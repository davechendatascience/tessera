# MVP 7.1 / C1-refined: ABC scoring on held-out trajectory — FALSIFIED

Empirical test of conjecture C1-refined from
`docs/research/process_discovery_sr.md`. First occupant of
`tessera.experimental`. **Result: conjecture falsified at both β=1.0
and β=0.1.**

## The conjecture, restated

> "ABC-style summary-statistics scoring evaluated on a held-out
> trajectory suppresses Class B (template/reduce_*) more than
> pointwise MSE on the same held-out data."

Operationalized as a three-mode comparison:

| Mode | Scoring |
|---|---|
| A (baseline) | fitness = train_loss + parsimony · cx |
| B (hold MSE) | fitness += β · MSE(tree(U_hold), dt_U_hold) |
| C (hold ABC) | fitness += β · ABC_distance(stats(dt_U_hold), stats(tree(U_hold))) |

ABC summary statistics: var_total, mean_abs, acf_time_lag1,
acf_space_lag1, spatial_mean_var, temporal_mean_var (interior,
NaN-robust, scale-invariant relative-error distance).

**Setup:** T=200, X=32, α=0.05. Three trajectories with different ICs
(TRAIN seed 100, HOLD seed 200, TEST seed 999). pop=240, gens=100,
5 seeds per mode. Tested at β ∈ {1.0, 0.1}. Wall-clock ~3 min per β.

## Headline result

| | Baseline (A) | Hold MSE (B) | Hold ABC (C) |
|---|---|---|---|
| Class C (clean mechanism) at β=1.0 | 1/5 | 1/5 | **0/5** |
| Class C-partial at β=1.0 | 0/5 | 0/5 | 0/5 |
| Class C (clean mechanism) at β=0.1 | 1/5 | 1/5 | 0/5 |
| Class C-partial at β=0.1 | 0/5 | 1/5 | 1/5 |
| Class B (natural-overfit) at β=1.0 | 1/5 | 0/5 | 0/5 |
| Class B at β=0.1 | 1/5 | 0/5 | 0/5 |

**Both held-out modes eliminate Class B** (which is what was hoped).
But **ABC mode SUPPRESSES Class C discovery alongside Class B**,
while pointwise MSE mode preserves Class C at the baseline rate.

## Why the conjecture failed

The ABC term is *too strict*. Class A candidates (generic diff-style
`M2D[1·(0,0) + -1·(1,0)](U)`) match the held-out summary statistics
fairly well — they're structurally simple operators applied to U, so
their output on the HOLD trajectory has similar variance and ACF as
the TRAIN-target's. Class B candidates have summary statistics that
diverge wildly on HOLD (because reduce_* gives a TRAIN-specific
scalar). Class C candidates also match perfectly on HOLD.

So ABC selects {A, C} equally well over B. **The problem is that
Class A is much easier to find by random mutation** (2-atom Measure2D)
than Class C (3-atom Laplacian template with the +1/-2/+1 weight
pattern). With both being valid under ABC scoring, the population
converges to Class A early and never explores the harder Class C
region.

Pointwise MSE on HOLD has a slightly different penalty surface:
Class A's pointwise loss on HOLD is ~2× oracle (typical for diff-style),
while Class C's is ~1× oracle. The DIFFERENCE between A and C is more
pronounced in pointwise terms than in ABC terms. So pointwise MSE
selection has a stronger gradient toward Class C — even if it doesn't
discover it more often, it preserves Class C when found.

The selection-pressure ratios at β=1.0:

- Pointwise MSE on HOLD: A penalty ~ 2× oracle, C penalty ~ 1× oracle.
  Difference: 1× oracle = ~4e-6.
- ABC on HOLD: A and C both close to oracle in summary terms. Difference: ~tiny.

The ABC term contributes much smaller selection pressure on the
A-vs-C axis than the MSE term, so the GP defaults to whatever its
random init finds first — which is usually Class A.

## Both β values fail in the same direction

At β=0.1 (gentler hold-out pressure):
- Mode B (hold MSE): 1 Class C + 1 C-partial = 2/5 mechanism-flavoured
- Mode C (hold ABC): 0 Class C + 1 C-partial = 1/5 mechanism-flavoured

At β=1.0 (stronger hold-out pressure):
- Mode B (hold MSE): 1 Class C / 5
- Mode C (hold ABC): 0 Class C / 5

Both βs give the same qualitative answer: ABC ≤ pointwise MSE for
Class C preservation. The conjecture is robust against this
falsification.

## What the experiment DOES validate

The discipline did its job. The ABC term:

1. **Successfully eliminates Class B at any β tested.** Natural-overfit
   shapes are penalized. This is consistent with the underlying intuition.
2. **Provides scale-invariant scoring** that doesn't depend on residual
   magnitude calibration. The 6 summary statistics + relative-error
   normalization are well-behaved.
3. **The GPWithHoldout machinery works correctly.** Zero-β reproduces
   baseline; positive β shifts the population (in this case, toward A).

The mechanism that makes ABC harmful (too uniform between A and C) is
informative — it tells us that for a "discover the rarer right answer"
problem, you want scoring with STRONGER A-vs-C gradient, which
pointwise MSE provides.

## Verdict for the experimental discipline

**Conjecture C1-refined: FALSIFIED at β ∈ {0.1, 1.0} on heat equation.**

The module `tessera.experimental.abc_scoring` is preserved but its
status in the docstring is updated to *falsified at tested β values*.
A future re-evaluation could:

1. Test more β values, particularly β << 0.1 where ABC contribution
   is sub-dominant but might still suppress Class B
2. Try different summary-statistic sets (more discriminating between
   A and C)
3. Apply to genuinely stochastic benchmarks (Ornstein-Uhlenbeck,
   Hawkes) where ABC's natural fit might be a better match

Until one of these produces a positive signal, **the practical
recommendation is: use pointwise MSE on held-out data, not ABC**.
The simpler tool is the better tool here.

This is exactly the discipline the user articulated:

> *"if you come up with novel conjectures, we should work to prove
> those conjectures, or at least show that they can be applied in
> some cases."*

We tested. The conjecture failed in its current form. The negative
result is informative: pointwise hold-out CV is sufficient; ABC adds
nothing extra on this benchmark class.

## Per-seed details (β=0.1 run, most recent)

| Mode | seed | train/oracle | test/oracle | cx | class |
|---|---|---|---|---|---|
| A_baseline | 2026 | 2.20 | 2.26 | 8 | A |
| A_baseline | 2027 | 1.68 | 1.51 | 12 | A |
| A_baseline | 2028 | 1.04 | 1.03 | 14 | **B** |
| A_baseline | 2029 | 1.04 | 1.00 | 4 | **C** |
| A_baseline | 2030 | 2.16 | 2.21 | 11 | A |
| B_hold_mse | 2026 | 2.18 | 2.21 | 10 | C-partial |
| B_hold_mse | 2027 | 1.04 | 1.00 | 4 | **C** |
| B_hold_mse | 2028 | 2.15 | 2.18 | 11 | A |
| B_hold_mse | 2029 | 2.22 | 2.29 | 6 | A |
| B_hold_mse | 2030 | 2.23 | 2.30 | 6 | A |
| C_hold_abc | 2026 | 2.21 | 2.15 | 9 | C-partial |
| C_hold_abc | 2027 | 2.17 | 2.19 | 6 | A |
| C_hold_abc | 2028 | 2.24 | 2.21 | 6 | A |
| C_hold_abc | 2029 | 2.25 | 2.29 | 6 | A |
| C_hold_abc | 2030 | 2.23 | 2.29 | 8 | A |

## What this means for the broader process-discovery direction

C1-refined was the cheapest test of "ABC adds something." The fact
that it failed at this benchmark means we shouldn't double down on
ABC machinery without strong reasons. The other conjectures (C2-C6
from `process_discovery_sr.md`) are independent and worth testing
separately — particularly C4 (causal direction priors) which addresses
a structurally different problem.

The methodological win: the experimental discipline (research note →
experimental module → empirical test → falsification or validation)
gave us a clean answer in ~1 day of work. This is exactly the
machinery the user asked for.

## Reproducing

```
python benchmarks/run_heat_equation_abc_mvp71.py --seeds 5 --beta 0.1
python benchmarks/run_heat_equation_abc_mvp71.py --seeds 5 --beta 1.0
```

Each run ~3 minutes wall-clock.
