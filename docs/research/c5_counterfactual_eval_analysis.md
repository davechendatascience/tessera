# C5 theoretical pre-analysis: counterfactual evaluation harness

**Status:** ? RESEARCH — theoretical analysis BEFORE implementation, per the methodological discipline established in commits ec98f12 / de0f6d0 / a6c6ced.

**Provenance:** user (2026-05-26), continuing the basket discipline after C1/C3/C4/C6. C5 is the last open conjecture besides C2 (which is the most expensive at ~1-2 weeks).

This document attempts to predict the outcome of C5 before any code, and to identify which operationalization (if any) is worth empirical test given what the pattern across four prior experiments has taught us.

---

## 1. The conjecture, restated

C5 (from process_discovery_sr.md §6.5):

> "Counterfactual / interventional evaluation harness. Penalizes candidates whose extrapolation degrades faster than expected for a true mechanism."

Reduced form: **evaluate candidates on systematically perturbed data (different IC, different parameters, different noise, different geometry) and use the divergence between candidate's predictions on the original vs perturbed data to discriminate mechanism (Class C) from natural-overfit (Class B) and tautology (Class A).**

## 2. The conjecture has two natural operationalizations

**Operationalization A: fitness-term version**

Add counterfactual-loss as a fitness penalty during search:

```
fitness = MSE_train + β · Σ MSE_counterfactual_k
```

This biases tournament selection toward candidates that generalize on counterfactuals during search.

**Operationalization B: post-hoc ranking version**

Run normal GP. Score each Pareto-front candidate on counterfactual data. Use this to RANK candidates for deployment, independent of how the search proceeded.

```
score_for_deployment = combine(MSE_train, MSE_counterfactual)
```

This is purely a diagnostic / selection tool. No effect on search trajectory.

These are very different in computational cost and in expected outcome.

## 3. The cross-experiment pattern strongly predicts Operationalization A will fail

Four experiments now in the basket:

| Conjecture | Layer | Status | Class C delta |
|---|---|---|---|
| C1 (ABC scoring) | Scoring/fitness | FALSIFIED | -1/5 |
| C4 (causal priors) | Hard search restriction | PARTIAL | 0/5 |
| C3 (MDL scoring) | Scoring/fitness | FALSIFIED | -1/5 |
| C6 (adaptive search) | Search direction | VALIDATED-AS-PREDICTED-NEGATIVE | 0/5 |

**Operationalization A is at the scoring/fitness layer** — same layer where C1, C3, C6 all failed to materially affect Class C discovery. The pattern strongly predicts:

> Operationalization A produces results similar to baseline. Class C count ≤ 1/5.

This is not a new prediction; it's the same prediction the basket would make for any new scoring-layer intervention. Running Operationalization A would add one more data point to the pattern but won't tell us anything genuinely new.

## 4. Operationalization B is at a different layer — and might work

Operationalization B does NOT modify search. It modifies *selection* — which candidate from the final Pareto front gets reported as "best" or "deployed."

This is the *evaluation* layer, which hasn't been tested in the basket so far.

**The question Operationalization B answers:** given a Pareto front containing Class A, Class B, and possibly Class C candidates, can counterfactual evaluation reliably IDENTIFY the Class C candidate?

If yes: counterfactual evaluation is a useful diagnostic tool even if it doesn't help search. We can run normal GP, get the front, and use counterfactual evaluation to identify which candidate captures mechanism.

If no: even at the selection layer, counterfactual evaluation doesn't add information beyond what's already in the (cx, train_loss) Pareto axes.

## 5. Provable claim about Operationalization B

**Provable (3a-style):** if Class B and Class A produce DIFFERENT counterfactual predictions than Class C, then a sufficiently strong counterfactual signal will discriminate them. This is essentially the definition of counterfactual generalization.

**Concretely for heat equation:**

Take a candidate tree on the Pareto front. Evaluate it on:
- Original TEST trajectory (different IC, same α) — typical TRAIN/TEST split
- Counterfactual A: TEST trajectory with α scaled by 2× (interventional change)
- Counterfactual B: TEST trajectory with X-domain reflected (geometric change)
- Counterfactual C: TEST trajectory with added correlated noise (regularity change)

For each:
- Class A (diff-style, e.g., `M2D[1·(0,0) + -1·(1,0)](U)`): outputs are *structurally generic* — same kind of approximation regardless of the perturbation. Counterfactual MSE ≈ proportional to perturbation magnitude.
- Class B (Laplacian/reduce_max): outputs are *trajectory-specific* — fail catastrophically on different ICs (already shown), but might fail differently on different perturbations.
- Class C (clean `α·Laplacian(U)`): outputs are *mechanism-correct* — counterfactual MSE preserves the mechanism's predictive structure. For α-doubled CF, the predicted dt_U scales correctly with α; for X-reflected CF, the Laplacian transforms symmetrically.

**The discriminating signal:** Class C's counterfactual losses should have *consistent ratios* (e.g., α-doubled gives 4× MSE because dt_U scales linearly with α and Laplacian is the same, so the residual is proportional). Class A's counterfactual losses should be more random. Class B's counterfactual losses should be MUCH HIGHER for most counterfactuals.

So a discriminator could be:
- High counterfactual MSE → likely Class B (penalize)
- Inconsistent counterfactual response → likely Class A (mild penalty)
- Consistent, predictable counterfactual response → likely Class C (accept)

## 6. Predicted outcomes for Operationalization B

**Best case:** counterfactual evaluation reliably distinguishes Class C from Class A and Class B. Useful as a post-hoc selector even if discovery rate is unchanged.

Predicted outcome: among the existing Pareto front from baseline runs, applying counterfactual evaluation correctly identifies the Class C candidate (when one exists in the front).

**Possible failure mode 1:** Class A's counterfactual losses are similar to Class C's because both are structurally generic over the perturbations. Then counterfactual eval can't discriminate them.

**Possible failure mode 2:** the chosen counterfactuals don't expose Class B's failure mode because reduce_max happens to be invariant under those specific perturbations. Then Class B passes counterfactual selection too.

**Possible failure mode 3:** Class C is so rare (1/5 baseline) that the post-hoc ranking question is "did the GP find Class C at all?" — counterfactual eval can confirm a Class C is Class C, but can't create one when none exists.

## 7. What experiment would tell us most

Given the analysis, the most informative experiment is **Operationalization B** as a post-hoc diagnostic:

1. Take Pareto fronts from previous baseline runs (we already have these from C1/C3/C4/C6 experiments)
2. For each candidate, compute counterfactual MSEs on a SET of perturbations
3. Score by counterfactual-aware criterion (e.g., median counterfactual MSE relative to perturbation magnitude)
4. Check: does the candidate with best counterfactual score correspond to the Class C tree (when one exists)?

This is essentially a meta-analysis on existing data + new counterfactual evaluation. Very cheap to run (no new GP runs).

**Operationalization A** is predicted to fail per the pattern; running it would add one more confirmation of the pattern but no new structural insight.

## 8. The cleanest minimal experiment

Implementation plan:

1. Create a function `counterfactual_perturbations(U_base, alpha_base)` that generates a SET of perturbed trajectories (different α, different IC, etc.)
2. Create a function `score_counterfactual(tree, U_perturbations)` that evaluates the tree on each perturbation and returns a vector of MSE ratios
3. Apply this to Pareto-front candidates from BASELINE runs (we have these from prior experiments)
4. Check: does counterfactual ranking identify Class C reliably?

This is purely post-hoc. ~half day to implement and run.

## 9. Predicted outcome (refined)

For the heat equation specifically, applying counterfactual evaluation to a Pareto front:

- **Class C trees** should have counterfactual MSE ≈ baseline TEST MSE for IC changes; should refit α cleanly for α changes
- **Class B trees** should have counterfactual MSE high (catastrophic) for IC changes — already shown
- **Class A trees** should have counterfactual MSE proportional to perturbation magnitude, similar to baseline

The discriminator: Class B has highest counterfactual cost; Class C has lowest. Class A is in the middle.

**Predicted ranking accuracy:** if we score by counterfactual MSE, Class C ranks #1 reliably (when present). Class B ranks last. Class A ranks middle.

**Predicted utility:** counterfactual evaluation IS useful as a post-hoc deployment selector. Not transformative (doesn't create Class C trees) but informative (correctly identifies them when present).

## 10. Decision

**Run Operationalization B** (post-hoc counterfactual ranking) as the cheap experiment.

- Skip Operationalization A (predicted negative per pattern; running it wastes compute)
- The post-hoc version tests something genuinely new (selection layer, not scoring/search layer)
- Cheap to implement (~half day) and run

This is the right next experiment because it sits at a layer that hasn't been falsified yet.

## 11. Falsification anchors

- If post-hoc counterfactual ranking correctly identifies Class C from existing baseline fronts → counterfactual evaluation is a useful diagnostic, validates the C5 conjecture in its selection-layer form
- If it doesn't reliably identify Class C → falsified; even at the selection layer, counterfactual eval doesn't add information beyond cx + train_loss
- If the rankings are noisy / inconsistent → suggests a different perturbation set is needed

## 12. What this experiment WILL NOT tell us

- Whether Operationalization A would help search (predicted not; not testing it)
- Whether counterfactual evaluation would help on different benchmarks (only heat eq)
- Whether the SPECIFIC counterfactuals we choose are the best ones

This is a focused minimal test. If it validates, we know counterfactual eval has selection value. If it falsifies, we have one more data point.

## Empirical outcome (2026-05-26)

Experiment: `benchmarks/results/heat_equation_counterfactual_mvp_c5.md`.

**Conjecture VALIDATED (selection-layer version).**

Results across 5 baseline seeds on heat equation:
- 2/5 seeds had mechanism candidates in the Pareto front
- CF ranking correctly identified them in 2/2 cases (no false negatives)
- 3/5 seeds had no mechanism in front; CF correctly picked best-available Class A (no false positives)
- Seed 2027: CF found a Pareto-strict improvement (picked equivalent-quality C-partial at cx=12 over cx=15)
- CF median score cleanly discriminates: mechanism candidates ≤ 1.5, Class A candidates > 1.7

**The selection layer is the productive layer.** This is the first VALIDATED-POSITIVE conjecture in the basket. The other four (C1, C3, C4, C6) operated at scoring/search-direction layers; all failed to materially affect Class C rates. C5 operates at the selection layer (post-hoc, after search completes) and works as predicted.

The deeper pattern: data-level and selection-layer interventions work; scoring-layer and search-direction interventions don't. Tessera's productive levers are now: vocabulary curation, mutation operator defaults, training data structure, and post-hoc counterfactual selection.

## Changelog

- 2026-05-26: initial pre-analysis. C5 has two operationalizations; the fitness-term version is predicted to fail per the cross-experiment pattern; the post-hoc ranking version is at a different (selection) layer and worth testing. Decision: run post-hoc only.
- 2026-05-26: empirical outcome appended. Conjecture validated; first positive basket result. Selection layer is the productive layer.
