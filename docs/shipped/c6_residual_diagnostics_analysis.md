# C6 theoretical pre-analysis: iterative strategy refinement via residual diagnostics

**Status update (2026-05-26 → `docs/shipped/`):** Moved from `docs/research/`. **VALIDATED-AS-PREDICTED-NEGATIVE** via `src/tessera/experimental/adaptive_search.py` — pre-analysis predicted adaptive mutation weights would not help on heat eq; experiment confirmed (`benchmarks/results/heat_equation_adaptive_mvp_c6.md`). Module preserved; conjecture's "iterative residual diagnostics" framing partially reused in the Stage 5 identification pipeline.

**Status:** ? RESEARCH — theoretical analysis BEFORE implementation, per the methodological discipline established in commits ec98f12 / de0f6d0.

**Provenance:** user (2026-05-26), after observing the pattern across three experiments (C1, C3 falsified at fitness-ranking layer; C4 partial at hard-constraint layer). Asked to apply discipline to C6.

This document attempts to predict the outcome of C6 BEFORE writing any code, and to identify the right operationalization (if any) for the empirical test.

---

## 1. The conjecture, restated

C6 (from process_discovery_sr.md §6.6):

> "After each generation (or every K generations), examine the best candidate's residuals for normality, heteroskedasticity, autocorrelation, outliers. Based on diagnostics, adjust the search prior. E.g., if residuals are heteroskedastic, increase mutation weight for distributional-output trees. If autocorrelated, favor trees with serial-correlation-handling primitives."

Reduced form: **adaptive mutation weights driven by residual diagnostics, vs static mutation weights**.

## 2. What's provable vs empirical

### 2.1 Provable claim (3a-style)

Under PERFECT diagnostic information — i.e., we know precisely what primitive is missing from the current best model — adaptive search converges in strictly fewer generations than static. This follows from basic optimization-theory arguments: directed gradients beat random walks.

**Status: weakly provable** (the argument is intuitive; formal proof requires assumptions about the GP search dynamics).

### 2.2 Empirical claim (3b-style)

In practice, with NOISY diagnostics and FINITE budget, adaptive mutation produces better Pareto fronts than static.

This depends entirely on:
- **Quality of diagnostic signal** — do the residuals actually tell us what's missing?
- **Cost of diagnostics** — does the per-generation overhead eat into search budget?
- **Mapping from diagnostic to corrective action** — does "residuals are autocorrelated" actually translate to a specific mutation weight adjustment?
- **Adaptation noise** — random weight changes based on noisy signal can be worse than fixed weights

**Status: empirical, depends on benchmark and operationalization.**

## 3. The mapping problem

The hardest part of C6 isn't the diagnostic computation; it's the **diagnostic → corrective action mapping**.

For heat equation specifically:

| Diagnostic finding | What it means | What corrective action would help |
|---|---|---|
| Residuals have spatial autocorrelation | Model misses spatial structure | Add 3-atom spatial operators (Laplacian) — but this is the specific answer, not a general rule |
| Residuals have heteroskedastic variance | Model doesn't capture state-dependence | Add distributional-output trees — but those don't exist in tessera today |
| Residuals are heavy-tailed | Model assumes Gaussian noise when noise is heavy-tailed | Change loss function to robust loss — but our scoring is MSE-based |
| Residuals have temporal autocorrelation | Model misses temporal structure | Add temporal operators — but C4 just FORBADE temporal operators for our target type |

**The mapping is benchmark-specific and often points at infrastructure that doesn't exist.** A generic C6 implementation cannot make these mappings; it would need either:

- Hand-engineered mappings (which is just specific biases, not "adaptive")
- Meta-learning of mappings from many SR runs (sample-complexity prohibitive)

## 4. The deeper issue — diagnostic-mapping circularity

The mappings that would help most are **specific to knowing the answer**. If we KNOW the heat equation needs Laplacian, we can write a diagnostic ("spatial autocorrelation in residuals → favor Laplacian"). But knowing this is essentially knowing the answer; the "adaptive" mechanism is just a delivery vehicle for engineered priors.

A truly generic C6 would identify diagnostic→corrective mappings WITHOUT knowing the answer. This is the meta-learning problem; it's hard and probably can't be solved in a single SR run.

## 5. Predicted outcomes for plausible operationalizations

### 5.1 Generic adaptive (most basic version)

"Adjust mutation weights to favor primitives that appear in the current best candidate" — a "double down on what works" heuristic.

**Predicted effect:** GP converges faster but to similar trees as baseline. Risks locking in early on Class A (because Class A is what the GP usually finds first). Might *hurt* Class C discovery by reducing exploration.

**Predicted Class C count:** ≤ baseline (1/5).

### 5.2 Bandit-style adaptive (more interesting)

At gen K, fork into two sub-populations with different mutation weight regimes. Track which produces better candidates over the next K generations; bias future runs toward the winning regime.

**Predicted effect:** if one regime is genuinely better, bandit identifies it. But the GP search dynamics are noisy; two regimes might not be distinguishable in finite time.

**Predicted Class C count:** ~baseline. Class C is rare enough (1/5) that even good adaptation might not discover it more often.

### 5.3 Residual-spectral adaptive (most specific)

For 2D dynamical data, compute FFT of residuals. If certain spatial-frequency content is present, bias toward M2D atoms with that frequency response (Laplacian has ω² response, so Laplacian-like operators would address ω² residuals).

**Predicted effect:** this IS engineering specific to PDE-discovery; it would work in proportion to how cleanly the diagnostic identifies the missing operator. For heat equation, this might actually find Class C more reliably.

**Predicted Class C count:** could be 2-3/5 if the implementation is clean. But this is essentially hand-engineered for PDE-discovery and not generic SR.

## 6. The methodological tension

C6 sits at an awkward methodological point:

- **Generic C6** (no specific knowledge): unlikely to outperform static baseline because the mapping problem is unsolved
- **Specific C6** (hand-engineered for a benchmark): would work but isn't really "adaptive" — it's just a delivery vehicle for the same engineered priors we could put in static weights

In other words, **the value of C6 depends on whether you can specify diagnostic→corrective mappings that DON'T essentially encode the answer**. If you can, those mappings could be put directly in static weights. If you can't, the adaptation can't help.

## 7. Where C6 might genuinely help

Despite the skepticism above, there are scenarios where adaptive matters:

- **Multi-stage problems**: where the "right answer" requires DIFFERENT primitives at different stages of search. E.g., start with rough features; refine with specialized operators.
- **Benchmark-agnostic adaptation**: where the diagnostics directly trigger broad parameter changes (e.g., decrease parsimony when fit plateaus; this is what `climb_then_anneal_parsimony` already does in tessera).
- **Annealing-style schedules**: where the optimal parameter changes over time and adaptation captures that schedule.

Tessera already has the parsimony-schedule mechanism (`climb_then_anneal_parsimony`); it's adaptive in a coarse way. The question for C6 is whether finer-grained, residual-driven adaptation adds anything.

## 8. Predicted experiment outcome

Given the analysis, the most likely outcome for any C6 operationalization on heat equation:

- **Generic adaptive**: 0-1/5 Class C (similar to baseline; possibly slightly worse due to noise)
- **Bandit-style**: 0-1/5 Class C (similar; insufficient signal in finite time)
- **PDE-specific (residual-spectral)**: could be 2-3/5 but requires hand-engineering

The most informative experiment would be either:
1. **The simplest generic C6** — confirms whether generic adaptation helps at all
2. **A PDE-specific residual-spectral version** — tests the upper bound of what C6 could achieve with hand-engineering

Of these, (1) is cheaper and more aligned with the conjecture as stated. (2) would essentially be a separate research direction.

## 9. Decision: run a minimal generic C6

Per the discipline, the experiment should test the generic conjecture, not hand-engineered priors.

**Minimal generic C6 implementation:**

- Every K generations, examine the population's MUTATION OPERATOR USAGE (which operators were used to create the current Pareto-front candidates)
- Compute the "winning operator distribution" from the front
- Bias future mutation weights toward winning operators
- Continue for the rest of the run

This is the simplest "double down on what works" version. It doesn't require physics knowledge or specific diagnostic mappings. Tests whether the GP can be HELPED by tracking its own success patterns.

**Predicted outcome (per the analysis):**

- If the GP's early-generation successes match its eventual best (likely for easy benchmarks): adaptive ≈ baseline (no harm)
- If early-gen successes lock in too quickly: adaptive < baseline (premature convergence)
- If adaptive correctly amplifies promising directions: adaptive > baseline (rare; would be the surprise)

I predict: **adaptive ≈ baseline** (within sampling noise at N=5 seeds). The Class C rate would be 0-1/5, similar to or slightly worse than the baseline 1/5.

## 10. Implementation specification

Module: `tessera.experimental.adaptive_search`

GP subclass that, every K=10 generations:
1. Identifies which operators (UN_OPS, BIN_OPS) appear most frequently in the current Pareto front's candidates
2. Adjusts mutation operator weights to favor those operators
3. Returns to standard breeding with adjusted weights

The adjustment factor: multiply each operator's effective weight by `(frequency_in_front + ε) / (uniform_frequency + ε)` so used operators get boosted and unused ones get suppressed.

This is "adaptive mutation prior" in the simplest sense.

## 11. What this experiment will tell us

| Outcome | What we learn |
|---|---|
| Adaptive ≈ baseline | Generic C6 doesn't help (likely outcome per analysis) |
| Adaptive > baseline | Surprising; would need to understand why |
| Adaptive << baseline | Adaptive introduces harmful noise; tells us the GP search is brittle to weight perturbations |

The most likely outcome is the first (no signal). That's not a wasted experiment — it confirms the analysis that GENERIC residual-driven adaptation doesn't materially help SR on this benchmark, and points future C6-like work toward either:
- Domain-specific diagnostic→corrective mappings (which is "engineering priors")
- Meta-learning approaches (which are out of scope for a single SR run)

## 12. Falsification anchors

- If generic adaptive consistently produces Class C at higher rate than baseline → my analysis is wrong; adaptive helps generically
- If generic adaptive consistently DESTROYS Class C (0/5) → adaptive introduces harmful noise
- If sampling variance dominates → need more seeds for a clear answer

## 13. Decision

**Proceed with the minimal generic C6 implementation** (§10). The predicted outcome is "no effect" but the experiment is cheap (~1 day) and confirms the analytical prediction. If somehow adaptive helps, that's a surprise worth investigating; if it doesn't, we've added one more data point to the pattern that "scoring-layer interventions don't materially change Class C discovery on this benchmark."

This is a sanity-check experiment, not a discovery experiment. The pre-analysis is the main intellectual contribution; the run validates or violates it.

## Empirical outcome (2026-05-26)

Experiment run on heat equation. Report: `benchmarks/results/heat_equation_adaptive_mvp_c6.md`.

**Prediction validated exactly.**

| | Baseline | Adaptive |
|---|---|---|
| Class C | 1/5 | 1/5 |
| Class B | 1/5 | 1/5 |
| Class A | 3/5 | 3/5 |
| Bit-identical to baseline | — | 3/5 seeds |

Adaptation triggered 10 events per run (every 10 gens for 100 gens) but produced essentially zero net effect on Pareto-front outcomes. Three of five seeds produced bit-for-bit identical results in baseline and adaptive modes.

The diagnostic→corrective mapping problem played out exactly as predicted: even when adaptation triggered on meaningful front content (seeds 2028, 2030), the resulting weight adjustments did not redirect search toward Class C.

This is the cleanest "validated-as-predicted" outcome of the four experiments: the pre-analysis correctly forecast that generic adaptation can't solve the mapping problem.

## Changelog

- 2026-05-26: initial theoretical pre-analysis for C6. Predicts generic residual-driven adaptive mutation weights will produce results similar to baseline.
- 2026-05-26: empirical outcome appended. Prediction validated exactly: adaptive matches baseline on every key metric.
