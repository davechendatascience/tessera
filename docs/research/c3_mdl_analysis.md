# C3 theoretical pre-analysis: MDL with explicit log-likelihood vs ad-hoc parsimony

**Status:** ? RESEARCH — theoretical analysis BEFORE empirical test, per the methodological discipline (commit 79a7f21 follow-up).

**Provenance:** user (2026-05-26), after C1 falsification and C4 partial validation:

> *"Shouldn't we try to prove them if there's a conjecture? So if it theoretically makes sense but fails at runtime, we know that it's either structural flaws or data problem that can be mitigated."*

This document is the theoretical pre-analysis for Conjecture C3 (process_discovery_sr.md §6.3). The goal: articulate precisely what's provable, what's empirical, predict the most likely outcome of the empirical test, and DECIDE whether the experiment is worth running.

---

## 1. The conjecture, restated precisely

C3 (from process_discovery_sr.md §6.3):

> "MDL with explicit log-likelihood (structure-function-aware) identifies the 'right amount of model' more accurately than ad-hoc parsimony."

The two scoring functions being compared:

| Scoring | Form |
|---|---|
| Ad-hoc parsimony (current tessera) | `fitness = MSE(tree, data) + α · cx(tree)` with α a hyperparameter |
| Proposed MDL | `fitness = −log p(data | tree) + α_MDL · DL(tree)` with α_MDL principled (typically 1 bit) |

The claim splits into two:

| Part | Claim | Provable? |
|---|---|---|
| **3a** | MDL has stronger theoretical foundation than ad-hoc parsimony | YES (see §2) |
| **3b** | MDL identifies the "right model" more accurately on tessera benchmarks | EMPIRICAL — depends on how "more accurately" is operationalized |

We have to disentangle 3a (theoretical claim) from 3b (empirical claim) to know what an experiment can tell us.

## 2. Proof of 3a — MDL has stronger theoretical foundation

### 2.1 The MDL principle

The MDL principle (Rissanen 1978; foundational refinement of Solomonoff 1964): the best model for data is the one that minimizes the total *description length* of (model + data given model).

Formally, for data x and model m drawn from some class M:

```
DL(m, x) = DL(m) + DL(x | m)
        = K(m) + (−log p(x | m))    [up to additive constants]
```

The first term is the model's description length (bits to specify m); the second is the conditional code length for x given m.

Under the universal Bayesian interpretation: prior `P(m) ∝ 2^(−K(m))`, posterior `P(m|x) ∝ P(x|m) · P(m)`. The MAP estimate minimizes `−log P(m|x) = −log P(x|m) − log P(m) = −log p(x|m) + K(m) · log 2`.

So MDL = MAP-estimate under a *universal* prior that prefers simpler models. This is principled in a sense ad-hoc parsimony is not.

### 2.2 Ad-hoc parsimony is NOT generally optimal

The ad-hoc form `MSE + α · cx`:

- `α` is a free hyperparameter; different α produce different Pareto fronts
- There's no information-theoretic interpretation of `α · cx`
- The MSE term IS proportional to a log-likelihood (under Gaussian noise) but with a different scaling: `−log p ∝ MSE / (2σ²)`, not MSE directly

So ad-hoc parsimony differs from MDL in two ways:
1. The data-loss term is MSE, not normalized log-likelihood (off by a factor of `1/(2σ²)`)
2. The complexity term uses node count `cx`, not bit-accurate description length `DL`

**Result of 3a:** MDL is the *Bayesian-correct* scoring under a universal-prior assumption. Ad-hoc parsimony is a HEURISTIC. **3a is proven.**

## 3. Empirical claim 3b — what would it take to validate?

3b says MDL identifies the right model "more accurately" than ad-hoc. To test this empirically, we need to define "more accurately."

Three possible operationalizations:

| Operationalization | Test |
|---|---|
| (i) Pareto front contains the canonical model at lower cx | Compare best Class-C-rate-at-low-cx across modes |
| (ii) Selection picks the canonical model more often | Compare class distribution at end of run |
| (iii) Out-of-distribution generalization improves | Test loss on held-out trajectory |

(ii) is the cleanest for our setup — same metric as C1 and C4.

## 4. The math of MDL under Gaussian likelihood

For tessera's GP, the trees are *deterministic* — they output a single value, not a probability distribution. To compute `−log p(data | tree)`, we add a noise model.

Standard choice: Gaussian noise with variance σ²:

```
p(y | tree, x) = N(y; tree(x), σ²)
log p(y | tree, x) = −(y − tree(x))² / (2σ²) − (1/2) log(2πσ²)
−log p(data | tree) = Σᵢ (yᵢ − tree(xᵢ))² / (2σ²) + (N/2) log(2πσ²)
                    = N · MSE / (2σ²) + N · log(σ) + const
```

So under Gaussian likelihood, `−log p` equals MSE up to a multiplicative factor `1/(2σ²)` plus an additive `N log σ` (which is constant in the tree, so doesn't affect selection).

**MDL fitness becomes:**

```
fitness_MDL = MSE / (2σ²) + α_MDL · DL(tree) / N    [per-sample form]
            ∝ MSE + (2σ² · α_MDL / N) · DL(tree)
```

Compare to ad-hoc:

```
fitness_adhoc = MSE + α · cx(tree)
```

**The two scorings have the same structural form: MSE + (something) × (complexity).** They differ in:

1. The COEFFICIENT on the complexity term:
   - Ad-hoc: arbitrary α (e.g., 0.005)
   - MDL: principled `2σ² · α_MDL / N` — depends on σ (noise) and N (samples)

2. The COMPLEXITY measure:
   - Ad-hoc: node count `cx`
   - MDL: bit-accurate `DL(tree)`

## 5. What MDL needs to estimate

Two estimation problems:

### 5.1 σ (noise variance)

For tessera benchmarks, σ is typically known approximately:
- Heat equation: σ ≈ 0.002 (the noise_std in simulate_heat_1d)
- Feynman: usually σ = 0 (deterministic targets) — MDL would be degenerate here

**Estimation choices:**
- (a) Fixed σ from prior knowledge (clean for synthetic data; brittle for real data)
- (b) Estimate σ from residuals of the current best tree (σ̂² = MSE_best)
- (c) Estimate σ as a hyperparameter, optimized jointly

Choice (b) is self-consistent: as the best tree improves, σ̂ shrinks, making the MSE term more dominant. This dynamic favors better-fit trees but can be unstable.

### 5.2 DL(tree) (description length in bits)

Bit-accurate description length requires choosing an encoding. A reasonable encoding for tessera trees:

```
For each node:
  - Type tag (Var/Const/BinOp/UnOp/FunctionalOp/FunctionalOp2D):  log2(6) ≈ 2.6 bits
  - Type-specific payload:
    - Var:        log2(|features|) bits to specify which feature
    - Const:      ~32 bits for a float64 mantissa, OR log2(K) for K discrete values
    - BinOp:      log2(|BIN_OPS|) bits to specify the op
    - UnOp:       log2(|UN_OPS|) bits
    - FunctionalOp: log2(num_functionals) + log2(num_measures) bits
    - FunctionalOp2D: similar
  - Recurse on children
```

Total: depends on tree shape and vocabulary size. For a typical tessera tree with cx=10, DL might be ~50-100 bits.

So if we use cx as a proxy for DL, the MDL/ad-hoc difference is at the COEFFICIENT level, not the COMPLEXITY-MEASURE level.

## 6. Predicting the empirical outcome

Now the actual question: will MDL produce a DIFFERENT Pareto front than ad-hoc parsimony?

**For heat equation (N = 5940 samples, σ ≈ 0.002):**

MDL coefficient: `2σ² · α_MDL / N` = `2 · 4e-6 · 1 / 5940` ≈ 1.35e-9 per bit.

Ad-hoc coefficient: `α = 0.005` per node ≈ 0.0025 per bit (if 2 bits/node).

**The MDL coefficient is ~6 orders of magnitude SMALLER than ad-hoc.** Under MDL, complexity is essentially free; the GP would be incentivized to fit data tightly with arbitrarily complex models.

This means MDL would produce LARGE, HIGH-CX trees on this benchmark, not the canonical cx=4 form.

This is the OPPOSITE of what the conjecture hopes.

**Why this happens:**

The MDL principle assumes the model and data live in COMPARABLE bit scales. For heat equation:
- σ = 0.002 is small noise
- N = 5940 samples
- Total data bits at this precision: `N · log2(range / σ)` ≈ N · 10 ≈ 60K bits

With 60K bits of data describing 1 unknown (α), the data dominates by orders of magnitude. Any tree that fits the data well (saves data bits) is worth MANY tree bits. MDL prefers high-cx solutions.

The conjecture's "right amount of model" intuition holds when DATA is the bottleneck. With N=5940 and σ=0.002, MODEL bits are virtually free relative to data bits. So MDL pushes toward overparameterization.

**Predicted empirical result:**

MDL with standard Gaussian-likelihood calibration on heat equation will produce trees with LARGER cx than ad-hoc, possibly with WORSE generalization (since high-cx trees overfit). The conjecture as stated would be FALSIFIED.

This is exactly opposite to the conjecture's intuition. The intuition was probably "MDL is theoretically better → better empirical results." But the calibration math shows MDL pushes toward overparameterization at typical tessera sample sizes.

## 7. Where the conjecture WOULD hold

For MDL to behave like a parsimony-enforcing prior, we'd need:

- LARGER σ (more noise → MSE term scales down → complexity becomes meaningful)
- SMALLER N (fewer samples → data bits compete with model bits)
- LARGER DL per node (e.g., considering Const values as full 32-bit floats rather than discrete)

Or: use a NON-Gaussian likelihood (e.g., heavy-tailed) where the noise model implicitly captures structural mismatch differently.

OR: use the structure function more carefully — only count BITS for things the model DETERMINES, not for noise inherent to the process.

The structure-function-aware version would say: model bits should count toward the *informative* part of the model only. Constants that match noise levels shouldn't count. This needs more careful design.

## 8. Decision: should we run the empirical test?

Based on the theoretical analysis:

| Outcome | Probability | Interpretation |
|---|---|---|
| MDL beats ad-hoc on Class C rate | LOW | Would be surprising; calibration math predicts opposite |
| MDL matches ad-hoc | MODERATE | Likely if effective α happens to be similar |
| MDL underperforms ad-hoc (more overfit, higher cx) | HIGH | What the calibration math predicts |

**The most likely outcome is MDL FAILS, but for an INFORMATIVE reason:** the calibration shows MDL with standard Gaussian likelihood under-penalizes complexity at typical tessera sample sizes. This is a useful finding because it identifies a specific PROBLEM with the naive MDL implementation and points at where the structure-function refinement would matter.

So the experiment IS worth running, but with revised expectations:
- We don't expect MDL to "beat" ad-hoc
- We DO expect to characterize HOW MDL fails (overfitting vs underfitting)
- The result informs the structure-function refinement direction

**This is a different kind of test than C1 (which we ran with hope of validation) and C4 (which we ran with hope of validation).** This is a SANITY CHECK: does naive MDL behave as the calibration math predicts? If yes, the math is validated and we know what to fix. If no, our calibration math is wrong and we learn something else.

## 9. Refined experimental protocol

Given the predicted outcome, the experiment should:

1. Implement MDL with two σ estimates:
   - (a) Known σ (use the simulator's noise_std for heat eq)
   - (b) Estimated σ from current best residual (self-consistent)

2. Compare against ad-hoc baseline (which we already have data for).

3. Measure NOT just Class A/B/C distribution, but:
   - Median cx of Pareto front (predict: MDL → higher)
   - Median train MSE (predict: MDL → lower)
   - Median test MSE (predict: MDL → similar or higher due to overfit)

4. Run a SECOND mode with structure-function-aware refinement:
   - Penalize K(tree) more aggressively (multiply by N or √N)
   - This tests whether RECALIBRATED MDL behaves better

5. **Falsification criterion (proven part):** if MDL with standard calibration does NOT push toward higher cx, our calibration math is wrong. Investigate.

6. **Validation criterion (empirical part):** if RECALIBRATED MDL produces a Pareto front comparable to or better than ad-hoc, we've found a principled alternative.

## 10. What this analysis adds vs not having it

WITHOUT this pre-analysis, we would:
- Implement MDL, run it, see high-cx overfit trees
- Conclude "MDL fails — falsified"
- Lose the insight about WHY (calibration mismatch at our N, σ)

WITH this pre-analysis:
- We predict the failure mode in advance
- The experiment becomes a sanity check on our calibration math
- Failure is informative — points at recalibration
- Success would be surprising and we'd know exactly why

This is the methodological win the user articulated: **predicting failure modes turns empirical falsification into structured learning, not noise**.

## 11. Decision

**Proceed with implementation, but with the refined protocol** (§9). Two modes (known σ vs estimated σ), measure cx growth, also test recalibrated MDL.

This is a less ambitious empirical claim than the original C3 ("MDL identifies right model better") but a more informative one ("MDL with naive calibration behaves as theory predicts; recalibration produces results comparable to ad-hoc").

The empirical artifact becomes:
- Tested MDL behavior matching theoretical prediction = math validated
- Tested recalibration as a path to principled scoring

If the experiment validates the prediction (MDL overfits), we have:
- Confirmation that ad-hoc parsimony α=0.005 is empirically reasonable (it equals MDL with effective σ much larger than 0.002)
- A clear next move (use structure-function-aware penalty)

If the experiment violates the prediction (MDL doesn't overfit), we have:
- A surprise that requires diagnosing
- Probably an implementation bug somewhere

Either is informative. **Run the experiment.**

## 12. What this note explicitly does NOT claim

- Not that the implementation will be correct on first attempt (testing required)
- Not that MDL is "wrong" — it's correct under its assumptions, and the assumptions don't match our N/σ regime
- Not that ad-hoc parsimony is "right" — it's a calibrated heuristic that happens to land near the recalibrated-MDL sweet spot for typical tessera problems
- Not that the structure function refinement is implementable in this experiment (it's theoretical; we'd need a separate research arc)

## Changelog

- 2026-05-26: initial document. Theoretical pre-analysis of C3 done BEFORE implementation per the methodological discipline (commit 79a7f21 follow-up). Provable: MDL has stronger theoretical foundation than ad-hoc parsimony. Empirical: MDL with naive Gaussian likelihood and typical tessera N/σ likely OVERFITS (calibration math). Refined experimental protocol: test naive MDL vs recalibrated MDL vs ad-hoc baseline; measure cx growth and overfit, not just Class C rate.
