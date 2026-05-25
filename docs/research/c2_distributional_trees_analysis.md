# C2 theoretical pre-analysis: distributional-output trees

**Status:** ? RESEARCH — pre-analysis only. **No empirical experiment to follow.** The analysis concludes that C2 cannot be meaningfully tested on tessera's current benchmark set; a new benchmark class would be required.

**Provenance:** user (2026-05-26), completing the basket discipline. After cross-benchmark validation of C5 produced a conditional-helpfulness verdict (C5 helps when benchmark has natural-overfit failure modes; neutral otherwise), the C2 pre-analysis is needed to determine whether C2 is on similar footing.

---

## 1. The conjecture, restated

C2 (from process_discovery_sr.md §6.1):

> Distributional-output trees (μ, σ) capture stochastic dynamics
> better than single-output trees.

Concretely: instead of trees outputting a single value `ŷ = tree(x)`,
trees output a parametric distribution `(μ, σ) = tree(x)` where the
prediction is `y ~ N(μ, σ²)`. The natural loss is Gaussian negative
log-likelihood:

```
−log p(y | μ, σ, x) = (y − μ)² / (2σ²) + log(σ) + const
```

The conjecture: this distributional formulation captures *state-dependent
noise* (heteroskedasticity) that single-output MSE-based regression
misses.

## 2. What's provable vs empirical

### 2.1 Provable (3a-style)

Under the assumption that the true generative process is Gaussian
with state-dependent variance `y_i ~ N(f(x_i), σ²(x_i))`, the
maximum-likelihood estimator for (f, σ²) factorizes:

```
(f̂, σ̂²) = argmax Σ_i log p(y_i | μ_i, σ²_i)
        = argmin Σ_i [(y_i − μ_i)²/(2σ²_i) + log(σ_i)]
```

This is strictly better than minimizing `Σ_i (y_i − μ_i)²` (which is
fixed-σ MLE) when σ varies across i. Provable by the standard MLE
optimality argument.

**Result of 3a: PROVEN** — distributional output is the right
formulation for state-dependent noise.

### 2.2 Empirical (3b-style)

In practice, does C2 help on benchmarks we care about? This depends
entirely on whether the benchmarks HAVE state-dependent noise.

## 3. The math of C2 implementation

For tessera, implementing C2 requires:

**Tree representation change.** Either:
- Two parallel trees (μ-tree and σ-tree), with separate or shared
  structure, OR
- A single tree whose output is interpreted as a 2-element vector
  (cx-counting changes accordingly)

**Loss function.** Gaussian NLL:
```
loss = (1/N) Σ_i [(y_i − μ_i(tree))² / (2σ²_i(tree)) + log(σ_i(tree))]
```

with σ output constrained positive (e.g., via `softplus(tree_σ)` or
`abs(tree_σ) + ε`).

**Mutation operators.** Must preserve the 2-output structure. Either:
- Mutate both heads independently (more search space)
- Mutate one at a time (smaller per-step change)

**Scoring path.** GP must evaluate trees on (μ, σ) pairs, compute
NLL, maintain Pareto front in (cx, NLL) space.

This is moderate but not trivial engineering. Estimated cost: ~1-2
weeks of focused work to ship a clean distributional-output GP path.

## 4. Why current benchmarks can't test C2

For C2 to differentiate from a point predictor + constant-σ MSE,
the benchmark must have non-trivial state-dependent noise structure.

| Benchmark | Noise model | State-dependent? | Would C2 differ from baseline? |
|---|---|---|---|
| Heat equation | y = α·∇²U + ε, ε ~ N(0, 0.002²) | **No — constant σ** | No — fixed σ predictor equivalent |
| Feynman (all targets) | y = f(x), deterministic | **No — σ = 0** | No — NLL ill-defined at σ → 0 |
| IK 3-DoF | y = forward_kinematics(θ), deterministic | **No** | No |
| MNIST features | y = pixel × kernel, image features | No (intrinsically deterministic per image) | No |
| Heat equation discovery (the C-experiments) | σ = 0.002 constant | **No** | No |

None of tessera's existing benchmarks have heteroskedastic targets.
Running C2 on them would give:
- **Best case:** C2 ≈ baseline (no signal; constant σ recovered as ~0.002)
- **Worst case:** C2 < baseline (extra σ-tree DoF overfits)

Per the conditional-helpfulness pattern from C5: an intervention
that addresses a specific failure mode is *neutral* when the failure
mode doesn't appear in the benchmark. C2 addresses heteroskedasticity;
none of our benchmarks have it.

## 5. What benchmark class WOULD test C2

To meaningfully test C2, we'd need benchmarks with one of:

### 5.1 Spatially-heteroskedastic heat equation

Variant of the heat equation simulator where noise variance is
position-dependent:
```
U[t, x] = (1 − α·∇²)U[t−1, x] + σ(x) · ξ[t, x]
```
with `σ(x)` varying smoothly (e.g., σ(x) = 0.001 + 0.005·|x|/X).

A distributional-output tree could learn `σ(x)` as a function of
state. A point-output tree would average over σ and either over- or
under-estimate noise.

**Cost to build:** ~1 day to modify the simulator + classify
candidates. Then ~1 day to run C2 vs baseline.

### 5.2 Ornstein-Uhlenbeck process

Stationary OU:
```
dx_t = −θ(x_t − μ) dt + σ_OU dW_t
```

The natural target for SR is `dx_t` as a function of `x_t`. With
correct mechanism, predictions should be `μ_pred = −θ(x − μ)` and
`σ_pred = σ_OU` (constant in this simplest form, but variants exist
where σ depends on x).

**Cost to build:** ~half day for simulator + classifier; ~1 day for
C2 experiment.

### 5.3 GARCH-style heteroskedastic time series

For financial-flavored data:
```
y_t = μ + ε_t,    ε_t ~ N(0, σ_t²),    σ_t² = ω + α·ε_{t−1}² + β·σ_{t−1}²
```

The natural target: predict (μ_t, σ_t²) as functions of past y.
Distributional output is essential because σ varies with state.

**Cost to build:** ~1 day for simulator + classifier; ~few days for
C2 experiment + analysis.

### 5.4 Hawkes self-exciting process

For event-time data with state-dependent intensity. More exotic;
~1 week to build.

The simplest of these is **5.1 (spatially-heteroskedastic heat eq)**
because it builds on existing tessera infrastructure (heat equation
benchmark, Measure2D primitives). The most natural would be 5.3
(GARCH) because financial benchmarks are tessera's adjacent domain
via market-analysis.

## 6. Predicted outcomes if C2 were tested

**On a heteroskedastic benchmark (5.1, 5.2, 5.3):**

- C2 with distributional output captures the noise structure
- Single-output baseline gets MSE that's higher than the true Gaussian
  log-likelihood would imply (because MSE pretends σ is constant)
- C2 should produce BETTER GENERALIZATION on held-out perturbations
  of the noise structure
- The "right answer" for the noise model is itself a discovery target

**Predicted outcome:** C2 helps materially on benchmarks designed
for it, the same conditional-helpfulness pattern as C5.

**On homoskedastic / noiseless benchmarks (existing tessera set):**

- C2 ≈ baseline at best
- C2 < baseline at worst (extra DoF)

This is the predicted outcome that motivates the decision NOT to run
C2 on existing benchmarks.

## 7. The cumulative basket pattern, after C2 analysis

| Conjecture | Status | Intervention layer | Predicted/observed effect |
|---|---|---|---|
| C1 (ABC scoring) | FALSIFIED | Scoring/fitness | No effect |
| C4 (causal priors) | PARTIAL | Hard search restriction | Eliminates failure modes |
| C3 (MDL scoring) | FALSIFIED | Scoring/fitness | No effect |
| C6 (adaptive search) | VAL-AS-NEG | Search direction | No effect |
| C5 (CF selection) | CONDITIONAL POS | Post-hoc selection | Helps when failure mode present |
| **C2 (distributional output)** | **PRE-ANALYSIS ONLY** | **Tree representation** | **Predicted: conditional pos; needs new benchmark** |

C2 sits at a NEW layer (tree representation / output type) not yet
tested. The conditional-helpfulness pattern predicts it would help
on benchmarks with state-dependent noise and be neutral on
homoskedastic / noiseless ones.

## 8. Decision

**Do not run C2 on existing benchmarks.** The pre-analysis predicts
the result: ≈ baseline at best, < baseline at worst, no useful signal.
Spending 1-2 weeks of implementation to confirm a predicted-null
result is poor allocation.

**Instead, this pre-analysis serves as the C2 deliverable** by
documenting:

1. The conjecture is provable in the right benchmark class (3a holds)
2. The empirical test (3b) requires a NEW benchmark class with state-
   dependent noise
3. Building the benchmark is itself a research project (~few days to
   ~weeks depending on scope)
4. The conditional-helpfulness pattern from C5 predicts the outcome:
   C2 helps on heteroskedastic benchmarks, neutral on homoskedastic

## 9. What this completes

The basket discipline:

- **5 conjectures empirically tested**: C1, C3, C4, C5, C6
- **1 conjecture pre-analyzed only**: C2 (this document)
- **Total**: 6 conjectures examined; ~5 days of focused experiment work
- **Outcomes**: 1 conditional positive (C5), 1 partial (C4), 3
  falsifications (C1, C3, C6), 1 pre-analysis-only (C2)

The structural finding from the basket:

> Tessera's tools have benchmark-class-dependent payoff. No universal
> scoring/selection improvement. Productive layers are: data-level
> (mutation defaults, training structure) and selection-layer
> (post-hoc evaluation). Unproductive layers are: scoring/fitness
> modification and search-direction tweaks.

C2 doesn't change this finding; it would extend it to a new layer
(tree representation) and predict it sits in the same
conditional-helpfulness regime as C5.

## 10. Why pre-analysis-only is honest, not lazy

In the discipline established by commits ec98f12 / de0f6d0 / a6c6ced:

> If a conjecture is theoretically sound but predicted to be
> empirically null on available benchmarks, running the experiment
> adds no information.

C2 is in this category. The 1-2 weeks of implementation would produce
the result "C2 ≈ baseline on benchmarks lacking heteroskedasticity"
— which the pre-analysis already predicts with high confidence.
Spending the time is justified ONLY if:

1. A new benchmark class is built first (which is its own project)
2. The pre-analysis prediction is somehow wrong (would need a different
   theoretical argument to override)

Neither holds. The pre-analysis is the right place to stop.

## 11. What would justify building C2 later

C2 becomes worth building when one of:

- Tessera adds a stochastic-dynamics benchmark (e.g., for SDE
  discovery in physics or for financial heteroskedastic modeling)
- A user case demands distributional outputs (e.g., uncertainty
  quantification on tessera predictions)
- The market-analysis project produces a benchmark that needs
  distributional treatment (GARCH-like volatility modeling)

Until then, C2 is a *known-not-yet-implementable* tool. The
analysis documents what it would do, when it would help, what
infrastructure is needed, and why running it now would be wasteful.

## 12. What this note explicitly does NOT claim

- Not that C2 doesn't work in principle — it does (3a is proven)
- Not that C2 wouldn't help on the right benchmark — it would (per
  conditional pattern)
- Not that the infrastructure is impossible — it's ~1-2 weeks of
  focused work
- Not that we should never build C2 — when the right benchmark
  arrives, it's the right tool

Just that on tessera's current benchmarks, C2 has no testable
opportunity to demonstrate its value.

## Changelog

- 2026-05-26: initial pre-analysis. Concludes that C2 cannot be
  meaningfully tested on tessera's current benchmarks (heat eq has
  constant noise; Feynman is noiseless). The conjecture is provable
  in 3a but its empirical test requires a heteroskedastic benchmark
  class that doesn't yet exist in tessera. Pre-analysis is the
  deliverable; no empirical experiment to follow.
