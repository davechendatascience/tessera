# Research note: process-discovery SR — studies + innovations

**Status:** ? RESEARCH. Substantial direction-setting note, half survey of developed theories and half tessera-specific innovation. Not promoted to PLANNED; stays in research until a concrete experiment validates the direction. New project-scale work, but proposed AS A TESSERA DIRECTION rather than as a separate codebase.

**Provenance:** user (2026-05-26), refining the natural-overfit + composite-dynamics threads:

> *"instead of always learning the equations from data, we have to figure out the math of data generation process. And use the theorems we've learned from this to guess an appropriate system dynamics. [...] data generation can be very interesting, it can be a function with a certain complexity, it can be a function with a certain complexity perturb by another function. [...] measuring the loss alone has friction. We are not treating it as a data generating process, but as sudoku slots. [...] we like to find the data generating process that fits the statistical properties, meaning that it doesn't fit every in-sample data points but broadly overparameterized and thus give very bad statistical assumptions."*

User direction: *"I want to still do it in tessera. Let's do it in tessera. I want the research to be half studies from developed theories and half innovation."*

This note formalizes the framework, surveys the literature, identifies the specific gap a tessera-shaped contribution could fill, and proposes tessera-specific innovations + MVP experiments.

---

## 1. The critique that motivates the direction

Current SR — including tessera — treats the problem as a Sudoku puzzle:

- Fixed vocabulary (the puzzle pieces)
- Fixed pointwise scoring (loss + parsimony)
- Search for the arrangement that satisfies constraints (TRAIN MSE minimization)
- Output a single solution

This frame is right for problems where:
- The solution is genuinely expressible in available pieces
- "Satisfying constraints" maps cleanly to "fits observed data"
- The data uniquely identifies the answer

It's wrong — or at best incomplete — for the scientific question:

> *What process generated this data, given that I might not have the right pieces, my prior is uncertain, and the data only partially constrains the answer?*

The Sudoku frame can fit ANY data point exactly given enough flexibility. A degree-N polynomial fits N points trivially while oscillating wildly between them. The Sudoku frame doesn't see this as a problem because it doesn't model "between the points." But the data-generating process IS what determines behavior between observed points; ignoring it amounts to free interpolation choices that the data didn't authorize.

**The process-discovery reframe asks**: instead of finding a function that interpolates points, find a generative process whose statistical properties match observed properties. Pointwise fit becomes one criterion among several; smoothness, distributional moments, autocorrelation, response functions, regularity all become evidence.

## 2. Connection to existing tessera framework

This direction extends two threads already in tessera's research notes:

- **`fit_as_perfect_info_game.md`**: covers the layer 1→2 game (data → equation). The note implicitly assumes "fits data" is the goal. Process-discovery adds layer 2→3 (equation → mechanism) as the actual goal, with statistical-property matching as the evaluation criterion.

- **`from_data_to_mechanism.md`** (closing synthesis): articulates the three-layer model and the natural-overfit reframe. The class taxonomy (A/B/C) was already pointing at this — `Class B` (template / reduce_*) fits pointwise but doesn't generalize because it's not a process model. Class C generalizes precisely because it IS a process model.

- **`randomized_recovery_bounds_for_sr.md`**: catalogs theorems for recovering generative operators (Boullé-Townsend etc.). The recovery-bounds framework is the *theoretical* version of what this note proposes operationally.

Process-discovery SR is the synthesis of these threads into a coherent methodology, with practical implementation paths.

## 3. Survey: developed theories (half 1)

The relevant published work, organized by approach.

### 3.1 Stochastic Differential Equation (SDE) discovery

Models: `dX = f(X)dt + g(X)dW` where `f` is drift, `g` is diffusion coefficient, `dW` is Wiener noise.

Key prior art:

- **Boninsegna, Nüske, Clementi 2018** — *"Sparse learning of stochastic dynamical equations"*: extends SINDy to identify drift AND diffusion from stochastic trajectories. Uses sparse regression in a library of candidate terms.

- **Li, Wong, Chen, Duvenaud 2020** — *"Scalable Gradients for Stochastic Differential Equations"* (AISTATS): adjoint method for backprop through SDE solutions; enables variational learning over SDE parameters.

- **Kidger, Foster, Li, Lyons 2021** — Neural SDEs as latent variable models with adversarial / variational training.

- **Doucet, de Freitas, Gordon 2001** — Sequential Monte Carlo for SDE parameter inference; particle filter methods.

- **Roberts, Stramer 2001** — *"On inference for partially observed nonlinear diffusion models"*: foundational Bayesian MCMC for SDEs.

These are genuinely process-discovery — they parameterize the *dynamics* (drift + diffusion), not a trajectory fit. **None are symbolic**; they're all neural or fixed-parametric. SR-style symbolic SDE discovery is largely absent.

### 3.2 Approximate Bayesian Computation (ABC)

When likelihood is intractable but you can simulate: match summary statistics of simulator vs data.

Key prior art:

- **Pritchard, Seielstad, Perez-Lezaun, Feldman 1999**: original ABC for population genetics.

- **Beaumont, Cornuet, Marin, Robert 2009-2010**: ABC-MCMC and ABC-SMC algorithms.

- **Marin, Pudlo, Robert, Ryder 2012** — *"Approximate Bayesian computational methods"* (Stat & Computing): theoretical foundations and convergence guarantees.

- **Wood 2010** — *"Statistical inference for noisy nonlinear ecological dynamic systems"* (Nature): synthetic likelihood approach.

- **Price, Drovandi, Lee, Nott 2018** — *"Bayesian Synthetic Likelihood"*: Gaussian approximation to simulator output, with applications to ecology and population dynamics.

ABC IS the methodology you're pointing at when you say *"fits the statistical properties, not every in-sample data point."* It's mature in ecology, epidemiology, population genetics. **Largely absent from the SR literature.**

### 3.3 Score-based generative models / diffusion models

Different tactic: learn `∇log p(x)` (the score function) — characterizes the distribution via its gradient field.

Key prior art:

- **Hyvärinen 2005** — *"Estimation of Non-Normalized Statistical Models by Score Matching"*: foundational score-matching paper.

- **Song & Ermon 2019** — *"Generative modeling by estimating gradients of the data distribution"* (NeurIPS): noise conditional score networks.

- **Ho, Jain, Abbeel 2020** — DDPM (denoising diffusion probabilistic models).

- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole 2021** — *"Score-based Generative Modeling Through Stochastic Differential Equations"*: unifies score-based and SDE formulations.

This IS process-discovery — the score parameterizes the distribution's gradient field, which determines the generative dynamics. But the score is *always* a neural network; symbolic versions don't exist.

### 3.4 Method of Moments / GMM

Match low-order moments of the data; closed-form for many parametric families.

- **Hansen 1982** — General Method of Moments: the canonical formulation.
- **Gourieroux, Monfort, Renault 1993** — Indirect inference using an auxiliary model.
- **Hall 2005** — *Generalized Method of Moments* (OUP): comprehensive textbook treatment.

Less popular than ABC for nonlinear systems but more tractable when moments are computable.

### 3.5 PDE / ODE discovery (current SR neighborhood)

- **Brunton, Proctor, Kutz 2016** — SINDy: sparse identification of nonlinear dynamics from a library of candidate terms.
- **Rudy, Brunton, Proctor, Kutz 2017** — PDE-FIND: extension to PDEs.
- **Raissi, Perdikaris, Karniadakis 2019** — PINNs: physics-informed neural networks with PDE residual as loss.
- **Chen, Rubanova, Bettencourt, Duvenaud 2018** — Neural ODEs: dynamics as a neural network in an ODE solver.

**These mostly use pointwise loss + sparsity prior.** They're closest to current SR; what they share with process-discovery is the dynamical-system framing, but they still optimize pointwise residual.

### 3.6 Information-theoretic equation discovery

- **Solomonoff 1964** — universal Bayesian inference with algorithmic prior.
- **Levin 1973** — computable approximation (Levin search).
- **Rissanen 1978** — MDL principle: shortest model + residual code.
- **Vereshchagin & Vitányi 2002, 2010** — *Algorithmic Statistics*: formalizes minimum sufficient statistic via Kolmogorov complexity.
- **Schmidhuber 2002** — *"Optimal Ordered Problem Solver"*: practical Solomonoff-flavored learning.

### 3.7 Causal discovery

- **Spirtes, Glymour, Scheines 1991** — PC algorithm (constraint-based).
- **Chickering 2002** — GES (score-based DAG learning).
- **Hoyer, Janzing, Mooij, Peters, Schölkopf 2009** — Additive Noise Models for causal discovery.
- **Pearl 2009** — *Causality*: comprehensive treatment of do-calculus and counterfactuals.

The bridge to SR: **axis-typed primitives can encode causal direction** (some variables are upstream of others; functional dependencies have a directionality). This is partially in tessera via `tessera.expression.axes`.

### 3.8 The specific topic — "function + perturbation by another function"

User's framing maps to **stochastic dynamical systems**. The constructive analytical math:

- **Itô calculus** — foundational SDE machinery.
- **Stratonovich calculus** — alternative SDE interpretation, useful for physics.
- **Random ODEs** — noise enters as a random parameter, not as Wiener.
- **Hawkes processes** — self-exciting point processes; event history drives future intensity.
- **Lévy processes** — includes jumps; generalizes Wiener.
- **Stochastic PDEs** — KPZ equation, stochastic heat equation, fluctuating hydrodynamics.
- **McKean-Vlasov SDEs** — mean-field interactions, large-population limits.

Almost all of this is *parametric* — the family is posited, parameters are inferred. The SR-style *discovery of the family* is sparser.

## 4. The Kolmogorov anchor

Beyond AIT (already covered in `randomized_recovery_bounds_for_sr.md`), Kolmogorov has work directly relevant to process-discovery:

### 4.1 Five Kolmogorov contributions, ordered by relevance

| # | Year | Contribution | Relevance to process-discovery |
|---|---|---|---|
| 1 | 1933 | Axiomatization of probability | Conceptual prerequisite — "process" requires probability measures |
| 2 | 1958 | Kolmogorov-Sinai entropy | Quantifies complexity of dynamical systems; relates to predictability |
| 3 | 1965 | "Three Approaches to the Concept of the Amount of Information" | Unifies probabilistic + combinatorial + algorithmic measures |
| 4 | 1965 | Kolmogorov complexity (with Solomonoff, Chaitin) | Universal compressibility; theoretical optimum for MDL |
| 5 | **1974** | **Kolmogorov Structure Function** | **Most directly relevant — formalizes structured-vs-random data decomposition** |

### 4.2 The Kolmogorov Structure Function in detail

Given data x and a complexity level k, define:

```
λ(k) = min { -log p(x | model) : K(model) ≤ k }
```

This is the smallest *log-loss* achievable by any probability model whose own complexity is bounded by k. Properties:

- λ(k) is non-increasing in k (more model complexity → better fit)
- As k grows, λ(k) approaches `log(1/p_true(x))` where `p_true` is the true generating distribution
- The **minimum sufficient statistic** is the smallest k* such that λ(k*) is essentially equal to its limit

**Why this matters for SR:**

This formalizes exactly the user's intuition: data has a *structured* part (modelable in K(model) bits) and a *random* part (incompressible residual). Fitting beyond k* is fitting random noise — your "over-parameterized" critique.

The structure function gives the *right amount of model* for any data — neither under- nor over-parameterized. Computing it exactly is impossible (K is uncomputable), but approximations exist:

- MDL with a chosen prior over models
- Cross-validation as empirical λ(k) estimation
- Bayesian model averaging weighted by description length

For SR specifically: **scoring candidates by `-log p(data | tree) + α·K(tree)`** (where K is approximated by tree description length) is the Kolmogorov-structure-function-aware analog of current `MSE + parsimony · cx`.

### 4.3 What's directly useful vs theoretical

| Kolmogorov concept | Usable in tessera? | How |
|---|---|---|
| Axiomatization of probability | Yes (framework) | All process-discovery work assumes this |
| K-S entropy | Yes (diagnostic) | Estimate from data; informs target complexity |
| AIT (algorithmic information) | Theoretical | MDL is the practical proxy |
| K complexity | Theoretical | Tree description length is the proxy |
| **Structure function** | **Yes (scoring)** | **MDL-with-explicit-likelihood scoring** |

The structure function is the operational anchor. It connects our existing parsimony machinery to a principled theoretical framework.

## 5. The gap that tessera could fill

Cross-referencing the survey against what the user articulated, the unsolved combination is:

> **Symbolic-regression-shaped methodology for stochastic dynamical systems, combining:**
> 1. SDE-style drift + diffusion model class
> 2. ABC-style summary-statistics fitness (not pointwise loss)
> 3. MDL / structure-function scoring with explicit log-likelihood
> 4. Causal direction enforcement via axis types
> 5. Counterfactual / interventional evaluation

No published system has all five. Pieces exist:
- SINDy: 1 (drift only) + sparsity prior
- Bayesian SDE: 1 + 3 (but neural, not symbolic)
- ABC for dynamics: 2 (but rarely with symbolic models)
- Kolmogorov structure function: 3 (theoretical, not implemented)
- Causal additive models: 4 (but not for dynamics)

The combination is genuinely novel and tessera-shaped because tessera already has:
- Symbolic tree representation (1)
- Axis-aware primitives (4 partial)
- Polynomial simplifier + sufficient-stat infrastructure (supports 3)
- A GP backbone that could drive structural search over drift/diffusion templates

What's missing is the scoring infrastructure, the simulation-based fitness, the structure-function-aware MDL, the explicit causal direction priors, and the counterfactual-evaluation harness.

## 6. Tessera-specific innovations (half 2)

What we would add to tessera to enable process-discovery SR. Ranked by tractability + impact.

### 6.1 Distributional-output trees

Current tessera: tree outputs a single value `y_hat = tree(x)`. Loss is `(y_hat - y)²` pointwise.

Innovation: tree outputs a *predictive distribution* parameterization, e.g., `(mu, sigma) = tree(x)`. Loss is `-log N(y; mu, sigma²)` — the negative log-likelihood under a Gaussian assumption.

Why useful: the second output (sigma) IS the diffusion coefficient in an SDE model. The tree must explain BOTH the mean and the variability. Fitting beyond the structure-function-implied complexity is automatically punished because Gaussian likelihood saturates.

Implementation cost: ~moderate. Trees need a second head; loss function changes; mutation operators need to respect two outputs.

### 6.2 ABC-style fitness via summary statistics

Current tessera: MSE on observed points.

Innovation: compute summary statistics on observed data (mean, variance, autocorrelation at multiple lags, spectral density at key frequencies, marginal distribution moments). For each candidate tree, simulate the implied dynamics and compute the same statistics. Score by distance between observed and simulated statistics.

Why useful: directly operationalizes user's "fits statistical properties, not every point." Robust against over-parameterized in-sample fitting. Validates trees by their statistical implications rather than their pointwise interpolation.

Implementation cost: ~high. Requires forward simulation from tree outputs. For polynomial/algebraic trees this is trivial (apply the tree); for SDE-style trees with diffusion outputs it requires Euler-Maruyama integration. The summary-statistic computation is cheap.

### 6.3 MDL with explicit log-likelihood scoring

Current tessera: `score = loss + parsimony · cx`. The `loss` is MSE; cx is node count.

Innovation: `score = -log p(data | tree) + α · description_length(tree)`. The first term is explicit log-likelihood under the tree's predictive distribution. The second is the MDL penalty with α calibrated to bit-counting (not arbitrary).

Why useful: rigorous structure-function-aware scoring. The trade-off between fit quality and model complexity becomes bit-accounting rather than ad-hoc penalty. Connects to Kolmogorov structure function directly.

Implementation cost: ~low. Just a different scoring function. Requires distributional output trees (6.1) for the log-likelihood term to make sense.

### 6.4 Causal direction priors at the tree level

Current tessera: axes module enforces dimensional/type compatibility (validation).

Innovation: extend axes module to include **causal direction**. Specific variables are marked as upstream (causes) or downstream (effects). Mutation operators enforce that downstream variables can't appear in expressions that compute upstream variables. Assign probability zero to trees that violate the causal structure.

Why useful: dramatically reduces effective search space; encodes scientific knowledge as a hard prior; eliminates many natural-overfit candidates (which often arise from acausal compositions).

Implementation cost: ~moderate. Existing axes module is a foundation. Need user-specified causal annotations + validator changes.

### 6.5 Counterfactual evaluation harness

Current tessera: TRAIN/TEST split is benchmark-by-benchmark; OOD probing is manual.

Innovation: standardized counterfactual evaluation. For each candidate, the harness:
- Evaluates on observed data (TRAIN)
- Evaluates on held-out same-distribution data (TEST)
- Evaluates on held-out OOD data (different IC, different parameter regime, etc.)
- Computes "extrapolation degradation" — how fast does loss grow as you move OOD?
- Penalizes candidates whose extrapolation degrades faster than expected for a true mechanism

Why useful: makes Class B (natural-overfit) candidates self-penalize. The class taxonomy from `from_data_to_mechanism.md` becomes a built-in evaluation feature.

Implementation cost: ~moderate. Each benchmark gets a "perturbation generator" that produces OOD samples. Generic perturbations: ±10% on each input range, different ICs for dynamical systems, etc.

### 6.6 Iterative strategy refinement based on residual diagnostics

Current tessera: one GP run; final Pareto front returned; user analyzes.

Innovation: after each generation (or every K generations), examine the best candidate's residuals for:
- Normality (Shapiro-Wilk / Anderson-Darling)
- Heteroskedasticity (Breusch-Pagan)
- Autocorrelation (Ljung-Box)
- Outliers / heavy tails

Based on diagnostics, adjust the search prior. E.g., if residuals are heteroskedastic, increase mutation weight for distributional-output trees. If autocorrelated, favor trees with serial-correlation-handling primitives.

Why useful: makes the search adaptive. The GP can detect that its current model is mis-specified (residuals tell you) and shift the search direction accordingly. Closer to how a scientist iteratively refines a model.

Implementation cost: ~moderate. Each diagnostic is small; the refinement logic is where the design work is.

> **Experimental home (added 2026-05-26):** implementations of the conjectures in this note live in `tessera.experimental.*`. Each module cites the specific conjecture (C1, C2, ...) and declares its graduation/removal criteria. See `tessera/experimental/__init__.py` for the discipline. No production code may import from there until a conjecture graduates with empirical evidence.

## 7. MVP sketches (three increasing scope levels)

To validate the direction without committing to the full architecture, three progressively-larger experiments.

### 7.1 Cheapest: ABC-style scoring on existing GP (~1 day)

Implementation:
- Take the heat equation benchmark
- Compute summary statistics on observed `dt_U`: variance, autocorrelation at lag 1-5, spectral density at key frequencies
- For each candidate tree (already evaluated for MSE), also compute these statistics on the tree's predictions
- Score = `MSE + β · summary_distance` for various β

Compare against the existing Class-A/B/C distribution. Question: does ABC-style scoring suppress Class B (natural-overfit) more effectively than pointwise MSE alone?

This requires no architecture changes; just an additional loss term computed alongside MSE.

### 7.2 Medium: distributional-output GP (~1-2 weeks)

Implementation:
- New tree type: `DistTree` with two outputs (`mu`, `sigma`)
- New loss: Gaussian negative log-likelihood
- Modified mutation operators that respect two-output structure
- New benchmark: a stochastic dynamical system (heat equation with state-dependent noise; Hawkes process; OU process)
- Evaluate: does the distributional output capture mechanism better than pointwise output?

This is medium because the GP infrastructure already handles single trees well; the modification is contained.

### 7.3 Maximal: full process-discovery framework (~3-6 months)

Implementation:
- All six innovations (6.1-6.6) integrated
- New benchmark suite specifically for stochastic dynamical systems
- Counterfactual evaluation as default
- MDL scoring with structure-function calibration
- Per-benchmark causal annotations
- Adaptive search strategy

This is the full project. Only worthwhile if 7.1 and 7.2 produce promising results.

## 8. Connection to existing tessera artifacts

How this direction builds on what tessera already has:

| Existing tessera capability | Process-discovery use |
|---|---|
| `tessera.expression.axes` | Foundation for 6.4 (causal direction priors) |
| GP backbone | Core search engine; modified for distributional output (6.1) |
| Polynomial simplifier | Reduces structurally-equivalent candidates; relevant for MDL bit-counting |
| `sufficient_stats` polish | Could be adapted for SDE coefficient inference (drift = linear combination of basis functions) |
| `reduce_*` downweight | Same justification: TRAIN-specific stats don't belong in per-sample prediction |
| `random_tree(enable_2d=True)` | Foundation for dynamical-system trees (Measure2D for spatial derivatives) |
| Class taxonomy (A/B/C) | Becomes the official evaluation framework with explicit Class C target |
| `simplify_full` pipeline | Extended with MDL-aware simplifications |

The direction is genuinely a *tessera direction* — it builds on existing architecture, doesn't require a separate codebase, and extends rather than replaces what works.

> **C1-refined empirically falsified (2026-05-26):** First-pass test on heat equation discovery. See `benchmarks/results/heat_equation_abc_mvp71.md`. ABC scoring on held-out trajectory eliminates Class B but ALSO suppresses Class C discovery; pointwise MSE on held-out data is strictly better at this benchmark. The conjecture in its current form does not hold. Module `tessera.experimental.abc_scoring` preserved for future re-evaluation (different β, different statistics, or genuinely-stochastic benchmark).

> **C4 partial validation (2026-05-26):** Test on heat equation. See `benchmarks/results/heat_equation_causal_axes_mvp_c4.md`. Causal-spatial constraint eliminates Class A-temporal (the temporal-derivative tautology) as designed AND doesn't lose the right answer (Class C still found at 1/5 with same canonical form). BUT does NOT boost Class C above baseline rate — the GP falls back to Class A-spatial or degenerate. Module `tessera.experimental.causal_axes` preserved; necessary but not sufficient intervention. Taxonomic refinement (Class A-temporal vs Class A-spatial) exposed as a useful diagnostic.

> **C3 falsified, calibration math partially validated (2026-05-26):** Pre-analysis in `docs/research/c3_mdl_analysis.md`; experiment in `benchmarks/results/heat_equation_mdl_mvp_c3.md`. MDL with naive Gaussian likelihood at heat eq's N/σ produced *identical* Pareto fronts to ad-hoc parsimony (median cx=8 for both); recalibrated mode forced smaller cx (6) but not Class C. Class C count: adhoc 1/5, naive_mdl 0/5, recal 0/5. **Math is directionally correct (α ordering matches derivation) but empirical effect is below noise floor.** Deeper insight: parsimony-scale scoring tweaks don't direct exploration on this benchmark; interventions need to operate at MSE-magnitude scale (the actual driver of GP search dynamics).

> **C6 validated-as-predicted (2026-05-26):** Pre-analysis in `docs/research/c6_residual_diagnostics_analysis.md`; experiment in `benchmarks/results/heat_equation_adaptive_mvp_c6.md`. Generic operator-usage-driven adaptive mutation weights produced results EXACTLY matching baseline: 1/5 Class C in both, 1/5 Class B in both, 3/5 seeds produced bit-identical results. Adaptation triggered 10 events per run but didn't change outcomes. The pre-analysis's diagnostic→corrective mapping argument is empirically confirmed: generic adaptation cannot solve the mapping problem; specific mappings essentially encode the answer. **Pattern across C1/C3/C4/C6:** all interventions at the scoring/search-modification layer produce similar Class C rates (-1/5 to 0/5 vs baseline); only data-level interventions (reduce_* downweight, multi-trajectory) move the needle.

> **C5 validated, selection-layer version (2026-05-26):** Pre-analysis in `docs/research/c5_counterfactual_eval_analysis.md`; experiment in `benchmarks/results/heat_equation_counterfactual_mvp_c5.md`. Counterfactual ranking applied as a post-hoc selector on baseline Pareto fronts: when mechanism candidates exist (2/5 seeds), CF ranking correctly identifies them (2/2); when absent, CF correctly picks best-available Class A. CF score cleanly discriminates mechanism (median ≤ 1.5) from non-mechanism (median > 1.7). Seed 2027: CF found a Pareto-strict improvement over train-loss selection (lower cx at same accuracy). **First VALIDATED-POSITIVE conjecture in the basket. The selection layer is the productive layer** — scoring-layer interventions don't help (C1/C3/C6) but post-hoc selection-layer evaluation does.

## 9. Falsification

What would tell us this direction is wrong:

1. **MVP 7.1 shows no improvement** over pointwise MSE on Class A/B/C distribution → ABC-style scoring isn't sufficient to suppress natural-overfit; need stronger interventions or the framework is mis-conceived.

2. **Distributional-output trees (7.2) overfit worse than single-output trees** → the additional degrees of freedom (modeling both mean and variance) hurt more than they help; over-parameterization isn't fixed by Gaussian likelihood.

3. **Counterfactual evaluation always agrees with same-distribution TEST** → the OOD probes aren't actually testing different things; the class taxonomy was an artifact of the heat equation specifically.

4. **Kolmogorov structure function approach doesn't measurably differ from current parsimony** → in practice, the bit-counting calibration doesn't change the Pareto front compared to ad-hoc parsimony coefficients.

If 7.1 fails (no improvement), the direction is likely not worth pursuing further. If 7.1 succeeds but 7.2 fails, the framework helps but distributional outputs aren't the right form. If both succeed, 7.3 becomes worth the investment.

## 10. What this note explicitly does NOT claim

- **Not that this is a complete theory.** The structure function is theoretical; bit-counting calibration is heuristic; counterfactual evaluation requires problem-specific perturbation generators.

- **Not that current tessera is broken.** Tessera's unit-dynamics SR works; the closing synthesis (`from_data_to_mechanism.md`) documents what it can do. Process-discovery SR is an EXTENSION, not a replacement.

- **Not that all six innovations are necessary.** The MVP path (7.1 → 7.2 → 7.3) lets us add them incrementally based on empirical signal.

- **Not that this is published-ready research.** This note is a direction-setting document. Publishability requires actual experimental work and comparison against baseline methods (SINDy, ABC, etc.).

- **Not that we have a timeline.** The MVPs are sized in days/weeks/months but no commitment is made about ordering or completion. This stays in `research/` until something promotes to `planned/`.

## 11. Reading order + next steps

If you're picking up this thread in the future:

1. Read `from_data_to_mechanism.md` first — establishes the framework this builds on.
2. Read this note (§3 for survey, §4 for Kolmogorov, §6 for the innovations, §7 for MVP options).
3. If considering implementation: start with 7.1 (cheapest, highest information). The result tells us whether to proceed to 7.2.
4. Read `randomized_recovery_bounds_for_sr.md` for the theoretical-bounds angle.
5. Read `algorithmic_information_for_sr.md` (Bowery / MDL critique) for the description-length philosophical foundations.

Next concrete experiment if user wants to test the direction:

**Run MVP 7.1 on heat equation benchmark.** Question: does adding an ABC-style summary-statistics-matching term to the scoring suppress Class B and promote Class C? Empirical signal in ~1 day of work. If positive, validates the direction; if neutral/negative, narrows scope.

## 12. Closing note

The user's instinct that current SR-as-Sudoku is the wrong frame for *scientific* equation discovery is sharp and correct. The literature has pieces of the right answer scattered across SDE inference, ABC, score-based modeling, AIT, and causal discovery — but no published system combines them into a unified process-discovery framework for SR.

Tessera is well-positioned to fill that gap because its architecture (axis-aware primitives, GP backbone, simplification infrastructure) already supports several pieces. The work to add the rest — distributional outputs, ABC fitness, MDL scoring, causal priors, counterfactual evaluation, adaptive search — is bounded and tractable.

This becomes the *unifying direction* for tessera's research agenda: less polish on within-Sudoku optimization, more pivoting toward what data should be telling us when we listen to it as samples from a process rather than as constraints on a fit.

The first move is MVP 7.1 — the cheapest test of whether the framework is operationally meaningful.

## Changelog

- 2026-05-26: initial document. Half survey of developed theories (SDE discovery, ABC, score-based models, K-S entropy, Kolmogorov structure function, PDE/ODE discovery, causal discovery, applied stochastic-dynamics math). Half tessera-specific innovations (distributional output trees, ABC fitness, MDL with explicit log-likelihood, causal direction priors, counterfactual evaluation, iterative strategy refinement). Three MVP sketches at increasing scope. Provenance: user direction to keep process-discovery work in tessera, with research half developed-theory survey and half innovation.
