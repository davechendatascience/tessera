# Research note: improving tessera's scores on existing SR benchmarks

**Status:** ? RESEARCH. Open exploration; not committed to ship. Parallel direction to [`high_dim_symbolic_regression.md`](high_dim_symbolic_regression.md).

**Provenance:** the 2026-05-24 extended Feynman benchmark (`benchmarks/results/feynman_extended.md`) gave 9 exact / 14 partial / 7 failed out of 30 equations. The user asked: *"benchmarks show the basic ability for our SR's system identification. The Feynman benchmark still has some missed points — I don't know if it's purely unattainable or our own fault."* This doc disentangles the failure modes.

Where `high_dim_symbolic_regression.md` is about *scaling to wide inputs* (MNIST), this doc is about *closing the score-gap on standard low-dim benchmarks* — the more conventional SR competency test.

---

## 1. Where we stand on Feynman (the empirical anchor)

From `benchmarks/results/feynman_extended.md` (30/100 equations, pop=400, gens=120):

| Bucket | Count | Equations |
|---|---|---|
| **Exact** (rel < 0.01) | 9 | I.6.20a, I.6.20, I.10.7, I.12.1, I.12.5, I.14.4, I.15.3t, I.25.13, I.29.4 |
| **Partial** (0.01 ≤ rel < 0.20) | 14 | I.11.19, I.12.4, I.14.3, I.16.6, I.18.4, I.24.6, I.27.6, I.30.3, I.32.5, I.34.8, I.39.22, I.43.43, I.47.23, I.48.2 |
| **Failed** (rel ≥ 0.20) | 7 | I.8.14, I.9.18, I.12.2, I.12.11, I.26.2, I.40.1, I.43.31 |

The 7 failures are the main subject. Are they tessera-fault (fixable) or SR-class limits (publishable as honest)?

## 2. Per-failure diagnosis

### 2a. Trigonometric: `I.12.11`, `I.26.2`, `I.30.3`

All three use `sin` (and `I.26.2` also uses `arcsin`). Tessera has `tanh`, `abs`, `sign`, `step`, `sqrt`, `log`, `exp` but **no trig**.

**Verdict:** pure vocabulary gap. Fixable in ~30 LOC by adding `sin`, `cos` to `UN_OP_FNS` (with interval bounds + simplifier folds analogous to the sqrt/exp/log/pow addition in `64`).

**Confidence: tessera-fault, high.** PySR ships sin/cos by default and Feynman-style physics demands them.

### 2b. 4+ variable inverse-products: `I.12.2`, `I.12.4`, `I.43.31`

`q1·q2/(4·π·ε·r²)` and `k·T/(6·π·η·r)`. Tessera has `mul`, `div`. The structure is reachable; the GP didn't find it.

**Verdict:** search-budget bounded. Empirically: when we ran the smaller subset at pop=120 gens=40 (commit `213b849`), I.43.31 also failed; bumping to pop=400 gens=120 still failed but with rel=0.27 instead of 0.998. More budget moves the needle but doesn't close.

**Hypothesis: combinatorial-search complexity.** Dropping 4 multiplications and 1 division into a tree of depth ≤ 5 = ~10^4 structurally-distinct trees with these op slots; GP samples roughly 400×120 = 48k trees, so on average each topology gets seen ~5 times. With realistic noise + parsimony, that's not enough to converge.

**Confidence: tessera-fault but only partially.** PySR also struggles on Feynman 4-variable inverses without `optimize_constants` (BFGS-style refinement) per its 2023 design paper. Our `optimize_constants` is Nelder-Mead and runs only every 3-5 generations. **Promoted-to-PLANNED §1.3 jax.grad const-opt would help here.**

### 2c. Sum-of-squares structure: `I.8.14` Euclidean `sqrt((x2-x1)² + (y2-y1)²)`

We have `sqrt` and `mul` now. The structure SHOULD be reachable. But the GP would need to find:
1. `x2 - x1` and `y2 - y1` (two subtraction subtrees)
2. Square each
3. Sum
4. Take sqrt

That's a 9-node tree minimum. Re-running at pop=400 gens=120 gave rel=0.295 — not even close.

**Hypothesis: random_tree distribution rarely produces this template.** Specifically: random_tree picks each op uniformly; the chance of generating `sqrt(... + ...)` where both `...` are `(...) * (...)` subtrees is small. Plus the GP's mutation operators don't have an "introduce sum-of-squares structure" primitive.

**Confidence: tessera-fault with a partial fix.** Adding `pow(x, 2)` recognition (or just biasing `random_tree` toward squaring patterns) might help. The deeper issue is that GP biases toward shallow trees; 9-node target with specific structure is search-budget-bound.

### 2d. 9-variable Newton gravity: `I.9.18` `G·m1·m2/((x2-x1)² + (y2-y1)² + (z2-z1)² )`

13-node minimum target with 9 distinct variables. Worst-case for both random_tree (low chance of finding the right structure) and the GP loop (parsimony pushes toward fewer vars).

**Verdict:** at this complexity, even with bigger budget, plain GP without architectural priors is unlikely to find this. This is the equation that AI Feynman (Udrescu & Tegmark 2020) specifically solves via *separability detection*: recognise that the function factorises as `G·m1·m2 · 1/r²` and search each factor independently.

**Confidence: 80% inherent-SR-limit / 20% tessera-fault.** Adding separability detection as a pre-processing step would help; pure GP probably can't.

### 2e. Compound exponential: `I.40.1` `n0·exp(-m·g·x/(k·T))`

We have `exp` now. The argument `-m·g·x/(k·T)` is a 4-variable compound. The chance of GP finding the exact form is similar to (2b) but with the extra constraint that the result lives inside `exp(...)`.

**Hypothesis:** the GP found `exp(approximation)` but the approximation has the wrong constants. With const-opt and more budget it should converge. We got rel=0.22, which is just on the failed/partial boundary.

**Confidence: tessera-fault, partial-fix likely.** Promoted-to-PLANNED §1.3 (jax.grad const-opt) would help here too.

## 3. The categorisation

| Failure mode | Equations | Severity | Fixability |
|---|---|---|---|
| **Vocabulary gap** (sin/cos missing) | 3 of 7 | trivial | ~30 LOC |
| **Search budget bound** (4-var inverse) | 2 of 7 | medium | bigger pop/gens + const-opt; partial fix expected |
| **Structural search bound** (sum-of-squares, Newton gravity) | 2 of 7 | high | needs AI-Feynman-style separability detection |
| **Constant-opt bound** (barometric) | 1 of 7? | low-medium | jax.grad Adam would help |

**Headline:** of the 7 failures:
- **3** are *certain tessera-fault* (vocabulary), fixable trivially.
- **3** are *probable tessera-fault* (search budget / const-opt), partially fixable via existing planned upgrades.
- **1** is *probably inherent* to pure-GP SR without architectural priors.

So the *honest cap* on score-improvement for our extended Feynman benchmark, with all planned improvements, is **~6 of 7 failures fixed**: 14 exact + 17 partial + 1 fail, roughly. That would put us at **~47% exact / ~57% within-partial-or-better** — competitive with PySR at modest budget.

To match AI Feynman's headline (~100% with their full pipeline) would require us to import separability detection too. That's a larger investment.

## 4. Hypotheses worth empirically testing

### 4.1 ○ PLANNED — Add `sin`, `cos` primitives

**What:** add `np.sin`, `np.cos` to `UN_OP_FNS`. Interval bounds: `[-1, 1]` (monotone on quarter-periods only — use conservative `[-1, 1]` for the interval evaluator). Simplifier folds: `sin(0) = 0`, `cos(0) = 1`, `sin(neg(x)) = neg(sin(x))`, etc.

**Why now:** trivial effort, knocks out 3 of 7 Feynman failures by construction, brings tessera's pointwise vocabulary in line with PySR.

**Effort:** ~30 LOC + 6 tests + a quick benchmark re-run on `I.12.11`, `I.26.2`, `I.30.3`.

**Acceptance criterion:** all three trig equations move from FAILED to PARTIAL or EXACT in the extended-Feynman re-run.

### 4.2 ? RESEARCH — Structural-template mutations

**What:** add new mutation operators that introduce common physics-style structures:
- `template_sum_of_squares(tree)`: wrap two subtrees as `sqrt(a² + b²)`
- `template_inverse_product(tree)`: wrap as `A / (B · C · D)`
- `template_exponential_decay(tree)`: wrap as `n0 · exp(-arg)`

**Why:** even if `random_tree` rarely generates these templates by chance, mutation can introduce them. This is biased random search toward forms physics actually uses.

**Effort:** ~3-5 days. Each template needs validation logic.

**Risk:** if we over-engineer the templates, we're essentially hand-crafting solutions and losing the "general SR" claim. Need to balance.

**Acceptance criterion:** I.8.14 Euclidean and I.40.1 barometric move from FAILED to EXACT on next run.

### 4.3 ? RESEARCH — Separability detection (AI-Feynman style)

**What:** before launching GP, test if the target `y(x1, x2, ..., xK)` factorises:
- Multiplicative: `y(x1, x2) = f(x1) · g(x2)` if `y(a, b) · y(c, d) = y(a, d) · y(c, b)` for sampled points
- Additive: `y(x1, x2) = f(x1) + g(x2)` if the mixed second derivative is ~0
- Composition: `y(x) = h(x · z)` for some scaling `z`

If separability is detected, run GP on each factor independently, then combine.

**Why:** standard AI-Feynman 2020 method. Closes the 9-variable Newton gravity gap structurally.

**Effort:** ~1-2 weeks. Detection algorithms are well-documented (Udrescu & Tegmark §3-5). Plumbing into tessera's GP loop is the bigger lift.

**Risk:** for noisy real-world data, separability tests are fragile. For Feynman (noiseless), they work well.

**Acceptance criterion:** I.9.18 (Newton gravity) and at least 5 of the 14 *partial* Feynman equations move to EXACT.

### 4.4 ○ PLANNED — `jax.grad` constant optimisation

**Origin:** already promoted in [`docs/planned/roadmap.md`](../planned/roadmap.md) §1.3.

**Expected Feynman impact:** I.40.1 (compound exp), I.12.2/I.12.4 (4-var inverse with the `1/(4π)` constant), and a quarter of the *partial* equations should improve.

### 4.5 ? RESEARCH — Full Feynman 100 benchmark

**What:** scale from our 30-equation subset to the full 100. Measure where we stand on the long tail of harder equations.

**Why:** the 30 we ship were hand-picked for diversity; some of the un-tested 70 may reveal new failure modes (or new successes).

**Effort:** ~1 day to author the runner (mostly hand-coded samplers per equation). Multi-hour run-time depending on budget.

**Acceptance criterion:** publishable Feynman-100 table comparable to PySR's. Decide based on the result whether we go for a paper.

### 4.6 ? RESEARCH — SRBench competition entry

**What:** participate in / submit to the SRBench symbolic-regression benchmark suite (https://github.com/cavalab/srbench). SRBench has ~250 problems across ground-truth + real-world tracks.

**Why:** standard community benchmark. Tessera's measure-theoretic ops would give a *distinct* position in SRBench (no other entry uses convolutional / functional operators natively).

**Effort:** multi-week. Includes packaging tessera as a SRBench-compatible runner, hyperparameter selection, full benchmark run.

**Risk:** SRBench heavily favours pointwise SR (most problems are smooth polynomial / rational). Tessera's strengths (time-series / functional) won't dominate; we'd be measured on the polynomial benchmarks where PySR already wins.

**Acceptance criterion:** ship a benchmark report comparing tessera against PySR / Operon / DSR on SRBench's ground-truth track.

## 5. The honest framing for the doc

Three possible empirical outcomes after 4.1 (trig) + 4.4 (jax.grad const-opt) + 4.2 (templates) all ship:

| Outcome | Feynman 30-subset score | Interpretation |
|---|---|---|
| **Optimistic** | 16 exact / 12 partial / 2 fail | Tessera matches PySR at modest budget; remaining 2 failures are AI-Feynman territory. Publishable. |
| **Modest** | 13 exact / 14 partial / 3 fail | Real improvement from the diagnostic work; competitive but not best-in-class. Documents the limits cleanly. |
| **Pessimistic** | 11 exact / 12 partial / 7 fail | Most of the §4 hypotheses didn't move the needle. Means failures are deeper than vocabulary/budget — possibly fundamental GP-class limits. Falsifies the "tessera-fault is fixable" framing. |

The pessimistic outcome is the most informative. If even with sin/cos + jax.grad + templates the score doesn't shift, we have empirical evidence that the failures are inherent to plain GP-on-pointwise-SR and the only real fix is separability detection or neural-symbolic hybrids.

## 6. Connection to high-dim SR direction

These two research notes are **complementary**, not redundant:

| Aspect | `high_dim_symbolic_regression.md` | This doc |
|---|---|---|
| **Input scale** | K = 100-784 features (image) | K = 1-9 features (physics) |
| **Output target** | Discriminative scalar (classification) | Exact analytical formula |
| **Ceiling type** | Hypothesis-class (can SR even *express* the model?) | Search + vocabulary (can SR *find* the model that's expressible?) |
| **Main lever** | Architectural enhancement (two-layer SR, sparsity) | Search-engine improvement (const-opt, templates, separability) |

Doing both directions in parallel gives tessera two distinct competency claims:
- **Pure SR on physics-style problems**: competitive Feynman scores → "tessera does what PySR does, plus measure-theoretic ops."
- **High-dim SR research**: 71-95% MNIST → "tessera is the only SR system attempting CV-scale problems with Knuth-style search machinery."

Neither alone is enough; both together is a stronger position.

## 7. Concrete next experiments (status-flagged)

| Item | Effort | Status |
|---|---|---|
| Add `sin`, `cos` to UN_OP_FNS + intervals + simplifier folds + Feynman re-run | ~half day | ○ PLANNED (§4.1) |
| Templated mutation operators for sum-of-squares / inverse-product | 3-5 days | ? RESEARCH (§4.2) |
| Separability detection (AI-Feynman style) | 1-2 weeks | ? RESEARCH (§4.3) |
| Full Feynman 100 benchmark runner | 1 day + multi-hour run | ? RESEARCH (§4.5) |
| SRBench submission | multi-week | ? RESEARCH (§4.6) |

Suggested order: **§4.1 first** (cheapest, clearest A/B test, knocks out 3 failures by construction). After §4.1 lands and a new Feynman score is published, decide whether §4.2 vs §4.3 is the next move based on whether the remaining failures look more search-budget or structural.

## 8. Reading list

- Udrescu & Tegmark, *AI Feynman: a Physics-Inspired Method for Symbolic Regression* (Science Advances 2020, arXiv:1905.11481) — separability detection, units analysis
- Udrescu et al., *AI Feynman 2.0* (arXiv:2006.10782) — neural-net oracle for compositional structure
- Cranmer, *Interpretable Machine Learning for Science with PySR/SymbolicRegression.jl* (arXiv:2305.01582) — PySR design including BFGS const-opt
- La Cava et al., *Contemporary Symbolic Regression Methods and their Relative Performance* (NeurIPS 2021 Datasets and Benchmarks track, SRBench) — community benchmark
- Eureqa: Schmidt & Lipson, *Distilling Free-Form Natural Laws from Experimental Data* (Science 2009) — historical baseline

## Changelog

- 2026-05-24: initial document. Empirical anchor: `benchmarks/results/feynman_extended.md` 30/100 result (9 exact / 14 partial / 7 fail). Diagnostic categorisation: 3 vocabulary gaps + 2 search-budget + 2 structural + 1 const-opt. Six hypotheses listed with status flags; §4.1 (sin/cos) flagged as PLANNED for immediate work.
