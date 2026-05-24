# Research note: algorithmic information theory and tessera's fitness function

**Status:** ? RESEARCH. Substantive critique of current SR practice; needs careful engagement before any commitment.

**Provenance:** the user shared (2026-05-24) a passage from "Mean Mr. Jim" (James Bowery, AIT-and-SR thinker) arguing that the SR community conflates *mutation bias* with *selection bias*, and that selection — specifically the *fitness function* — must be a real algorithmic-information-theory (AIT) / minimum-description-length (MDL) count, not a summary statistic like MSE.

This doc records the critique, audits tessera against it, and frames what a response would entail.

---

## 1. The critique, restated

The Bowery passage makes two distinct claims that should not be conflated:

**Claim 1 (the easy one): mutation bias is fine.**

> *"Go ahead and impose whatever arbitrary mutation bias you want in terms of whatever language you choose. ... The way you mutate within this language is up to you as well — just have at it."*

Tessera's grammar (the operator alphabet, the `random_tree` distribution, the mutation operators in `OP_WEIGHTS`) is one such mutation bias. Bowery's claim: there's *no philosophical problem* with this. Pick whatever language is convenient; mutate however you like.

**Claim 2 (the hard one): selection bias must be MDL-grounded.**

> *"When you compute your fitness in preparation for model SELECTION — that is to say, when you compute the measure of complexity — that is to say, when you compute your 'loss' function — do your goddamn homework and encode your 'arbitrary language' as a minimum length interpreter written in an 'instruction set' chosen to be reasonable. ... You must also encode the errors as program literals so that the resulting algorithm outputs the dataset exactly bit for bit, rather than summarizing the errors as 'noise' in 'metrics' like sum of squared errors."*

What this means concretely:

For a candidate `tree` fitted to data `(X, y)`, the AIT-correct fitness is:

```
fitness(tree) = bits(interpreter)
              + bits(tree | interpreter)
              + bits(residuals | tree, interpreter)
```

where:
- `interpreter` is the minimum-length program that can EXECUTE trees in the candidate's language (op alphabet)
- `tree | interpreter` is the candidate's bit-string under that interpreter's encoding
- `residuals` is the EXACT bit-for-bit error stream (not summarised by MSE), encoded with whatever model of the residual structure the candidate IMPLIES

**The key insight Bowery is pushing:** if your model summarises errors as MSE, you have *thrown away* the structure of those errors. Bit-for-bit residuals encode the actual information you failed to capture; ignoring them is confirmation bias dressed up as "noise."

## 2. Where tessera currently stands

Tessera's fitness function (from `tessera.search.gp`):

```python
fitness = loss + parsimony * cx
```

where `loss = mse_loss(y_pred, y_true)` and `cx = complexity(tree)` (= node count).

Auditing this against Bowery's MDL ideal, term by term:

| AIT term | Tessera equivalent | Honest assessment |
|---|---|---|
| `bits(interpreter)` | NONE — assumed constant | We don't count the interpreter. Bowery's homework assignment not done. |
| `bits(tree \| interpreter)` | `parsimony * cx` — heuristic proportional to node count | Loose approximation of model bit-length. Not formally derived from the grammar's encoding. |
| `bits(residuals \| tree)` | `mse_loss` — quadratic summary | **The exact thing Bowery rejects.** MSE assumes IID Gaussian residuals; if residuals have structure, MSE flattens it out. |

**Verdict:** tessera's fitness is a *heuristic* with the same shape as MDL (one term for model complexity, one term for fit) but without the AIT grounding. The `parsimony` coefficient is a hand-tuned weight, not an information-theoretic count; the `mse_loss` is a summary statistic, not a bit-encoding.

This is not unique to tessera — PySR, Operon, DSR, and most published SR systems use the same heuristic shape. Bowery's critique applies to the SR community as a whole, not specifically to us.

## 3. When MSE is a near-optimal MDL approximation (and when it isn't)

The MSE-as-MDL approximation isn't *wrong* everywhere — it's exactly right under a specific assumption:

**MSE = MDL when residuals are IID Gaussian with known variance.**

For Gaussian residuals `r_i ~ N(0, σ²)`, the optimal lossless encoding has bit-length `~ N/2 · log(MSE) + const`. So minimising MSE *is* minimising bit-length-of-residuals under the Gaussian model. The Bayesian information criterion (BIC), Akaike information criterion (AIC), and minimum description length all collapse to "minimise MSE + complexity penalty" in this regime.

**Where MSE diverges from MDL:**

1. **Structured residuals.** If the residuals carry information the model failed to capture (e.g., class-correlation in MNIST after K=10 features; quadrant-shaped error after partial-fit IK), MSE flattens that structure into a single scalar. MDL would encode the residuals *as their actual structure* — a model that produces noticeably-structured residuals would be penalised more by MDL than by MSE.

2. **Non-Gaussian noise.** Financial returns are heavy-tailed; image classification errors are categorical (right class vs nearby class); robotics sensor noise is often multimodal. MSE assumes Gaussian; MDL would use the actual residual distribution's entropy.

3. **Sparse / outlier-dominated residuals.** Most residuals near zero with a few large; MSE dominated by the outliers. MDL with the right code would treat outliers as their own short codewords.

**Where tessera lives empirically:**

| Tessera benchmark | Residual structure | MSE-as-MDL approximation |
|---|---|---|
| Feynman (deterministic targets) | Residuals ≈ 0 by construction | Approximation tight (no structure to flatten) |
| MNIST 10-class via K=10 SR features | Residuals are CLASS-CORRELATED (e.g., 4/9 confusions) | Approximation LOOSE — MSE misses the confusion structure |
| 3-DoF planar IK after partial-fit | Residuals shaped by missing atan2 quadrants | Approximation LOOSE — residual structure is exactly the missing model |
| Trading PnL (`pnl_loss_hard/smooth`) | Residuals non-Gaussian, regime-dependent | Approximation VERY LOOSE — MSE wrong by design (we use pnl_loss instead) |

So Bowery's critique BITES most for MNIST/IK/trading and barely matters for Feynman. The tessera benchmarks where we've struggled most (MNIST 71% ceiling; IK Tier D) are exactly where MSE-as-fitness is most wrong.

## 4. What an MDL-compliant fitness would look like (the homework)

Bowery's homework assignment, made concrete for tessera:

### 4.1 Choose an instruction set

Pick a *reasonable* instruction set for encoding tessera's trees. Candidates:

- **Lambda calculus / SKI combinators**: minimum, axiom-aligned, computer-science-classical
- **Brainfuck-like**: pedagogically simple but inefficient for arithmetic
- **A small typed assembly**: more practical; pick a few opcodes (add, mul, branch, load, store, call) and count instruction bits
- **The tessera grammar itself**: simplest from a tessera-internal perspective, but then we're not really doing the homework — the "interpreter" is just the existing evaluator

A reasonable choice for tessera-research-purposes: a small typed lambda calculus with primitives for `+, -, *, /, sin, cos, ...` — each primitive has a fixed bit-cost equal to `log2(num_primitives_in_alphabet)`. This counts each op selection as `log2(K)` bits where K is alphabet size.

### 4.2 Encode the tree under the chosen instruction set

For a tree with `n` nodes, the bit cost is approximately:

```
bits(tree) ~ n * (log2(K) + log2(max_arity)) + n_const * bits_per_const_literal
```

where each node selects an op (log2(K) bits), some child slots may be lazy/eager (log2(max_arity) bits), and constants need their own literal encoding (32 bits for float32, or a custom prior on common values).

### 4.3 Encode the residuals as literals (the hard part)

Given `y_pred = evaluate(tree, X)` and `r = y - y_pred`, we want `bits(r)`. Options:

- **Gaussian assumption** (MSE-equivalent): `N/2 * log(MSE) + const`. Trivial.
- **Empirical entropy**: quantise `r`, estimate the distribution histogram, compute Shannon entropy. Better but still summary.
- **Arithmetic coding under a fitted model**: fit a residual model (e.g., Student's t, mixture of Gaussians), use its negative log-likelihood. Closer to AIT but computationally non-trivial.
- **Per-sample literal encoding**: literally write down each residual to the desired precision. Worst case (no residual model): `N * bits_per_sample`.

The Bowery position favours the latter end: if you can't compress the residuals, they belong as program literals at full precision.

### 4.4 Add them up

```
fitness_mdl(tree) = bits(tree) + bits(residuals_under_tree)
```

Compare against the trivial baseline `bits(no_model) = N * bits_per_sample + bits(zero_tree)`. A model is INFORMATION-USEFUL iff its total bit-cost is below the trivial baseline.

## 5. Why this hasn't been adopted as standard practice

Several honest reasons the SR community uses MSE despite Bowery's critique:

1. **Computability.** The "minimum-length interpreter" is *uncomputable* in general (Kolmogorov complexity is uncomputable). Practical approximations exist (BIC, NML, prequential) but each makes its own choices.

2. **Gaussian-residual assumption is often good enough.** For physics, engineering, controlled experiments — most data IS Gaussian-ish. MSE is fine there.

3. **Differentiability.** MSE has clean gradients; arithmetic-coding-based MDL doesn't. For const-opt loops (Levenberg-Marquardt, Adam), MSE is differentiable; many MDL approximations aren't.

4. **Computational cost.** Even cheap MDL approximations (BIC = `log(N) · cx`) cost more than naive parsimony. Per-sample residual encoding scales with N.

5. **It's not common knowledge in ML.** Bowery's critique of LeCun and Pearl is partly about this: even very senior ML researchers haven't engaged with AIT seriously.

None of these is a knock-down argument against doing the homework. They're reasons to *approximate* MDL rather than reject it entirely.

## 6. What it would mean for tessera specifically

A falsifiable experiment: replace tessera's fitness function with an explicit MDL approximation and see whether discovery improves on benchmarks where MSE was a poor proxy (MNIST, IK).

**Step 1 (cheapest MDL approximation):** swap `parsimony * cx` for `log(N) * cx / 2` (the BIC form). This costs ~1 line of code and gives a more theoretically-grounded complexity penalty.

**Step 2 (residual-model based):** replace `mse_loss` with `-log_likelihood` under a fitted residual model. For Gaussian residuals this is equivalent. For Student's t / mixture of Gaussians it's different.

**Step 3 (full per-sample literal encoding):** the Bowery-extreme. The fitness becomes the literal bit-count of the dataset given the model. Computationally heavy but the most honest.

**What we'd LEARN by trying this:**

- If step 1 alone (BIC-style complexity) closes the MNIST gap meaningfully, the existing `parsimony` value is just hand-tuned wrong; cheap fix.
- If step 2 (better residual model) closes more of the gap, the GP was discarding structured residual information.
- If step 3 changes the result substantially vs step 2, the residual distribution itself was being misspecified.

Each step is a falsifiable comparison against the current MSE baseline.

## 7. Practical concerns and open questions

1. **For Pareto-front maintenance**: tessera's HoF tracks (loss, cx) pairs. With MDL fitness, both axes become "bits" — the front collapses to a single number. Is the Pareto front still meaningful? Probably yes, since `bits(tree)` and `bits(residuals)` are independently controllable.

2. **For const-opt**: scipy.minimize and jax.grad both want differentiable losses. BIC-style is differentiable. Residual-model-based is differentiable if the residual model is. Per-sample literal is not. Trade-off.

3. **For the perfect-info game framing** (`fit_as_perfect_info_game.md`): an MDL fitness DOESN'T change the deterministic perfect-info nature. The landscape becomes `(tree, MDL_fitness)` instead of `(tree, MSE)`; both are deterministic; B&B still applies.

4. **Lower-bound oracle for B&B**: interval arithmetic gives bounds on `y_pred`. To bound `MDL_fitness` we'd need to also bound `bits(residuals)`. This is harder but not impossible (interval gives bounds on the residual distribution).

5. **Cross-benchmark generalisation**: if MDL improves MNIST but not Feynman, the right interpretation is "MSE is fine for low-noise-Gaussian benchmarks; MDL helps where residuals are structured." That's an empirically clean separation.

## 8. Connection to existing research notes

- `fit_as_perfect_info_game.md`: the perfect-info framing is UNCHANGED by switching fitness. MDL is a different loss landscape over the same state space.
- `network_sr_and_budget_allocation.md` §4: this doc's "deterministic admissible search" applies identically to an MDL landscape.
- `high_dim_symbolic_regression.md` §3 (theoretical analysis #2: inductive-bias mismatch): the MDL view *sharpens* the mismatch — if image residuals have structure that SR's pointwise alphabet can't represent, the MDL fitness will see that as a residual-bits cost and penalise the model accordingly.
- `benchmark_score_improvement.md` §4.3 (AI-Feynman separability detection): AI Feynman's symmetry tests are themselves an MDL-flavoured move — find structure in residuals, factor it out.

## 9. The honest verdict

The Bowery critique is **substantively right** for the benchmarks where tessera struggles most (MNIST, IK, trading). It is **substantively wrong as a knock-down argument** because computability + differentiability make full MDL implementation impractical; approximations are necessary.

The right tessera response is **NOT** "rewrite the entire fitness function" but rather: **try the cheapest MDL approximation (BIC-style)** and see if the empirical result changes meaningfully on a benchmark where MSE-as-fitness is suspect. If yes, invest in step 2. If no, the heuristic was fine.

This is the same falsification-criterion discipline used in the other research notes. We don't commit to Bowery's framing on philosophy; we test the *cheapest* implementable consequence and let the data decide.

## 10. Concrete next steps (status-flagged)

| Item | Effort | Status |
|---|---|---|
| Swap `parsimony * cx` → `log(N) * cx / 2` (BIC); rerun Feynman + MNIST | 1 hour + multi-hour runs | ○ PLANNED candidate |
| Add Student's-t residual loss as alternative to mse_loss | half day | ? RESEARCH |
| Full per-sample literal encoding as fitness (Bowery-extreme) | 1-2 days; needs arithmetic coding | ? RESEARCH |
| Document explicitly that tessera's current fitness is heuristic-MDL, not full-AIT | already done (this note) | ✓ DONE |

The first item is the cheapest test. If you direct it, it would be the next promotion to PLANNED.

## Changelog

- 2026-05-24: initial document. Provenance: user's Mean Mr. Jim (James Bowery) quote. Audited tessera's current fitness against the MDL ideal; identified MNIST/IK/trading as the benchmarks where MSE is most suspect; proposed BIC-style complexity penalty as the cheapest falsifiable test.
