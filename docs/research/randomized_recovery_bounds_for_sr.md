# Research note: randomized recovery bounds for symbolic regression

**Status:** ? RESEARCH. The bounds catalogued here exist in the literature; what's open is which of them transfer cleanly to tessera sub-problems and what their predictions would say about our empirical results.

**Provenance:** user (2026-05-25), after watching a video on what's solvable and what isn't with operator learning: *"there seems to be a lot of theorem available already with functional querying bounds [...] modern applied maths."* The video framed a randomized algorithm constructing Green's functions for a PDE class up to a probability. The follow-up direction was: draft a research note cataloguing these theorems and identifying which tessera sub-problems they apply to.

The motivating observation is sharp: **modern applied math has built mature probabilistic sample-complexity theory for linear and smooth operator settings.** Symbolic regression has resisted similar treatment because the search is combinatorial and the hypothesis class is discrete. But where SR's sub-problems reduce to the same shape — convex / low-rank / linear-algebra-flavoured — we get the bounds for free. We just have to recognize that we're in that regime.

This note catalogues the relevant landscape, identifies which tessera benchmarks live in which regime, and proposes a calibration experiment using `run_heat_equation_discovery.py` to see whether predicted and observed sample complexity agree.

---

## 1. The "what's solvable" partition

Operator learning's solvability landscape is sharply characterized:

- **Solvable with tight rates**: smooth operators on compact function spaces with low effective dimension. Polynomial sample complexity in 1/ε.
- **Solvable but rate is bad**: high-effective-dimension or low-smoothness operators. Curse of dimensionality on function spaces is real and worse than on ℝ^d.
- **Provably hard**: operators that depend on hidden state or have no smoothness; non-polynomial sample complexity.

The SR-side partition lines up:

- **Solvable with tight rates** (SR sub-problems that reduce to convex/linear/sparse-recovery): polynomial in 1/ε.
- **Solvable but rate is empirically slow**: general tree-search SR. We know representability is fine (Stone-Weierstrass) but search hardness blocks tight bounds.
- **Provably hard**: arbitrary Kolmogorov-complex targets in unrestricted vocabularies.

Tessera benchmarks distribute across all three rows. The question this note asks: **for benchmarks in row 1, can we cite published bounds rather than relying on empirical reassurance?**

## 2. Catalog of relevant theorems

The following are reference points, not faithful theorem statements. Constants and exact forms vary across versions; see the cited papers for precise statements. The point of the catalogue is the *shape* of each result.

### 2.1 Randomized NLA foundations

**Halko, Martinsson, Tropp 2011** — *"Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions"* (SIAM Review).

For a matrix A with effective rank k, randomized SVD recovers the top-k singular vectors to ε accuracy with probability 1 − δ using O(k + p) random projections (p = small oversampling). The error analysis is sharp; this is the toolkit Boullé-Townsend uses.

Relevance to SR: when a sub-problem can be cast as low-rank kernel recovery (e.g., Green's function with hierarchical off-diagonal low-rank structure), randomized SVD is the right tool.

### 2.2 PDE Green's function recovery

**Boullé & Townsend 2022** — *"Learning elliptic partial differential equations with randomized linear algebra"* (likely Foundations of Computational Mathematics or SIAM Review; arxiv:2105.06035).

For 2nd-order self-adjoint uniformly elliptic linear PDEs on a bounded Lipschitz domain in d dimensions, the Green's function G(x, y) can be recovered from O((log(1/ε))^{d+1} · ε^{-d}) input-output pairs (u_i, Lu_i) with probability 1 − δ. The algorithm exploits that G has hierarchically off-diagonal low-rank structure for elliptic operators.

**Boullé, Halikias, Townsend** — extensions to parabolic and certain other PDE classes.

Relevance to SR: this is **the** theorem the user's video framing pointed at. Tessera's `run_heat_equation_discovery.py` is a parabolic-PDE-discovery task; the parabolic extensions apply directly to our setting (with appropriate caveats about the discovery task being structural recovery, not just functional approximation).

### 2.3 Neural operator rates

**Lanthaler, Mishra, Karniadakis 2022** — *"Error estimates for DeepONets: A deep learning framework in infinite dimensions"* (Transactions of Mathematics and Its Applications).

For DeepONets approximating a Lipschitz operator G : X → Y between Banach spaces, error bounds in terms of the network width and depth. Effective dimension of the operator determines the rate.

**Kovachki et al. 2023** — Fourier Neural Operator universal approximation + rate bounds.

Relevance to SR: indirect. These bounds describe what neural operators can achieve; the SR-side question is whether symbolic operator approximation could match these rates given the right vocabulary. Open question.

### 2.4 Sparse recovery / compressed sensing

**Candes & Tao 2005-2007** — sparse signal recovery with random matrices satisfying restricted isometry property. O(s log(N/s)) samples suffice to recover an s-sparse signal in N dimensions with high probability.

**Schaeffer, Caflisch and others** — SINDy convergence theorems for sparse PDE recovery; conditions on the candidate library + noise level give exact recovery guarantees.

Relevance to SR: **the planned §4.4 SINDy-style hybrid GP** lives entirely inside this theoretical framework. When the target IS sparse in a precomputed basis, polynomial-in-1/ε sample complexity is guaranteed. This is the cleanest tessera/theory bridge.

### 2.5 Polynomial sample complexity for sparse basis fits

**Cohen, Davenport, Leviatan and others** — sample complexity for polynomial approximation in high dimensions. Best-N-term approximation rates from random samples.

Relevance to SR: directly bounds tessera's §2.3 (polynomial-basis sufficient stats). When polish discovers a polynomial fit, the sample complexity needed for that fit to *generalize* (not just minimize TRAIN MSE) is governed by these bounds.

## 3. Tessera sub-problems and theorem applicability

Crisp mapping:

| Tessera sub-problem | Theorem class | Applies? | Reference |
|---|---|---|---|
| Heat equation discovery (`run_heat_equation_discovery.py`) | Boullé-Townsend parabolic extension | **Yes**, with caveats — see §4 | §2.2 |
| General PDE rediscovery (elliptic/parabolic) | Boullé-Townsend family | Yes (with smoothness assumptions) | §2.2 |
| Linear-in-parameters hybrid GP (planned §4.4) | SINDy convergence + LASSO bounds | **Yes, cleanest match** | §2.4 |
| Polynomial-basis sufficient stats (§2.3) | Sparse polynomial recovery | Yes — applies to the GENERALIZATION question | §2.5 |
| Branch-and-bound pruning (§2.2 planned) | Randomized NLA for kernel approximation | Indirectly — the lower-bound computation is itself a kernel-bound problem | §2.1 |
| General tree SR (Feynman general subset, IK, MNIST) | None directly | **No** — combinatorial search, no smoothness or sparsity to exploit | — |
| Symbolic operator discovery with hidden state | None — open problem | No | — |

The pattern is sharp: **wherever the SR sub-problem reduces to a convex / low-rank / sparse-linear problem, the applied-math theorems give us bounds for free**. Where it stays combinatorial, the theorems don't bind.

## 4. The vocabulary-restriction perspective

The conventional framing of Boullé-Townsend etc. is "the function/operator lives in some smooth function space; we recover it to ε accuracy from O(...) samples." The tessera-shaped re-framing:

> We commit to a finite vocabulary V (the set of tessera primitives). Functions/operators representable in V form a SMALLER hypothesis class than "all smooth Green's functions on the domain." Under appropriate identifiability conditions, the SAME randomized algorithm should recover the V-representable target with at least as good sample complexity as the unrestricted version — often better, because the hypothesis class is smaller.

This is the SR analog of the dimensionality-reduction story in numerical PDEs: if you know the solution lives in a low-dimensional subspace, you can recover it with fewer samples than the ambient dimension would suggest. **Tessera's vocabulary is the analog of that low-dimensional subspace** — committing to it is a strong prior that should translate into improved sample complexity, even though we lose universal approximation in cases where the target isn't V-representable.

This re-framing is not currently in any paper I'm aware of. The cleanest way to operationalize it would be:

1. Pick a tessera benchmark with known target (heat equation discovery)
2. Compute the Boullé-Townsend predicted sample complexity for the unrestricted Green's function recovery
3. Measure tessera's actual sample complexity at the same accuracy
4. If tessera uses substantially FEWER samples, the vocabulary-restriction advantage is empirically real
5. If tessera uses MORE, the combinatorial-search overhead is dominating and we're underutilizing the prior

This is a falsifiable, well-defined experiment. See §6.

## 5. Where these theorems DON'T transfer

Honest boundary, because the failure modes matter as much as the successes:

- **Nonlinear operators with no convex structure.** Most SR targets are nonlinear in the parameters; the linear-algebra bounds don't apply.
- **Discrete-state / piecewise problems.** The IK Run-3 failure ("piecewise / modular structure can't be expressed in current ops") is exactly this case. No randomized linear algebra recovers a piecewise function.
- **Combinatorial structure learning.** Tree shape search is NP-hard in general; sample-complexity bounds don't bound *search time*. The applied-math theorems are about query complexity, not computational complexity, and SR's bottleneck is often the second.
- **Vocabularies that fail to span the target.** If `sin` is needed but not in V, no amount of samples helps. The Boullé-Townsend bounds assume the target lives in their hypothesis class.

This last point is particularly important. The bounds presuppose target ∈ hypothesis class. For SR with a restricted vocabulary, this is a *strong* assumption that's not always met. The IK Run 1 ("vocab gap") case shows what happens when it isn't.

## 6. Calibration plan — heat equation discovery as the test case

The most concrete proposed experiment:

`benchmarks/run_heat_equation_discovery.py` already runs PDE discovery. It uses a fixed sample size and reports whether the heat equation is recovered. To turn this into a calibrated theoretical check:

1. **Vary sample count** across multiple orders of magnitude (N = 100, 300, 1k, 3k, 10k, 30k).
2. **Run repeated seeds** at each N (5-10 seeds) to estimate the success probability.
3. **Plot success rate vs N** on log axes.
4. **Overlay Boullé-Townsend's predicted N(ε, δ) curve** for the parabolic case — the prediction is roughly O((log(1/ε))^{d+1} · ε^{-d}).
5. **Compare**: do tessera's empirical samples-to-success curve match the slope predicted by the theorem? If we use d=1 (1-D heat equation) and ε corresponding to our acceptance threshold, the predicted N is computable.

Three possible outcomes:

- **Match within constant factor**: theorem applies cleanly; tessera is empirically near the optimal sample complexity. We've validated the theorem-tessera bridge.
- **Tessera uses substantially FEWER samples than predicted**: vocabulary-restriction advantage is real. The note's §4 reframing is the right next move.
- **Tessera uses substantially MORE samples**: either we're failing to exploit available structure (the bounds expose an engineering opportunity), or the GP search dominates the sample complexity (combinatorial overhead, not query overhead).

Each outcome is informative. The experiment is ~1 day to implement (extend the existing benchmark with the sample-count sweep) and ~few hours to run.

**Acceptance criterion** for this note → planned conversion:

A `benchmarks/results/heat_equation_sample_complexity.md` report showing the success-rate-vs-N curve and the Boullé-Townsend overlay, with a clear verdict (match / vocab-restriction-better / search-overhead-worse). The verdict itself is the deliverable, not a specific number.

## 7. What else this note opens up

Once the heat equation calibration lands, three natural follow-ons:

### 7.1 §4.4 hybrid GP with explicit SINDy convergence guarantee

When the planned §4.4 hybrid GP ships (`docs/planned/roadmap.md`), we can cite the SINDy convergence theorem directly as the guarantee for the *coefficient-fitting* step. The *structure-search* step remains combinatorial and not theoretically bounded, but at least one half of the procedure carries a probabilistic guarantee.

### 7.2 Sample-complexity-aware budget allocation

`docs/research/network_sr_and_budget_allocation.md` proposes deterministic budget allocation across SR units. If each unit has a known target type (polynomial, linear-in-params, PDE Green's function, opaque), we could allocate samples proportional to the *predicted* sample complexity rather than uniform. Closing the gap between "what theory predicts" and "what budget we actually spend."

### 7.3 Vocabulary-restriction advantage formalized

The §4 re-framing — that committing to a finite vocabulary V gives better-than-unrestricted sample complexity when the target ∈ V — could be formalized into a published-style theorem. **This is genuinely a research direction**, not engineering. It would be the kind of result that gives tessera academic legs. Out of scope here; flagged for future intellectual investment.

## 8. Connections to existing notes

- `algorithmic_information_for_sr.md` — Bowery's MDL critique. The applied-math sample-complexity bounds are the *frequentist* version of what MDL describes Bayesianly. Both point at: shorter description / smaller hypothesis class → better generalization. Same intuition, different mathematical machinery.

- `analytical_delta_loss.md` — Regime B (sufficient-statistic precomputation) is exactly the convex/sparse sub-problem these theorems can bound. The 196× speedup measured there is computational; the present note's question is whether it ALSO has favourable sample complexity. Yes — that's the SINDy-style bound from §2.4.

- `network_sr_and_budget_allocation.md` — budget allocation framework. Sample-complexity awareness is a natural extension; see §7.2.

- `high_dim_symbolic_regression.md` — the curse-of-dim story. Randomized methods are one of the named §5 directions. The present note makes that direction more theoretically grounded.

- `fit_as_perfect_info_game.md` — sample complexity = "how many queries does the searcher need." The present note quantifies that, for the sub-problems where the answer is known.

- `benchmark_difficulty_and_climb_then_simplify.md` §3 — the difficulty score the user requested. Sample complexity from these theorems is a *principled* difficulty estimator for the subset of benchmarks they apply to (PDE discovery, sparse linear, polynomial). For Feynman/IK, no theoretical estimate is available; we fall back to the empirical proxies in that note.

## 9. Falsification

This note's central claim — "the catalogued theorems apply to specific tessera sub-problems and predict their sample complexity; the SR-side equivalents are less developed outside that class" — is falsifiable by any of:

1. **Finding a theorem with tight bounds for general structural SR** (would invalidate the "outside that class" partition). Possible but unlikely given current state of art.

2. **Finding that heat equation discovery doesn't actually satisfy Boullé-Townsend's hypotheses** (e.g., our discretization breaks the regularity assumptions). Would mean even the PDE benchmark is outside the theory. Would require careful check of the regularity conditions during the §6 calibration.

3. **Empirical sample complexity vastly off from predicted** (10× more or fewer samples) — would tell us either the conditions are violated, OR our implementation is far from optimal, OR the vocabulary-restriction advantage (§4) is real and dominant. The §6 experiment is exactly this falsification test.

If none of the three triggers, the note's claim is validated by the calibration experiment.

## 10. Concrete sub-ideas with effort estimates

In priority order:

| Item | Effort | Expected payoff |
|---|---|---|
| Heat equation sample-complexity calibration (§6) | 1 day | Empirical validation or falsification of theorem applicability |
| Cite SINDy convergence theorem in §4.4 ship docs | 0.5 day at ship time | Theoretical guarantee on coefficient-fitting half of hybrid GP |
| Sample-complexity-aware budget allocator | 2-3 days | Closer match between theoretical and engineering optima |
| Vocabulary-restriction theorem (formalize §4 reframing) | 1-2 weeks research | Publishable theoretical contribution; genuine research |

The first item is the cheapest and the highest-yield empirical test. Defer 2-4 until 1 produces a clear verdict.

## 11. What this note does NOT claim

- **That bounds exist for general structural SR.** They don't. The partition in §1 is honest about the no-bounds row.

- **That tessera's empirical methods are sub-optimal.** We don't know yet. The §6 calibration is what would tell us.

- **That these theorems make SR sample-efficient.** They apply to specific sub-problems. Tessera's broader value (Feynman, IK, MNIST features) sits outside their reach.

- **That the vocabulary-restriction advantage (§4) is theoretically established.** It's a re-framing that *would need* a new theorem to support. Listed as a future research direction, not a claim.

- **That this note is faithful to the exact constants and conditions of the cited theorems.** I worked from general knowledge of the literature; specific applications need a careful re-read of the original papers to confirm hypothesis conditions are met by tessera benchmarks.

The catalogue is the artifact; the calibration is the bridge; the partition is the honest framing of where theory ends and empirical work begins.

## Changelog

- 2026-05-25: initial document. Provenance: user's video-watching observation that applied math has mature sample-complexity theory for operator learning. Cataloged Boullé-Townsend, randomized NLA, SINDy convergence, sparse polynomial recovery. Mapped 7 tessera sub-problems to theorem applicability. Proposed heat-equation-discovery calibration (§6) as the cheapest falsification test.
