# Research note: high-dimensional symbolic regression

**Status:** ? RESEARCH. Open exploration, not committed to ship. Active direction as of 2026-05-24 following the MNIST experimental result.

**Provenance:** the K=10 MNIST run on 2026-05-24 produced 71.1% test accuracy with a 0.4pt train-test gap — the textbook signature of a *hypothesis-class ceiling*, not an overfitting problem. Adding training data won't help; the feature class is the bottleneck. This doc surveys why high-dim SR is hard, what others have tried, and where tessera's unique position offers a research opening.

---

## 1. The problem (anchored empirically)

**Tessera's MNIST result, 2026-05-24:**

| Metric | Value | Note |
|---|---|---|
| Train accuracy | 71.5% | K=10 one-vs-rest SR features + LogisticRegression |
| Test accuracy | 71.1% | Held-out 1K samples |
| **Train − test gap** | **0.4pt** | **Effectively zero** |
| CNN baseline | ~99% | Reference |
| Random baseline | 10% | 10-class chance |

The gap-to-CNN is 28pt. The gap-to-target (95%) is 24pt. **The zero train-test gap is the smoking gun:** the model is at the ceiling of what the feature class supports on this data size; this is not overfitting, it's underfitting from a hypothesis-class that's structurally smaller than CNN's.

This empirical anchor is consistent with the broader SR literature's struggle on high-dim data.

## 2. Cranmer / PySR's documented stance

Miles Cranmer (PySR author) has repeatedly stated that PySR is designed for *low-dimensional* problems (typically < 20 features) with smooth, low-arity formulas. The PySR documentation explicitly recommends:

- Use **dimensionality reduction first** (PCA, autoencoders) — find a low-dim representation, then run SR on that.
- For images / sensor arrays: **use neural-net-extracted features as input to SR** (the "neural-symbolic hybrid" position).
- The AI Feynman pipeline (Udrescu & Tegmark, *AI Feynman*, Science Advances 2020, arXiv:1905.11481) preprocesses physics-style data: factor out units, find separability, then SR on the residual.

**The implication for tessera:** if we want to compete on high-dim *without* a neural front-end, we are explicitly outside what the dominant SR systems target. That's either a research opportunity or a structural mistake. The MNIST result says it's at least partially the latter at single-feature scale; the open question is whether structural changes to the SR engine itself can close the gap before any neural front-end is added.

## 3. Recent literature (citations harvested via Perplexity 2026-05-24)

A sampling from 2021-2025 SR research relevant to high-dim:

| Direction | Representative work | What it does |
|---|---|---|
| **Deep Symbolic Regression (DSR)** | Petersen et al., 2021 (arXiv:2006.11287) | RL-style policy generates SR candidates; scales better than GP for low-D but still struggles past ~20 features |
| **Neural-Symbolic hybrids** | Biggio et al., *Neural Symbolic Regression that Scales*, 2021 | Pre-train a transformer on synthetic SR problems; use it to propose candidates |
| **CNN-then-SR pipeline** | Cranmer et al., *Discovering Symbolic Models from Deep Learning with Inductive Biases*, NeurIPS 2020 | Train a GNN/CNN, extract its message-passing functions, run PySR on those |
| **AI Feynman 2.0** | Udrescu et al., 2020 (arXiv:2006.10782) | Adds neural network as an oracle to suggest separable structure |
| **Mundhenk et al.**, 2021 | RL-warm-started GP seeding | Use a neural net to seed GP populations |
| **uDSR** (uniform DSR) | Landajuela et al., 2022 | Larger-scale extension of DSR; still not image-scale |
| **Symbolic Regression for Computer Vision** | sporadic attempts | No published SR pipeline reaches >95% on CIFAR-10 or competitive accuracy on standard CV benchmarks without a CNN front-end |

**Published accuracy ceilings for pure SR (no neural augmentation):**
- MNIST 10-class: no consistently-published numbers I could find. Anecdotal reports range from 70-85% depending on preprocessing.
- CIFAR-10: no credible published number for pure SR; the SR community treats CIFAR as out-of-scope.
- Tabular benchmarks (≤50 features): SR matches or beats trees on a subset (the AI Feynman benchmark, SRBench's Penn data), but degrades sharply past ~100 features.

**Theoretical analyses of why SR struggles with high-dim** (cited in survey papers but not always primary):
1. **Combinatorial explosion in tree space.** Even at modest depth, the number of syntactic trees over K features grows as roughly `O((|Ops| · K)^depth)`. For K=784 (MNIST flat), tree-space at depth=6 is enormous; the GP can't search it densely.
2. **Inductive-bias mismatch.** SR grammars favour smooth low-arity formulas (polynomials, transcendentals); image discrimination requires structured spatial operators (convolutions, pooling). PySR's standard alphabet doesn't natively express CNN-style filters.
3. **Loss landscape pathology.** High-dim regression typically has many degenerate-equivalent local minima (different feature subsets achieving similar accuracy); GP gets stuck in them. Lexicase-selection mitigates partially but not fully.

The literature converges on the conclusion that *pure* SR at image scale is fundamentally limited by (1) and (2). Solutions are nearly always to add a neural front-end (CNN/autoencoder) and run SR on its features.

## 4. The tessera-specific opportunity: Knuth-style search + GPU parallelism

Three observations frame what's novel here:

**4a.** Tessera has a unique structural commitment to **measure-theoretic primitives** (LinearFunctional, FunctionalOp2D, Volterra2). These ARE the convolutional / spatial operators that PySR lacks. Tessera's MNIST features at 71% accuracy DID discover Laplacian-like edge detectors and vertical-stroke detectors — the inductive-bias mismatch is partially addressed already. The remaining gap is in *composing* these primitives at scale.

**4b.** The broader SR community has not seriously combined **Knuth-style combinatorial search machinery** (branch-and-bound, equality saturation, BDDs/ZDDs, equivalence-class enumeration) with **GPU parallelism**. The two threads exist independently:
- Knuth's *Combinatorial Algorithms* (TAOCP Vol 4) targets serial CPU asymptotics. The classical algorithms (DLX, alpha-beta, BDD ops) are pointer-chasing-heavy.
- GPU parallelism dominates modern ML (transformers, CNNs) but operates on dense tensor algebra, not combinatorial enumeration.
- GPU-parallel implementations of *individual* Knuth-style algorithms exist (parallel SAT, parallel BDDs), but they're niche and not connected to SR.

**4c.** Tessera has shipped Tiers 1-3 of a GPU backend (May 2026), making per-eval cost cheap. The implementation is missing the **fewer-evals lever** — per the `fit_as_perfect_info_game.md` §12 self-criticism, our current FunctionalCache + B&B + simplify_canonical implementations are partially-realised versions of Knuth's tools.

**The synthesis:** tessera has the substrate (GPU + measure-theoretic ops) to run Knuth-style combinatorial-search algorithms at scales the SR community hasn't tried, on problems (high-dim sensor data) the SR community has mostly punted on. Whether this combination *closes the gap to CNN-class accuracy* is the open research question.

## 5. Scalable upgrade directions (not simple case-by-case fixes)

The user's framing 2026-05-24: "*I want scalable upgrades, not simple case-by-case fixes.*" This rules out:

- Adding more features per class (incremental quantity, not quality)
- Hand-tuning per-problem grammars (one-shot, not transferable)
- Adopting a CNN front-end (concedes the unique-pitch point)

Five directions that are *scalable* in the sense of "each one buys orders-of-magnitude more search reach, not constant-factor improvements":

### 5.1 ? GPU-parallel branch-and-bound on equivalence classes

**The idea:** evaluate INTERVAL BOUNDS (not full losses) for thousands of candidates in one batched GPU launch. Skip the full O(N) loss eval for candidates whose bound exceeds the Pareto incumbent.

**Why scalable:** the bound-check is cheap (~constant per candidate); the GPU vmap-batches it across the entire population at once. If 80% of candidates fail the bound check, we get a 5× reduction in expensive eval count *per generation*, on top of the per-eval Tier-3 speedup. The compound is multiplicative.

**Knuth anchor:** branch-and-bound + alpha-beta (TAOCP Vol 4B §7.2.2). On GPU: parallel-branch pruning.

**Tessera status:** scalar `prune_by_lower_bound=True` flag exists, off by default. The GPU-batched version is unimplemented.

### 5.2 ? Equality saturation with topology-canonical-form hashing

**The idea:** before each GP generation, run a fixed-budget e-graph rewrite saturation over the population. Identify equivalence classes; evaluate only one representative per class. Use GPU-parallel hashing to detect equivalents.

**Why scalable:** the `|E_K|/|T_K|` ratio (per the perfect-info game framing) determines theoretical speedup. Conservative estimate from the doc: `|E_K|` is 0.1-0.5× `|T_K|` for tessera's grammar — i.e., 2-10× fewer evals. The hashing scales O(K log K) with GPU; the saturation scales with rewrite-budget but is amortizable across generations.

**Knuth anchor:** equivalence-class enumeration, *Concrete Mathematics* chapter on counting, Pólya/Burnside orbit counting.

**Tessera status:** `simplify_canonical` is a partial implementation. Full e-graph (snake-egg or hand-rolled) is research-noted but not shipped.

### 5.3 ? ZDD-compressed enumeration of bounded-complexity trees

**The idea:** instead of *random* tree generation, enumerate all syntactically valid trees of complexity ≤ K via a ZDD (Zero-suppressed Decision Diagram). ZDDs compress sets of bitstrings polynomially when the set has structure. For tessera's grammar at K ≤ 20, the ZDD might be tractable in memory.

**Why scalable:** if the ZDD fits, we get *exhaustive coverage* of the search space at K=20, not random sampling. The "GP as random walk through tree space" gives way to "deterministic visit each canonical form once." Combined with §5.2's equivalence collapse, the visited count drops by another factor.

**Knuth anchor:** TAOCP Vol 4A and 4B fascicles on BDDs/ZDDs. *The Art of Computer Programming Volume 4 Fascicle 1: Bitwise Tricks and Techniques; BDDs* (2009).

**Tessera status:** not implemented. Open question: how large is the ZDD for tessera's grammar at K=10, 15, 20? Empirical measurement needed before committing.

### 5.4 ? Hierarchical / multi-resolution SR (two-layer composition)

**The idea:** discovered features at one scale feed into a SECOND-LAYER SR that composes them. The second layer's "variables" are the first-layer's outputs. CNN-like deep composition but at the symbolic level.

**Why scalable:** linear classifier on K features asymptotes; symbolic composition of K features doesn't (the search space at layer 2 is multiplicative on layer 1's quality). If layer 1 gives ~10 useful primitives and layer 2 composes them into combinations of ≤ 5, the effective feature space is roughly C(10, 5) ≈ 250 features without growing the input dimensionality.

**Knuth anchor:** none directly — this is closer to grammar-based program synthesis. But the *evaluation* of layer-2 trees becomes a perfect-info game over layer-1 outputs.

**Tessera status:** not implemented. Closest existing pattern is the materialize layer (cross-tree caching, May 2026) which is a degenerate one-layer-deep version.

### 5.5 ? Sparsity-inducing search distribution

**The idea:** for high-dim inputs (image flattened to 784 pixels, audio waveform), most discovered features should reference *a few specific positions*, not the full input. Bias `random_tree` to produce trees with Var references concentrated on a sparse subset.

**Why scalable:** the effective search space at sparsity-s is roughly `C(K, s) × (small_tree_space)` instead of `K^depth`. For K=784, s=10, this is 10^15 instead of 10^36 — gain of 21 orders of magnitude. Combined with the other levers, the effective search space becomes tractable.

**Knuth anchor:** combinatorial enumeration of k-subsets (TAOCP Vol 4A §7.2.1.3). The "knapsack-style" structure means we should explore *important* k-subsets via Knuth-style branching, not uniform sampling.

**Tessera status:** not implemented. `random_tree` currently has no sparsity prior.

## 6. The novelty claim (what tessera could publish)

Combined, the five directions above form a research program with a distinct position:

> *Symbolic regression via GPU-parallel Knuth-style combinatorial search over measure-theoretic primitives.*

Each piece exists independently in the literature:
- GPU-parallel SAT/SMT: yes
- BDDs/ZDDs in algorithm libraries: yes
- Equality saturation in compiler optimisation: yes (egg)
- Measure-theoretic operators in numerical analysis: yes

The combination targeting SR has not been published. The empirical question is whether the *combined* leverage (multiplicative across the five) closes the high-dim SR gap that has stymied the field.

## 7. Honest expectations and falsification criteria

The research doc would be dishonest without an explicit "what would falsify this." Three honest possibilities:

**A. The synthesis works (the optimistic case).** Combined leverage closes the gap to ~90-95% on MNIST 10-class with pure SR. This would be a notable result: the first pure-SR system to compete on a standard CV benchmark. Publishable as a methods paper.

**B. The synthesis improves but doesn't close (the modest case).** Reaches ~80-85% on MNIST. Still meaningful improvement over the 71% baseline, demonstrates the techniques work, but doesn't beat the dominant neural-symbolic hybrid position. Publishable as a benchmark / engineering paper.

**C. The synthesis hits the same hypothesis-class ceiling (the pessimistic case).** Even with all five levers, the *symbolic* hypothesis class can't represent enough of CV's required structure. The ceiling lives at ~75-80% regardless of search budget. This would be a *negative result* worth publishing: a principled limit on what pure SR can do, with the Knuth-GPU machinery as proof we tried hard.

Falsification criterion: if §5.1 (GPU B&B) ships and produces no measurable accuracy improvement at MNIST, that's evidence against the synthesis. We should reassess after each direction independently before committing to the full program.

## 8. Concrete next experiments (status-flagged)

| Item | Direction | Effort | Status |
|---|---|---|---|
| Enable `prune_by_lower_bound=True` by default for MSE; measure pruning rate | §5.1 | ½ day | ○ PLANNED |
| Batched B&B bound check on GPU | §5.1 | 2 days | ○ PLANNED |
| Sparsity-bias `random_tree` (configurable max Var references per tree) | §5.5 | 1 day | ○ PLANNED |
| Two-layer SR prototype (use K=10 MNIST features as inputs to a second SR) | §5.4 | 2-3 days | ○ PLANNED |
| Burnside-bound empirical experiment: count `\|E_K\|` for tessera grammar at K≤8 | §5.2 | 1-2 days | ○ PLANNED |
| ZDD size measurement for tessera grammar | §5.3 | 1 week (uncertain payoff) | ? RESEARCH |
| Equality-saturation prototype (snake-egg or hand-rolled) | §5.2 | 1-2 weeks | ? RESEARCH |

Suggested order: §5.5 (sparsity) and §5.1 (B&B default-on) first — cheapest, fastest feedback on whether scaling SR alphabets matters. If both yield ≥5pt MNIST improvement, commit to §5.4 (two-layer). The e-graph + ZDD directions are the most theoretically interesting but also the most uncertain in payoff.

## 9. Reading list (to be expanded)

- Cranmer et al., *Discovering Symbolic Models from Deep Learning with Inductive Biases* (NeurIPS 2020, arXiv:2006.11287)
- Udrescu & Tegmark, *AI Feynman: A Physics-Inspired Method for Symbolic Regression* (Sci. Adv. 2020, arXiv:1905.11481)
- Udrescu et al., *AI Feynman 2.0* (arXiv:2006.10782)
- Biggio et al., *Neural Symbolic Regression that Scales* (ICML 2021, arXiv:2106.06427)
- Petersen et al., *Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients* (ICLR 2021, arXiv:1912.04871)
- Knuth, *The Art of Computer Programming Volume 4A: Combinatorial Algorithms, Part 1*
- Knuth, *Volume 4B: Combinatorial Algorithms, Part 2* (the backtracking + B&B chapters)
- Knuth, *Dancing Links* (2000, arXiv:cs/0011047)
- Willsey et al., *egg: Fast and Extensible Equality Saturation* (POPL 2021, arXiv:2004.03082)

## Changelog
- 2026-05-24: initial document. Empirical anchor: MNIST 71% test acc with K=10 features. Theoretical anchor: the Knuth-style serial combinatorial search has not been combined with GPU parallelism in the SR literature. Five scalable upgrade directions enumerated; five planned + two research-only.
