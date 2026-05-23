# tessera roadmap — GP improvements, search-algorithm directions

Snapshot as of v0.1.2 (after const-opt polish step).

This document captures (1) what's missing in tessera's GP relative to
the best-in-class open-source SR engine (PySR), (2) what literature is
worth reading, and (3) where the next research directions are — including
Hamiltonian / Ising / annealing methods, which are a natural extension
of tessera's measure-theoretic core.

It is a living document — feel free to revise as priorities shift.

---

## 1. Gap analysis vs PySR

PySR is the most engineered open-source SR system (Cranmer, 2023). After
benchmarking tessera against it on BTC 1h, here's where the gaps are,
ranked by likely impact on TRAIN-loss-reaching:

| Feature | PySR | Tessera | Impact | Effort |
|---|---|---|---|---|
| `optimize_constants` | BFGS via Optim.jl every gen | ✓ (Nelder-Mead, every K gens, v0.1.2) | **HIGH** | done |
| Algebraic simplifier | `do_simplification` mutation | ✓ (v0.1.1) | medium | done |
| **Hall of Fame** (per-cx best ever, immune from mutation) | yes | **missing** | **HIGH** | medium |
| **Lexicase / loss-biased selection** | yes (`LossSelect`) | tournament only | **HIGH** | small |
| Multi-population + migration | P populations of N each, top migrate every K gens | single pop | medium | medium |
| Annealed mutation temperature | T schedule biases exploration → exploitation | fixed weights | medium | small |
| Operator weight scheduling | `do_simplification` weight increases late | fixed | low-medium | small |
| Adaptive complexity penalty | `parsimony` adjusted by population stats | fixed | low | small |
| Constraint system (max complexity per op, etc.) | yes | partial | low | medium |
| Custom loss in Julia (JIT) | yes (loss_function string) | py loss_fn callable | n/a (already there in Python) | n/a |

**Top three to implement next**, in priority order:

### 1.1 Hall of Fame (per-complexity protected store)

PySR maintains a Hall of Fame separate from the active populations: for
each complexity `cx`, the best-ever candidate at that cx is stored.
Mutations and tournament selection cannot destroy these. When the final
Pareto front is reported, it comes from the HoF, not the population.

**Why it matters:** A classic GP failure mode is to find a great
discovery at cx=8, then have the population drift to cx=15-20
"improvements" that overfit, losing the cx=8 winner forever. Without
HoF, the cx=8 might only appear by accident at the final generation
sampling. With HoF, it's guaranteed to survive.

**Implementation sketch** (tessera-specific):
```python
class HallOfFame:
    best_per_cx: dict[int, Candidate]  # cx -> best ever seen at that cx

    def update(self, candidate):
        cx = candidate.complexity
        if cx not in self.best_per_cx or \
           candidate.train_loss < self.best_per_cx[cx].train_loss:
            self.best_per_cx[cx] = candidate

    def pareto_front(self) -> list[Candidate]:
        # Run pareto_front over self.best_per_cx.values()
        ...
```
Update on every successful scoring; return from `GP.run` at the end.
~50 LOC. High impact, low risk.

### 1.2 Loss-biased / lexicase selection

Tessera's `_tournament` picks the best of K random candidates by
`fitness` (loss + parsimony·cx). PySR uses two stronger selection
strategies:

- **`LossSelect`**: Same idea but weighted by `loss_function` exponent —
  candidates with very low loss have higher probability of being picked
  as parent. Heavier exploitation pressure.
- **ε-lexicase selection**: Treat each TRAIN sample as a test case. A
  candidate "passes" sample t if its prediction is within ε of the
  median for that sample. Best-passing candidates win. This naturally
  preserves diversity — candidates that nail HARD samples (which most
  others miss) are kept even if their average loss is mediocre.

**Why it matters:** Plain tournament converges fast to a local minimum;
lexicase explores wider for longer. For PnL+flip-style losses where
the optimum involves complex sign-flip structure, the diversity helps.

**Effort:** ~80 LOC for both. Could be added as `selection_method`
GPConfig option.

### 1.3 Annealed mutation temperature

PySR's mutation operators have weights that change over generations.
Early: more exploration (large `subtree_swap` and `term_insert`); late:
more exploitation (more `constant_jitter` and `do_simplification`).
Implemented via a temperature parameter `T(gen)` that decays.

**Why it matters:** Fixed `OP_WEIGHTS` in tessera means a 50-gen run
behaves the same on gen 1 as gen 49. Many real searches benefit from
"settling down" toward refinement.

**Effort:** ~30 LOC. Multiply the weights by a per-gen schedule.

---

## 2. Reading list

In priority order for tessera-relevant topics.

### 2.1 GP / SR foundations

1. **Poli, Langdon, McPhee — "A Field Guide to Genetic Programming"** (2008). Free PDF
   at `gp-field-guide.org.uk`. The practical bible. Read chapters 1-5
   first; chapter 6 (Bloat) and chapter 9 (Real-life applications) are
   directly relevant to what we're hitting on BTC.
   *Already in `books/AFieldGuideToGeneticProgramming.pdf`.*
2. **Banzhaf, Nordin, Keller, Francone — "Genetic Programming: An Introduction"** (1998).
   The standard textbook. Heavier on theory; good for understanding
   schema theorem / Price's theorem / why crossover works.
3. **Koza — "Genetic Programming"** (1992) and "GP IV" (2003). Foundational
   if you want the original framing; long.

### 2.2 Const optimisation in SR (directly addresses our recent commit)

4. **Topchy & Punch (2001) "Faster GP based on Local Gradient Search of
   Numeric Leaf Values"** — the foundational paper showing why polishing
   leaf constants accelerates GP. Short, readable.
5. **Kommenda, Kronberger, Winkler, Affenzeller, Wagner (2013) "Effects
   of constant optimization by nonlinear least squares minimization in
   symbolic regression"** — modern treatment with Levenberg-Marquardt.
   Tessera currently uses Nelder-Mead because PnL+flip is non-smooth,
   but for smooth losses LM is much faster.
6. **Worm & Chiu (2013) "Prioritized Grammar Enumeration"** — an
   alternative to GP that builds expressions by dynamic programming,
   with constant optimisation built-in at every step. Worth knowing as
   an architectural alternative to GP.

### 2.3 PySR specifically

7. **Cranmer (2023) "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"**
   — `arxiv.org/abs/2305.01582`. The PySR paper. Read the algorithm
   section + the appendices.
8. **PySR source code**: `github.com/MilesCranmer/SymbolicRegression.jl`.
   Key files: `src/Mutate.jl`, `src/ConstantOptimization.jl`,
   `src/Population.jl`, `src/HallOfFame.jl`, `src/Migration.jl`.
   Mostly Julia but readable.

### 2.4 Multi-objective / Pareto-front methods

9. **Smits & Kotanchek (2005) "Pareto-Front Exploitation in Symbolic
   Regression"** — how to use the Pareto front not just as output but as
   parent pool. Tessera currently uses it only as output.
10. **Deb, Pratap, Agarwal, Meyarivan (2002) NSGA-II** — the canonical
    multi-objective evolutionary algorithm. The tessera Pareto-front
    builder is closely related.

### 2.5 Modern data-driven SR

11. **Schmidt & Lipson (2009, Science) "Distilling Free-Form Natural Laws
    from Experimental Data"** — Eureqa paper. Read for the
    fitness-predictor coevolution idea (might be useful to make tessera's
    search cheaper on large datasets).
12. **Udrescu & Tegmark (2020) "AI Feynman"** — different architecture
    (recursive decomposition + symmetry detection). Useful for
    physics-style problems.
13. **Cranmer, Sanchez-Gonzalez, Battaglia, Xu, Cranmer, Spergel, Ho (2020)
    "Discovering Symbolic Models from Deep Learning with Inductive Biases"**
    — frames SR as the explanation layer on top of GNN predictions. The
    inductive-bias framing motivates tessera's measure-theoretic core.

---

## 3. Hamiltonian / Ising / QUBO methods

You asked: *"are they purely pointwise? They use Hamiltonian. What
would be our next direction?"* Honest answer:

### 3.1 They're NOT pointwise — they're quadratic

A QUBO objective is `min_x x^T Q x` over `x ∈ {0, 1}^n`. The Ising
Hamiltonian is the same with `s ∈ {-1, +1}`:

$$H(s) = -\sum_{i<j} J_{ij}\, s_i s_j - \sum_i h_i s_i$$

This is a **second-order polynomial** in the variables — a *bilinear
form* plus a *linear form*. In tessera's vocabulary, that's exactly:

- The linear term `Σ h_i s_i` ↔ `LinearFunctional(measure_signed_sum)`
- The bilinear term `Σ J_ij s_i s_j` ↔ `SeparableBilinear` / `Volterra2`

So an Ising Hamiltonian is **already expressible in tessera's existing
expression algebra**. What it adds is a different *domain* (discrete
variables, binary or ±1) and a different *search procedure* (annealing
instead of GP).

### 3.2 Four directions where annealing/Hamiltonian methods could enter tessera

**(a) Replace Nelder-Mead with Simulated Annealing for const-opt**

Const-opt currently uses scipy's Nelder-Mead because PnL+flip is
non-smooth. SA is also gradient-free, has provable convergence to global
optimum (Geman & Geman 1984), and handles discontinuities cleanly. On
the BTC PnL benchmark — where Nelder-Mead got stuck — SA might find
better constant configurations.

*Effort: ~50 LOC. Just swap the optimiser. Low risk, contained.*

**(b) Use Simulated Annealing as the GP loop's search algorithm**

GP and SA are different metaphors for the same problem: search a
discrete state space (here, tree structures) by making local moves.
Cranmer's PySR is GP-style; you could equally implement an SA-style
search where:
- Current state = a single tree (not a population)
- Proposal = a tessera mutation
- Accept with probability `min(1, exp(-(L_new - L_curr) / T))`
- Cool T over time

Pros: simpler than GP; provable convergence theory; some problems are
better suited to SA than GP.
Cons: loses the diversity benefit of populations; can't parallelise as
naturally; harder to combine with Hall of Fame.

*Effort: ~200 LOC. Multi-day. Worth trying as a comparison to GP.*

**(c) QUBO formulation of tree structure search**

The most ambitious framing: encode tree-structure choices as binary
variables and solve as a QUBO problem. For each "slot" in a fixed-depth
tree template, a one-hot binary vector selects which operator goes
there. The objective is the loss of the assembled tree. Pairwise
interactions encode constraint satisfaction (e.g., "if slot 3 is a
BinOp, slots 7 and 8 must be valid subtrees").

If this works, you could solve on classical SA *or* on a real D-Wave
quantum annealer. Some recent papers do this for neural-architecture
search (NAS) on small problems.

Pros: maps to actual quantum hardware. Provides a principled framing.
Cons: tree-structure search is genuinely combinatorial, and encoding
constraints in QUBO is fiddly. Probably small-scale only.

*Effort: research-level. Multi-week. Interesting but uncertain payoff.*

**(d) Hamiltonian-style smooth losses**

Most trading losses (PnL+flip-rate) are non-smooth because of
`sign(prediction)`. A Hamiltonian-style relaxation would smooth this:

$$H(\text{tree}) = -\langle \text{tanh}(\beta\, \hat y) \cdot y_{\text{fwd}} \rangle + \lambda \langle \text{tanh}(\beta\,\hat y)^2 - 1 \rangle$$

where the second term softly pushes positions toward ±1 (like an Ising
double-well potential). As `β → ∞`, this recovers the PnL+flip loss; at
finite β it's smooth, so BFGS const-opt works.

Pros: makes const-opt much faster (BFGS vs Nelder-Mead) and might let
tessera find the flip-rich regime PySR found.
Cons: changes the loss function — must be careful not to change the
optimum.

*Effort: ~30 LOC for the loss function + benchmark comparison. Low risk.*

### 3.3 My recommendation for next direction

Three honest candidates ordered by tessera-relevance + impact:

1. **Option (b) from earlier conversation: indicator / threshold
   primitives.** Closes the *vocabulary gap* so tessera can natively
   express EML primitives (and Hamiltonian energies, since those involve
   binary indicators of state). High value, medium effort.

2. **Hall of Fame (§1.1).** Pure GP engineering, high impact, low risk.
   Probably the single most-impactful tessera improvement available
   right now.

3. **Hamiltonian-smooth losses (§3.2(d)).** A natural bridge from
   tessera's measure-theoretic core to QUBO/Ising methods, without
   committing to a full annealing search yet. Tests whether the BTC
   1h closure is the *loss landscape* (smooth vs non-smooth) or the
   *search algorithm* (GP vs annealing).

Doing **(2) → (1) → (3)** in that order would close the engine gap
vs PySR (HoF), then close the vocabulary gap (indicators), then open
a new direction (Hamiltonian losses) you can build a research program
around. Each takes 1-2 days.

If you want to jump straight to annealing methods, do **(3) first** —
it answers the most interesting question without committing to a full
architectural rewrite.

---

## 4. Long-term: tessera as a research workbench

Beyond the SR engine itself, tessera's measure-theoretic vocabulary is
useful for:
- **PDE discovery** (already shown in `benchmarks/run_heat_equation_discovery.py`)
- **Multi-asset signal extraction** (cross-section × time as 2D)
- **Aggtrade-microstructure** (time × trade-size-bin as 2D)
- **Hamiltonian / Lagrangian discovery** (energy functions of
  multi-dimensional state — adds to the SR-on-physics literature
  alongside AI Feynman)

These are 6-month directions. The roadmap above covers the next
2-4 weeks.

---

## Changelog
- 2026-05-24: initial document. Synthesis of PySR study + BTC 1h
  benchmarks (k_tessera_btc_1h*) + user request for QUBO/Ising direction.
