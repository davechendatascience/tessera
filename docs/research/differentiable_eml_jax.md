# Differentiable SR via a JAX EML super-graph (Conjecture D1)

**Goal.** A *scalable* symbolic-regression engine: JAX/GPU-native,
differentiable end-to-end, so each optimization step does O(P)
credit-assignment across all parameters at once — the property that lets
deep learning scale and that genetic programming structurally lacks
(GP's per-generation update carries ~O(1) bits of fitness signal,
regardless of expression size).

**Provenance.** Ported from `market-analysis/src/lib/diff_eml`
(PyTorch), itself built on Yarotsky's universal operator
`eml(x,y) = exp(x) − ln(y)` with grammar `S → 1 | eml(S,S)` (every
elementary function is a finite eml-tree). Architecturally this is
DARTS/EQL-style differentiable architecture search: a fixed-depth DAG
where each node holds a soft mixture over a static operator dictionary,
annealed from soft → hard, then snapped to a discrete program. The
eml-tree is the canonical *output* form; the forward pass uses standard
operator functions (faster, more stable than nesting exp/log).

## The central problem this track must solve: local minima

In market-analysis, **speed was not the bottleneck — local minima
were.** This is not incidental; it is the fundamental wall *relocated*.
Soft relaxation turns combinatorial structure search into non-convex
continuous optimization, and the non-convexity *is* the discreteness
reappearing. Concretely (visible in the torch supergraph):

- **Discretization gap (DARTS' core disease).** Early, with high
  temperature τ, each node's input is a near-uniform *blend* of all
  previous nodes. A blend fed into `mul`/`exp` is a function unrelated to
  any *hard* wiring, so the soft landscape's minimum need not lie in the
  basin of any discrete program. By the time τ hardens, the logits have
  settled, and `argmax` reads off a poor program.
- **Single init, no restarts** → one basin, no escape.
- **Entropy sparsity applied early** → premature commitment.
- **Hard `clamp` on exp/log** → zero-gradient dead zones where the
  optimizer stalls.

So "EML + better optimizer" does not dissolve the hardness; it *bets*
the relaxed landscape is more navigable than the discrete one. That bet
pays off in the **decomposable / low-order regime** (benign landscape)
and fails under high epistasis (local minima as bad as GP's plateaus) —
the same matched-constraint boundary as the rest of tessera.

## Conjecture D1

> A JAX EML super-graph with **parallel random restarts** (vmap'd over
> the GPU), delayed-sparsity + cosine-τ annealing, and smooth singular
> operators recovers low-order symbolic forms reliably; and **restart
> count trades GPU parallelism for local-minima robustness** — i.e.
> "scale buys robustness." Single-init success may be low, but
> best-of-R rises sharply with R at near-zero marginal wall-clock.

**Falsification.** If best-of-R hit-rate does not rise materially above
single-init on simple targets (`x²`, `sin(2x)`, `x₀·x₁`), or the engine
cannot recover them at all, the parallel-restart thesis is falsified for
the easy regime and the track needs the Tier-2 hybrid before it is worth
pursuing.

## Engineering plan (prioritized)

**Tier 1 — cheap, GPU-parallel (this module):**
- **vmap'd parallel restarts** — the headline lever. Local minima are
  init-dependent; run R inits in parallel, select by *hard-program* MSE
  (not soft loss). This is where scale directly buys robustness.
- **Annealing done right** — slow cosine τ; **delay sparsity** (λ=0
  early, ramp late) so the net fits before it must commit.
- **Smooth singular ops** — `div`, `inv`, `sqrt` as smooth surrogates
  (e.g. `a·b/(b²+ε)`), not `clamp`, so gradients flow everywhere.

**Tier 2 — the tessera-native hybrid (next, if D1 holds):**
- A *population* of EML graphs, each gradient-refined in parallel, with
  periodic GP-style recombination/mutation of the **hardened structure**
  (the non-local move gradients can't make), then re-soften + refine.
  Gradients do the O(P) continuous tuning; discrete moves cross basins.
- Simulated annealing on the discrete program as the barrier-crossing
  move (the energy/annealing work, correctly scoped — a structure-jump
  operator, not the main engine).

**Tier 3 — the conditional guarantee:**
- Run structure detectors (power-law, C8, coordinate-discovery) first;
  if they certify decomposability, **warm-start** the super-graph in
  that basin (detect-then-seed sidesteps the worst minima). This is
  where any scalability *guarantee* lives — conditional on structure,
  not generic.

## Update (2026-05-29): restarts are a DIAGNOSTIC, not the answer

First run: `x²` recovers trivially; `x0*x1` recovers by R=64; `sin(2x)`
recovers the exact form `sin(x0+x0)` only at R≥256 (per-init success
~1%). So restarts "work" — but this is the wrong kind of work.

**Why parallel restarts are not intelligent search.** GP uses fitness
to *guide* structural moves: good subtrees survive selection and
recombine via crossover — the population is cross-attempt memory.
Restarts have none: each init is independent (restart k+1 learns nothing
from restart k), and within a run the gradient guides the soft *blend*,
not the discrete pick. A ~1% per-init success rate is the signature of
blind search — 99% of inits land in a wrong basin and nothing learns
from them. "Scale buys robustness" is really "buy more lottery tickets":
fine at p~1%, useless when p→0 on real targets. The per-init success
rate is therefore a **diagnostic of landscape blindness**, not a method.

**Structure issue or optimization issue? — Primarily a structure-
paradigm issue that manifests as an optimization problem.** The
relaxation (softmax over operators) makes a bet: that the continuous
loss gradient is a good guide to the discrete optimum. It isn't, for a
structural reason — *the relaxation has no building-block operator.* It
cannot represent "x0+x0 is a good motif, preserve it and build on it";
it only slides logits over a fixed template, and softmax makes operators
co-adapt into good *blends* rather than good *discrete picks*. GP's
crossover IS a building-block operator. So the relaxation discards the
exact mechanism that makes GP intelligent — and no amount of restart /
annealing / basin-hopping recreates building-block intelligence from a
representation that threw it away.

**The blurry zone (honest):** some "optimization" interventions are
structure-representation fixes in disguise, and those *do* help —
Gumbel-softmax / straight-through makes the forward pass commit to a
*sampled discrete* program (closing the discretization gap); hard-
concrete L0 actually drives one-hot. These change what the gradient
sees. Pure restart/annealing do not.

### The three paradigms (the real design choice)

| | structure search | param tuning | intelligent? | differentiable in |
|---|---|---|---|---|
| **A. Relaxation** (current diff_eml; EQL/DARTS) | continuous logits + SGD | same logits | **No** (blind blend) | the structure logits |
| **B. Learned policy** (DSR) | discrete programs sampled from a learned controller, policy-gradient | — | Yes (policy learns from reward) | the controller |
| **C. Evolution + gradient** (hybrid) | GP evolves structure (building blocks) | gradients tune constants O(P) | Yes (fitness-guided) | the parameters |

- **A** is weakest on the intelligence axis. Gumbel/STE + L0 are the real
  (non-restart) upgrades, but they still add no explicit building blocks.
- **B** keeps structure discrete (no gap) and *learns* the search — a
  differentiable controller that gets smarter. GPU-scalable; but policy
  gradient is high-variance and the literature is mixed vs evolution.
- **C** fuses GP's intelligent structure search with gradient param
  tuning — the most direct answer to "GP evolves intelligently, diff_eml
  doesn't," and the most tessera-native (GP + const-opt already exist).
  Honest limit: its *structure* search is still GP, so it inherits GP's
  wall under high epistasis; what it fixes is making parameter tuning
  O(P) (scales to many params) instead of O(1).

**Decision pending (user).** Lane A-upgraded / B / C. The benchmark
`run_diff_eml_jax.py` is kept only as the landscape-blindness diagnostic,
not as a method. Restarts will NOT be the engine.

### Method D: structure search as a CSP (Knuth TAOCP 4F7, §7.2.2.3)

The sequential bottleneck is the discrete STRUCTURE search (the search
loop — GPU-immune; GPU only parallelizes the const-refinement across the
population). Knuth's *Constraint Satisfaction* fascicle is the efficient-
algorithm source for it, because **the structure search IS a CSP**:

- **Variables**: per node i, `(op_i, left_i, right_i)`.
- **Domains**: `op_i ∈ operators`; `left_i, right_i ∈` valid earlier slots.
- **Constraints**: wiring validity; arity (unary ⇒ canonical right);
  **connectivity** (the root must transitively depend on the inputs —
  forbids the constant-collapse nogood we watched A+ fall into,
  `sqrt(1.16)`); **symmetry-breaking** (commutative ops ⇒ `left ≤ right`;
  no `neg(neg ·)`; no dead nodes; canonical associative chains —
  eliminates the redundant equivalents GP/CEM re-sample endlessly);
  optional dimensional/type consistency; complexity bound.

Knuth's toolkit for solving it efficiently — **backtracking with
constraint propagation, dancing-cells (reversible sparse-set state for
O(1) backtrack steps), and dynamic variable-ordering heuristics**. Why
this is the right call, grounded in the fascicle:

- **Smart backtracking beats general/blind search on STRUCTURED
  problems.** Knuth's 3-coloring benchmark: plain backtracking
  (7.2.2.1X) beat the best SAT solver by ~600× because the problem is
  structured; and the *encoding* changed runtime by orders of magnitude.
  SR structure is highly constrained ⇒ the same lesson applies.
- **Completeness on bounded complexity** — backtracking will find the
  form if it exists within the node budget; GP/CEM/relaxation give no
  such guarantee. This is the home for the *conditional* scalability
  guarantee discussed earlier: Knuth's **tractable families / dichotomy
  theorem** rigorously characterize which CSPs are poly-time.
- **Unification**: the fascicle explicitly frames the **Ising model as a
  CSP** (statistical mechanics, p.4). So the E1 energy work and this
  structure search are the *same* formal object — CSP ⊇ {Ising, SAT,
  XCC, structure search}.

**Architecture (Method D):** a CSP backtracking *structure enumerator*
(symmetry-breaking + connectivity + dynamic ordering + dancing-cells,
host-side, sequential, smart) streams DISTINCT valid skeletons in an
intelligent order → the GPU `vrefine` batch-refines their constants in
parallel. Sequential search becomes systematic + pruned + complete
instead of blind; GPU does the parallel numerical work.

**Honest limit**: backtracking is still worst-case exponential in program
size. The win is pruning + symmetry-breaking (large constant factors) +
completeness on bounded structure + the right encoding — exactly where
Knuth shows it beats blind/general methods. It does not repeal the
exponential; it makes the *structured, low-complexity* regime fast and
guaranteed, which is the regime tessera's detectors certify.

## Status

- Status: **substrate built, restart-only optimization rejected**;
  awaiting paradigm decision (A-upgraded / B / C) before real
  optimization engineering.
- Graduation criterion: D1 holds on the three simple targets AND on a
  handful of Feynman equations, with best-of-R clearly beating
  single-init; then a head-to-head vs GP on speed + recovery.
- Removal criterion: parallel restarts don't beat single-init, or the
  engine can't recover the simple targets.
- Results: `benchmarks/results/diff_eml_jax.md` (after first run).

## Notes / open questions

- JAX chosen (user directive) over staying in torch: vmap makes the
  parallel-restart lever trivial, and it fits tessera's JAX investment.
- Forward uses a standard operator set; the eml-tree canonical-form
  rendering (snap-to-compressed) is deferred — orthogonal to the
  local-minima question, which is about the selection logits.
- Variable-depth: the template is fixed-depth; simple equations
  over-parameterize (harder to snap), complex ones may not fit. Open.
