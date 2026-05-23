# Research note: SR-for-fit as energy minimisation with full data information

**Status:** open research direction, not yet implemented. Captured 2026-05-24.

## The reframing

Classical search problems split along an information axis:

- **Adversarial / partial information** (chess, poker): you can't see the
  opponent's plan; minimax-style algorithms guard against worst-case
  responses. The "game" is intrinsically about uncertainty.
- **Optimisation / full information** (SR-for-fit on a fixed dataset):
  every candidate expression has a *deterministically computable* loss
  on the data. There's no hidden adversary. The "search" is energy
  minimisation over a known landscape.

These two regimes look superficially similar — both involve a state
space, a step / move, and a fitness function. But the second admits
exploits the first cannot:

1. **Subexpression caching.** Same subtree on same data → same value.
   In an adversarial game positions never repeat in a useful way; in
   fit-to-data, repeated work is free to skip.
2. **Algebraic equivalence as free budget.** Two expressions that
   compute the same function on every input have identical loss.
   Rewriting a 50-node tree to a 1-node equivalent before evaluation
   converts mathematical knowledge into runtime savings.
3. **Provable lower bounds (branch-and-bound).** With the data fixed
   you can sometimes prove a candidate CANNOT beat the current best
   without evaluating it (e.g., if the tree's range is bounded and
   the resulting loss-lower-bound exceeds the incumbent's loss,
   prune). Adversarial games allow no such guarantee.

The framing "we have full information from the data because we want
to fit the data" lets us treat SR as **energy minimisation under a
move budget**, where every algebraic-equivalence rewrite is **budget
recovered**.

## Where simplification fits in this framing

Tessera's current `simplify()` is a small finite term-rewriting
system: ~15 hand-coded identities (`X-X→0`, `X*0→0`, `X/0→0` under
safe-divide, double-negation, constant folding, etc.). It costs ≪1
operation per node and saves the evaluation of the collapsed
subtree.

The state-of-the-art for compiler optimisation has moved well past
hand-coded rewrite systems:

- **Knuth-Bendix completion** (1970) — turns a set of equations into
  a confluent terminating rewrite system. Gives normal forms.
- **Gröbner bases** (Buchberger, 1965) — normal-form reduction for
  polynomial ideals. The right tool for polynomial SR.
- **E-graphs and equality saturation** (Willsey et al., 2021; the
  `egg` Rust library) — represent ALL equivalent expressions in a
  compact congruence-closure data structure, then extract the
  *cheapest* one for a given cost function. Used in production by
  Cranelift, the WebAssembly compiler, and Herbie (floating-point
  precision).
- **SAT/SMT for equivalence** — z3 / cvc5 can prove two expressions
  are equivalent in the theory of reals (or any decidable
  fragment). Slow but provably correct.

**SR-specific use of these:** essentially nobody is using
equality-saturation-style simplification in SR engines today. PySR's
`do_simplification` is hand-coded; AI Feynman uses SymPy
(rule-based, not e-graph-based). The closest published work is
WSINDy on PDEs, which uses moment-matching rather than algebraic
equivalence. There's a real research-paper-shaped hole here.

## Why this is "elegant"

The user's framing — "where you were spending X operations, now
you're spending 1" — is exactly the elegance metric a working SR
engine implicitly optimises. Every rewrite that COLLAPSES a tree
is one piece of mathematical knowledge converted to one unit of
evaluation budget. The Pareto front in (complexity, loss) is the
*observable* image of this: trees that exploit more identities sit
further to the left at the same loss level.

For trading-style SR specifically, where the loss is non-smooth
(PnL+flip) and the search budget is large but finite, **budget
spent on equivalent-by-algebra subtrees is wasted budget**. A
better simplifier could:

- Reduce the effective search-space cardinality (fewer distinct
  trees to evaluate, more cache hits)
- Improve the Pareto front's interpretability (no `(x-x)/(x-x) +
  abs(y)` ghosts hiding the actual structure)
- Lower the cost per generation, freeing budget for deeper search

## Concrete research directions

1. **Tessera-egg integration.** Wrap the Rust `egg` library (or a
   pure-Python equivalent like `quiver`) so tessera trees can be
   reduced to normal form via equality saturation. Cost function =
   tessera's complexity measure. Expected impact: 10-30%
   reduction in effective tree size on real-world Pareto fronts.
2. **Polynomial normal form.** For polynomial subexpressions
   (no `tanh`, `abs`, etc.), use Gröbner-basis reduction
   (SymPy can do this) before scoring. PDE discovery benefits
   most.
3. **Branch-and-bound on Pareto fronts.** Before evaluating a
   high-complexity candidate, compute a loss lower bound from its
   structure. If the bound exceeds the current Pareto-frontier's
   loss at the candidate's complexity, prune.
4. **Provable equivalence checking with z3.** Optional, slow,
   useful for verifying that a discovered expression matches a
   target physics equation symbolically.

## Reading list

- Willsey, Nandi, Wang, Flatt, Tatlock, Panchekha (2021)
  *egg: Fast and Extensible Equality Saturation.* POPL.
- Tate, Stepp, Tatlock, Lerner (2009) *Equality Saturation:
  A New Approach to Optimization.* POPL.
- Buchberger (1965, thesis) *Gröbner basis algorithm.* Standard
  treatment in Cox-Little-O'Shea, *Ideals, Varieties, and
  Algorithms* (2015, 4th ed.) chs. 2-3.
- Knuth & Bendix (1970) *Simple word problems in universal
  algebras.* (foundational; original term-rewriting paper.)
- Panchekha, Sanchez-Stern, Wilcox, Tatlock (2015) *Automatically
  Improving Accuracy for Floating Point Expressions* (Herbie —
  uses equality saturation for numerical accuracy).

## Perplexity research query

Paste-able prompt to dig further:

> In symbolic regression (SR), what is the state of the art for
> algebraic simplification of candidate expressions during search?
> Specifically: (1) equality saturation and E-graphs (egg, quiver,
> Willsey et al. POPL 2021) — has anyone integrated these into
> a working SR engine? (2) Knuth-Bendix completion for confluent
> term rewriting in SR — papers, implementations? (3) Gröbner basis
> reduction for polynomial subexpressions during SR — used in
> Eureqa, PySR, AI Feynman, Operon, or any other engine?
> (4) Branch-and-bound pruning using algebraic lower bounds, given
> that SR-for-fit has full information from the dataset (the
> "energy minimisation with known landscape" framing) — is there
> any work exploiting the fact that the search problem is NOT
> adversarial? (5) Any results quantifying the "budget recovery"
> tradeoff — i.e. how much evaluation budget is freed by better
> simplification, measured on standard SR benchmarks like the
> Feynman dataset or SRBench? Bonus: what's the deepest theoretical
> connection between term-rewriting systems and the parsimony
> regularisation term in SR (Occam's razor, MDL, structural risk
> minimisation)?

This is a multi-part research question; perplexity_research is
probably the right tool (vs perplexity_ask) because it'll dig
through multiple papers.

## Findings from perplexity_research (2026-05-25)

Ran the query above with `perplexity_research` at `reasoning_effort=medium`.
Synthesised findings, with the answers grouped by question:

### Q1 — Equality saturation / E-graphs in SR

**No widely-used SR engine integrates `egg` or equivalents in production.**
Eureqa, PySR, AI Feynman, Operon all use shallow hand-coded rule sets,
sometimes via a CAS bridge (SymPy). A few research prototypes have
applied equality saturation to small-scale SR-like problems and shown
it can compress the search space and find simpler final models, but
these have not reached the maturity or evaluation breadth of standard
benchmarks (Feynman / SRBench).

Two adoption blockers:
1. **E-graph size explosion** when many AC/D rules are active. Saturation
   can cost more than the SR loop it's embedded in unless heuristically
   scheduled.
2. **Cost-agnostic extraction.** Equality saturation finds equivalence
   classes; choosing the BEST representative requires a cost function.
   In SR that cost is data-dependent (loss + parsimony), and propagating
   it into the e-graph extractor is non-trivial.

### Q2 — Knuth-Bendix completion in SR

**Not used in production SR.** Theoretical appeal but two practical
problems:
- Completion may not terminate (or produces an unwieldy TRS) for the
  full theory of real arithmetic + elementary functions.
- A small hand-crafted rule set is "good enough" in practice — the
  marginal gain from formal convergence rarely justifies the cost.

Research-prototype level work embeds SR in theorem-proving environments
(Maude, ELAN) that support completion, but they're domain-restricted
to algebraic specifications, not data-driven SR.

### Q3 — Gröbner bases in SR

**Not used in mainstream SR engines.** Used in adjacent fields:
- Algebraic system identification — finding *implicit* polynomial
  constraints `p(x_1, ..., x_d) ≈ 0` from data
- Invariant / conserved-quantity discovery
- SINDy-adjacent sparse regression on polynomial bases

For *explicit* y = f(x) SR (the standard formulation), Gröbner bases
are too expensive to run per-candidate. They'd justify themselves in
a hybrid where polynomial subtrees get reduced separately, but no
mainstream engine does this.

### Q4 — Branch-and-bound exploiting full data info

**Explicitly identified as a "significant research opportunity"** by
the report — the user's framing is real and underexplored. Existing
work:
- MIP formulations of SR / feature selection that use branch-and-bound
  on the combinatorial structure choice
- Lower bounds for linear regression error are sometimes used to prune
  feature subsets

What's missing:
- Generic **dataset-informed algebraic lower bounds** on the loss of
  large structural classes of candidate trees
- Interval-arithmetic / convex-relaxation / Lipschitz-based bounds
  applied per-candidate as a pre-filter before full eval
- Branch-and-bound integrated with the GP/SA outer loop

This is the most actionable gap from the report.

### Q5 — Budget recovery measurement

**Largely missing from the literature.** Feynman / SRBench report
accuracy + expression size + runtime but rarely break down evaluation
count vs simplification effort. A few studies confirm "turning off
simplification → bloated populations" qualitatively, but no
systematic measurement of how much budget more aggressive
simplification recovers.

This is itself a paper-shaped hole: a benchmark study that
*quantifies* budget recovery from progressively more sophisticated
simplification (none → tessera's hand-coded → SymPy → e-graph) on a
fixed SR task.

### Bonus — TRS ↔ parsimony connection

The deepest theoretical link: a confluent TRS defines equivalence
classes of expressions and a canonical representative for each.
Parsimony regularisation picks the minimal-cost representative —
which is exactly what MDL / Kolmogorov-approximating complexity
penalties try to do.

**Operational implication for tessera:** the parsimony penalty
should ideally be computed on the *normal form* of a tree, not the
raw tree. Without that, parsimony is "distorted by arbitrary
syntactic differences" — `a + b` and `b + a` get different fitness
even though they're semantically identical.

Tessera's current `simplify()` is a partial canonicalisation. AC
normalisation (sort children of associative-commutative ops by a
total order) would close most of this gap with ~30 LOC and is a
strict improvement over the status quo.

## Concrete next experiments for tessera

Ranked by tractability + alignment with the energy-min framing:

### Exp 1: AC normalisation (commutativity + associativity)

**Cheap, high-value, no external deps.** Extend `simplify()` to:
- Sort children of `add`, `mul`, `min`, `max` by a total order
  (e.g. `complexity(child)` then `str(child)`)
- Flatten nested associative ops: `add(add(a, b), c) → add(a, b, c)`
  (needs n-ary add/mul; currently tessera only has binary)

The flatten step is the bigger refactor. Without it, just SORTING
children of binary ops makes `a + b` and `b + a` identical in the
cache + Pareto front. That alone should improve hit rates and reduce
"duplicate" tree pollution.

Estimated impact: 5-15% wall-clock saving from cache hit improvements;
cleaner Pareto fronts.

Effort: ~50 LOC (sort), ~200 LOC (full n-ary flatten).

### Exp 2: Dataset-informed lower-bound pruning

**Direct test of the user's "energy minimisation with known landscape"
framing. Validated by perplexity as a significant gap.**

For each candidate tree:
1. Compute interval bounds on `y_pred` via interval-arithmetic
   evaluation of the tree on the data's range.
2. From `(y_pred_lo, y_pred_hi)` and known `y_true`, compute a
   loss lower bound. For MSE: `lower_bound = Σ_i max(0, dist(y_pred_interval_i, y_true_i))²`
3. If `lower_bound + parsimony * cx > current Pareto front loss at cx`,
   skip full evaluation.

Measure:
- Fraction of candidates pruned
- Wall-clock saved
- Pareto front equivalence (must produce identical final fronts)

Effort: ~200 LOC. Tessera-internal, no external deps.

This is the most aligned with the "perfect information" framing — it
explicitly converts dataset knowledge (input range, target) into
search-space pruning.

### Exp 3: Selective Gröbner reduction on polynomial subtrees

**Validated by perplexity as a hybrid pattern.** Detect subtrees that
are pure polynomial (no `tanh`, `abs`, `sign`, FunctionalOp, indicator
ops). For each:
1. Convert to SymPy
2. Call `sympy.expand` + `sympy.simplify` + optional `sympy.groebner`
3. Convert canonical form back to a tessera tree

Effort: ~150 LOC + optional sympy dep.

Trade-off: SymPy conversion is slow (10-100ms per tree). Only worth
applying to LARGE polynomial subtrees, not every tree.

### Exp 4: snake-egg / egglog wrapper

**Highest ceiling, highest risk.** Use a Python wrapper around
egg / egglog to do real equality saturation on tessera trees.

Effort: ~400 LOC + external dep (snake-egg or egglog). Need to write
all the rewrite rules for tessera's BIN_OP_FNS / UN_OP_FNS / measure
algebra.

Pays off mainly if Exps 1-3 are insufficient — start with cheaper
options first.

### Recommendation

Do **Exp 1 (AC normalisation)** first — it's the cheapest, has
guaranteed semantic correctness (sort is unambiguous), and reduces
search-space cardinality measurably. Pair with **Exp 2 (lower-bound
pruning)** as the bigger experiment — that's the operationalisation
of the user's framing and the perplexity-validated "significant gap"
direction.

Skip Exp 3 / Exp 4 until Exp 1 + Exp 2 have been measured and we
know how much budget they recovered.

## Changelog
- 2026-05-24: initial document
- 2026-05-25: perplexity_research findings added, 4 concrete
  experiments proposed (AC normalisation, lower-bound pruning,
  Gröbner subtree, e-graph wrapper)
