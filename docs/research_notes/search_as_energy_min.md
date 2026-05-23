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
