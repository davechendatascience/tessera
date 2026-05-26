# Research note: benchmark difficulty and the climb-then-simplify path problem

**Status:** ? RESEARCH. Thinking-aloud document; not formalised. Captures two connected user observations from 2026-05-24:

> *"We can actually construct a raw difficulty score for each of our benchmark. Rough guess is how many operations we need to get to it. And that's for within the vocabulary. What if there's approximations that are real good just like how eml trees can describe most operations. So we'll know beforehand that IK is difficult."*
>
> *"If we have to get to cx=30 and then simplify to get to a cx=5 solution for IK, this will be hard."*
>
> *"The game theoretic approach is what if we can search multiple steps ahead and know the evaluations? This depends on our specific data, but that's where the math should come in."*

The IK rerun result (Tier-D-via-search-explosion documented in `network_sr_and_budget_allocation.md`) made the user notice these two threads. Both point at real failure modes that the current GP design doesn't address.

---

## 1. The raw difficulty score idea

For each benchmark, a heuristic difficulty number computed as:

- `D_min` — minimum tree complexity required to express the target *exactly* in the current vocabulary
- `D_approx` — minimum cx for an approximation within some tolerance ε
- `D_random` — expected cx of a *random* tree that achieves ε-tolerance under uniform random sampling (rough proxy: how many trees you'd have to draw to hit it by chance)
- `D_gap = D_random - D_min` — how much "search-space dilution" the GP faces

The user's framing: this isn't an official metric, but a working number. The point is to know **before** running SR which benchmarks are easy vs hard.

### 1.1 Rough difficulty estimates for our existing benchmarks

| Benchmark | D_min (current vocab) | Approximation quality | D_approx | Honest difficulty |
|---|---|---|---|---|
| Feynman I.12.1 `μ·Nn` | 3 (two Vars, one mul) | Exact | 3 | trivial — D_random small |
| Feynman I.6.20a `exp(-θ²/2)` | ~7 (`exp(neg(mul(θ, θ)) / 2)`) | Exact | 7 | easy after exp shipped |
| Feynman I.43.31 `kT/(6πηr)` | ~9 (4-arg inverse product) | Tight (need exact constants) | 9 | medium — search budget bound |
| MNIST per-feature (one-vs-rest) | unknown — likely > 50 for >95% accuracy | tessera found cx≈8-25 trees @ ~80% AUC | ~20 (approx) | **hard** — many possible features; no exact form |
| IK q2 = `acos((r²-2)/2)` | ~7 with acos | Exact | 7 | **hard for current GP** because of D_gap (see below) |
| IK q1 with atan2 | ~12 (two `atan2` calls + subtree wiring) | Exact | 12 | **hard for the same reason** |

The pattern: **D_min is often small (5-12 nodes), but D_random is huge** because random_tree's uniform op sampling makes the specific composition appear with vanishingly low probability. The IK benchmark's Tier-D-via-search-explosion is exactly this: D_min ≈ 7-12, but D_random (the expected cx the GP actually finds via random walk) is much larger.

### 1.2 Pre-flight value

If we knew D for each benchmark, the framework could:
- **Allocate search budget** proportional to D (easy problems get cheap runs; hard ones get aggressive search)
- **Choose strategy** by difficulty class (D_min < 10 → pure GP; 10 < D_min < 30 → templated-mutations or seeded init; D_min > 30 → multi-feature ensemble or two-layer SR)
- **Set acceptance criteria honestly** — for D_random > 10⁶, "Tier D" is the expected outcome of plain GP, and we'd know it before running

This converts "tessera attempts every benchmark with the same config" into "tessera adapts its strategy to the benchmark's expected difficulty." That's exactly the unit-architecture framing from `high_dim_symbolic_regression.md` §6, but informed by an empirical difficulty estimate instead of a hand-picked category.

## 2. The climb-then-simplify path problem

The user's other thread is sharper still:

> *"If we have to get to cx=30 and then simplify to get to a cx=5 solution for IK, this will be hard."*

Consider the IK q2 case. The clean form is `acos((r² - 2) / 2)`, cx ≈ 7. But the GP, exploring randomly, might construct an intermediate form like:

```
(((cos(q2_apx) + cos(q2_apx)) * (-0.5)) + 1.0)  ≈ cx=14
```

which is an algebraic re-expression of `1 - cos(q2_apx)`. From cx=14, a CHAIN OF MUTATIONS that involves:
- Recognising `1 - cos(x)` ~ `x²/2` via Taylor expansion → wraps in a `pow(..., 0.5)` to invert
- Spotting the `acos` structure indirectly
- Replacing the whole chain with `acos((r² - 2) / 2)`

The MUTATION PATH from `cx=14 algebraic` to `cx=7 acos` involves intermediate forms that are *more complex than either endpoint*. The GP under monotone parsimony has no incentive to explore those intermediate forms — they'd be immediately rejected by selection.

This is the **mutation-path problem**:

> *In a search space where the optimum is reachable only by going UP in complexity first, monotone-parsimony GP fails to find it. The optimum is "behind" a complexity ridge that the search can't climb.*

This is well-known in continuous optimisation (saddle points; need momentum to escape) and in evolutionary biology (Wright's fitness landscape; valleys between peaks require neutral drift). For SR with parsimony pressure, it's structurally the same.

### 2.1 Three known responses (none of which tessera currently uses)

**(a) Non-monotone parsimony schedule.** Start with NEGATIVE parsimony (reward complexity → explore deep trees) for the first 20-30% of gens. Anneal to positive (reward simplification). Lets the GP "climb the ridge" before being forced down.

**(b) Multi-modal Pareto.** Maintain multiple Pareto fronts in parallel ("exploration fronts" at different cx ranges). The GP works on the BEST candidate in each front. Some fronts are at cx=30; others at cx=5. Cross-front mutation lets discoveries flow between cx levels.

**(c) Free-move budget (the existing perfect-info game §11.2 framing).** Two-budget GP: EVAL budget for expensive moves, REWRITE budget for cheap simplifications. When the GP commits a candidate to eval, FIRST run a free simplification pass (algebraic equivalences, AC normalisation, e-graph saturation if available). This lets cx=30 candidates evaluate as their cx=5 simplified equivalents — same evaluation cost, but Pareto front sees the simpler tree.

(c) is the "tessera-native" answer — uses the existing simplify_canonical machinery on the eval input. The current GP only simplifies AT scoring time; it doesn't aggressively pre-simplify before mutation either. So a complex tree's mutation-children are also complex, and the only path to a simpler form is via op-swap or term-delete.

### 2.2 Why this is harder than it looks

For the IK case specifically: there's no straightforward simplification rule that takes `(1 - cos(q2)) / 2 → q2²/4` (which is the Taylor approximation). Algebraic identities operate on EXACT equivalences (like `X·1 = X`), not approximations.

So the climb-then-simplify failure is structural for non-algebraic targets. The ONLY way to find `acos((r²-2)/2)` from a `cos`-only approximation is to *replace* the cos-based subtree with an `acos`-based one — which is a structural mutation, not a simplification.

Three options:
- **Template-based mutations** (already in `benchmark_score_improvement.md §4.2`): explicit `template_acos_replace` that swaps a `cos`-shaped subtree with its `acos` inverse.
- **Equality saturation with approximate equivalences** (research): e-graphs typically use EXACT rewrites; an approximation-aware variant would record "X is *approximately* equivalent to Y" and let the search prefer the simpler Y when both fit the data.
- **Seed the initial population** with known-good IK formula shapes (`atan2`-based skeletons). Cheap but problem-specific.

## 3. Game-theoretic lookahead: when can we know an eval without computing it?

The user's third thread:

> *"What if we can search multiple steps ahead and know the evaluations? This depends on our specific data, but that's where the math should come in."*

This is the **chess-engine analog**. In chess, you can look ahead because:
- Position evaluation is cheap (~100 ns)
- The game tree is enumerable
- Heuristic evaluators (material + position) give reliable signal without full simulation

For SR-for-fit, can we look ahead? Three cases:

### 3.1 Cheap lookahead via interval arithmetic (already shipped)

`tessera.expression.interval` gives a LOWER BOUND on the loss without full O(N) eval. The bound is tight when the input intervals are narrow and the operators are monotone. For composite trees, the bound is conservative but cheap.

This IS lookahead: "I can KNOW the lower bound of evaluating this tree without paying O(N)." Branch-and-bound (per `network_sr_and_budget_allocation.md`) uses this.

### 3.2 Gradient-based lookahead (partial)

For a tree with parameters (constants), we can compute the GRADIENT of the loss w.r.t. parameters via jax.grad (shipped in `optimize_constants_jax`). This tells us how the loss would change for small perturbations to constants — a one-step lookahead in CONSTANT space.

But it does NOT lookahead in STRUCTURE space (you can't differentiate through `op = "mul"`).

### 3.3 Structural lookahead: the open problem

The user's "math" hint points here. Can we lookahead through MUTATION CHAINS?

For a simulated mutation `tree → tree'`, we'd want to estimate `loss(tree')` without computing it. Requirements:

- A **loss-landscape model** that interpolates between known evaluated trees
- A **mutation-effect model** that predicts what kind of trees a given mutation produces
- A **bound oracle** that's tighter than the interval bound (which is often loose)

Candidates that might work:
- **Gaussian process regression over tree-embedding space.** Embed trees in a vector space (e.g., learned embedding); fit a GP on (embedding, loss) pairs; predict new tree's loss via GP posterior mean + variance. This DOES require uncertainty (Bayesian); brings back some of the bandit machinery the user rejected — but it's a TOOL for budget allocation, not a stochastic-game framework.
- **Algebraic loss-surface model.** For specific operator classes (polynomial, rational), the loss surface has algebraic structure that can be exploited. AI Feynman's separability tests work this way: if `y = f(x1) · g(x2)`, detect it by sampling and exploit.
- **Symbolic differentiation of the loss landscape.** For pointwise targets, `∂L/∂tree` is well-defined w.r.t. the tree's parameters. For STRUCTURE changes, the loss landscape over tree-space has a combinatorial gradient (which mutation reduces loss the most?) — measurable empirically but not analytically.

The honest verdict: **lookahead in structure space is the open problem**. The community has tried (DSR, neural-symbolic hybrids) but no clean solution exists. The user's intuition is that "data-specific math" might solve this for particular problem classes (e.g., the algebraic structure of polynomial regression makes lookahead tractable; the algebraic structure of trigonometric IK does too, in principle).

## 4. Connecting the three threads

The three threads of this note connect:

```
Difficulty score (§1)   ─┐
                          ├──→  Adaptive search strategy
Climb-then-simplify (§2) ─┤
                          │
Lookahead (§3)           ─┘     (the unifying mechanism)
```

A tessera that **knows the benchmark's difficulty** (§1) can **choose its strategy** appropriately. A tessera that can **climb-then-simplify** (§2) is the right strategy for hard-but-reachable problems. A tessera that can **lookahead through mutation chains** (§3) is what makes the climb-then-simplify viable without exploding the eval budget.

The math the user pointed to ("but that's where the math should come in") is the LOSS LANDSCAPE STRUCTURE — for problems where the loss has algebraic structure that can be exploited analytically. This is the same family as Knuth's branch-and-bound (admissible heuristic from structural properties of the search space).

## 5. Five open questions

1. **Can we compute D_min for our benchmarks empirically?** Take the analytical formula for each Feynman/IK target; count its tree-cx under tessera's vocabulary; tabulate. Half day; gives the difficulty distribution we should design for.

2. **Does non-monotone parsimony (climb-then-anneal) close the IK Tier-D gap?** Run benchmarks/run_ik_planar_3dof.py with a parsimony schedule that starts at -0.005 (REWARD complexity) and anneals to +0.005. Half day.

3. **Does free-move budget (aggressive simplification before scoring) help?** Run the IK benchmark with simplify_canonical applied to EVERY candidate at every scoring call (more aggressive than the current "every K gens"). Cheap to test.

4. **Can we build a tighter loss lower-bound oracle than interval arithmetic?** Currently the interval-based bound is conservative because it loses dependency info between operators. A tighter bound would make B&B prune more aggressively. Research-class question.

5. **For specific problem classes, can we exploit algebraic structure for lookahead?** E.g., for polynomial regression, the loss surface IS the residual sum of squares of a polynomial fit; we KNOW its minimum analytically without GP. For SR over polynomial-rational functions, there's a closed-form best fit per topology. This is the "math" the user pointed to.

## 6. What this note doesn't propose

This is intentionally a thinking-aloud doc, not a planned-item factory. The user has noted they're "running out of new ideas to do the optimization part with a mind of perfect information game related concepts" — meaning further theoretical work has diminishing returns. The right move is empirical:

- Test (Q2 above) first — non-monotone parsimony schedule is ~30 LOC and a multi-hour rerun
- Then (Q3) free-move budget — implementing means changing the scorer call site
- Then (Q1) the difficulty tabulation — gives us a clearer picture of which benchmarks to invest in

The deeper theoretical work (Q4, Q5) is a multi-month direction; not blocking anything urgent.

## 7. Connection to existing research notes

- `fit_as_perfect_info_game.md` §11.2 (eval-vs-rewrite two-budget) — this note's §2's "free-move budget" is the same framing applied to the climb-then-simplify problem.
- `network_sr_and_budget_allocation.md` §4 — this note's §3 fits the "deterministic admissible search under budget" answer; the algebraic structure IS the admissible heuristic.
- `high_dim_symbolic_regression.md` §6 (unit-architecture) — the difficulty-adapt strategy in this note's §1.2 is one route to network-SR: pick the unit by D.
- `benchmark_score_improvement.md` §4.2 (template mutations) — this note's §2.2 promotes that direction empirically.

## Changelog

- 2026-05-24: initial document. Captures three connected user threads (difficulty estimation, climb-then-simplify path problem, lookahead math) raised in the wake of the IK Tier-D rerun. Acknowledges the user's framing that further perfect-info-game *theoretical* work has diminishing returns and the next moves are empirical.
