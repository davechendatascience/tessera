# Framework synthesis: how every tessera piece maps to the Knuth-grounded perfect-info game

**Status:** living document. Written 2026-05-25 in response to:
*"how do we integrate Knuth's work with our diverging implementations?"*

The answer is **the implementations aren't actually diverging — they're
each filling a slot in the same framework.** This doc makes that mapping
explicit so the picture stops feeling scattered.

## The framework, in one paragraph

SR-for-fit is a single-agent perfect-information game (Knuth-style
combinatorial search). The player navigates an expression-tree space
to minimise loss + parsimony, with a finite move budget. Three
structural properties (F1: perfect info, F2: decomposable evaluation,
F3: algebraic equivalence) distinguish it from chess. The two
load-bearing techniques are **branch-and-bound** (TAOCP Vol 4B §7.2.2)
and **equivalence-class collapse** (canonical forms). Sophisticated
exploits add axis-semantic constraints, measure-algebra identities,
and group-quotient reasoning (Burnside) on top.

## The map

Every tessera component shipped or sketched falls into one of seven
framework roles. Listing every implementation against its role:

### Role 1 — The state space (expression trees)

The discrete space the player navigates.

| Shipped | Implementation |
|---|---|
| `tessera.expression.tree` Node types | The state representation |
| `random_tree` | Initial-state sampler |
| `mutate` operators | Move set; defines the state-transition graph |
| `validate_tree` | Move legality check |

This is the *board* and *legal moves* layer. Comparable to chess's
piece + move rules, except the move set is much richer (subtree swap,
crossover, op swap, measure mutation, etc.).

### Role 2 — The evaluation function (the "loss")

How positions get their numerical score.

| Shipped | Implementation |
|---|---|
| `evaluate(tree, env)` | Pointwise tree walker |
| `tessera.expression.measure.Measure.apply` | Convolution semantics |
| `tessera.expression.functional` | Bilinear / Volterra wrappers |
| `tessera.search.losses` | mse_loss, _prediction_is_valid |
| `tessera.search.losses_trading` | pnl_loss_hard, pnl_loss_smooth (Hamiltonian) |

This is the **F2 (decomposable evaluation)** property realised. Each
loss factors per-sample; tree eval is O(N) in data size. Comparable
to a chess heuristic evaluator — except F1 (perfect info) means the
result is deterministic, not adversarial.

### Role 3 — The search strategy (the player)

How the player chooses moves.

| Shipped | Implementation |
|---|---|
| `tessera.search.gp` | Population-based ES (μ+λ) with tournament |
| `tessera.search.sa` | Single-state Metropolis annealing |
| `tessera.search.random_search` | i.i.d. baseline |
| `optimize_constants` | Inner-loop scipy on Const leaves (PySR-style polish) |

Three different search strategies sharing the same state space +
evaluation. Reasoning behind each in the Knuth taxonomy:
- GP ↔ evolutionary / population-genetic style
- SA ↔ Geman-Geman convergence in probability
- Random ↔ baseline (any directed search should beat it)

### Role 4 — Branch-and-bound (the framework's load-bearing technique)

Skip provably-suboptimal candidates without evaluation.

| Shipped | Implementation |
|---|---|
| `tessera.expression.interval` | Sound interval arithmetic |
| `measure_l1_norm`, `measure_2d_l1_norm` | Hölder-style bounds for LinearFunctional / SeparableBilinear / Volterra2 / FunctionalOp2D |
| `tessera.search.bounds.mse_lower_bound` | Closed-form MSE bound from prediction interval |
| `tessera.search.bounds.pareto_threshold` | Incumbent at each complexity |
| `GPConfig.prune_by_lower_bound` | Wired into `_score` |

This is **Knuth's TAOCP Vol 4B branch-and-bound** specialised to
tessera. The L1-norm bound (step c) was the empirical breakthrough —
unbounded fraction dropped 47% → 19% on the tightness benchmark.

### Role 5 — Equivalence-class collapse (the framework's other big lever)

Canonical forms shrink |E_K| relative to |T_K|.

| Shipped | Implementation |
|---|---|
| `tessera.expression.simplify.core` | Rule-based folds (X−X, X*0, safe-divide) |
| `tessera.expression.simplify.ac` | AC normalisation (sort children of comm ops) |
| `simplify_canonical` | Composition: AC then rules |
| `Measure._canonicalise_atoms` | Per-Measure canonical form |
| `Measure.compose` + `collapse_functional_chain` | Algebraic exit from nested LinearFunctional |
| `HallOfFame` | Protects discoveries from drift (per-cx best-ever) |

This is what the perfect-info framing CALLS FOR. F3 (algebraic
equivalence) is operationalised here. Empirically validated: at cx=7
in restricted grammar, |E_K|/|T_K| = 0.077 (92% of trees are
duplicates).

### Role 6 — The grammar / inductive bias (the constraint layer)

Restricting the state space to physically-meaningful candidates.

| Shipped | Implementation |
|---|---|
| `BIN_OPS`, `UN_OPS` tables | Operator alphabet |
| `MAX_DEPTH`, `MAX_COMPLEXITY` | Tree-size constraints |
| `pointwise_only` mode | For ODE rediscovery (no FunctionalOps) |
| `enable_2d` mode | 2-D PDE-style grammar |
| **reduce ops** (new today) | Aggregation primitives — let GP convert array→scalar |
| **tessera.axes** (new today) | Type-level invariance declarations |

**The axes module is the SHARPEST tool in this role.** It lets the
user declare invariance for time, space, asset-axis, etc. The
compatibility checker validates trees against axis declarations.
Future: enforce constraints at random_tree / mutate time → Burnside-
flavoured |E_K| / |G| compression of the framework's conjecture.

### Role 7 — Implementation substrate (the "how", not the "what")

Where the math runs.

| Shipped | Implementation |
|---|---|
| numpy + numba kernels | Default CPU backend |
| `FunctionalCache` | Subexpression caching |
| Multi-worker `ProcessPoolExecutor` | Limited GP parallelism |
| **`tessera.backend`** (new today) | Switchable CPU/GPU API |
| `docs/milestones/gpu_backend.md` | Tracking the GPU port (8-10 days work) |

GPU support is the next-big-milestone, but the framework's
THEORETICAL claims don't depend on it. The backend abstraction makes
it switchable; the milestone doc tracks what's left.

## The four research notes, mapped

Each open research direction maps to one or more roles:

| Note | Primary role | Status |
|---|---|---|
| `search_as_energy_min.md` | Role 4 (B&B) + Role 5 (equivalence) | partly implemented |
| `fit_as_perfect_info_game.md` | The framework itself | foundational doc |
| `measure_theory_and_perfect_info.md` | Role 4 specialised to measures (L1 bounds, Fubini) | step (c) shipped |
| `gpu_and_cv_via_sr.md` | Role 7 (substrate) + Role 6 (CV-grammar) | scaffold shipped, full work scoped |
| `invariance_in_sr.md` | Role 6 (axis-aware grammar) | minimal version shipped today |

**There's no divergence between the implementations — there are 7
slots in the same framework, and we've been filling them at
different speeds.**

## What's MISSING from the framework

Not every role is fully built out. Honest gaps:

| Role | Gap |
|---|---|
| Role 1 | The `mutate` operators don't bias toward axis-respecting trees. Future: `wrap_in_reduce` mutation, axis-aware random_tree |
| Role 3 | No multi-population GP with migration; no lexicase selection. PySR has both; tessera doesn't |
| Role 4 | FunctionalOp2D bounds shipped today; per-row interval evaluation would be tighter. Branch-and-bound on partial / mid-mutation trees not implemented |
| Role 5 | Equality saturation (e-graphs) not implemented; Knuth-Bendix completion not pursued |
| Role 6 | Axes module shipped today but doesn't enforce in search. `WeightedIndicatorSum` primitive (closing the EML vocabulary gap) not implemented |
| Role 7 | GPU backend tier 1-4 deferred (~8-10 days) |

## How the user can use this map

Three honest ways:

1. **When picking the next task:** look at the seven roles. The
   weakest role today is Role 6 (grammar / inductive bias). Adding
   axis-enforcement to the GP search would fill the biggest gap.

2. **When evaluating "is this divergence":** every commit can be
   tagged with its primary role. If a commit doesn't fit one of the
   seven roles, that's a yellow flag (maybe it's not framework-aligned).

3. **When writing a research note:** lead with WHICH role the note
   contributes to. The four existing notes all do this implicitly; the
   next one could do it explicitly.

## The user's question, answered directly

> *"How do we integrate Knuth's work with our diverging implementations?"*

The implementations aren't diverging — they're each filling one of
seven slots in the Knuth-grounded framework. The PROBLEM was that
we hadn't written down the mapping. This doc IS the mapping.

Going forward: any new implementation should tag itself with its
role. New research notes should declare which role they extend or
which gap they fill. The framework becomes the organising lens; the
implementations are concrete realisations.

## Sequencing recommendation (no new code today)

Stop opening new directions. Consolidate. Re-read the four research
notes + this synthesis. Then pick ONE next implementation step from
the gap table above. Suggested priority:

1. **Axis-enforcement in `random_tree`** (Role 6) — extends today's
   `tessera.expression.axes` to actually constrain search. ~80 LOC.
2. **`wrap_in_reduce` mutation** (Role 1) — addresses the MNIST
   benchmark's "GP doesn't discover aggregation" finding. ~30 LOC.
3. **Multi-population GP with migration** (Role 3) — PySR's main
   diversity tool. ~200 LOC.

Anything else listed in the gap table is BIGGER (weeks of work) or
RESEARCHY (open theoretical questions). The three above are well-
scoped engineering tasks that each close a known gap.

## Changelog
- 2026-05-25: initial synthesis. Maps every shipped tessera component
  to its role in the Knuth-grounded perfect-info game framework.
