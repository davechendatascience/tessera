# Heat equation Mode-1 minimal experiment — parsimony=0 vs default

Tests whether removing parsimony pressure exposes the α·Laplacian
candidate on the Pareto front alongside diff_t-style candidates.

**Setup:** T=200, X=32, 5 seeds each, pop=120, gens=40. Oracle loss
= 4.008e-06. Default parsimony = 3.2e-08 (0.001 × target_var); zero
parsimony = 0.0.

## Verdict

**Mode 1 alone is insufficient for this benchmark.** Both parsimony=0
and parsimony=default produce essentially identical Pareto fronts:
0/5 seeds find Laplacian-shape candidates in either setting. The
issue isn't scoring; it's that **multi-atom compositions (the 5-point
Laplacian needs 3 atoms with specific weights +1/−2/+1) are not
discoverable by random mutation in the given budget**, regardless of
whether parsimony rewards or punishes them.

## What the GP actually finds

The trees contain raw Measure2D operators with 2 atoms, which ARE
discrete differential operators — just spelled as raw `M2D[w1·(t1,x1)
+ w2·(t2,x2)]` rather than the named factory `measure_2d_diff_t`.
Examples on the Pareto front:

- `M2D[1·(0,0) + -1·(1,0)](U)` = `U[t,x] − U[t+1,x]` — backward time
  difference (1-step). Loss/oracle ≈ 2.32 at cx=2.
- `M2D[1·(0,0) + -1·(3,0)](0.31 · U)` — 3-step time difference,
  scaled. Loss/oracle ≈ 1.78 at cx=4.

These are the same physics-content as `measure_2d_diff_t(lag=1)` and
`measure_2d_diff_t(lag=3)`, discovered through random sampling of
2-atom Measure2D weights and offsets.

**What's NOT found** at either parsimony setting:
- `M2D[1·(0,-1) + -2·(0,0) + 1·(0,+1)](U)` — the 5-point spatial
  Laplacian (3 atoms, weights +1/−2/+1)
- Any 3-atom Measure2D with non-trivial weights

The random Measure2D generator clearly samples 2-atom operators (one
positive, one negative — a "difference operator" template) freely,
but the probability of generating the specific 3-atom Laplacian
template by random combination of atom-weight-offset triples is near
zero in the budget tested.

## What this tells us about the framework

This is a clean instance of **search bottleneck dominating scoring
choice**. The user's hypothesis from the brainstorm — "Mode 1 minimal
(parsimony=0) might be enough" — falsifies cleanly: removing
parsimony doesn't help when the candidate isn't being SEARCHED to
begin with. The proper next move is one of:

1. **Mode 2 (derivation grammars)** — bias mutations toward
   constructing canonical Measure2D templates. A "build a Laplacian"
   macro would compose `(±1, 0, ∓1)` weights at `(±1, 0)` offsets in
   one step, dramatically raising the probability of the Laplacian
   template appearing in the population.

2. **Vocabulary expansion at the Measure2D-factory level** — pre-
   register `measure_2d_laplacian_5pt` (and other canonical templates)
   as primitives that random_tree can sample directly, rather than
   requiring random Measure2D atom-weight generation to discover them.

3. **Plan B — accept the limitation.** This benchmark needs a
   composition the random search doesn't reach in reasonable budget.
   Recognize that "PDE discovery with arbitrary stencils" is harder
   than "PDE discovery with curated stencils" and scope tessera
   accordingly.

Of these, (2) is by far the cheapest and most aligned with how the
existing benchmark was *designed* — the original
`run_heat_equation_discovery.py` already used `measure_2d_laplacian_5pt`
as the oracle. If the random_tree generator can sample factory
templates directly, the Laplacian becomes a single-step random pick,
not a multi-step composition.

## Parsimony=default fronts (per seed)

**seed=2026** (9.6s)
| cx | loss/oracle | tree |
|---|---|---|
| 1 | 7.66 | `-0.00171094` (just predict zero) |
| 2 | 2.32 | `M2D[1·(0,0) + -1·(1,0)](U)` |

**seed=2027** (1.8s)
| cx | loss/oracle | tree |
|---|---|---|
| 1 | 7.66 | `-0.00171094` |
| 2 | 2.32 | `M2D[1·(0,0) + -1·(1,0)](U)` |
| 6 | 2.14 | `M2D[1·(0,0) + -1·(1,0)]((U + atan2(3.94, U)))` |
| 7 | 1.79 | (added atan2 stack) |
| 11 | 1.73 | (added more atan2) |
| 15 | 1.68 | (4-deep atan2 stack) |

**seed=2028** (1.3s)
| cx | loss/oracle | tree |
|---|---|---|
| 1 | 7.66 | `-0.00171107` |
| 4 | 1.78 | `M2D[1·(0,0) + -1·(3,0)]((0.31 * U))` |

**seed=2029, 2030** — only predict-zero candidates (cx=1, loss/oracle
= 7.66). Random init didn't generate any Measure2D-using trees that
survived.

## Parsimony=zero fronts (per seed)

Nearly identical to default. The only differences: seed=2028 finds
more cx variants (3, 4, 6, 8, 11) but they all use the same 2-atom
M2D structure, just with different inner expressions. No Laplacian
template appears.

## Why parsimony=0 didn't help

In retrospect, the prediction was wrong. The hypothesis assumed:

> "Laplacian candidates exist in the search but get parsimony-suppressed
> in favor of shorter equivalent diff_t shapes."

The actual situation:

> "Laplacian candidates don't exist in the search at all — random
> Measure2D atom generation doesn't reach the 3-atom +1/-2/+1 pattern
> in the budget tested."

So removing parsimony doesn't help because there's nothing to un-
suppress. The scoring layer can only decide among candidates the
search produces.

## Implications for the Mode 1 / Mode 2 brainstorm

- **Mode 1 (compression-aware scoring) was the wrong first investment.**
  The candidate this benchmark needs isn't being created by search
  in the first place. Scoring fixes apply to candidates that EXIST.

- **Mode 2 (derivation grammars / template biases) is the necessary
  next move.** Without it, this benchmark stays underfit at ~1.6×
  oracle indefinitely.

- **Vocabulary curation at the random_tree level is the cheapest
  Mode-2-flavoured fix.** If `measure_2d_laplacian_5pt` is registered
  as a single-step random_tree choice, the Laplacian-template
  candidate becomes a one-pick generation, not a multi-step
  combinatorial discovery.

The general principle this confirms: **search reach is the binding
constraint when the target requires multi-step composition the random
generator doesn't reach efficiently.** This is the optimizer-ceiling
problem made operational.

## Next steps (not in this experiment's scope)

1. Modify `random_tree(enable_2d=True)` to sample from the
   factory-registered Measure2D templates (Laplacian_5pt, diff_t,
   diff_x) with non-zero probability, alongside random atom
   generation. Predicted effect: structural Laplacian recovery rate
   rises from 0/5 to ≥3/5 seeds at the same budget.

2. Then re-run the parsimony=0 vs default experiment with this fix.
   The honest test of Mode 1's effect requires the candidate to be
   generatable; once it is, the comparison becomes meaningful.

3. Only after that, evaluate whether Mode 1 (compression-aware
   scoring) adds value on top of vocabulary-fixed search.

This is a clean experimental insight even though the headline result
is "no change between settings": **the diagnostic narrowed the problem
from scoring to search, redirecting the next investment.**
