# MVP / C4: causal direction priors — PARTIAL VALIDATION

Empirical test of conjecture C4 from
`docs/research/process_discovery_sr.md` §6.4. Second occupant of
tessera.experimental. **Result: partially validated — does what it
was designed to do, but doesn't deliver a transformative boost.**

## The conjecture, restated

> "Causal direction priors at the tree level reduce effective search
> space WITHOUT losing the right answer."

Operationalization: forbid Measure2D operators whose atoms span
multiple time offsets (`lag_t` values). For dynamical-system targets
where the target is a temporal derivative of the state, this kills
the temporal-derivative tautology shortcut (e.g., `M2D[1·(0,0) +
-1·(1,0)](U) = U[t] − U[t+1] ≈ −dt_U`).

Implementation: hard rejection at scoring — violators get
`train_loss = fitness = +inf`. They're excluded from both Pareto
front AND tournament selection.

## Headline result

| Mode | Class C (clean) | Class A-temporal (violators>0) | Class A-spatial (violators=0) | Degenerate (cx=1) | Class B |
|---|---|---|---|---|---|
| Baseline | 1/5 | 4/5 | 0/5 | 0/5 | 0/5 |
| With C4 | **1/5** | **0/5** | 2/5 | 2/5 | 0/5 |

**Class C rate unchanged.** C4 eliminates Class A-temporal entirely
(as designed) but the GP falls back to *either*:
- Class A-spatial: 2-atom spatial first-derivative (e.g., `M2D[1·(0,0) +
  -1·(0,1)](U) ≈ −∂U/∂x`); still wrong mechanism but at least HONEST
  about being wrong (no target leakage)
- Degenerate: predict-zero, stuck at the trivial fixed point

## What the verdict means

**Both parts of the conjecture hold:**

1. *Reduce effective search space*: ✓ — 0/5 temporal-tautology trees
   remain (vs 4/5 in baseline). The constraint successfully prunes
   the tautology branch of the search space.

2. *Without losing the right answer*: ✓ — Class C still found at
   1/5 (seed 2026 with C4 produced the exact same canonical form
   `(M2D[Laplacian](U) * 0.05)` that seed 2029 produced in baseline).

**What the conjecture did NOT promise** (and we hoped for):

- Class C rate would *rise* above baseline. It didn't. The constraint
  doesn't *help* the GP find the right mechanism; it only prevents
  it from settling on the wrong one.

## The taxonomic refinement this exposed

The class taxonomy needed an A-temporal vs A-spatial distinction:

| Sub-class | Shape | Train fit | Honesty |
|---|---|---|---|
| A-temporal | `M2D[different lag_t](U)` | ~2.0× oracle (low) | Dishonest — target leakage via time-difference |
| A-spatial | `M2D[same lag_t, different lag_x](U)` | ~2.5× oracle (higher) | Honest — first-derivative approximation, wrong mechanism but no leakage |

C4 forces A-spatial over A-temporal. The baseline GP, given the choice,
prefers A-temporal because it fits TRAIN better — but that fit comes
from tautology, not mechanism. C4 makes the GP honest at the cost of
slightly worse train loss on the non-mechanism Pareto candidates.

This is actually a methodologically useful outcome: the constraint
*reveals* the GP's previous reliance on tautology by forcing it to
acknowledge what it actually has.

## Why C4 doesn't boost Class C discovery

The 3-atom Laplacian template (Class C) has a specific structure:
weights +1/-2/+1 at lag_x = -1/0/+1, all at lag_t = 0. Random
Measure2D generation reaches 2-atom shapes easily (Class A-spatial
emerges naturally) but the 3-atom +1/-2/+1 pattern requires a more
specific combinatorial discovery. C4's constraint doesn't bias TOWARD
3-atom; it just FORBIDS temporal atoms.

So the constraint is *necessary but not sufficient* for Class C
discovery. To boost Class C rate, would need to also bias random_tree
toward 3-atom Measure2D templates (which gets close to "factory
primitive" territory we explicitly rejected).

## Per-seed details

| Mode | seed | train/oracle | test/oracle | cx | class | violators | tree summary |
|---|---|---|---|---|---|---|---|
| baseline | 2026 | 2.20 | 2.26 | 8 | A-temporal | 1 | `M2D[1·(0,0)+(-1)·(1,0)]`-flavoured |
| baseline | 2027 | 1.68 | 1.51 | 12 | A-temporal | 3 | nested temporal M2Ds |
| baseline | 2028 | 2.19 | 2.20 | 10 | A-temporal | 1 | temporal diff |
| baseline | **2029** | **1.04** | **1.00** | **4** | **C** | **0** | `(Laplacian(U) * 0.05)` |
| baseline | 2030 | 2.20 | 2.26 | 8 | A-temporal | 1 | temporal diff |
| with_c4 | **2026** | **1.04** | **1.00** | **4** | **C** | **0** | `(Laplacian(U) * 0.05)` |
| with_c4 | 2027 | 16.15 | inf | 1 | degenerate | 0 | predict-zero |
| with_c4 | 2028 | 16.15 | inf | 1 | degenerate | 0 | predict-zero |
| with_c4 | 2029 | 2.77 | 3.63 | 9 | A-spatial | 0 | spatial diff structure |
| with_c4 | 2030 | 2.55 | 3.91 | 7 | A-spatial | 0 | spatial diff structure |

## Comparison with prior experiments

| Intervention | Class C rate | Class B rate | Class A behaviour |
|---|---|---|---|
| Baseline (with reduce_* downweight default) | 1/5 | 1/5 | A-temporal dominant |
| Multi-trajectory training (from earlier session) | 1/3 (cx=4 canonical) | 0/3 | Eliminated |
| Held-out MSE in scoring (C1 Mode B) | 1/5 | 0/5 | A-temporal dominant |
| Causal-axes constraint (C4) | 1/5 | 0/5 | A-temporal eliminated; A-spatial emerges |

**Multi-trajectory training remains the most effective single
intervention** for boosting Class C discovery. C4 in isolation
doesn't add to it.

## Possible refinements (not committed)

C4 might combine well with other interventions:

1. **C4 + 3-atom-biased random_measure_2d**: also bias toward 3-atom
   spatial templates. Together: forbid temporal AND prefer 3-atom.
2. **C4 + held-out MSE (C1 Mode B)**: stack both interventions.
3. **C4 + multi-trajectory**: would C4 add anything on top of
   multi-trajectory (which already eliminates Class A)?

Each is testable in ~half day. Not committed to ship; included as
possible follow-on.

## Verdict for the experimental discipline

**Conjecture C4: PARTIALLY VALIDATED.**

- ✓ Reduces effective search space (Class A-temporal eliminated)
- ✓ Doesn't lose the right answer (Class C rate unchanged)
- ✗ Doesn't deliver a transformative boost to mechanism discovery

The module `tessera.experimental.causal_axes` is preserved with
status updated to *partially-validated; eliminates temporal-tautology
shortcut without boosting mechanism discovery in isolation*. The
discipline worked: we made a conjecture, tested it, got a clear
nuanced result, documented it honestly.

The taxonomic refinement (A-temporal vs A-spatial) is itself a
contribution — it makes visible a distinction that pointwise loss
alone doesn't surface.

## Two-experiment summary so far

| Conjecture | Status | Class C delta |
|---|---|---|
| C1 (ABC scoring) | FALSIFIED | -1/5 vs baseline (hurt mechanism) |
| **C4 (causal priors)** | **PARTIAL** | **+0/5 vs baseline (preserved, didn't boost)** |

The basket of conjectures gives us cumulative knowledge. Each test
narrows what works and what doesn't.

## Reproducing

```
python benchmarks/run_heat_equation_causal_axes_mvp_c4.py --seeds 5
```

Wall-clock ~80 seconds.
