# Heat eq paired diagnostic — TRAIN/TEST split + compute scaling

Two diagnostics in one experiment:
- TRAIN/TEST split: fit on trajectory A, evaluate on B (different IC, same α). Detects overfit if TRAIN < TEST.
- Compute scaling: vary budget at fixed N. Detects underfit (improves with compute) vs fundamental search limit (flat in compute).

**Setup:** T=200, X=32, α=0.05. TRAIN ic_seed=100, TEST ic_seed=200 (different bump locations/amplitudes/counts).

**TRAIN:** var=6.863e-05, oracle=4.008e-06
**TEST:**  var=7.779e-05, oracle=3.981e-06

**Wall-clock:** 98.6s for 12 runs (4 budgets × 3 seeds).

## Headline finding — the Laplacian IS being discovered, just in OVERFIT WRAPPING

Two of 12 runs (budgets 6000 and 43200, both seed 2026) produced trees containing the exact 5-point Laplacian template:

```
M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U)
```

This is the canonical heat-equation operator, found by raw atom-weight generation without any factory primitive. **Yesterday's "0/5 structural recovery" was a detection-function bug** (looking for the substring "laplacian" in the atom notation, which doesn't appear there).

But both Laplacian-finding runs **catastrophically overfit**:

| Budget | Seed | Train/oracle | Test/oracle | Tree (truncated) |
|---|---|---|---|---|
| 6000 | 2026 | **1.08** | **4.17** | `(Laplacian(U) / reduce_max(...))` |
| 43200 | 2026 | **1.04** | **31.79** | `(Laplacian(U) / reduce_std(B[...]))` |

The Laplacian is divided by a `reduce_*(...)` operator — a TRAIN-specific scalar that doesn't transfer to TEST. On TRAIN: the divisor happens to make the fit nearly oracle-perfect. On TEST: the divisor is different (recomputed on TEST data), and the prediction blows up.

**The clean mechanism `c · Laplacian(U)` is never produced.** When the Laplacian template appears, it's always wrapped in TRAIN-specific reductions. This is a *bona fide* natural-sense overfit — argument tweaks (the divisor) to make data fit, while making the mechanism non-portable.

## Three classes of candidates discovered

Inspection of the per-seed trees reveals three structurally distinct families on the Pareto fronts:

### Class A — Diff-style (most common)
Examples: `M2D[1·(0,0) + -1·(1,0)](U)`, `M2D[1·(0,0) + -1·(3,0)](0.32 · U)`
- 2-atom Measure2D (a discrete time-difference operator)
- No reduction operators
- Train ≈ Test ≈ 2× oracle (consistent across IC variations)
- **These generalize cleanly** — diff_t is structurally generic; identical on any smooth trajectory

### Class B — Mechanism with TRAIN-specific contamination (2 of 12 runs)
Examples: `(Laplacian(U) / reduce_max(...))`, `(Laplacian(U) / reduce_std(...))`
- Contains the correct 3-atom Laplacian template
- Wrapped in `reduce_*` operators that compute trajectory-specific scalars
- Train ≈ 1× oracle (near-perfect on TRAIN)
- Test ≈ 4-32× oracle (catastrophic OOS)
- **These are the natural-sense overfits**

### Class C — Clean mechanism `c · Laplacian(U)` (never observed)
Would be: `0.05 · M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U)`
- The canonical, generalizable answer
- Would give Train ≈ Test ≈ 1× oracle (oracle-level on both)
- **NOT produced by GP in any of 12 runs**

The GP can find Class A reliably (every seed has them) and Class B occasionally (2/12), but **never converges to Class C**. Class C requires multiplying the Laplacian template by a *constant*, not by a reduction. The search is finding `(template / reduction)` shapes much more easily than `(constant · template)` shapes.

## Compute scaling table (medians across seeds)

| pop × gens | budget | median train/oracle | median test/oracle | median best-by-test/oracle | median cx |
|---|---|---|---|---|---|
| 60×25 | 1500 | 2.34 | 2.22 | 2.22 | 2 |
| 120×50 | 6000 | 1.79 | 1.94 | 1.94 | 10 |
| 240×100 | 24000 | 1.91 | 1.84 | 1.84 | 7 |
| 360×120 | 43200 | 2.15 | 2.17 | 2.17 | 10 |

The MEDIANS are dominated by Class A candidates (~2× oracle, stable across compute). The Class B runs are outliers; they pull individual seeds toward train=1× oracle but blow up test.

Best-by-test (last column) is essentially flat at 1.84-2.22× oracle across 28× compute scaling. **The Pareto-front's mechanism-recovery ceiling is ~1.8× oracle even with optimal post-hoc selection.**

## Real diagnosis (corrected from auto-verdict)

The original four-quadrant decision tree was incomplete because it didn't account for the existence of partial-mechanism candidates (Class B). The accurate diagnosis:

- **NOT underfit-fixable-by-compute**: median train/oracle doesn't trend down with budget (2.34 → 2.15 across 28× compute).
- **NOT pure search limit**: the Laplacian template IS reached at higher budgets (2/12 runs); search reach isn't the binding constraint at 6000+ budget.
- **NOT generic overfit**: most candidates (Class A) have train ≈ test; they generalize.
- **IS class-B overfit AND class-C absence**: the GP finds the mechanism template but only inside non-generalizable wrappers, and never produces the clean `c · template` form that would be ideal.

The honest conclusion: **the search reaches the right operator template but the surrounding context (arithmetic wrappers) is what determines generalization, and the GP's mutation operators don't reliably converge to "constant times template" — they converge to "template divided by trajectory-specific reduction".**

## Why this matters for the Mode-1/Mode-2 brainstorm

- **Mode 1 (compression-aware scoring)** as I proposed wouldn't help here. Class B candidates have *lower train loss* than Class A, so removing parsimony doesn't expose Class C — it just makes Class B preferred. Class C doesn't exist in the search to be selected.

- **Mode 2 (derivation grammars)** would help directly: a grammar "multiply discovered template by Const" would systematically construct Class C candidates from any Class B candidate that already has the template. This is the natural completion of the partial discovery.

- **Regularization against trajectory-specific operators** is a third, simpler alternative: penalize `reduce_*` operators in mutation weights, or test candidates on a held-out portion of TRAIN before promoting to the Pareto front. This kills Class B without needing grammar machinery.

The user's framing — **"we trade too much loss for simplicity"** — actually plays out the OPPOSITE way here: parsimony isn't punishing the right things (Class B has low-cx with reduce_*), and selection-by-train is REWARDING the wrong things (Class B's TRAIN-fit). The fix isn't "less parsimony"; it's a different REGULARIZATION axis (anti-trajectory-specific-operators) or a different SELECTION criterion (cross-validation rather than train).

## Per-seed details

| budget | seed | train/oracle | test/oracle | cx | best tree (truncated) | class |
|---|---|---|---|---|---|---|
| 60×25 | 2026 | 16.15 | inf | 1 | `-0.00226855` (predict zero) | (degenerate) |
| 60×25 | 2027 | 2.34 | 2.22 | 2 | `M2D[1·(0,0) + -1·(1,0)](U)` | **A** |
| 60×25 | 2028 | 1.87 | 1.94 | 4 | `M2D[1·(0,0) + -1·(3,0)](0.323 · U)` | **A** |
| 120×50 | 2026 | **1.08** | **4.17** | 10 | `(M2D[Laplacian](U) / reduce_max(...))` | **B** ★ |
| 120×50 | 2027 | 1.79 | 1.65 | 20 | `M2D[1·(0,0) + -1·(1,0)]((U + atan2(2.12, M2D[...])))` | A (deep) |
| 120×50 | 2028 | 1.85 | 1.94 | 8 | `M2D[1·(0,0) + -1·(3,0)]((0.32 * (-0.02 + max(0...))))` | A |
| 240×100 | 2026 | 2.00 | 2.00 | 11 | `M2D[1·(0,0) + -1·(1,0)]((U * tanh(sqrt(sqrt(...)))))` | A |
| 240×100 | 2027 | 1.91 | 1.84 | 4 | `M2D[1·(0,0) + -1·(2,0)]((U / 2.08))` | A |
| 240×100 | 2028 | 1.83 | 1.68 | 7 | `M2D[1·(0,0) + -1·(1,0)]((U + pow(0.62, M2D[...])))` | A |
| 360×120 | 2026 | **1.04** | **31.79** | 10 | `(M2D[Laplacian](U) / reduce_std(B[...]))` | **B** ★ |
| 360×120 | 2027 | 2.15 | 2.11 | 10 | `min(min(0.009, (0.008 · U)), M2D[1·(0,0) + -1·(...](...)))` | A |
| 360×120 | 2028 | 2.23 | 2.17 | 6 | `M2D[1·(0,0) + -1·(1,0)]((U - min(0.21, U)))` | A |

★ = Class B (Laplacian template wrapped in trajectory-specific reduction)

## What this opens

The clean experimental finding suggests three concrete next moves, in priority:

| Move | Effort | What it tests |
|---|---|---|
| **Regularize against `reduce_*` operators in the GP search** (lower their mutation weights when target is per-sample, not per-trajectory) | ~30 LOC | Does penalizing trajectory-specific reductions push the GP toward Class C (clean `c · template`)? |
| **Mode 2 grammar for "wrap discovered template in constant multiply"** | ~half day | Would systematically produce Class C from any Class B by completion |
| **Cross-validation scoring instead of train-only** (split TRAIN into k folds, average) | ~1 day | Would make Class B unselectable since each fold's reduce_* gives a different scalar |

The first is the cheapest by far. It addresses the *immediate* cause of Class B overfit (the GP loves `template / reduce_*` shapes) by reducing the probability the GP generates them. If Class C appears after this change, we've validated the diagnosis.

Note that all three are *correctives* on the existing architecture — they don't require the full Mode 1+2 redesign. The experiment narrowed the problem enough that a small targeted fix may suffice.

## Reproducing

```
python benchmarks/run_heat_equation_traintest_computescale.py
```

Wall-clock ~100 seconds at default settings.
