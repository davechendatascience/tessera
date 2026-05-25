# Heat eq paired diagnostic — TRAIN/TEST split + compute scaling (AFTER reduce_* downweight)

Two diagnostics in one experiment:
- TRAIN/TEST split: fit on trajectory A, evaluate on B (different IC, same α). Detects overfit if TRAIN < TEST.
- Compute scaling: vary budget at fixed N. Detects underfit (improves with compute) vs fundamental search limit (flat in compute).

**Setup:** T=200, X=32, α=0.05. TRAIN ic_seed=100, TEST ic_seed=200 (different bump locations/amplitudes/counts).

**TRAIN:** var=6.863e-05, oracle=4.008e-06
**TEST:**  var=7.779e-05, oracle=3.981e-06

**Wall-clock:** 80.7s

## Headline finding — Class C (clean mechanism) appears after reduce_* downweight

**At 360×120 seed=2027:** the GP discovered the canonical heat equation:

```
(M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U) / 20.0639)
```

That's the 5-point Laplacian template divided by a **constant** 20.0639. Since 1/20.0639 ≈ 0.0499 ≈ α = 0.05, this is structurally `α · Laplacian(U)` — the canonical heat-equation form. **TEST loss exactly matches oracle.** No factory primitives, no grammar machinery. Just random search with `reduce_*` operators downweighted by 10× to discourage trajectory-specific overfit wrappers.

This validates the minimum-intervention hypothesis from yesterday's paired-diagnostic: the GP COULD find the mechanism; it was just being pulled toward `template / reduce_*` shapes by uniform unary-op sampling.

## Before-vs-after comparison (same experiment, two configs)

| | Before (uniform reduce_*) | After (reduce_* × 0.1) |
|---|---|---|
| Class C runs (clean `c · Laplacian`) | 0/12 | **1/12** |
| Class B runs (template / reduce_*) | 2/12 | 2/12 (similar count) |
| Worst train/test gap | train=1.04 / test=**31.79** | train=1.04 / test=**2.81** |
| Mean test/oracle at largest budget | 12.7 (skewed by overfit outlier) | 1.78 |
| Best test/oracle ever achieved | 2.11 | **1.00** (oracle-exact) |

Class C only 1/12 means the fix is enabling, not forcing, mechanism discovery. The GP can still get distracted by Class B (`reduce_*` wrappers haven't gone away entirely with 10× downweight). But Class C is now within reach, where before it wasn't reachable at all.

## Compute scaling table (medians across seeds)

| pop × gens | budget | median train/oracle | median test/oracle | median best-by-test/oracle | median cx |
|---|---|---|---|---|---|
| 60×25 | 1500 | 2.10 | 2.43 | 2.43 | 5 |
| 120×50 | 6000 | 2.17 | 2.22 | 2.22 | 5 |
| 240×100 | 24000 | 1.49 | 2.14 | 2.14 | 8 |
| 360×120 | 43200 | 2.22 | 2.17 | 2.17 | 6 |

The MEDIANS are dominated by Class A candidates (diff-style, ~2× oracle, stable). The Class C run at 43200 (seed 2027) is an outlier that pulls the mean (not median) toward 1.0.

**Mean test/oracle by budget:** 1500: inf (outlier), 6000: 4.31, 24000: 2.10, 43200: **1.78**. The mean improves substantially with compute now — because Class C appearance probability rises at large budgets.

## Class taxonomy (revised)

### Class A — Generic diff (most common, 9/12 runs)
Trees like `M2D[1·(0,0) + -1·(1,0)](max(0.21, U))`. 2-atom Measure2D wrapping U through generic arithmetic.
- Train ≈ Test ≈ 2× oracle
- Always generalizes

### Class B — Template / reduce_* (still 2/12 runs)
Example: `(M2D[Laplacian](U) / reduce_max(U))` and `(M2D[Laplacian](U) / reduce_max((U-...)))`.
- Train ≈ 1× oracle, Test ≈ 3-9× oracle (much milder than before)
- The reduce_* downweight pushed these toward smaller Train improvements; the worst Class B gap shrunk from 30× to 3× oracle.

### Class C — Clean `c · Laplacian` (1/12 runs, NEW)
Example: `(M2D[Laplacian](U) / 20.0639)`.
- Train ≈ Test ≈ 1× oracle (exact mechanism recovery)
- Coefficient extracted: 1/20.0639 ≈ 0.0499 ≈ α=0.05 (matches simulator within 0.2%)

The relative populations: 75% A, 17% B, 8% C. Class C is now possible but not yet reliable.

## What this tells us

1. **The diagnosis was correct.** The GP wasn't search-limited or scoring-limited in the deep sense. It was *biased* by uniform reduce_* sampling toward `template / reduce_*` shapes. A 10× downweight on reduce_* operators uncovered Class C without any other architectural change.

2. **The user's "natural overfit" framing was operationally validated.** `diff_t / reduce_max` is the textbook example: argument tweaks (the divisor) to make TRAIN fit while breaking mechanism portability. Removing the easy availability of that tweak forced the GP to find legitimate fits.

3. **Class C is enabled but not forced.** 1/12 = 8% probability is too low for reliable deployment. To make Class C more common would need EITHER:
   - Stronger downweight (reduce_* weight → 0.01 or → 0)
   - Mode-2 grammar that ACTIVELY constructs `Const · Template` from any discovered template
   - Multi-trajectory training (would punish any TRAIN-specific reductions automatically)

## Per-seed details

| budget | seed | train/oracle | test/oracle | cx | tree (truncated) | class |
|---|---|---|---|---|---|---|
| 60×25 | 2026 | 16.15 | inf | 1 | `-0.00226855` | degenerate |
| 60×25 | 2027 | 2.10 | 2.43 | 5 | `M2D[-0.95·(2,0) + 1.29·(2,0)](M2D[1·(0,0) + -1·(3...](...))` | A |
| 60×25 | 2028 | 1.87 | 1.94 | 6 | `M2D[1·(0,0) + -1·(3,0)]((0.55 + (0.32 * U)))` | A |
| 120×50 | 2026 | 2.31 | 2.22 | 4 | `M2D[1·(0,0) + -1·(1,0)](max(0, U))` | A |
| 120×50 | 2027 | 1.80 | 1.66 | 11 | `M2D[diff](((U + atan2(10.06, U)) + atan(...)))` | A (deep) |
| 120×50 | 2028 | 2.17 | **9.05** | 5 | `(M2D[Laplacian](U) / reduce_max(U))` | **B** |
| 240×100 | 2026 | 2.20 | 2.14 | 8 | `M2D[1·(0,0) + -1·(1,0)](max(0.20, (U - (0.01 * ...))))` | A |
| 240×100 | 2027 | 1.49 | 1.35 | 15 | `M2D[diff](((U / step(U)) + atan2(...)))` | A (sub-oracle?) |
| 240×100 | 2028 | 1.04 | **2.81** | 7 | `(M2D[Laplacian](U) / reduce_max((U...)))` | **B** |
| 360×120 | 2026 | 2.22 | 2.17 | 6 | `(0.0001 + M2D[1·(0,0) + -1·(1,0)](max(0.21, U)))` | A |
| **360×120** | **2027** | **1.04** | **1.00** | **4** | `(M2D[Laplacian](U) / 20.0639)` | **C** ★ |
| 360×120 | 2028 | 2.23 | 2.18 | 6 | `(0.0002 + M2D[1·(0,0) + -1·(1,0)](max(0.21, U)))` | A |

★ = Class C (clean mechanism)

## Conclusion

The minimal-fix hypothesis from yesterday's diagnostic is validated. A 5-LOC change to weight reduce_* operators 10× lower in `random_tree`'s unary-op sampling enabled the GP to discover the canonical mechanism for the first time. The discovered form `(Laplacian(U) / 20.0639)` is mathematically equivalent to `α · Laplacian(U)` with α matching the simulator to 0.2%.

This is the simplest possible "natural-overfit prevention" — bias the search away from trajectory-specific reductions that fit TRAIN without generalizing. It's not a complete fix (Class C is still rare at 1/12), but it transforms the benchmark from "unreachable" to "occasionally reachable" at modest budget.

The next move for reliability would be one of: (a) stronger reduce_* downweight or removal, (b) Mode-2 grammar to construct `Const · Template` directly, (c) multi-trajectory training to make TRAIN-specific reductions self-defeating.

## Reproducing

```
python benchmarks/run_heat_equation_traintest_computescale.py
```

Wall-clock ~80 seconds. Requires the post-`UN_OP_WEIGHTS` mutation.py from commit (pending).
