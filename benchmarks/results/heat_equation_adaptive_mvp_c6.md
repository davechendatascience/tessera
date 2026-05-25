# MVP / C6: adaptive mutation weights — empirical test

Empirical test of conjecture C6 (residual-diagnostic-driven
adaptive mutation weights), following the theoretical
pre-analysis in `docs/research/c6_residual_diagnostics_analysis.md`.

**Pre-analysis prediction:** generic adaptive ≈ baseline.
Diagnostic→corrective mapping problem prevents generic
adaptation from materially helping.

**Setup:** T=200, X=32, single-trajectory training,
pop=240, gens=100, 5 seeds/mode.
Adaptation: every 10 gens, adapt_strength=0.7.
Wall-clock: 69.4s

## Class distribution comparison

| Mode | Class A | Class B | C-partial | **Class C** | Degenerate |
|---|---|---|---|---|---|
| baseline | 3/5 | 1/5 | 0/5 | **1/5** | 0/5 |
| adaptive | 3/5 | 1/5 | 0/5 | **1/5** | 0/5 |

## Adaptation activity

- Total adaptation events: 50
- Meaningful adaptations (front had UN_OPS to count): 20
- No-op adaptations (front had no UN_OPS): 30

## Verdict against the pre-analysis prediction

- Predicted: adaptive ≈ baseline on Class C count
- Observed: baseline Class C = 1/5, adaptive Class C = 1/5

**PREDICTION VALIDATED.** Adaptive matches baseline within sampling
noise. Generic operator-usage-driven adaptation provides no
material benefit on Class C discovery. The diagnostic→corrective
mapping problem (predicted by the pre-analysis) is confirmed.

## Per-seed details

| Mode | seed | train/oracle | test/oracle | cx | class | tree (truncated) |
|---|---|---|---|---|---|---|
| baseline | 2026 | 2.20 | 2.26 | 8 | A | `M2D[1??(0,0) + -1??(1,0)](max(0.200517, (U - (0.014178 * ...` |
| baseline | 2027 | 1.68 | 1.51 | 12 | A | `M2D[1??(0,0) + -1??(1,0)]((U + atan2(1.1148, M2D[1??(0,0)...` |
| baseline | 2028 | 1.04 | 1.03 | 14 | B | `max((0.253896 - reduce_std(U)), (M2D[1??(0,-1) + -2??(0,0...` |
| baseline | 2029 | 1.04 | 1.00 | 4 | C | `M2D[1??(0,-1) + -2??(0,0) + 1??(0,1)]((0.049841 * U))` |
| baseline | 2030 | 2.16 | 2.21 | 11 | A | `M2D[1??(0,0) + -1??(1,0)](((0.936372 * U) * tanh(sqrt(max...` |
| adaptive | 2026 | 2.20 | 2.26 | 8 | A | `M2D[1??(0,0) + -1??(1,0)](max(0.200517, (U - (0.014178 * ...` |
| adaptive | 2027 | 1.68 | 1.51 | 12 | A | `M2D[1??(0,0) + -1??(1,0)]((U + atan2(1.1148, M2D[1??(0,0)...` |
| adaptive | 2028 | 1.04 | 1.03 | 9 | B | `(M2D[1??(0,-1) + -2??(0,0) + 1??(0,1)](U) / reduce_max((U...` |
| adaptive | 2029 | 1.04 | 1.00 | 4 | C | `M2D[1??(0,-1) + -2??(0,0) + 1??(0,1)]((0.049841 * U))` |
| adaptive | 2030 | 1.86 | 1.78 | 14 | A | `M2D[1??(0,0) + -1??(1,0)]((sqrt((U * U)) * tanh((M2D[0.69...` |

## Four-experiment basket state (running totals)

| Conjecture | Status | Class C delta vs baseline | Insight |
|---|---|---|---|
| C1 (ABC scoring) | FALSIFIED | -1/5 | ABC structure-distance doesn't discriminate B from C |
| C4 (causal priors) | PARTIAL | 0/5 | A-temporal vs A-spatial distinction useful diagnostic |
| C3 (MDL scoring) | FALSIFIED | -1/5 | Parsimony-scale tweaks below empirical noise floor |
| **C6 (adaptive)** | **VALIDATED-AS-PREDICTED** | +0/5 | Generic adaptation can't solve mapping problem |

**Cross-experiment pattern:** all four interventions at the
scoring/search-modification layer produce similar Class C rates.
The interventions that genuinely moved the needle in prior work
(reduce_* downweight, multi-trajectory training) operate at the
data/vocabulary level, not at the search-direction level.

## Reproducing

```
python benchmarks/run_heat_equation_adaptive_mvp_c6.py --seeds 5
```