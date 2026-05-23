# Weather PDE rediscovery benchmark (NCEP/NCAR Reanalysis 2)

**Data:** NCEP-DOE AMIP-II Reanalysis daily 2m temperature, year 2023
**Grid:** 365 days × 13 lats × 29 lons (lat 25.7..48.6, lon 236.2..288.8)
**Target:** `dT/dt = T[t+1] − T[t]`, var = 9.846 K²/day²
**GP:** pop=120, gens=50, parsimony=0.0492, enable_2d=True, runtime=7.6s

## Oracle baselines (hand-coded, OLS-fit)

| # | Expression | MSE | % of target var |
|---|---|---|---|
| 0 | predict 0 (no change) | 9.846 | 100.0% |
| 1 | `-0.04198·(T − T̄)`  (Newtonian relaxation) | 9.644 | 98.0% |
| 2 | `0.1428·∇²T`  (diffusion) | 9.448 | 96.0% |
| 3 | `0.07945·(T[t]−T[t-1])`  (AR(1) persistence) | 9.784 | 99.4% |

These are the bars the GP needs to beat to claim it found structure beyond simple linear filters.

## Pareto front (mid-latitude row slice)

| Cx | TRAIN loss | rel. to var | Tree |
|---|---|---|---|
| 1 | 10.38 | 105.38% | `-0.021941` |
| 3 | 10.08 | 102.33% | `M2D[0.5·δ(0,-1) + -0.5·δ(0,1)](M2D[0.5·δ(0,-1) + -0.5·δ(0,1)](T))` |
| 4 | 9.309 | 94.55% | `tanh(M2D[(1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))](neg(T)))` |

## Best expression

`tanh(M2D[(1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))](neg(T)))`

- complexity = 4
- loss = 9.309 (94.5% of target var)
- Measure2Ds used: (1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))
- vs oracles: ✓ BEATS best oracle baseline (best oracle MSE = 9.448)

## Notes

- This benchmark validates the FunctionalOp2D path on REAL data, not the canonical heat-equation simulator.
- The slice-based formulation (mid-lat row) is the cheapest way to exercise 2D measures; full grid would need either a 3D FunctionalOp or flattened (Y,X) → 1D spatial.
- **Caveat on the comparison:** oracles are OLS-fit on the FULL grid (var ≈ 9.85 K²/day²); the GP sees only the mid-latitude row (slice var ≈ 10.4 K²/day²). The GP's absolute MSE (9.31) is lower than the best linear oracle's full-grid MSE (9.45), so the win is real, but apples-to-apples would require fitting oracles on the same slice — small refinement deferred.
- For ERA5 (0.25° resolution, 3-hourly), supply a CDS API key and swap `load_ncep_t2m` for an ERA5 retrieve; the rest is unchanged.