# Weather PDE rediscovery benchmark (NCEP/NCAR Reanalysis 2)

**Data:** NCEP-DOE AMIP-II Reanalysis daily 2m temperature, year 2023
**Grid:** 365 days × 13 lats × 29 lons (lat 25.7..48.6, lon 236.2..288.8)
**Target:** `dT/dt = T[t+1] − T[t]`, var = 9.846 K²/day² (full grid); 10.38 K²/day² (mid-lat row slice)
**GP:** pop=120, gens=50, parsimony=0.0492, enable_2d=True, runtime=7.4s

## Oracle baselines — full grid (context)

Fit on all (T, Y, X) interior points. Provided for context; the GP only sees the slice, so use the next table for the apples-to-apples comparison.

| # | Expression | MSE | % of full var |
|---|---|---|---|
| 0 | predict 0 (no change) | 9.846 | 100.0% |
| 1 | `-0.04198·(T − T̄)`  (Newtonian relaxation) | 9.644 | 98.0% |
| 2 | `+0.1428·∇²T`  (diffusion, 2D) | 9.448 | 96.0% |
| 3 | `+0.07945·(T[t]−T[t-1])`  (AR(1) persistence) | 9.784 | 99.4% |

## Oracle baselines — mid-latitude row slice (apples-to-apples)

Same baselines refit on the (T, X) slice the GP sees (var = 10.38 K²/day²). Diffusion drops to a 1D horizontal Laplacian (no y-direction in a single row).

| # | Expression | MSE | % of slice var |
|---|---|---|---|
| 0 | predict 0 (no change) | 10.38 | 100.0% |
| 1 | `-0.05752·(T − T̄)`  (Newtonian relaxation) | 10.09 | 97.2% |
| 2 | `+0.3218·∂²T/∂x²`  (1D horizontal diffusion) | 10.01 | 96.5% |
| 3 | `+0.1092·(T[t]−T[t-1])`  (AR(1) persistence) | 10.25 | 98.8% |

These are the bars the GP needs to beat.

## Pareto front (mid-latitude row slice)

| Cx | TRAIN loss | rel. to slice var | Tree |
|---|---|---|---|
| 1 | 10.38 | 100.00% | `-0.021941` |
| 3 | 10.08 | 97.10% | `M2D[0.5·δ(0,-1) + -0.5·δ(0,1)](M2D[0.5·δ(0,-1) + -0.5·δ(0,1)](T))` |
| 4 | 9.309 | 89.72% | `tanh(M2D[(1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))](neg(T)))` |

## Best expression

`tanh(M2D[(1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))](neg(T)))`

- complexity = 4
- loss = 9.309 (89.7% of slice var)
- Measure2Ds used: (1·δ(0) + -1·δ(3)) ⊗ (1·δ(0) + -1·δ(6))
- vs slice oracles: BEATS best slice oracle (diff_1D, MSE=10.01) by 7.0%

## Notes

- This benchmark validates the FunctionalOp2D path on REAL data, not the canonical heat-equation simulator.
- The slice-based formulation (mid-lat row) is the cheapest way to exercise 2D measures; full grid would need either a 3D FunctionalOp or flattened (Y,X) → 1D spatial.
- Slice oracles are computed on the SAME (T, X) array the GP sees, so the verdict above is apples-to-apples.
- For ERA5 (0.25° resolution, 3-hourly), supply a CDS API key and swap `load_ncep_t2m` for an ERA5 retrieve; the rest is unchanged.