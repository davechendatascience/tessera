# Weather PDE rediscovery: TRAIN/TEST split + Class A/B/C taxonomy

**Data:** NCEP-DOE AMIP-II Reanalysis daily 2m temperature, year 2023
**Grid:** 365 days × 13 lats × 29 lons (lat 25.7..48.6, lon 236.2..288.8)
**Split:** chronological 75/25 — TRAIN = first 273 days (days 1-273), TEST = remaining (days 274-365)
**Target:** `dT/dt = T[t+1] − T[t]`; var(TRAIN) = 9.702 K²/day², var(TEST) = 12.33 K²/day²
**GP:** pop=300, gens=120, parsimony=0.0485, enable_2d=True, runtime=7.7s

## Oracle baselines (TRAIN-fit, TRAIN+TEST evaluated)

Each baseline's coefficient is fit on TRAIN only, then evaluated on both halves.
If the baseline form is genuine physics, TRAIN ≈ TEST. If it's TRAIN-overfit, TEST blows up.

| Oracle | TRAIN MSE | TRAIN %var | TEST MSE | TEST %var | TEST/TRAIN |
|---|---|---|---|---|---|
| predict 0 | 9.703 | 100.0% | 12.37 | 100.3% | 1.27 |
| Newton relax (α·(T−T̄)) | 9.428 | 97.2% | 12.13 | 98.4% | 1.29 |
| 1D diffusion (α·∂²T/∂x²) | 9.343 | 96.3% | 12.06 | 97.8% | 1.29 |
| AR(1) (β·(T[t]−T[t-1])) | 9.618 | 99.1% | 12.16 | 98.6% | 1.26 |

## Pareto front — TRAIN/TEST + class verdict

Each candidate's TRAIN loss is what the GP optimized; TEST loss is from
evaluating the same tree on the held-out slice. Class verdict per the
taxonomy in `docs/research/from_data_to_mechanism.md` §5.

| Cx | TRAIN | TRAIN %var | TEST | TEST %var | TEST/TRAIN | Verdict | Tree |
|---|---|---|---|---|---|---|---|
| 1 | 9.702 | 100.0% | 12.39 | 100.4% | 1.28 | trivial | `0.0361187` |
| 3 | 9.65 | 99.5% | 12.27 | 99.5% | 1.27 | trivial | `M2D[1·δ(0,-1) + -2·δ(0,0) + 1·δ(0,1)](sqrt(T))` |
| 4 | 9.345 | 96.3% | 12.04 | 97.6% | 1.29 | A-generic | `(0.385496 * M2D[1·δ(0,-1) + -2·δ(0,0) + 1·δ(0,1)](T))` |
| 6 | 9.343 | 96.3% | 12.06 | 97.8% | 1.29 | A-generic | `(0.0393012 + (0.385631 * M2D[1·δ(0,-1) + -2·δ(0,0) + 1·δ(0,1)](T)))` |
| 9 | 9.343 | 96.3% | 12.05 | 97.7% | 1.29 | A-generic | `((0.385496 * M2D[1·δ(0,-1) + -2·δ(0,0) + 1·δ(0,1)](T)) + neg((-7.11765 / T)))` |

## Class tally

- **A-generic**: 3
- **trivial**: 2

## Best Class-A candidate (generic, no mechanism found)

```
(0.385496 * M2D[1·δ(0,-1) + -2·δ(0,0) + 1·δ(0,1)](T))
```

- cx = 4
- TRAIN MSE = 9.345  (96.3% of TRAIN var)
- TEST MSE = 12.04  (97.6% of TEST var)
- TEST/TRAIN ratio = 1.29
- verdict: **A-generic**

## vs best oracle

- best TRAIN oracle: **diff_1D** at MSE=9.343
- best TEST oracle: **diff_1D** at MSE=12.06
- GP TRAIN does NOT beat best oracle
- GP TEST beats best oracle by 0.2%

## Reading the result

- A Class-C result is the goal: TRAIN beats the trivial baselines AND
  TEST follows along (TEST/TRAIN ≤ 1.3). That's the closest equivalent
  on real data of the heat-equation 'canonical mechanism' verdict.
- A Class-B result means the GP found a TRAIN-specific pattern that
  doesn't transfer. On weather this often shows up as a candidate that
  uses scalar reductions or seasonally-biased windowing — same failure
  mode the `reduce_*` downweight (CHANGELOG 2026-05-25) was designed for.
- A Class-A result is honest underfit: nothing meaningful found, but
  also no false positive. Better than B; worse than C.
- 'trivial' = predict-zero level. Means the GP search didn't find any
  signal at all, even on TRAIN.

## Reproducing

```
python benchmarks/run_weather_pde_traintest.py --year 2023 --pop 300 --gens 120
```