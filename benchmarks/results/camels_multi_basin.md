# CAMELS multi-basin generalization sweep

**Same polish pipeline (sufficient_stats on lagged P features)
applied to 5 reference basins spanning humid/arid/snowmelt regimes.**

**Config:** pop=300, gens=120, polish-every=5, lags=[1, 3, 7, 14], seed=2026, period=1990-2020

## Summary table

| Basin | Climate | persist tr | persist te | engin tr | engin te | GP cx | GP tr | GP te | ratio | Verdict | Δ vs persist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01013500 | humid temp | 0.007191 | 0.008276 | 0.004691 | 0.00553 | 21 | 0.005028 | 0.005944 | 1.18 | A-generic | +30.1% |
| 02479155 | humid subtrop | 0.3108 | 0.3076 | 0.1556 | 0.1685 | 25 | 0.2008 | 0.2217 | 1.10 | A-generic | +35.4% |
| 06614800 | semi-arid mtn | 0.01217 | 0.008435 | 0.012 | 0.008289 | 21 | 0.01203 | 0.00832 | 0.69 | trivial | +1.2% |
| 09497500 | arid | 0.04678 | 0.0402 | 0.0348 | 0.03314 | 21 | 0.03597 | 0.03387 | 0.94 | C-mechanism-ish | +23.1% |
| 11264500 | Med mtn | 0.03466 | 0.04247 | 0.02957 | 0.03441 | 21 | 0.02979 | 0.03478 | 1.17 | C-mechanism-ish | +14.0% |

`Δ vs persist` = TRAIN improvement over persistence (positive is better).

## Verdict tally

- **C-mechanism-ish**: 2/5
- **A-generic**: 2/5
- **trivial**: 1/5

## Per-basin discovered forms

### 01013500 — Fish River near Fort Kent, ME

- area = 2253 km², n = 11,315 daily samples
- persistence: TRAIN=0.007191, TEST=0.008276
- engineering: TRAIN=0.004691, TEST=0.00553
- GP best: cx=21, TRAIN=0.005028, TEST=0.005944, ratio=1.18, verdict=**A-generic**

```
((((((0.00757653 * P) + (-0.000326823 * P_lag1)) + (-0.00174155 * P_lag14)) + (-0.000331433 * P_lag3)) + (-0.00112111 * P_lag7)) + Q)
```

### 02479155 — Black Creek at Wiggins, MS

- area = 520 km², n = 11,314 daily samples
- persistence: TRAIN=0.3108, TEST=0.3076
- engineering: TRAIN=0.1556, TEST=0.1685
- GP best: cx=25, TRAIN=0.2008, TEST=0.2217, ratio=1.10, verdict=**A-generic**

```
(((((((0.00590708 * P) + (-0.0163651 * P_lag1)) + (-0.00241925 * P_lag14)) + (-0.00872997 * P_lag3)) + (-0.00284011 * P_lag7)) + Q) + atan2(P, T))
```

### 06614800 — Cache la Poudre, CO

- area = 379 km², n = 11,315 daily samples
- persistence: TRAIN=0.01217, TEST=0.008435
- engineering: TRAIN=0.012, TEST=0.008289
- GP best: cx=21, TRAIN=0.01203, TEST=0.00832, ratio=0.69, verdict=**trivial**

```
((((((0.00156042 * P) + (-0.00276059 * P_lag1)) + (0.000574333 * P_lag14)) + (1.58036e-05 * P_lag3)) + (0.00130044 * P_lag7)) + Q)
```

### 09497500 — Salt River, AZ

- area = 11154 km², n = 11,315 daily samples
- persistence: TRAIN=0.04678, TEST=0.0402
- engineering: TRAIN=0.0348, TEST=0.03314
- GP best: cx=21, TRAIN=0.03597, TEST=0.03387, ratio=0.94, verdict=**C-mechanism-ish**

```
((((((0.0260779 * P) + (-0.0084768 * P_lag1)) + (-0.00115466 * P_lag14)) + (-0.00766132 * P_lag3)) + (-0.000381218 * P_lag7)) + Q)
```

### 11264500 — Merced River, CA

- area = 469 km², n = 11,315 daily samples
- persistence: TRAIN=0.03466, TEST=0.04247
- engineering: TRAIN=0.02957, TEST=0.03441
- GP best: cx=21, TRAIN=0.02979, TEST=0.03478, ratio=1.17, verdict=**C-mechanism-ish**

```
((((((0.00957109 * P) + (-0.00580994 * P_lag1)) + (1.35745e-05 * P_lag14)) + (-0.000619296 * P_lag3)) + (-0.000505247 * P_lag7)) + Q)
```

## Reading the sweep

- Class C across all basins would mean the polish-on-lagged-P
  pattern captures the catchment's IUH at all climate regimes.
  Realistic expectation: A-generic across the board, with the gap
  to engineering reflecting the fixed lag basis {0,1,3,7,14}.
- The *shape* of the discovered IUH weights is interpretable: 
  large positive coefficient on today's P + slow decay on lagged
  P matches storm-runoff catchments; large positive coefficients
  on long-lagged P (week+) suggest snowmelt buffering.
- A Class-B verdict on any basin would mean the polish injected a
  TRAIN-specific scalar pattern (failure mode the reduce_*
  downweight already addressed for non-trading benchmarks).

## Reproducing

```
python benchmarks/run_camels_multi_basin.py --pop 300 --gens 120
```