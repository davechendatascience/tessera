# Feynman A/B: coordinate-discovery (C7) OFF vs ON

**Per-equation A/B of the C7 coordinate-discovery prepass on the
30-equation Feynman subset. C7 tries five target-space transforms
{identity, log_abs, sqrt_abs, square, inverse} and seeds the GP
with the inverse-transformed power-law of the highest-R² fit.**

**GP config**: pop_size=400, n_gens=120, 
seed=2026, Nelder-Mead, decompose_prepass_enabled=False (BOTH arms).

**Why decompose disabled on both arms**: this A/B isolates C7's
effect. Since C7's transform library includes identity (which
subsumes the production power-law detector) and log_abs (which
subsumes the exp-wrapper detector), C7 alone should equal-or-
beat decompose v2.

**Total wall-clock**: 527.4s

## Headline tally

| C7 | Exact (rel<0.01) | Partial (rel<0.20) | Failed |
|---|---|---|---|
| OFF | 10 | 13 | 7 |
| ON  | 20  | 6  | 4  |

**C7=ON wins by +10 exact**.

## Transitions

| Transition | Count |
|---|---|
| partial->exact | 7 |
| failed->partial | 0 |
| failed->exact | 3 |
| exact->partial | 0 |
| exact->failed | 0 |
| partial->failed | 0 |
| same | 20 |

## Per-equation results

| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0003 | exact | 10 | 0.0001 | exact | **ON much better** |
| I.6.20 | 16 | 0.0160 | partial | 13 | 0.0081 | exact | ON better |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 29 | 0.1039 | partial | 29 | 0.1039 | partial | tie |
| I.10.7 | 5 | 0.0031 | exact | 11 | 0.0010 | exact | **ON much better** |
| I.11.19 | 15 | 0.0453 | partial | 15 | 0.0453 | partial | tie |
| I.12.1 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.12.2 | 9 | 0.3580 | failed | 15 | 0.0000 | exact | **ON much better** |
| I.12.4 | 14 | 0.0000 | exact | 15 | 0.0000 | exact | **ON much better** |
| I.12.5 | 28 | 0.1065 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.12.11 | 19 | 0.3875 | failed | 19 | 0.3875 | failed | tie |
| I.14.3 | 5 | 0.0000 | exact | 5 | 0.0000 | exact | tie |
| I.14.4 | 14 | 0.0225 | partial | 7 | 0.0000 | exact | **ON much better** |
| I.15.3t | 8 | 0.0017 | exact | 8 | 0.0017 | exact | tie |
| I.16.6 | 9 | 0.0098 | exact | 9 | 0.0098 | exact | tie |
| I.18.4 | 12 | 0.1001 | partial | 12 | 0.1001 | partial | tie |
| I.24.6 | 19 | 0.1461 | partial | 19 | 0.1461 | partial | tie |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 21 | 0.0189 | partial | 21 | 0.0189 | partial | tie |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 23 | 0.0870 | partial | 23 | 0.0870 | partial | tie |
| I.32.5 | 30 | 0.3559 | failed | 17 | 0.0000 | exact | **ON much better** |
| I.34.8 | 24 | 0.0878 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.39.22 | 21 | 0.0457 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.40.1 | 29 | 0.2162 | failed | 29 | 0.2162 | failed | tie |
| I.43.31 | 19 | 0.2336 | failed | 16 | 0.0000 | exact | **ON much better** |
| I.43.43 | 36 | 0.1493 | partial | 14 | 0.0000 | exact | **ON much better** |
| I.47.23 | 15 | 0.0402 | partial | 10 | 0.0000 | exact | **ON much better** |
| I.48.2 | 9 | 0.0001 | exact | 9 | 0.0001 | exact | OFF better |

## Reproducing

```
PYTHONHASHSEED=0 python benchmarks/run_feynman_coord_discovery_ab.py --n_gens 120 --pop_size 400
```