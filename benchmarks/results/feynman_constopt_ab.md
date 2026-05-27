# Feynman A/B: scipy Nelder-Mead vs scipy BFGS const-opt

**Per-equation A/B comparison of constant optimization methods on
the 30-equation Feynman subset.**

**GP config**: pop_size=400, n_gens=120, 
init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, 
pointwise_only=True, seed=2026

**Total wall-clock**: 257.9s

## Headline tally

| Method | Exact (rel<0.01) | Partial (rel<0.20) | Failed |
|---|---|---|---|
| Nelder-Mead | 12 | 12 | 6 |
| BFGS | 11 | 14 | 5 |

**BFGS loses on exact-count by -1**. Investigate.

## Transitions (NM → BFGS)

| Transition | Count |
|---|---|
| partial→exact | 1 |
| failed→partial | 2 |
| failed→exact | 0 |
| exact→partial | 2 |
| exact→failed | 0 |
| partial→failed | 1 |
| same | 24 |

## Per-equation results

| Eq | NM cx | NM rel | NM verdict | BFGS cx | BFGS rel | BFGS verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0003 | exact | 19 | 0.0005 | exact | NM better |
| I.6.20 | 12 | 0.0270 | partial | 13 | 0.0360 | partial | NM better |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 12 | 0.3585 | failed | 16 | 0.3604 | failed | tie |
| I.10.7 | 5 | 0.0031 | exact | 5 | 0.0031 | exact | tie |
| I.11.19 | 23 | 0.0895 | partial | 25 | 0.1478 | partial | NM better |
| I.12.1 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.12.2 | 12 | 0.0000 | exact | 19 | 0.0525 | partial | NM much better |
| I.12.4 | 12 | 0.3385 | failed | 12 | 0.1801 | partial | BFGS better |
| I.12.5 | 26 | 0.0537 | partial | 27 | 0.0812 | partial | NM better |
| I.12.11 | 12 | 0.0000 | exact | 12 | 0.0000 | exact | tie |
| I.14.3 | 5 | 0.0000 | exact | 5 | 0.0000 | exact | tie |
| I.14.4 | 20 | 0.0253 | partial | 26 | 0.0311 | partial | NM better |
| I.15.3t | 8 | 0.0022 | exact | 8 | 0.0022 | exact | tie |
| I.16.6 | 10 | 0.0086 | exact | 10 | 0.0086 | exact | tie |
| I.18.4 | 8 | 0.0982 | partial | 19 | 0.1365 | partial | NM better |
| I.24.6 | 24 | 0.1714 | partial | 26 | 0.0928 | partial | BFGS better |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 14 | 0.0334 | partial | 7 | 0.0000 | exact | **BFGS much better** |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 13 | 0.0797 | partial | 14 | 0.0688 | partial | BFGS better |
| I.32.5 | 15 | 0.6073 | failed | 15 | 0.6073 | failed | tie |
| I.34.8 | 29 | 0.1483 | partial | 24 | 0.1051 | partial | BFGS better |
| I.39.22 | 20 | 0.0768 | partial | 20 | 0.0370 | partial | **BFGS much better** |
| I.40.1 | 26 | 0.2251 | failed | 28 | 0.1608 | partial | BFGS better |
| I.43.31 | 15 | 0.0136 | partial | 9 | 0.6888 | failed | NM much better |
| I.43.43 | 15 | 0.0099 | exact | 22 | 0.0290 | partial | NM much better |
| I.47.23 | 19 | 0.0363 | partial | 23 | 0.0809 | partial | NM much better |
| I.48.2 | 8 | 0.0006 | exact | 8 | 0.0006 | exact | tie |

## Reproducing

```
python benchmarks/run_feynman_constopt_ab.py --n_gens 120 --pop_size 400
```