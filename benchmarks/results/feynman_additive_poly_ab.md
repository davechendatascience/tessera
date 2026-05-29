# Feynman A/B: decompose v2 vs decompose v2 + C8 additive polynomial

**Tests Conjecture C8 (additive polynomial structure detector). Both arms
enable production decompose v2; ON arm adds a C8 polynomial seed via
`precomputed_seed_trees`. The delta isolates C8's incremental contribution.**

**GP config**: pop_size=400, n_gens=120, seed=2026, 
Nelder-Mead, decompose_prepass_enabled=True (BOTH arms), 
C8 max_degree=3, top_n=8.

**Total wall-clock**: 621.6s

## Headline tally

| Arm | Exact | Partial | Failed |
|---|---|---|---|
| OFF (decompose v2 only) | 20 | 6 | 4 |
| ON (decompose v2 + C8)  | 21 | 5 | 4 |

**C8 ON wins by +1 exact** beyond decompose v2.

## Transitions

| Transition | Count |
|---|---|
| partial->exact | 1 |
| failed->partial | 0 |
| failed->exact | 0 |
| exact->partial | 0 |
| exact->failed | 0 |
| partial->failed | 0 |
| same | 29 |

## Per-equation results

| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0000 | exact | 7 | 0.0000 | exact | tie |
| I.6.20 | 13 | 0.0000 | exact | 13 | 0.0000 | exact | tie |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 29 | 0.1039 | partial | 29 | 0.1039 | partial | tie |
| I.10.7 | 11 | 0.0010 | exact | 11 | 0.0010 | exact | tie |
| I.11.19 | 15 | 0.0453 | partial | 11 | 0.0000 | exact | **ON much better** |
| I.12.1 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.12.2 | 15 | 0.0000 | exact | 15 | 0.0000 | exact | tie |
| I.12.4 | 15 | 0.0000 | exact | 15 | 0.0000 | exact | tie |
| I.12.5 | 9 | 0.0000 | exact | 9 | 0.0000 | exact | tie |
| I.12.11 | 19 | 0.3875 | failed | 19 | 0.3875 | failed | tie |
| I.14.3 | 5 | 0.0000 | exact | 5 | 0.0000 | exact | tie |
| I.14.4 | 7 | 0.0000 | exact | 7 | 0.0000 | exact | tie |
| I.15.3t | 8 | 0.0017 | exact | 8 | 0.0017 | exact | tie |
| I.16.6 | 9 | 0.0098 | exact | 9 | 0.0098 | exact | tie |
| I.18.4 | 12 | 0.1001 | partial | 12 | 0.1001 | partial | tie |
| I.24.6 | 19 | 0.1461 | partial | 19 | 0.1461 | partial | tie |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 21 | 0.0189 | partial | 21 | 0.0189 | partial | tie |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 23 | 0.0870 | partial | 23 | 0.0870 | partial | tie |
| I.32.5 | 17 | 0.0000 | exact | 17 | 0.0000 | exact | tie |
| I.34.8 | 9 | 0.0000 | exact | 9 | 0.0000 | exact | tie |
| I.39.22 | 9 | 0.0000 | exact | 9 | 0.0000 | exact | tie |
| I.40.1 | 29 | 0.2162 | failed | 29 | 0.2162 | failed | tie |
| I.43.31 | 16 | 0.0000 | exact | 16 | 0.0000 | exact | tie |
| I.43.43 | 14 | 0.0000 | exact | 14 | 0.0000 | exact | tie |
| I.47.23 | 10 | 0.0000 | exact | 10 | 0.0000 | exact | tie |
| I.48.2 | 9 | 0.0001 | exact | 9 | 0.0001 | exact | tie |

## Reproducing

```
PYTHONHASHSEED=0 python benchmarks/run_feynman_additive_poly_ab.py --n_gens 120 --pop_size 400 --c8_max_degree 3
```