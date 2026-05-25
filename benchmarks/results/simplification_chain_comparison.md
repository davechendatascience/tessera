# Simplification chain head-to-head comparison

**Question:** does CAS (sympy) catch redundancies our existing
hand-rolled chain misses on REAL GP-discovered trees?

Chains compared:
  (a) `ac_only`: just AC normalization
  (b) `canonical`: AC + rule-based (current GP default)
  (c) `full`: canonical + polynomial like-term collection
  (d) `full+cas`: full → sympy CAS fallback

**CAS backend:** sympy
**Total wall-clock:** 1.8s

## Aggregate findings

Across 19 trees (handcrafted + real GP outputs):

| Chain | Total cx | Saved vs orig | Time (ms) | Reductions beyond prev |
|---|---|---|---|---|
| ac_only | 90 | 0 | 0.3 | 0 |
| canonical | 84 | 6 | 0.3 | 3 |
| full | 76 | 14 | 0.5 | 2 |
| full+cas | 62 | 28 | 345.5 | 2 |

## Headline

- **`full+cas` reduces cx beyond `full` in 2/19 trees**
- **`full` reduces cx beyond `canonical` in 2/19 trees**
- **Total cx saved by adding CAS to the chain: 14** (across 19 trees)

CAS overhead: 344.9 ms total (647.1× cost of `full`)

## Per-tree comparison

| Target | Case | orig cx | ac | canonical | full | full+cas |
|---|---|---|---|---|---|---|
| handcrafted | polynomial 2x+3x | 7 | 7 | 7 | **3** | **3** |
| handcrafted | constant_fold 2*3 | 3 | 3 | **1** | **1** | **1** |
| handcrafted | ac (c+a)+b | 5 | 5 | 5 | 5 | 5 |
| handcrafted | x*1 | 3 | 3 | **1** | **1** | **1** |
| handcrafted | polynomial_x2 + 2x2 | 9 | 9 | 9 | **5** | **5** |
| handcrafted | trig sin²+cos² | 11 | 11 | 11 | 11 | **1** |
| handcrafted | rational (2x)/x | 5 | 5 | 5 | 5 | **1** |
| handcrafted | log(exp(x)) | 3 | 3 | **1** | **1** | **1** |
| gaussian_I.6.20a | front[0] | 1 | 1 | 1 | 1 | 1 |
| gaussian_I.6.20a | front[1] | 2 | 2 | 2 | 2 | 2 |
| gaussian_I.6.20a | front[2] | 3 | 3 | 3 | 3 | 3 |
| gaussian_I.6.20a | front[3] | 5 | 5 | 5 | 5 | 5 |
| reduced_I.27.6 | front[0] | 1 | 1 | 1 | 1 | 1 |
| reduced_I.27.6 | front[1] | 5 | 5 | 5 | 5 | 5 |
| reduced_I.27.6 | front[2] | 7 | 7 | 7 | 7 | 7 |
| reduced_I.27.6 | front[3] | 9 | 9 | 9 | 9 | 9 |
| stokes_I.43.31 | front[0] | 1 | 1 | 1 | 1 | 1 |
| stokes_I.43.31 | front[1] | 3 | 3 | 3 | 3 | 3 |
| stokes_I.43.31 | front[2] | 7 | 7 | 7 | 7 | 7 |

## Cases where CAS reduced cx BEYOND `full`

### handcrafted / trig sin²+cos²

- Original (cx=11): `((sin(x) * sin(x)) + (cos(x) * cos(x)))`
- After `full` (cx=11): `((cos(x) * cos(x)) + (sin(x) * sin(x)))`
- After `full+cas` (cx=1): `1`

### handcrafted / rational (2x)/x

- Original (cx=5): `((2 * x) / x)`
- After `full` (cx=5): `((2 * x) / x)`
- After `full+cas` (cx=1): `2`

## Verdict

**CAS adds value.** Saved 14 cx beyond
the existing chain on 2/19 trees.
Use case: catches redundancies in trees containing trig/log/
div/etc. that our hand-rolled passes miss.

## Implication for tessera's simplification pipeline

Current GP scoring path uses `simplify_canonical`. The data here
tells us:

- **`simplify_full` saves cx vs `simplify_canonical`** (8 nodes saved).
  Worth changing the GP default to `simplify_full` (the polynomial pass).

- CAS catches 2 trees `full` missed.
  Worth keeping CAS as an opt-in for redundancy-rich workflows
  (PDE discovery, signal processing). Not necessarily a default.

## Reproducing

```
python benchmarks/run_simplification_chain_comparison.py --pop 80 --gens 30
```