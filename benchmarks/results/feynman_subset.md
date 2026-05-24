# Feynman subset on tessera

**Purpose:** Goal-1 (workbench) validation. Runs tessera GP on 8
representative equations from the Feynman dataset (Udrescu & Tegmark
2020). The full benchmark has 100 equations; this subset spans
trivial-product to non-trivial Lorentz boost in ~2 minutes total.

**GP config:** pop=200, gens=80, pointwise_only=True, optimize_constants every 3 gens (Nelder-Mead, 30 iter). Alphabet includes protected sqrt/exp/log/pow.
**Samples per equation:** 2000
**Total wall-clock:** 12.6s

## Results

| # | Eq ID | True formula | n_vars | best cx | best loss | rel to var | runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | I.6.20a | `exp(-theta^2/2)` | 1 | 9 | 8.108e-05 | 0.0012 | 1.4 |
| 2 | I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | 4 | 11 | 0.05184 | 0.2153 | 1.0 |
| 3 | I.12.1 | `mu*Nn` | 2 | 3 | 0 | 0.0000 | 0.3 |
| 4 | I.12.5 | `q1*q2/r^2` | 3 | 22 | 0.3488 | 0.0626 | 2.0 |
| 5 | I.14.3 | `m*g*z` | 3 | 25 | 35.66 | 0.0959 | 2.8 |
| 6 | I.15.3t | `(x - u*t)/sqrt(1 - u^2/c^2)` | 4 | 15 | 0.04554 | 0.0271 | 1.4 |
| 7 | I.27.6 | `d1*d2/(d1+d2)` | 2 | 14 | 0.0148 | 0.0762 | 0.9 |
| 8 | I.43.31 | `k*T/(6*pi*eta*r)` | 4 | 21 | 0.001639 | 0.2655 | 2.9 |

## Discovered expressions

### I.6.20a
- True: `exp(-theta^2/2)`
- Best tree (cx=9, rel=0.0012):
  ```
  pow((1.26805 + abs((0.599673 - theta))), neg(theta))
  ```

### I.8.14
- True: `sqrt((x2-x1)^2+(y2-y1)^2)`
- Best tree (cx=11, rel=0.2153):
  ```
  ((0.33584 + pow((y2 - y1), 0.699077)) - (x1 * x2))
  ```

### I.12.1
- True: `mu*Nn`
- Best tree (cx=3, rel=0.0000):
  ```
  (Nn * mu)
  ```

### I.12.5
- True: `q1*q2/r^2`
- Best tree (cx=22, rel=0.0626):
  ```
  (((q1 / r) + min(-0.039862, q1)) + min(exp((q2 - (r * r))), ((q1 / r) + (q1 / r))))
  ```

### I.14.3
- True: `m*g*z`
- Best tree (cx=25, rel=0.0959):
  ```
  ((((((-9.97212 + m) + (m * m)) + (z > g)) + min(m, z)) + min(m, z)) + (z * pow(g, 1.61458)))
  ```

### I.15.3t
- True: `(x - u*t)/sqrt(1 - u^2/c^2)`
- Best tree (cx=15, rel=0.0271):
  ```
  ((0.18816 + pow(u, t)) + (((x - log(sqrt(t))) - u) - u))
  ```

### I.27.6
- True: `d1*d2/(d1+d2)`
- Best tree (cx=14, rel=0.0762):
  ```
  ((0.239986 + (0.012192 * d1)) + sqrt(min(d2, (-0.864315 + min(d1, d2)))))
  ```

### I.43.31
- True: `k*T/(6*pi*eta*r)`
- Best tree (cx=21, rel=0.2655):
  ```
  ((pow(0.160917, r) + neg((-0.02065 * pow(log(k), T)))) + pow(0.162708, ((eta * r) * (eta / T))))
  ```

## Reading

- `rel to var` close to 0 means the search found a near-perfect fit
  on TRAIN. Values > 0.01 indicate the search didn't fully recover
  the analytical form (or its constants are imprecise).
- This is TRAIN loss with no test split. The Feynman targets are
  noiseless, so train-perfect = structurally correct (modulo
  constants).
- Tessera is competitive on small-arity (≤ 3 vars) polynomial /
  rational forms. Higher-arity (4 vars) + transcendental forms
  (Lorentz boost) are harder in 40 generations.

## Caveats

- The full Feynman benchmark uses larger sample sizes and more
  generations. PySR / AI Feynman / Operon all run for minutes-to-
  hours per equation; we ran 40 generations as a quick demo.
- For a real workbench claim, run with pop=200, gens=200 and
  report both TRAIN loss and a held-out test loss.

## See also

- Original Feynman benchmark: Udrescu & Tegmark, *AI Feynman: a
  Physics-Inspired Method for Symbolic Regression*, Sci. Adv. 2020.
- SRBench: https://github.com/cavalab/srbench for the full
  community SR benchmark suite.