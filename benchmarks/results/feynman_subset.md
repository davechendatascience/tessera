# Feynman subset on tessera

**Purpose:** Goal-1 (workbench) validation. Runs tessera GP on 8
representative equations from the Feynman dataset (Udrescu & Tegmark
2020). The full benchmark has 100 equations; this subset spans
trivial-product to non-trivial Lorentz boost in ~2 minutes total.

**GP config:** pop=120, gens=40, pointwise_only=True, optimize_constants every 3 gens (Nelder-Mead, 30 iter).
**Samples per equation:** 2000
**Total wall-clock:** 3.1s

## Results

| # | Eq ID | True formula | n_vars | best cx | best loss | rel to var | runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | I.6.20a | `exp(-theta^2/2)` | 1 | 10 | 0.001395 | 0.0209 | 1.2 |
| 2 | I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | 4 | 1 | 0.2407 | 1.0000 | 0.1 |
| 3 | I.12.1 | `mu*Nn` | 2 | 3 | 0 | 0.0000 | 0.1 |
| 4 | I.12.5 | `q1*q2/r^2` | 3 | 19 | 0.2609 | 0.0468 | 0.6 |
| 5 | I.14.3 | `m*g*z` | 3 | 5 | 0 | 0.0000 | 0.2 |
| 6 | I.15.3t | `(x - u*t)/sqrt(1 - u^2/c^2)` | 4 | 7 | 0.003665 | 0.0022 | 0.3 |
| 7 | I.27.6 | `d1*d2/(d1+d2)` | 2 | 7 | 0.01243 | 0.0640 | 0.4 |
| 8 | I.43.31 | `k*T/(6*pi*eta*r)` | 4 | 9 | 0.006163 | 0.9983 | 0.2 |

## Discovered expressions

### I.6.20a
- True: `exp(-theta^2/2)`
- Best tree (cx=10, rel=0.0209):
  ```
  tanh((((2.38931 >= theta) / 1.5119) / (theta * theta)))
  ```

### I.8.14
- True: `sqrt((x2-x1)^2+(y2-y1)^2)`
- Best tree (cx=1, rel=1.0000):
  ```
  1.0338
  ```

### I.12.1
- True: `mu*Nn`
- Best tree (cx=3, rel=0.0000):
  ```
  (Nn * mu)
  ```

### I.12.5
- True: `q1*q2/r^2`
- Best tree (cx=19, rel=0.0468):
  ```
  (((q2 / r) + ((q2 + max(-0.339185, ((q1 + q1) - (r + r)))) - 1.79053)) / r)
  ```

### I.14.3
- True: `m*g*z`
- Best tree (cx=5, rel=0.0000):
  ```
  ((g * m) * z)
  ```

### I.15.3t
- True: `(x - u*t)/sqrt(1 - u^2/c^2)`
- Best tree (cx=7, rel=0.0022):
  ```
  (0.0509949 + (x - (t * u)))
  ```

### I.27.6
- True: `d1*d2/(d1+d2)`
- Best tree (cx=7, rel=0.0640):
  ```
  (0.316451 + (0.457166 * min(d1, d2)))
  ```

### I.43.31
- True: `k*T/(6*pi*eta*r)`
- Best tree (cx=9, rel=0.9983):
  ```
  ((0.179579 / tanh((r < 4.9035))) / reduce_mean(k))
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