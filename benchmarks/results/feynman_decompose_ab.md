# Feynman A/B: decomposition pre-pass OFF vs ON

**Per-equation A/B of multiplicative-separability pre-pass on
the 30-equation Feynman subset.**

**GP config**: pop_size=400, n_gens=120, 
init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, 
Nelder-Mead, pointwise_only=True, seed=2026

**Pre-pass config**: R² threshold = 0.99 for accepting a power-law fit.
Detected fits are converted to seed trees `C · ∏ x_i^{a_i}` and injected
into the initial population. Selection determines if seeds survive.

**Total wall-clock**: 315.1s

## Headline tally

| Prepass | Exact (rel<0.01) | Partial (rel<0.20) | Failed |
|---|---|---|---|
| OFF (baseline) | 9 | 16 | 5 |
| ON             | 18 | 8 | 4 |

**Prepass=ON wins on exact-count by +9**.

## Transitions (OFF -> ON)

| Transition | Count |
|---|---|
| partial->exact | 8 |
| failed->partial | 0 |
| failed->exact | 1 |
| exact->partial | 0 |
| exact->failed | 0 |
| partial->failed | 0 |
| same | 21 |

## Per-equation results

| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0003 | exact | 7 | 0.0003 | exact | tie |
| I.6.20 | 14 | 0.0591 | partial | 14 | 0.0591 | partial | tie |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 14 | 0.3700 | failed | 14 | 0.3700 | failed | tie |
| I.10.7 | 5 | 0.0031 | exact | 11 | 0.0010 | exact | **ON much better** |
| I.11.19 | 27 | 0.1095 | partial | 27 | 0.1095 | partial | tie |
| I.12.1 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.12.2 | 10 | 0.3594 | failed | 15 | 0.0000 | exact | **ON much better** |
| I.12.4 | 29 | 0.1730 | partial | 15 | 0.0000 | exact | **ON much better** |
| I.12.5 | 32 | 0.0414 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.12.11 | 31 | 0.1912 | partial | 31 | 0.1912 | partial | tie |
| I.14.3 | 5 | 0.0000 | exact | 5 | 0.0000 | exact | tie |
| I.14.4 | 15 | 0.0218 | partial | 7 | 0.0000 | exact | **ON much better** |
| I.15.3t | 14 | 0.0320 | partial | 14 | 0.0320 | partial | tie |
| I.16.6 | 10 | 0.0086 | exact | 10 | 0.0086 | exact | tie |
| I.18.4 | 16 | 0.0874 | partial | 16 | 0.0874 | partial | tie |
| I.24.6 | 29 | 0.1032 | partial | 29 | 0.1032 | partial | tie |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 15 | 0.0722 | partial | 15 | 0.0722 | partial | tie |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 17 | 0.1852 | partial | 17 | 0.1852 | partial | tie |
| I.32.5 | 19 | 0.1758 | partial | 17 | 0.0000 | exact | **ON much better** |
| I.34.8 | 24 | 0.1338 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.39.22 | 7 | 0.0000 | exact | 9 | 0.0000 | exact | OFF better |
| I.40.1 | 23 | 0.2100 | failed | 23 | 0.2100 | failed | tie |
| I.43.31 | 13 | 0.0479 | partial | 16 | 0.0000 | exact | **ON much better** |
| I.43.43 | 33 | 0.1758 | partial | 14 | 0.0000 | exact | **ON much better** |
| I.47.23 | 22 | 0.0347 | partial | 10 | 0.0000 | exact | **ON much better** |
| I.48.2 | 9 | 0.0001 | exact | 9 | 0.0001 | exact | OFF better |

## Discovered expressions (prepass=ON arm)

### I.6.20a
- cx=7, rel=0.0003
  ```
  atan2(0.956843, pow((1.35075 * theta), theta))
  ```

### I.6.20
- cx=14, rel=0.0591
  ```
  min(min(sigma, tanh(sigma)), atan2(((sigma * sigma) / theta), (theta * theta)))
  ```

### I.8.14
- cx=17, rel=0.2525
  ```
  (((-0.036391 + neg((x1 * x2))) + (asin(x2) / x2)) + neg((y1 * sin(y2))))
  ```

### I.9.18
- cx=14, rel=0.3700
  ```
  pow(atan2(min(G, m1), 2.58525), (neg(m2) + min(min(x2, y2), z2)))
  ```

### I.10.7
- cx=11, rel=0.0010
  ```
  (((1.1078 * m0) * pow(c, -0.0580633)) * pow(v, 0.0410907))
  ```

### I.11.19
- cx=27, rel=0.1095
  ```
  ((((((neg(x1) + (2.09355 * x3)) + (x1 * y1)) + neg((x2 >= x1))) + pow(-1.63039, y3)) + pow(-1.66503, x2)) + pow(1.63219, y2))
  ```

### I.12.1
- cx=3, rel=0.0000
  ```
  (Nn * mu)
  ```

### I.12.2
- cx=15, rel=0.0000
  ```
  ((((0.0795775 * q1) * q2) * (1 / eps)) * (1 / (r * r)))
  ```

### I.12.4
- cx=15, rel=0.0000
  ```
  ((((0.0795775 * q1) * q2) * (1 / eps)) * (1 / (r * r)))
  ```

### I.12.5
- cx=9, rel=0.0000
  ```
  ((q1 * q2) * (1 / (r * r)))
  ```

### I.12.11
- cx=31, rel=0.1912
  ```
  ((((((-2.57479 + B) + Ef) + (1.74902 * (q * v))) + ((B * q) * asin(theta))) + (theta * (Ef + (-2.69311 * theta)))) + neg((q > Ef)))
  ```

### I.14.3
- cx=5, rel=0.0000
  ```
  ((g * m) * z)
  ```

### I.14.4
- cx=7, rel=0.0000
  ```
  (((0.5 * k) * x) * x)
  ```

### I.15.3t
- cx=14, rel=0.0320
  ```
  (((x + neg((u * u))) + neg(log(t))) + pow(0.002791, u))
  ```

### I.16.6
- cx=10, rel=0.0086
  ```
  ((neg(pow(-0.027296, c)) + tanh(u)) + tanh(v))
  ```

### I.18.4
- cx=16, rel=0.0874
  ```
  (((0.0585424 * r1) + exp(atan2(r1, atan2(m2, sin(sin(r2)))))) + neg(sin(r2)))
  ```

### I.24.6
- cx=29, rel=0.1032
  ```
  ((((((-1.76523 + x) * pow(omega, 3.48598)) + exp(x)) + neg((m / -0.0260687))) + neg(reduce_mean(m))) + pow(min(omega0, x), (1.14634 + min(3.49955, m))))
  ```

### I.25.13
- cx=3, rel=0.0000
  ```
  (q / C)
  ```

### I.26.2
- cx=3, rel=nan
  ```
  (n * theta2)
  ```

### I.27.6
- cx=15, rel=0.0722
  ```
  (((0.099351 * d1) + atan2(-0.056674, d1)) + atan2(pow(d2, d1), (d1 * d2)))
  ```

### I.29.4
- cx=3, rel=0.0000
  ```
  (omega / c)
  ```

### I.30.3
- cx=17, rel=0.1852
  ```
  (((-3.74874 + (2.43254 * (I0 * n))) + (-13.0046 * (theta * theta))) + (n / theta))
  ```

### I.32.5
- cx=17, rel=0.0000
  ```
  ((((((0.0530516 * a) * a) * q) * q) * (1 / eps)) * pow(c, -3))
  ```

### I.34.8
- cx=9, rel=0.0000
  ```
  (((B * q) * v) * (1 / p))
  ```

### I.39.22
- cx=9, rel=0.0000
  ```
  (((T * k) * n) * (1 / V))
  ```

### I.40.1
- cx=23, rel=0.2100
  ```
  (((n0 / sqrt((m + min(m, (g * x))))) + atan2(T, g)) + neg(max(min(0.010436, g), min(g, x))))
  ```

### I.43.31
- cx=16, rel=0.0000
  ```
  ((((0.0530516 * T) * k) * (1 / r)) * (sign((eta + r)) / eta))
  ```

### I.43.43
- cx=14, rel=0.0000
  ```
  ((((kappa * v) * v) * (1 / sigma)) * (sign(kappa) / n))
  ```

### I.47.23
- cx=10, rel=0.0000
  ```
  ((sqrt(gamma) * sqrt(pr)) * (1 / sqrt(rho)))
  ```

### I.48.2
- cx=9, rel=0.0001
  ```
  ((m * pow(c, 2.00426)) + pow(v, c))
  ```

## Reproducing

```
python benchmarks/run_feynman_decompose_ab.py --n_gens 120 --pop_size 400
```