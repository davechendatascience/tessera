# Feynman A/B: decomposition v2 (power-law + exp-wrapper) OFF vs ON

**Per-equation A/B of multiplicative-separability + exp-wrapper
pre-pass on the 30-equation Feynman subset.**

**GP config**: pop_size=400, n_gens=120, 
init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, 
Nelder-Mead, pointwise_only=True, seed=2026

**Pre-pass config**: R² threshold = 0.99. Orchestrator tries
power-law `C · ∏ x_i^{a_i}` first; falls back to exp-wrapper
`±exp(C · ∏ x_i^{a_i})` if power-law rejects.

**Total wall-clock**: 433.4s

## Headline tally

| Prepass | Exact (rel<0.01) | Partial (rel<0.20) | Failed |
|---|---|---|---|
| OFF (baseline) | 10 | 14 | 6 |
| ON             | 20 | 6 | 4 |

**Prepass=ON wins on exact-count by +10**.

## Transitions (OFF -> ON)

| Transition | Count |
|---|---|
| partial->exact | 8 |
| failed->partial | 0 |
| failed->exact | 2 |
| exact->partial | 0 |
| exact->failed | 0 |
| partial->failed | 0 |
| same | 20 |

## Per-equation results

| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0003 | exact | 7 | 0.0000 | exact | **ON much better** |
| I.6.20 | 16 | 0.0160 | partial | 13 | 0.0000 | exact | **ON much better** |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 29 | 0.1039 | partial | 29 | 0.1039 | partial | tie |
| I.10.7 | 5 | 0.0031 | exact | 11 | 0.0010 | exact | **ON much better** |
| I.11.19 | 27 | 0.0437 | partial | 27 | 0.0437 | partial | tie |
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
| I.24.6 | 33 | 0.0844 | partial | 33 | 0.0844 | partial | tie |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 21 | 0.0189 | partial | 21 | 0.0189 | partial | tie |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 23 | 0.0870 | partial | 23 | 0.0870 | partial | tie |
| I.32.5 | 30 | 0.3559 | failed | 17 | 0.0000 | exact | **ON much better** |
| I.34.8 | 24 | 0.0878 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.39.22 | 21 | 0.0457 | partial | 9 | 0.0000 | exact | **ON much better** |
| I.40.1 | 29 | 0.2162 | failed | 29 | 0.2162 | failed | tie |
| I.43.31 | 21 | 0.1129 | partial | 16 | 0.0000 | exact | **ON much better** |
| I.43.43 | 31 | 0.1518 | partial | 14 | 0.0000 | exact | **ON much better** |
| I.47.23 | 19 | 0.0402 | partial | 10 | 0.0000 | exact | **ON much better** |
| I.48.2 | 9 | 0.0001 | exact | 9 | 0.0001 | exact | OFF better |

## Discovered expressions (prepass=ON arm)

### I.6.20a
- cx=7, rel=0.0000
  ```
  exp(neg(((0.5 * theta) * theta)))
  ```

### I.6.20
- cx=13, rel=0.0000
  ```
  exp(neg((((0.5 * theta) * theta) * (1 / (sigma * sigma)))))
  ```

### I.8.14
- cx=17, rel=0.2525
  ```
  (((-0.036391 + neg((x1 * x2))) + (asin(x2) / x2)) + neg((y1 * sin(y2))))
  ```

### I.9.18
- cx=29, rel=0.1039
  ```
  (atan2(G, (log(z2) * (y2 / m2))) / sqrt((((log(y2) * (-2.47065 + m1)) * (neg(x2) + y1)) * atan2((y1 / z1), (-1.15625 + x1)))))
  ```

### I.10.7
- cx=11, rel=0.0010
  ```
  (((1.1078 * m0) * pow(c, -0.0580633)) * pow(v, 0.0410907))
  ```

### I.11.19
- cx=27, rel=0.0437
  ```
  ((((-0.0137364 + (2.64659 * x3)) + (x1 * y1)) + (x2 * (((-3.81719 + y2) + y3) + ((y2 * y3) < y2)))) + max(x2, y3))
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
- cx=19, rel=0.3875
  ```
  (((-1.51235 + (B * q)) + (1.70403 * (q * v))) + ((q * theta) / (theta < Ef)))
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
- cx=8, rel=0.0017
  ```
  ((1.01833 * x) + neg((t * u)))
  ```

### I.16.6
- cx=9, rel=0.0098
  ```
  (v + (u * (1.03011 + (v / -2.30067))))
  ```

### I.18.4
- cx=12, rel=0.1001
  ```
  ((atan2(m2, m2) / (3.34664 * r2)) + sqrt((r1 * r2)))
  ```

### I.24.6
- cx=33, rel=0.0844
  ```
  (((((neg(x) + ((-1.27203 + x) * pow(omega, 3.39431))) + (4.82017 * min(exp(x), pow(m, omega0)))) + exp(x)) + max(omega0, exp(m))) + pow(min(m, x), omega0))
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
- cx=21, rel=0.0189
  ```
  ((-0.211232 + (log(d1) * (((0.0795175 * d2) * log(d1)) + (0.0319307 >= d2)))) + atan2(pow(d1, d2), d1))
  ```

### I.29.4
- cx=3, rel=0.0000
  ```
  (omega / c)
  ```

### I.30.3
- cx=23, rel=0.0870
  ```
  ((((I0 * pow(-1.78132, n)) + (I0 / theta)) + log(theta)) + neg(pow((-13.534 + (I0 * pow(-1.90784, n))), theta)))
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
- cx=29, rel=0.2162
  ```
  ((neg(x) + (((g < k) + max(n0, x)) / asin(sqrt(atan2(sqrt(m), 1.07717))))) + neg(pow(T, ((m / n0) * (neg(k) + x)))))
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
  ((-0.227845 + v) + (m * pow(c, 2.00888)))
  ```

## Reproducing

```
python benchmarks/run_feynman_decompose_v2_ab.py --n_gens 120 --pop_size 400
```