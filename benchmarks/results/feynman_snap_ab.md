# Feynman A/B: constant snapping OFF vs ON

**Per-equation A/B of canonical-constant snapping on the
30-equation Feynman subset.**

**GP config**: pop_size=400, n_gens=120, 
init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, 
Nelder-Mead, pointwise_only=True, seed=2026

**Snap config**: rel_tol=0.005 (0.5%), ~94 candidate values
(integers, small rationals, pi/e/sqrt forms).

**Total wall-clock**: 320.0s

## Headline tally

| Snap | Exact (rel<0.01) | Partial (rel<0.20) | Failed |
|---|---|---|---|
| OFF (baseline) | 12 | 12 | 6 |
| ON             | 11 | 12 | 7 |

**Snap=ON loses on exact-count by -1**. Snap is safe-by-construction so this indicates a bug; investigate.

## Transitions (OFF -> ON)

| Transition | Count |
|---|---|
| partial->exact | 0 |
| failed->partial | 2 |
| failed->exact | 0 |
| exact->partial | 0 |
| exact->failed | 1 |
| partial->failed | 2 |
| same | 25 |

## Per-equation results

| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |
|---|---|---|---|---|---|---|---|
| I.6.20a | 7 | 0.0016 | exact | 7 | 0.0016 | exact | tie |
| I.6.20 | 14 | 0.0034 | exact | 11 | 0.0037 | exact | OFF better |
| I.8.14 | 17 | 0.2525 | failed | 17 | 0.2525 | failed | tie |
| I.9.18 | 17 | 0.3818 | failed | 17 | 0.3818 | failed | tie |
| I.10.7 | 5 | 0.0031 | exact | 5 | 0.0031 | exact | tie |
| I.11.19 | 11 | 0.0000 | exact | 11 | 0.0000 | exact | tie |
| I.12.1 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.12.2 | 27 | 0.9195 | failed | 27 | 0.9195 | failed | tie |
| I.12.4 | 16 | 0.0420 | partial | 16 | 0.0420 | partial | tie |
| I.12.5 | 23 | 0.0266 | partial | 23 | 0.0266 | partial | tie |
| I.12.11 | 12 | 0.0000 | exact | 23 | 0.3735 | failed | OFF much better |
| I.14.3 | 5 | 0.0000 | exact | 5 | 0.0000 | exact | OFF much better |
| I.14.4 | 12 | 0.0261 | partial | 12 | 0.0261 | partial | tie |
| I.15.3t | 10 | 0.0018 | exact | 10 | 0.0018 | exact | tie |
| I.16.6 | 10 | 0.0129 | partial | 10 | 0.0129 | partial | tie |
| I.18.4 | 11 | 0.0876 | partial | 11 | 0.0876 | partial | tie |
| I.24.6 | 25 | 0.1356 | partial | 25 | 0.1172 | partial | ON better |
| I.25.13 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.26.2 | 3 | nan | failed | 3 | nan | failed | tie |
| I.27.6 | 18 | 0.0495 | partial | 18 | 0.0495 | partial | tie |
| I.29.4 | 3 | 0.0000 | exact | 3 | 0.0000 | exact | tie |
| I.30.3 | 25 | 0.2303 | failed | 17 | 0.1024 | partial | **ON much better** |
| I.32.5 | 25 | 0.1964 | partial | 36 | 0.2080 | failed | OFF better |
| I.34.8 | 24 | 0.0666 | partial | 17 | 0.0836 | partial | OFF better |
| I.39.22 | 7 | 0.0000 | exact | 10 | 0.0000 | exact | tie |
| I.40.1 | 35 | 0.1967 | partial | 27 | 0.1734 | partial | ON better |
| I.43.31 | 17 | 0.2527 | failed | 18 | 0.0821 | partial | **ON much better** |
| I.43.43 | 23 | 0.0898 | partial | 25 | 0.2883 | failed | OFF much better |
| I.47.23 | 20 | 0.0398 | partial | 17 | 0.0402 | partial | tie |
| I.48.2 | 11 | 0.0001 | exact | 11 | 0.0001 | exact | tie |

## Discovered expressions (snap=ON arm)

### I.6.20a
- cx=7, rel=0.0016
  ```
  (pow(0.118952, theta) / (0.197866 / theta))
  ```

### I.6.20
- cx=11, rel=0.0037
  ```
  (-0.000167984 + pow(tanh((sigma / sqrt(theta))), (1.94789 * theta)))
  ```

### I.8.14
- cx=17, rel=0.2525
  ```
  (((-0.036391 + neg((x1 * x2))) + (asin(x2) / x2)) + neg((y1 * sin(y2))))
  ```

### I.9.18
- cx=17, rel=0.3818
  ```
  ((-0.062986 + atan2(((G * min(m1, m2)) / x2), asin(m2))) + neg(pow(-0.028364, y1)))
  ```

### I.10.7
- cx=5, rel=0.0031
  ```
  (m0 + (v * v))
  ```

### I.11.19
- cx=11, rel=0.0000
  ```
  (((x1 * y1) + (x2 * y2)) + (x3 * y3))
  ```

### I.12.1
- cx=3, rel=0.0000
  ```
  (Nn * mu)
  ```

### I.12.2
- cx=27, rel=0.9195
  ```
  ((0.0422371 * sqrt(q1)) / (max(abs(q2), ((exp(eps) > -0.000342034) < eps)) >= (((eps + neg(q2)) * (2.29651 * q1)) / max(q1, r))))
  ```

### I.12.4
- cx=16, rel=0.0420
  ```
  (((atan2(0.349436, eps) * (min(q1, q2) / r)) / r) + neg((0.019543 / q2)))
  ```

### I.12.5
- cx=23, rel=0.0266
  ```
  ((q1 + (((q2 / r) * log((q1 * q2))) / r)) + neg(min(q1, ((2 * r) + neg(tanh(r))))))
  ```

### I.12.11
- cx=23, rel=0.3735
  ```
  (((((-3.32625 + B) + (1.88706 * (q * v))) + abs(max(-0.0622697, (B * q)))) + neg(sin(Ef))) + sin(theta))
  ```

### I.14.3
- cx=5, rel=0.0000
  ```
  (m * (g * z))
  ```

### I.14.4
- cx=12, rel=0.0261
  ```
  ((-2.69978 + ((2.42103 * x) * min(k, x))) + cos(x))
  ```

### I.15.3t
- cx=10, rel=0.0018
  ```
  (((0.199446 * u) + x) + neg((t * u)))
  ```

### I.16.6
- cx=10, rel=0.0129
  ```
  (atan2(c, (c / u)) + sin(sin(sin(v))))
  ```

### I.18.4
- cx=11, rel=0.0876
  ```
  (sqrt(r1) * pow(max(r2, (neg(m2) + r1)), 0.516057))
  ```

### I.24.6
- cx=25, rel=0.1172
  ```
  (((((-126.778 + ((-1.79744 + x) * pow(omega, 3.57575))) + (m / 0.0134261)) + exp(omega0)) + exp(x)) + pow(min(m, omega0), x))
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
- cx=18, rel=0.0495
  ```
  (((-0.485858 + (0.229571 * min(d1, d2))) + atan2(pow(d1, d2), d1)) + neg((-0.08539 / d2)))
  ```

### I.29.4
- cx=3, rel=0.0000
  ```
  (omega / c)
  ```

### I.30.3
- cx=17, rel=0.1024
  ```
  (((I0 * pow(n, atan2(1.10259, pow(theta, I0)))) + (n / theta)) + neg(sin(I0)))
  ```

### I.32.5
- cx=36, rel=0.2080
  ```
  ((atan2(min((3.01391 <= a), atan2(q, abs((c * eps)))), c) + pow(c, -4.19123)) + pow(min(min((3.29757 <= q), atan2(a, abs((c * eps)))), atan2(a, abs((c * eps)))), c))
  ```

### I.34.8
- cx=17, rel=0.0836
  ```
  (-1.94756 + (min(B, v) * (q + ((q + ((B + q) / p)) / p))))
  ```

### I.39.22
- cx=10, rel=0.0000
  ```
  (((k * n) * (T / V)) + acos(n))
  ```

### I.40.1
- cx=27, rel=0.1734
  ```
  (((0.831621 + neg(x)) + ((n0 + (sqrt((g * m)) < k)) / asin(sqrt(atan2(n0, x))))) + neg((log((g * m)) / T)))
  ```

### I.43.31
- cx=18, rel=0.0821
  ```
  (sqrt(((T + neg(eta)) + k)) * atan2(((T + neg(eta)) + k), (r / 0.0305389)))
  ```

### I.43.43
- cx=25, rel=0.2883
  ```
  ((((((neg(sigma) + (kappa * v)) + (1.91246 > sigma)) + (v * sin(n))) + cos(n)) + neg(log(sigma))) + sin(kappa))
  ```

### I.47.23
- cx=17, rel=0.0402
  ```
  ((-0.197108 + acos(min(0.915578, atan2(2.24858, pr)))) + neg((abs(gamma) * neg(pow(rho, -0.873908)))))
  ```

### I.48.2
- cx=11, rel=0.0001
  ```
  ((-0.141776 + v) + (m * max(c, (c * c))))
  ```

## Reproducing

```
python benchmarks/run_feynman_snap_ab.py --n_gens 120 --pop_size 400
```