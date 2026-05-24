# Feynman extended on tessera

**Equations:** 30 (representative subset of the canonical 100)
**GP config:** pop=400, gens=120, pointwise_only=True, optimize_constants every 3 gens, Nelder-Mead 30 iter.
**Samples per equation:** 2000
**Total wall-clock:** 369.5s (6.2 min)
**Alphabet:** add/sub/mul/div/min/max, gt/lt/ge/le, tanh/abs/sign/neg/step, sqrt/exp/log/pow, reduce_mean/max/sum/std.

## Headline

- **Exact** (rel < 0.01): 8 / 30
- **Partial** (0.01 ≤ rel < 0.20): 15 / 30
- **Failed** (rel ≥ 0.20): 7 / 30

## Results table

| # | Eq ID | True formula | n_vars | best cx | best rel | runtime (s) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | I.6.20a | `exp(-theta^2/2)` | 1 | 9 | 0.0007 | 4.3 | **exact** |
| 2 | I.6.20 | `exp(-(theta/sigma)^2/2)` | 2 | 14 | 0.0113 | 4.1 | partial |
| 3 | I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | 4 | 22 | 0.2857 | 2.6 | failed |
| 4 | I.9.18 | `G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)` | 9 | 13 | 0.4391 | 3.3 | failed |
| 5 | I.10.7 | `m0/sqrt(1-v^2/c^2)` | 3 | 9 | 0.0024 | 0.9 | **exact** |
| 6 | I.11.19 | `x1*y1+x2*y2+x3*y3` | 6 | 21 | 0.0709 | 11.0 | partial |
| 7 | I.12.1 | `mu*Nn` | 2 | 3 | 0.0000 | 1.2 | **exact** |
| 8 | I.12.2 | `q1*q2/(4*pi*eps*r^2)` | 4 | 11 | 0.4654 | 1.8 | failed |
| 9 | I.12.4 | `q1*q2/(4*pi*eps*r^2)` | 4 | 13 | 0.5062 | 2.2 | failed |
| 10 | I.12.5 | `q1*q2/r^2` | 3 | 7 | 0.0000 | 3.6 | **exact** |
| 11 | I.12.11 | `q*(Ef+B*v*sin(theta))` | 5 | 25 | 0.0762 | 18.1 | partial |
| 12 | I.14.3 | `m*g*z` | 3 | 5 | 0.0000 | 5.2 | **exact** |
| 13 | I.14.4 | `0.5*k*x^2` | 2 | 13 | 0.0120 | 10.5 | partial |
| 14 | I.15.3t | `(x-u*t)/sqrt(1-u^2/c^2)` | 4 | 11 | 0.0169 | 10.7 | partial |
| 15 | I.16.6 | `(u+v)/(1+u*v/c^2)` | 3 | 7 | 0.0128 | 7.3 | partial |
| 16 | I.18.4 | `(m1*r1+m2*r2)/(m1+m2)` | 4 | 11 | 0.1861 | 7.7 | partial |
| 17 | I.24.6 | `0.5*m*(omega^2+omega0^2)*x^2` | 4 | 23 | 0.1202 | 18.5 | partial |
| 18 | I.25.13 | `q/C` | 2 | 3 | 0.0000 | 1.3 | **exact** |
| 19 | I.26.2 | `arcsin(n*sin(theta2))` | 2 | 5 | nan | 2.3 | failed |
| 20 | I.27.6 | `d1*d2/(d1+d2)` | 2 | 12 | 0.0214 | 9.2 | partial |
| 21 | I.29.4 | `omega/c` | 2 | 3 | 0.0000 | 3.7 | **exact** |
| 22 | I.30.3 | `I0*sin(n*theta/2)^2/sin(theta/2)^2` | 3 | 21 | 0.2095 | 11.5 | failed |
| 23 | I.32.5 | `q^2*a^2/(6*pi*eps*c^3)` | 4 | 26 | 0.1683 | 45.1 | partial |
| 24 | I.34.8 | `q*v*B/p` | 4 | 31 | 0.0109 | 29.8 | partial |
| 25 | I.39.22 | `n*k*T/V` | 4 | 22 | 0.0783 | 22.0 | partial |
| 26 | I.40.1 | `n0*exp(-m*g*x/(k*T))` | 6 | 18 | 0.0485 | 26.6 | partial |
| 27 | I.43.31 | `k*T/(6*pi*eta*r)` | 4 | 5 | 0.9361 | 3.3 | failed |
| 28 | I.43.43 | `kappa*v^2/(n*sigma)` | 4 | 31 | 0.0415 | 42.2 | partial |
| 29 | I.47.23 | `sqrt(gamma*pr/rho)` | 3 | 33 | 0.0210 | 52.3 | partial |
| 30 | I.48.2 | `m*c^2/sqrt(1-v^2/c^2)` | 3 | 9 | 0.0001 | 7.1 | **exact** |

## Discovered expressions

### I.6.20a  (`exp(-theta^2/2)`)
- best cx=9, rel=0.0007, runtime=4.3s
  ```
  (0.610208 / pow((theta / pow(theta, -0.03271)), theta))
  ```

### I.6.20  (`exp(-(theta/sigma)^2/2)`)
- best cx=14, rel=0.0113, runtime=4.1s
  ```
  (tanh(((0.961692 * min(sigma, theta)) / (theta * (theta / sigma)))) - 0.099391)
  ```

### I.8.14  (`sqrt((x2-x1)^2+(y2-y1)^2)`)
- best cx=22, rel=0.2857, runtime=2.6s
  ```
  (((min(min(-0.0659032, y2), sign(x2)) <= -0.000208124) - (x1 * x2)) - (y1 * min(y2, (0.904846 - (y1 * y2)))))
  ```

### I.9.18  (`G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)`)
- best cx=13, rel=0.4391, runtime=3.3s
  ```
  ((0.143942 * min(G, m2)) * min(m1, pow(max(x1, z1), z1)))
  ```

### I.10.7  (`m0/sqrt(1-v^2/c^2)`)
- best cx=9, rel=0.0024, runtime=0.9s
  ```
  ((m0 + (-0.0671877 / m0)) + (v * v))
  ```

### I.11.19  (`x1*y1+x2*y2+x3*y3`)
- best cx=21, rel=0.0709, runtime=11.0s
  ```
  ((((((x1 + x1) + x3) + y3) + (x2 * y2)) + (y1 * min(x3, y3))) - (x2 / x1))
  ```

### I.12.1  (`mu*Nn`)
- best cx=3, rel=0.0000, runtime=1.2s
  ```
  (Nn * mu)
  ```

### I.12.2  (`q1*q2/(4*pi*eps*r^2)`)
- best cx=11, rel=0.4654, runtime=1.8s
  ```
  pow(max(max(eps, (r - -0.13886)), min(q2, r)), -2.67733)
  ```

### I.12.4  (`q1*q2/(4*pi*eps*r^2)`)
- best cx=13, rel=0.5062, runtime=2.2s
  ```
  (q1 * (pow(-0.114913, r) + (((0.0455824 >= q1) <= 0.000291526) <= 0.0139668)))
  ```

### I.12.5  (`q1*q2/r^2`)
- best cx=7, rel=0.0000, runtime=3.6s
  ```
  (q2 / ((r * r) / q1))
  ```

### I.12.11  (`q*(Ef+B*v*sin(theta))`)
- best cx=25, rel=0.0762, runtime=18.1s
  ```
  ((q * ((v + min(max(-0.016965, Ef), 4.32067)) + min(((v + v) + v), pow((B * (-3.02645 + theta)), theta)))) - q)
  ```

### I.14.3  (`m*g*z`)
- best cx=5, rel=0.0000, runtime=5.2s
  ```
  ((g * m) * z)
  ```

### I.14.4  (`0.5*k*x^2`)
- best cx=13, rel=0.0120, runtime=10.5s
  ```
  (((x + log(x)) + (x * pow(k, log(x)))) - 0.655788)
  ```

### I.15.3t  (`(x-u*t)/sqrt(1-u^2/c^2)`)
- best cx=11, rel=0.0169, runtime=10.7s
  ```
  (((0.0993045 / u) + (t / -3.17813)) + (x - u))
  ```

### I.16.6  (`(u+v)/(1+u*v/c^2)`)
- best cx=7, rel=0.0128, runtime=7.3s
  ```
  (sin(sin(u)) + tanh(sin(v)))
  ```

### I.18.4  (`(m1*r1+m2*r2)/(m1+m2)`)
- best cx=11, rel=0.1861, runtime=7.7s
  ```
  ((r2 + ((r1 / r2) - 1.07663)) - pow(0.422078, r1))
  ```

### I.24.6  (`0.5*m*(omega^2+omega0^2)*x^2`)
- best cx=23, rel=0.1202, runtime=18.5s
  ```
  ((((-345.952 + omega) + exp(omega0)) + (sqrt((m * sqrt((omega * omega0)))) / neg((-0.0134729 / x)))) - pow(1.44385, m))
  ```

### I.25.13  (`q/C`)
- best cx=3, rel=0.0000, runtime=1.3s
  ```
  (q / C)
  ```

### I.26.2  (`arcsin(n*sin(theta2))`)
- best cx=5, rel=nan, runtime=2.3s
  ```
  (theta2 * pow(n, n))
  ```

### I.27.6  (`d1*d2/(d1+d2)`)
- best cx=12, rel=0.0214, runtime=9.2s
  ```
  ((0.109076 * d1) + (sqrt(min(d1, d2)) - pow(0.749417, d2)))
  ```

### I.29.4  (`omega/c`)
- best cx=3, rel=0.0000, runtime=3.7s
  ```
  (omega / c)
  ```

### I.30.3  (`I0*sin(n*theta/2)^2/sin(theta/2)^2`)
- best cx=21, rel=0.2095, runtime=11.5s
  ```
  ((-6.12499 * theta) + ((((I0 * n) + (n / theta)) + (I0 * min(n, (I0 / theta)))) - theta))
  ```

### I.32.5  (`q^2*a^2/(6*pi*eps*c^3)`)
- best cx=26, rel=0.1683, runtime=45.1s
  ```
  max(0.088164, ((((a + min(-2.21811, a)) - (-0.0401207 < c)) >= sin(q)) / ((pow(c, c) / min(a, q)) / (c / eps))))
  ```

### I.34.8  (`q*v*B/p`)
- best cx=31, rel=0.0109, runtime=29.8s
  ```
  (((B * q) + (((v - p) * ((q * min(B, q)) / p)) / p)) + ((v - p) * min(B, ((B * min(B, q)) / p))))
  ```

### I.39.22  (`n*k*T/V`)
- best cx=22, rel=0.0783, runtime=22.0s
  ```
  (cos(V) + max(k, (((T * k) + (k * (n - V))) + (T * (n - min(V, n))))))
  ```

### I.40.1  (`n0*exp(-m*g*x/(k*T))`)
- best cx=18, rel=0.0485, runtime=26.6s
  ```
  ((-0.050442 / k) + (n0 / max(1.14786, pow(((g * m) / (k * log(T))), x))))
  ```

### I.43.31  (`k*T/(6*pi*eta*r)`)
- best cx=5, rel=0.9361, runtime=3.3s
  ```
  (0.0775533 - pow(-0.096972, k))
  ```

### I.43.43  (`kappa*v^2/(n*sigma)`)
- best cx=31, rel=0.0415, runtime=42.2s
  ```
  ((((v + ((19.797 / n) * (v >= 2.8619))) / sigma) + (((sqrt(pow(kappa, v)) / n) + (sqrt(pow(kappa, v)) / n)) / sigma)) - (n >= kappa))
  ```

### I.47.23  (`sqrt(gamma*pr/rho)`)
- best cx=33, rel=0.0210, runtime=52.3s
  ```
  (((((-0.0880808 + (-0.396996 / gamma)) + (-0.402885 / gamma)) + pow(-0.535233, rho)) + pow(-0.537037, rho)) + pow(pr, min((0.112809 * reduce_std(pow(gamma, pr))), tanh(pow(0.416334, (rho > gamma))))))
  ```

### I.48.2  (`m*c^2/sqrt(1-v^2/c^2)`)
- best cx=9, rel=0.0001, runtime=7.1s
  ```
  ((v / 1.66205) + ((c * c) * m))
  ```

## Known limitations

- **No `sin` / `cos`**: equations I.12.11, I.26.2, I.30.3 use
  trigonometric ops that tessera's vocabulary doesn't include.
  These will fail by construction — documenting the gap.
- **Multi-variable inverse products (4+ vars)**: even with
  pop=400, gens=120, the search may not converge cleanly on
  forms like `q1*q2/(4*pi*eps*r^2)` where the constant must
  be fitted alongside structural symbol choice.

## See also

- `run_feynman_subset.py` — 8-equation quick run (~10 s)
- Udrescu & Tegmark, *AI Feynman*, Sci. Adv. 2020
- SRBench: https://github.com/cavalab/srbench