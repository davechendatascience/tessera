# Feynman extended on tessera

**Equations:** 30 (representative subset of the canonical 100)
**GP config:** pop=400, gens=120, pointwise_only=True, optimize_constants every 3 gens, Nelder-Mead 30 iter.
**Samples per equation:** 2000
**Total wall-clock:** 94.5s (1.6 min)
**Alphabet:** add/sub/mul/div/min/max, gt/lt/ge/le, tanh/abs/sign/neg/step, sqrt/exp/log/pow, reduce_mean/max/sum/std.

## Headline

- **Exact** (rel < 0.01): 9 / 30
- **Partial** (0.01 ≤ rel < 0.20): 14 / 30
- **Failed** (rel ≥ 0.20): 7 / 30

## Results table

| # | Eq ID | True formula | n_vars | best cx | best rel | runtime (s) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | I.6.20a | `exp(-theta^2/2)` | 1 | 7 | 0.0010 | 3.2 | **exact** |
| 2 | I.6.20 | `exp(-(theta/sigma)^2/2)` | 2 | 11 | 0.0000 | 2.6 | **exact** |
| 3 | I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | 4 | 15 | 0.2954 | 2.1 | failed |
| 4 | I.9.18 | `G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)` | 9 | 25 | 0.2614 | 8.6 | failed |
| 5 | I.10.7 | `m0/sqrt(1-v^2/c^2)` | 3 | 5 | 0.0031 | 0.6 | **exact** |
| 6 | I.11.19 | `x1*y1+x2*y2+x3*y3` | 6 | 22 | 0.1459 | 4.5 | partial |
| 7 | I.12.1 | `mu*Nn` | 2 | 3 | 0.0000 | 0.8 | **exact** |
| 8 | I.12.2 | `q1*q2/(4*pi*eps*r^2)` | 4 | 11 | 0.2656 | 2.7 | failed |
| 9 | I.12.4 | `q1*q2/(4*pi*eps*r^2)` | 4 | 25 | 0.1942 | 3.3 | partial |
| 10 | I.12.5 | `q1*q2/r^2` | 3 | 7 | 0.0000 | 1.8 | **exact** |
| 11 | I.12.11 | `q*(Ef+B*v*sin(theta))` | 5 | 28 | 0.3227 | 4.5 | failed |
| 12 | I.14.3 | `m*g*z` | 3 | 19 | 0.0454 | 3.7 | partial |
| 13 | I.14.4 | `0.5*k*x^2` | 2 | 14 | 0.0087 | 3.0 | **exact** |
| 14 | I.15.3t | `(x-u*t)/sqrt(1-u^2/c^2)` | 4 | 9 | 0.0018 | 1.3 | **exact** |
| 15 | I.16.6 | `(u+v)/(1+u*v/c^2)` | 3 | 9 | 0.0117 | 1.7 | partial |
| 16 | I.18.4 | `(m1*r1+m2*r2)/(m1+m2)` | 4 | 12 | 0.1452 | 2.2 | partial |
| 17 | I.24.6 | `0.5*m*(omega^2+omega0^2)*x^2` | 4 | 20 | 0.0768 | 5.7 | partial |
| 18 | I.25.13 | `q/C` | 2 | 3 | 0.0000 | 0.8 | **exact** |
| 19 | I.26.2 | `arcsin(n*sin(theta2))` | 2 | 17 | nan | 1.4 | failed |
| 20 | I.27.6 | `d1*d2/(d1+d2)` | 2 | 14 | 0.0722 | 3.1 | partial |
| 21 | I.29.4 | `omega/c` | 2 | 3 | 0.0000 | 0.8 | **exact** |
| 22 | I.30.3 | `I0*sin(n*theta/2)^2/sin(theta/2)^2` | 3 | 23 | 0.1632 | 4.9 | partial |
| 23 | I.32.5 | `q^2*a^2/(6*pi*eps*c^3)` | 4 | 20 | 0.0596 | 5.1 | partial |
| 24 | I.34.8 | `q*v*B/p` | 4 | 19 | 0.0411 | 4.4 | partial |
| 25 | I.39.22 | `n*k*T/V` | 4 | 21 | 0.0949 | 3.4 | partial |
| 26 | I.40.1 | `n0*exp(-m*g*x/(k*T))` | 6 | 20 | 0.2187 | 3.5 | failed |
| 27 | I.43.31 | `k*T/(6*pi*eta*r)` | 4 | 17 | 0.2732 | 3.5 | failed |
| 28 | I.43.43 | `kappa*v^2/(n*sigma)` | 4 | 21 | 0.0676 | 3.6 | partial |
| 29 | I.47.23 | `sqrt(gamma*pr/rho)` | 3 | 24 | 0.1541 | 3.2 | partial |
| 30 | I.48.2 | `m*c^2/sqrt(1-v^2/c^2)` | 3 | 17 | 0.0200 | 4.6 | partial |

## Discovered expressions

### I.6.20a  (`exp(-theta^2/2)`)
- best cx=7, rel=0.0010, runtime=3.2s
  ```
  pow(tanh(tanh((0.847232 / theta))), theta)
  ```

### I.6.20  (`exp(-(theta/sigma)^2/2)`)
- best cx=11, rel=0.0000, runtime=2.6s
  ```
  pow(pow(-0.606531, (theta / sigma)), (theta / max(0.175903, sigma)))
  ```

### I.8.14  (`sqrt((x2-x1)^2+(y2-y1)^2)`)
- best cx=15, rel=0.2954, runtime=2.1s
  ```
  ((max((0.24031 - x1), (y2 >= -1.46241)) - (y1 * y2)) - (x1 * x2))
  ```

### I.9.18  (`G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)`)
- best cx=25, rel=0.2614, runtime=8.6s
  ```
  (((tanh(G) / x2) + (max(0.288569, pow((G * m1), (max(y1, z1) - (z1 >= y2)))) / exp(y2))) - pow(0.206276, m2))
  ```

### I.10.7  (`m0/sqrt(1-v^2/c^2)`)
- best cx=5, rel=0.0031, runtime=0.6s
  ```
  (m0 + (v * v))
  ```

### I.11.19  (`x1*y1+x2*y2+x3*y3`)
- best cx=22, rel=0.1459, runtime=4.5s
  ```
  ((((((-2.71606 + x2) + x2) + x3) + x3) + (x2 - y3)) + ((y1 * y2) - neg((x1 * y3))))
  ```

### I.12.1  (`mu*Nn`)
- best cx=3, rel=0.0000, runtime=0.8s
  ```
  (Nn * mu)
  ```

### I.12.2  (`q1*q2/(4*pi*eps*r^2)`)
- best cx=11, rel=0.2656, runtime=2.7s
  ```
  ((pow(r, (-6.35691 + q2)) / eps) - min(-0.011353, r))
  ```

### I.12.4  (`q1*q2/(4*pi*eps*r^2)`)
- best cx=25, rel=0.1942, runtime=3.3s
  ```
  exp(min(((q2 / (q1 >= r)) - (2.30655 + eps)), (exp(min((1.15306 - r), ((q1 - r) - (eps + r)))) - r)))
  ```

### I.12.5  (`q1*q2/r^2`)
- best cx=7, rel=0.0000, runtime=1.8s
  ```
  ((q1 / (r / q2)) / r)
  ```

### I.12.11  (`q*(Ef+B*v*sin(theta))`)
- best cx=28, rel=0.3227, runtime=4.5s
  ```
  ((((tanh(Ef) + (8.67512 * q)) + (B * v)) + ((B * theta) * (Ef - theta))) + (((B * theta) * (v - theta)) - 18.4546))
  ```

### I.14.3  (`m*g*z`)
- best cx=19, rel=0.0454, runtime=3.7s
  ```
  ((((g * z) + (min(g, z) * pow(m, 1.79257))) - (m - g)) - (m / z))
  ```

### I.14.4  (`0.5*k*x^2`)
- best cx=14, rel=0.0087, runtime=3.0s
  ```
  (((x * (k + log(pow(x, k)))) - k) - (1.69866 <= x))
  ```

### I.15.3t  (`(x-u*t)/sqrt(1-u^2/c^2)`)
- best cx=9, rel=0.0018, runtime=1.3s
  ```
  ((x - (t * u)) - (-0.154723 * u))
  ```

### I.16.6  (`(u+v)/(1+u*v/c^2)`)
- best cx=9, rel=0.0117, runtime=1.7s
  ```
  ((u + v) - ((u * v) * v))
  ```

### I.18.4  (`(m1*r1+m2*r2)/(m1+m2)`)
- best cx=12, rel=0.1452, runtime=2.2s
  ```
  (min(reduce_mean(m2), tanh(r1)) + log(max(0.043536, (r1 * r2))))
  ```

### I.24.6  (`0.5*m*(omega^2+omega0^2)*x^2`)
- best cx=20, rel=0.0768, runtime=5.7s
  ```
  ((omega0 * (((m * x) - m) / 0.063235)) + (pow(x, omega) * min(min(0.446469, log(m)), 1)))
  ```

### I.25.13  (`q/C`)
- best cx=3, rel=0.0000, runtime=0.8s
  ```
  (q / C)
  ```

### I.26.2  (`arcsin(n*sin(theta2))`)
- best cx=17, rel=nan, runtime=1.4s
  ```
  ((theta2 * pow(n, n)) + (max(n, (n < theta2)) < max(-0.00580758, (n * theta2))))
  ```

### I.27.6  (`d1*d2/(d1+d2)`)
- best cx=14, rel=0.0722, runtime=3.1s
  ```
  (log((min(d1, d2) + min(d2, (d1 - (d2 < d1))))) - 0.0756316)
  ```

### I.29.4  (`omega/c`)
- best cx=3, rel=0.0000, runtime=0.8s
  ```
  (omega / c)
  ```

### I.30.3  (`I0*sin(n*theta/2)^2/sin(theta/2)^2`)
- best cx=23, rel=0.1632, runtime=4.9s
  ```
  ((((I0 * n) + (I0 * n)) + (n / theta)) + (I0 - pow(((I0 * n) + (I0 * n)), theta)))
  ```

### I.32.5  (`q^2*a^2/(6*pi*eps*c^3)`)
- best cx=20, rel=0.0596, runtime=5.1s
  ```
  abs(((q / pow(c, c)) / ((-1.66875 * eps) / (a - max((a / q), (a > 0.0155282))))))
  ```

### I.34.8  (`q*v*B/p`)
- best cx=19, rel=0.0411, runtime=4.4s
  ```
  ((((B + q) / (p / (v + min(min(B, q), v)))) - 1.32223) - (3.26736 - q))
  ```

### I.39.22  (`n*k*T/V`)
- best cx=21, rel=0.0949, runtime=3.4s
  ```
  ((((T - 2.67637) + (T - 3.39244)) + (((T + n) + n) / (V / k))) - (n <= 2.71905))
  ```

### I.40.1  (`n0*exp(-m*g*x/(k*T))`)
- best cx=20, rel=0.2187, runtime=3.5s
  ```
  (((x <= 0.50762) + abs(((n0 * ((m >= T) - k)) / (g + k)))) - pow(0.157828, T))
  ```

### I.43.31  (`k*T/(6*pi*eta*r)`)
- best cx=17, rel=0.2732, runtime=3.5s
  ```
  (-0.023436 * (min(k, pow(T, (r <= pow(k, (r <= T))))) / (-0.215705 * eta)))
  ```

### I.43.43  (`kappa*v^2/(n*sigma)`)
- best cx=21, rel=0.0676, runtime=3.6s
  ```
  ((kappa - (v - ((v - (sigma - kappa)) * ((v - (sigma - v)) / n)))) - (kappa > sigma))
  ```

### I.47.23  (`sqrt(gamma*pr/rho)`)
- best cx=24, rel=0.1541, runtime=3.2s
  ```
  (reduce_std(gamma) + (min((pr / rho), pow((-1.05324 * min(min(gamma, pr), pr)), (log(rho) < sqrt(gamma)))) / sign(log(rho))))
  ```

### I.48.2  (`m*c^2/sqrt(1-v^2/c^2)`)
- best cx=17, rel=0.0200, runtime=4.6s
  ```
  ((exp(c) + (m * log(pow(min(min(c, m), (c * m)), m)))) - c)
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