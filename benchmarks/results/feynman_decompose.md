# Feynman: `discover` vs `discover_decompose` (held-out)

Single-layer csp_sr vs Strategy A (outer-op peel + polynomial-STLSQ
leaf + separability). Fit on a 70% train split, `rel = mse/var(y)`
on the 30% held-out test. exact `rel<1e-8`, approx `rel<1e-2`.

Config max_size=3, vocab `['neg', 'sqrt', 'exp', 'sin', 'cos', 'log', 'add', 'sub', 'mul', 'div']`, decompose max_depth=2.

**EXACT: baseline 8/30 -> decompose 24/30.** gains=['I.6.20a', 'I.6.20', 'I.8.14', 'I.10.7', 'I.11.19', 'I.12.2', 'I.12.4', 'I.15.3t', 'I.16.6', 'I.24.6', 'I.32.5', 'I.34.8', 'I.39.22', 'I.43.31', 'I.43.43', 'I.48.2'], losses=[]. Wall-clock 1293s.

| eq | formula | base | decompose | method |
|---|---|---|---|---|
| I.6.20a | `exp(-theta^2/2)` | approx (8e-05) | exact (4e-33) | `peel[square]->base` |
| I.6.20 | `exp(-(theta/sigma)^2/2)` | approx (0.001) | exact (4e-23) | `peel[square]->peel[log_abs]->base` |
| I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | fail (0.03) | exact (4e-29) | `peel[square]->base` |
| I.9.18 | `G*m1*m2/((x2-x1)^2+(y2-y1)^2+(` | fail (0.06) | approx (2e-05) | `peel[square]->peel[log_abs]->base` |
| I.10.7 | `m0/sqrt(1-v^2/c^2)` | approx (6e-05) | exact (1e-09) | `base` |
| I.11.19 | `x1*y1+x2*y2+x3*y3` | fail (0.03) | exact (0) | `base` |
| I.12.1 | `mu*Nn` | exact (0) | exact (0) | `base` |
| I.12.2 | `q1*q2/(4*pi*eps*r^2)` | fail (0.02) | exact (3e-32) | `peel[square]->powerlaw` |
| I.12.4 | `q1*q2/(4*pi*eps*r^2)` | fail (0.04) | exact (9e-31) | `peel[square]->powerlaw` |
| I.12.5 | `q1*q2/r^2` | exact (2e-32) | exact (2e-32) | `base` |
| I.12.11 | `q*(Ef+B*v*sin(theta))` | fail (0.07) | approx (3e-06) | `base` |
| I.14.3 | `m*g*z` | exact (3e-32) | exact (0) | `base` |
| I.14.4 | `0.5*k*x^2` | exact (3e-31) | exact (3e-31) | `base` |
| I.15.3t | `(x-u*t)/sqrt(1-u^2/c^2)` | approx (9e-05) | exact (3e-10) | `base` |
| I.16.6 | `(u+v)/(1+u*v/c^2)` | approx (0.0001) | exact (7e-10) | `base` |
| I.18.4 | `(m1*r1+m2*r2)/(m1+m2)` | approx (0.002) | approx (0.0002) | `peel[square]->peel[square]->base` |
| I.24.6 | `0.5*m*(omega^2+omega0^2)*x^2` | fail (0.02) | exact (1e-31) | `base` |
| I.25.13 | `q/C` | exact (0) | exact (0) | `base` |
| I.26.2 | `arcsin(n*sin(theta2))` | fail (nan) | fail (nan) | `base` |
| I.27.6 | `d1*d2/(d1+d2)` | exact (9e-32) | exact (9e-32) | `base` |
| I.29.4 | `omega/c` | exact (0) | exact (0) | `base` |
| I.30.3 | `I0*sin(n*theta/2)^2/sin(theta/` | approx (0.008) | approx (1e-08) | `base` |
| I.32.5 | `q^2*a^2/(6*pi*eps*c^3)` | fail (0.06) | exact (1e-28) | `peel[square]->peel[log_abs]->base` |
| I.34.8 | `q*v*B/p` | fail (0.05) | exact (3e-32) | `peel[square]->powerlaw` |
| I.39.22 | `n*k*T/V` | fail (0.05) | exact (3e-32) | `peel[square]->powerlaw` |
| I.40.1 | `n0*exp(-m*g*x/(k*T))` | fail (0.05) | fail (0.01) | `base` |
| I.43.31 | `k*T/(6*pi*eta*r)` | fail (0.02) | exact (2e-31) | `peel[square]->powerlaw` |
| I.43.43 | `kappa*v^2/(n*sigma)` | fail (0.02) | exact (3e-32) | `peel[square]->powerlaw` |
| I.47.23 | `sqrt(gamma*pr/rho)` | exact (5e-32) | exact (5e-32) | `base` |
| I.48.2 | `m*c^2/sqrt(1-v^2/c^2)` | approx (6e-05) | exact (1e-10) | `base` |

## Reading

- **gains** are equations the single-layer enumeration could not
  reach but decomposition does — the deep-structure forms broken by
  outer-op peel (e.g. sqrt(1-v^2/c^2) -> peel sqrt -> 1 - v^2/c^2,
  a 2-term linear fit) and polynomial-after-peel (STLSQ).
- **losses** would be regressions (decomposition's high threshold
  or a wrong peel beating the base fit) — expected ~none, since
  decompose tries base first and only overrides on a verified
  higher-precision result.
- Decomposition does NOT manufacture Class-B approximations: it
  short-circuits only at machine precision (rel<1e-9), so a smooth
  target is recovered in its true form (via peel), not as a
  high-degree polynomial fit.

## Reproducing

```
python benchmarks/run_feynman_decompose.py
```