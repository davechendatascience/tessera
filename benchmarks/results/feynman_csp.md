# csp_sr on the extended Feynman benchmark (general mode)

CSP-enumerated const-free tessera trees (rich vocab) + sparse
linear (beam) fit, gradient-free. `rel = mse/var(y)`, verified via
tessera.evaluate.

Config: max_size=3, beam_width=12, max_features=25000, vocab=['neg', 'sqrt', 'exp', 'sin', 'cos', 'log', 'add', 'sub', 'mul', 'div'].

**Honest scoring.** A flexible multi-term linear basis APPROXIMATES
smooth functions well enough to pass the AI-Feynman `rel<0.01`
threshold WITHOUT being the true symbolic form (Class-B natural
overfit). We separate:
- **exact** (`rel < 1e-8`): genuine symbolic recovery — the fit is
  machine-precision because the true form IS in the dictionary;
- **approx** (`1e-8 ≤ rel < 0.01`): passes AI-Feynman but is a
  multi-term numerical approximation, not the true form;
- **partial** (`0.01 ≤ rel < 0.2`), **failed** otherwise.

**Genuine exact 8/30** (parsimonious symbolic match); approx 8/30; partial 13/30.
AI-Feynman-threshold count (exact+approx) = 16/30. Wall-clock 50s.

| eq | formula | verdict | rel | terms | feats | found |
|---|---|---|---|---|---|---|
| I.6.20a | `exp(-theta^2/2)` | approx | 7.6e-05 | 4 | 961 | `add(add(add(add(mul(0.01582, cos(add(log(theta), theta))), m` |
| I.6.20 | `exp(-(theta/sigma)^2/2)` | approx | 0.00145 | 4 | 5098 | `add(add(add(add(mul(-0.08492, sin(sqrt(sub(theta, sigma)))),` |
| I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | partial | 0.0328 | 4 | 14446 | `add(add(add(add(mul(-0.9255, cos(sub(y1, y2))), mul(0.2605, ` |
| I.9.18 | `G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z` | partial | 0.0528 | 4 | 25000 | `add(add(add(add(mul(0.08374, mul(mul(G, m1), m2)), mul(0.042` |
| I.10.7 | `m0/sqrt(1-v^2/c^2)` | approx | 5.94e-05 | 3 | 12244 | `add(add(add(mul(0.05866, exp(mul(div(m0, c), v))), mul(-0.20` |
| I.11.19 | `x1*y1+x2*y2+x3*y3` | partial | 0.0348 | 4 | 17527 | `add(add(add(add(mul(3.414, sqrt(mul(mul(x1, x2), y1))), mul(` |
| I.12.1 | `mu*Nn` | exact | 0 | 1 | 5101 | `mul(Nn, mu)` |
| I.12.2 | `q1*q2/(4*pi*eps*r^2)` | partial | 0.0209 | 4 | 14185 | `add(add(add(add(mul(0.02544, exp(sub(div(q2, r), eps))), mul` |
| I.12.4 | `q1*q2/(4*pi*eps*r^2)` | partial | 0.035 | 4 | 14185 | `add(add(add(add(mul(0.04286, sqrt(exp(div(q2, r)))), mul(0.0` |
| I.12.5 | `q1*q2/r^2` | exact | 2.21e-32 | 1 | 12262 | `mul(div(q1, r), div(q2, r))` |
| I.12.11 | `q*(Ef+B*v*sin(theta))` | partial | 0.0667 | 4 | 15803 | `add(add(add(add(mul(8.992, mul(sin(theta), q)), mul(0.6064, ` |
| I.14.3 | `m*g*z` | exact | 2.65e-32 | 1 | 12262 | `mul(mul(m, z), g)` |
| I.14.4 | `0.5*k*x^2` | exact | 0 | 1 | 5101 | `mul(0.5, mul(mul(k, x), x))` |
| I.15.3t | `(x-u*t)/sqrt(1-u^2/c^2)` | approx | 8.16e-05 | 3 | 14152 | `add(add(add(mul(-0.9952, sub(mul(t, u), x)), mul(-0.2492, co` |
| I.16.6 | `(u+v)/(1+u*v/c^2)` | approx | 0.000113 | 4 | 12234 | `add(add(add(add(mul(-0.004367, div(log(c), v)), mul(-0.08402` |
| I.18.4 | `(m1*r1+m2*r2)/(m1+m2)` | approx | 0.00209 | 4 | 14185 | `add(add(add(add(mul(0.08559, div(sub(r2, r1), m1)), mul(0.36` |
| I.24.6 | `0.5*m*(omega^2+omega0^2)*x^2` | partial | 0.0193 | 4 | 14185 | `add(add(add(add(mul(8.781, exp(sqrt(mul(omega, x)))), mul(-0` |
| I.25.13 | `q/C` | exact | 0 | 1 | 5101 | `div(q, C)` |
| I.26.2 | `arcsin(n*sin(theta2))` | failed | nan | 0 | 5094 | `—` |
| I.27.6 | `d1*d2/(d1+d2)` | exact | 9.57e-32 | 1 | 5101 | `mul(div(d1, add(d1, d2)), d2)` |
| I.29.4 | `omega/c` | exact | 0 | 1 | 5101 | `div(omega, c)` |
| I.30.3 | `I0*sin(n*theta/2)^2/sin(theta/2)^2` | approx | 0.00749 | 4 | 12247 | `add(add(add(add(mul(1.633, cos(add(sub(n, I0), n))), mul(-4.` |
| I.32.5 | `q^2*a^2/(6*pi*eps*c^3)` | partial | 0.0536 | 4 | 14148 | `add(add(add(add(mul(0.0003134, exp(add(sub(q, eps), a))), mu` |
| I.34.8 | `q*v*B/p` | partial | 0.0285 | 4 | 14181 | `add(add(add(add(mul(-1.97, exp(sub(div(v, p), q))), mul(3.25` |
| I.39.22 | `n*k*T/V` | partial | 0.0285 | 4 | 14181 | `add(add(add(add(mul(-1.97, exp(sub(div(k, V), n))), mul(3.25` |
| I.40.1 | `n0*exp(-m*g*x/(k*T))` | partial | 0.0505 | 4 | 17528 | `add(add(add(add(mul(0.8533, sqrt(mul(add(T, k), n0))), mul(-` |
| I.43.31 | `k*T/(6*pi*eta*r)` | partial | 0.0241 | 4 | 14185 | `add(add(add(add(mul(0.05719, cos(log(div(k, T)))), mul(-0.23` |
| I.43.43 | `kappa*v^2/(n*sigma)` | partial | 0.0243 | 4 | 14181 | `add(add(add(add(mul(1, mul(div(kappa, n), v)), mul(0.9976, m` |
| I.47.23 | `sqrt(gamma*pr/rho)` | exact | 4.35e-32 | 1 | 12253 | `sqrt(mul(div(gamma, rho), pr))` |
| I.48.2 | `m*c^2/sqrt(1-v^2/c^2)` | approx | 5.85e-05 | 1 | 12243 | `add(mul(1.005, add(mul(mul(c, m), c), v)), -0.2228)` |

## Reading

- **Genuine recoveries** are 1-term, machine-precision fits of
  products / ratios / sqrt-of-product forms — the linear-in-one-
  feature class (mu·Nn, q1q2/r², m·g·z, q/C, d1d2/(d1+d2), ω/c,
  sqrt(γ·pr/ρ), …). These are real symbolic recoveries.
- **`approx`** are the honest false-positive warning: 4-term fits
  that pass `rel<0.01` but are NOT the true form (e.g. sqrt(1−v²/c²)
  approximated by exp/cos terms on the benign sampling range). A
  loose threshold + flexible basis manufactures these; machine-
  precision separation exposes them.
- **Boundary (expected):** constants buried inside nonlinearities
  (the `1` in sqrt(1−v²/c²), `/2` in exp(-x²/2)) aren't a linear
  combo of const-free features → only approximated; high-variable
  / deep equations exceed the enumeration cap (I.9.18, 9 vars).
- **Honest verdict:** csp_sr cleanly recovers linear-in-parameter
  symbolic forms with small features; it APPROXIMATES (does not
  recover) embedded-constant nonlinear forms. The latter needs the
  per-feature nonlinear-constant refine (1-step Gauss-Newton) or
  deeper structured enumeration — the clear next extension.

## Reproducing

```
python benchmarks/run_feynman_csp.py
```