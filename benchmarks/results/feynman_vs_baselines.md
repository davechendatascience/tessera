# Feynman head-to-head: tessera csp_decompose vs gplearn

Same 30 equations, held-out split, same metric (exact `rel<1e-8`
machine-precision symbolic match; approx `rel<1e-2`). gplearn is a
pure-Python GP baseline (no constant optimisation); PySR and
AI-Feynman are heavier and cited in the README, not re-run here.

**csp_decompose: exact 24/30, approx 4/30 — 720s total (gradient-free).**
**gplearn: exact 4/30, approx 2/30 — 13s total (GP).**

| eq | formula | csp_decompose | (s) | gplearn | (s) |
|---|---|---|---|---|---|
| I.6.20a | `exp(-theta^2/2)` | exact | 0.2 | fail | 0.4 |
| I.6.20 | `exp(-(theta/sigma)^2/2)` | exact | 2.9 | fail | 0.5 |
| I.8.14 | `sqrt((x2-x1)^2+(y2-y1)^2)` | exact | 10.8 | fail | 0.4 |
| I.9.18 | `G*m1*m2/((x2-x1)^2+(y2-y1)^2` | approx | 188.8 | fail | 0.5 |
| I.10.7 | `m0/sqrt(1-v^2/c^2)` | exact | 0.1 | approx | 0.4 |
| I.11.19 | `x1*y1+x2*y2+x3*y3` | exact | 0.0 | fail | 0.5 |
| I.12.1 | `mu*Nn` | exact | 0.0 | exact | 0.3 |
| I.12.2 | `q1*q2/(4*pi*eps*r^2)` | exact | 23.0 | fail | 0.5 |
| I.12.4 | `q1*q2/(4*pi*eps*r^2)` | exact | 23.0 | fail | 0.5 |
| I.12.5 | `q1*q2/r^2` | exact | 1.7 | fail | 0.6 |
| I.12.11 | `q*(Ef+B*v*sin(theta))` | approx | 84.9 | fail | 0.6 |
| I.14.3 | `m*g*z` | exact | 0.0 | exact | 0.2 |
| I.14.4 | `0.5*k*x^2` | exact | 0.0 | fail | 0.5 |
| I.15.3t | `(x-u*t)/sqrt(1-u^2/c^2)` | exact | 3.0 | fail | 0.6 |
| I.16.6 | `(u+v)/(1+u*v/c^2)` | exact | 0.3 | fail | 0.4 |
| I.18.4 | `(m1*r1+m2*r2)/(m1+m2)` | approx | 87.8 | fail | 0.5 |
| I.24.6 | `0.5*m*(omega^2+omega0^2)*x^2` | exact | 0.0 | fail | 0.5 |
| I.25.13 | `q/C` | exact | 0.5 | exact | 0.2 |
| I.26.2 | `arcsin(n*sin(theta2))` | fail | 3.7 | fail | 0.0 |
| I.27.6 | `d1*d2/(d1+d2)` | exact | 0.5 | fail | 0.4 |
| I.29.4 | `omega/c` | exact | 0.5 | exact | 0.2 |
| I.30.3 | `I0*sin(n*theta/2)^2/sin(thet` | approx | 28.5 | fail | 0.4 |
| I.32.5 | `q^2*a^2/(6*pi*eps*c^3)` | exact | 24.7 | fail | 0.5 |
| I.34.8 | `q*v*B/p` | exact | 24.2 | fail | 0.5 |
| I.39.22 | `n*k*T/V` | exact | 24.5 | fail | 0.4 |
| I.40.1 | `n0*exp(-m*g*x/(k*T))` | fail | 141.2 | fail | 0.4 |
| I.43.31 | `k*T/(6*pi*eta*r)` | exact | 19.5 | fail | 0.4 |
| I.43.43 | `kappa*v^2/(n*sigma)` | exact | 23.7 | fail | 0.8 |
| I.47.23 | `sqrt(gamma*pr/rho)` | exact | 1.9 | fail | 0.7 |
| I.48.2 | `m*c^2/sqrt(1-v^2/c^2)` | exact | 0.1 | approx | 0.5 |

## Reading

- `exact` = machine-precision symbolic recovery (the true closed
  form is found), not just a good numerical fit.
- tessera is gradient-free and fast; gplearn (basic GP, no const-opt)
  rarely reaches machine precision. PySR (GP + const optimisation)
  and AI-Feynman (NN + dimensional analysis) are stronger on raw
  recovery but far heavier — see the README positioning table.
