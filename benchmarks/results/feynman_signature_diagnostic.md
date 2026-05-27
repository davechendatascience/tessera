# Feynman signature diagnostic

Workbench signatures applied to each of the 30 Feynman equations.
Goal: identify whether any equation shows hidden structure (non-
algebraic ACF, mode multiplicity, surprising symmetry) that
would route to a different identification path in Stage 5.

**Total wall-clock:** 9.3s

## Headline

- **29 of 29** equations classified as ALGEBRAIC by Tier A (expected)

## Per-equation signatures

| Eq | Formula | n_vars | Inferred | perm_inv | t_acf | smooth | modes | eff_dim |
|---|---|---|---|---|---|---|---|---|
| I.6.20a | exp(-theta^2/2) | 1 | algebraic |  0.338 |  0.049 |  0.000 |    5 |  1.000 |
| I.6.20 | exp(-(theta/sigma)^2/2) | 2 | algebraic |  0.391 |  0.024 |  0.000 |    5 |  1.000 |
| I.8.14 | sqrt((x2-x1)^2+(y2-y1)^2) | 4 | algebraic |  0.515 |  0.047 |  0.000 |    1 |  1.000 |
| I.9.18 | G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1 | 9 | algebraic |  0.360 |  0.078 |  0.000 |    2 |  1.000 |
| I.10.7 | m0/sqrt(1-v^2/c^2) | 3 | algebraic |  0.265 |  0.050 |  0.000 |    4 |  1.000 |
| I.11.19 | x1*y1+x2*y2+x3*y3 | 6 | algebraic |  0.823 |  0.056 |  0.001 |    1 |  1.000 |
| I.12.1 | mu*Nn | 2 | algebraic |  0.252 |  0.067 |  0.000 |    2 |  1.000 |
| I.12.2 | q1*q2/(4*pi*eps*r^2) | 4 | algebraic |  0.280 |  0.073 |  0.000 |    4 |  1.000 |
| I.12.4 | q1*q2/(4*pi*eps*r^2) | 4 | algebraic |  0.672 |  0.052 |  0.003 |    4 |  1.000 |
| I.12.5 | q1*q2/r^2 | 3 | algebraic |  0.356 |  0.043 |  0.000 |    3 |  1.000 |
| I.12.11 | q*(Ef+B*v*sin(theta)) | 5 | algebraic |  0.406 |  0.043 |  0.001 |    3 |  1.000 |
| I.14.3 | m*g*z | 3 | algebraic |  0.906 |  0.046 |  0.001 |    3 |  1.000 |
| I.14.4 | 0.5*k*x^2 | 2 | algebraic |  0.254 |  0.071 |  0.000 |    3 |  1.000 |
| I.15.3t | (x-u*t)/sqrt(1-u^2/c^2) | 4 | algebraic |  0.270 |  0.047 |  0.000 |    2 |  1.000 |
| I.16.6 | (u+v)/(1+u*v/c^2) | 3 | algebraic |  0.187 |  0.069 |  0.000 |    1 |  1.000 |
| I.18.4 | (m1*r1+m2*r2)/(m1+m2) | 4 | algebraic |  0.797 |  0.045 |  0.002 |    2 |  1.000 |
| I.24.6 | 0.5*m*(omega^2+omega0^2)*x^2 | 4 | algebraic |  0.470 |  0.043 |  0.004 |    4 |  1.000 |
| I.25.13 | q/C | 2 | algebraic |  0.270 |  0.045 |  0.000 |    2 |  1.000 |
| I.26.2 | arcsin(n*sin(theta2)) | — | ERROR | — | — | — | — | — |
| I.27.6 | d1*d2/(d1+d2) | 2 | algebraic |  0.257 |  0.060 |  0.000 |    2 |  1.000 |
| I.29.4 | omega/c | 2 | algebraic |  0.270 |  0.045 |  0.000 |    2 |  1.000 |
| I.30.3 | I0*sin(n*theta/2)^2/sin(theta/2)^2 | 3 | algebraic |  0.293 |  0.048 |  0.000 |    4 |  1.000 |
| I.32.5 | q^2*a^2/(6*pi*eps*c^3) | 4 | algebraic |  0.264 |  0.054 |  0.000 |    3 |  1.000 |
| I.34.8 | q*v*B/p | 4 | algebraic |  0.989 |  0.053 |  0.003 |    3 |  1.000 |
| I.39.22 | n*k*T/V | 4 | algebraic |  0.989 |  0.053 |  0.003 |    3 |  1.000 |
| I.40.1 | n0*exp(-m*g*x/(k*T)) | 6 | algebraic |  0.367 |  0.046 |  0.000 |    2 |  1.000 |
| I.43.31 | k*T/(6*pi*eta*r) | 4 | algebraic |  0.254 |  0.061 |  0.000 |    4 |  1.000 |
| I.43.43 | kappa*v^2/(n*sigma) | 4 | algebraic |  0.259 |  0.060 |  0.000 |    5 |  1.000 |
| I.47.23 | sqrt(gamma*pr/rho) | 3 | algebraic |  0.351 |  0.051 |  0.000 |    2 |  1.000 |
| I.48.2 | m*c^2/sqrt(1-v^2/c^2) | 3 | algebraic |  0.249 |  0.072 |  0.000 |    2 |  1.000 |

## Reading

Feynman targets are PURE algebraic functions of iid inputs. The
Tier A signature should classify all as ALGEBRAIC; any that don't
indicate either (a) the input sampling has hidden correlation
we haven't accounted for, or (b) the formula has structural
regularity that the autocorrelation test picks up as spurious
temporal structure.

Smoothness ≈ 0 is expected (iid inputs → no temporal smoothness).
Effective dim should track n_vars roughly — high-n_var equations
with low effective dim may have a redundancy SR can exploit.
Mode count > 1 on iid data is a GMM-clustering artifact, not
a topological mode count — already documented limitation.

## What this diagnostic does NOT measure

- **Which equations the GP will actually find.** That's the
  Feynman benchmark itself; this is just a data characterization.
- **What const-opt would unlock.** The diagnostic suggests
  whether an equation has structure SR can identify; the
  constant-optimization step is independent.
- **Whether vocabulary is sufficient.** sin/cos/atan2 are in
  the vocabulary; whether they'd be discovered is a GP-search
  question, not a signature question.

## Implication for Step 2 (const-opt upgrade)

Equations classified correctly as algebraic with low complexity
signatures (low effective dim, few modes) are the ones most
likely to benefit from BFGS const-opt — they have clean structure
and the GP's bottleneck is parameter precision, not form discovery.

Equations with surprising signatures (misclassified, high
effective dim, weird mode counts) may need other interventions
(vocabulary expansion, better operator support).

## Reproducing

```
python benchmarks/run_feynman_signature_diagnostic.py
```