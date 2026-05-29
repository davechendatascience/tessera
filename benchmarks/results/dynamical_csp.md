# csp_sr on dynamical systems (Lorenz, Rössler)

CSP-enumeration + sparse linear fit (gradient-free) recovering
dX/dt = f(state) from trajectory samples with analytic RHS targets.
This is the SINDy regime: sparse polynomial dynamics.

**Recovered 6/6 components** (R² > 0.9999, verified via
tessera.evaluate).

## Lorenz

| component | truth | R² | terms | found |
|---|---|---|---|---|
| dx/dt | `10*(y - x)` | 1.0000 | 2 | `add(mul(-10, x), mul(10, y))` |
| dy/dt | `x*(28 - z) - y  =  28x - xz - y` | 1.0000 | 3 | `add(add(mul(28, x), mul(-1, y)), mul(-1, mul(x, z)))` |
| dz/dt | `xy - 2.667 z` | 1.0000 | 2 | `add(mul(-2.667, z), mul(x, y))` |

## Rössler

| component | truth | R² | terms | found |
|---|---|---|---|---|
| dx/dt | `-y - z` | 1.0000 | 2 | `add(mul(-1, y), mul(-1, z))` |
| dy/dt | `x + 0.2 y` | 1.0000 | 2 | `add(x, mul(0.2, y))` |
| dz/dt | `0.2 + z(x - 5.7)  =  0.2 + xz - 5.7z` | 1.0000 | 2 | `add(add(mul(-5.7, z), mul(x, z)), 0.2)` |

## Reading

- Sparse polynomial RHS (products, linear terms, constants) are
  recovered exactly and parsimoniously — coefficients via linear
  least squares, structure via enumeration. No gradients.
- Constants like Rössler's `b` (intercept) and `-c·z` (coefficient)
  fall out of the linear fit; `xz` from enumerated `mul(x,z)`.

## Reproducing

```
python benchmarks/run_dynamical_csp.py
```