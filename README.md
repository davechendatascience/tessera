<p align="center">
  <img src="https://raw.githubusercontent.com/davechendatascience/tessera/main/resources/logo.png" alt="tessera" width="480">
</p>

# tessera

> *"Each piece of a mosaic is small and simple. The whole is composed."*

**Tessera** is a Python library of **applied-math primitives for compositional ML**:
measures, symbolic operators, state-space inference, differentiable feature
engineering — designed to fit together as building blocks for dynamical-systems
discovery, time-series modeling, PDE identification, and more.

## Design goals

1. **Compositional.** Every primitive is a small, hashable, immutable object
   that combines cleanly with others.
2. **Cacheable.** Subexpressions in any search are computed once and shared.
3. **Domain-agnostic.** The same `Measure` abstraction handles time-series
   convolutions, image filters, graph kernels, point-cloud aggregations.
4. **Honest about edges.** Linear functionals are first-class via measure theory.
   Nonlinear (Volterra) and time-varying extensions are exposed where useful
   and clearly marked where they fall outside the measure-theoretic core.

## Modules

| Module | Status | Purpose |
|---|---|---|
| [`tessera.expression`](src/tessera/expression/README.md) | shipping (v0.1) | Symbolic operators + measure-theoretic kernels + GP search |
| `tessera.koopman` | planned | Latent-state recovery via Koopman EVD |
| `tessera.ssm` | planned | Kalman / state-space filtering |
| `tessera.mts` | planned | Multi-timescale analysis |
| `tessera.diff_eml` | planned | Differentiable feature engineering |

## Install

```bash
pip install -e .             # editable, for development
pip install -e .[dev]        # + pytest, ruff, mypy
pip install -e .[all]        # + sympy, pysr (optional)
```

After this, any consuming project can simply `import tessera`.

## Usage example (expression module)

```python
import numpy as np
from tessera.expression import (
    measure_ema, measure_diff, measure_signed_sum,
    SeparableBilinear, Volterra2,
    FunctionalCache, apply_with_cache,
)

x = np.random.randn(10_000)

# Pure linear functional via a measure
y = measure_ema(halflife=24).apply(x)            # exact pandas EMA, O(N)

# Signed atomic measure — arbitrary weighted sum of lags
m = measure_signed_sum([(0.5, 0), (-0.3, 24), (0.1, 168)])
y = m.apply(x)

# Bilinear (separable, via Fubini)
bil = SeparableBilinear(
    measure_a=measure_diff(1),
    measure_b=measure_ema(24),
)
y_cross = bil.apply(x, x)   # r(t) × ema(x)[t]

# Quadratic Volterra-2: e.g. squared returns
vol = Volterra2(measure_a=measure_diff(1), measure_b=measure_diff(1))
realised_var = vol.apply(x)

# Subexpression caching — drop-in for any search loop
cache = FunctionalCache(mem_size=10_000, disk_dir="cache/feat")
y = apply_with_cache(bil, cache, var_ids=("x", "x"), xs=(x, x))
```

## Status

- **v0.1** — `tessera.expression` ships with `Measure`, `Functional`,
  `FunctionalCache`, Numba-JIT kernels, Expr `tree`, `mutation`
  operators, and a population-based `GP` search loop.
  **117/117 tests passing.**  See [`src/tessera/expression/README.md`](src/tessera/expression/README.md).
- Next: port the koopman / ssm / mts / diff_eml modules under the same
  packaging.

## License

MIT.
