# `tessera.expression`

> Measure-theoretic operators, n-ary functionals, an Expr tree, and a GP
> search engine — all designed to fit together as building blocks for
> compositional, interpretable ML on any kind of structured signal.

## What this module is

A self-contained symbolic-regression-style stack with three layers:

1. **Primitives** — `Measure`, `Functional`. The mathematical building blocks.
2. **Plumbing** — `FunctionalCache`, JIT-compiled kernels. The performance layer.
3. **Search** — `tree`, `mutation`, `gp`. The discovery layer.

Each layer is independently testable and useful. You can use just the
primitives (no search), just primitives + plumbing (manual symbolic
construction with caching), or the full stack (run a GP to discover
expressions automatically).

## When to use it

- You have **time-series data** and want to discover compact symbolic
  formulae that predict a target.
- You want **interpretability** over raw accuracy — a 6-node expression
  that captures 90% of the signal beats a 49-coefficient linear model.
- Your problem has **measure-theoretic structure** — convolutions,
  windowed averages, lagged sums, multi-scale interactions.
- You'd use **PySR** but want temporal/functional operators that PySR's
  pointwise-only alphabet doesn't represent.

## When NOT to use it

- Your problem is **purely pointwise** (no time-series structure) → use PySR; it's faster.
- You need **deep-learning-grade accuracy** on a hard target → use a neural net.
- You're doing **theorem-style derivation** from physics → use SymPy directly.
- You need **GPU acceleration** → not supported (CPU/numba only).

## Quick start

```python
import numpy as np
from tessera.expression import (
    GP, GPConfig,
    measure_ema, measure_diff, measure_signed_sum,
    LinearFunctional, SeparableBilinear, Volterra2,
)

# Synthetic problem: discover the relationship y = 0.5 * ema(x, 24) − 0.3 * diff(x, 6)
rng = np.random.default_rng(0)
x = rng.standard_normal(5000)
y_true = (
    0.5  * measure_ema(24).apply(x, fill_warmup=0.0)
    - 0.3 * measure_diff(6).apply(x, fill_warmup=0.0)
    + 0.05 * rng.standard_normal(5000)
)

# Run the GP
cfg = GPConfig(pop_size=100, n_gens=30, parsimony=0.005, seed=42)
gp = GP(cfg)
front = gp.run(env={"x": x}, y_true=y_true, feature_names=["x"])

# Walk the Pareto front: low complexity → high complexity, accuracy improves
for cand in front:
    print(f"cx={cand.complexity:2d}  loss={cand.train_loss:.4g}  {cand.tree}")
```

## API map

### 1. Primitives — measures and functionals

```python
from tessera.expression import (
    # Measure: a signed measure on non-negative integer lags
    Measure, Atom,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
    # Functional: n-ary wrapper (Fubini-decomposed bilinear / Volterra-2)
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
    apply_with_cache,
)
```

**Measures** are signed measures `μ = atoms ⊕ density` on lag positions
`s ∈ {0, 1, 2, ...}`. Two-line examples:

| Construction | Mathematical meaning |
|---|---|
| `measure_lag(24)` | δ(s−24), pure shift by 24 |
| `measure_diff(1)` | δ(s) − δ(s−1), one-period return |
| `measure_ema(24)` | exponential density with halflife 24 |
| `measure_roll_mean(168)` | uniform [0, 168) — rolling 168-period mean |
| `measure_power_law(scale=24, alpha=1.5)` | long-memory (1+s/24)⁻¹·⁵ |
| `measure_signed_sum([(0.5,0),(-0.3,24),(0.1,168)])` | arbitrary signed atomic mass |

Apply: `m.apply(x)` returns the convolution `(m·x)(t) = Σ_s κ(s)·x(t−s)`.

**Functionals** are n-ary wrappers around measures:
- `LinearFunctional(measure=μ)`: `(μ·x)(t)` — n=1
- `SeparableBilinear(μa, μb)`: `(μa·x)(t) · (μb·y)(t)` — n=2 (Fubini-decomposed)
- `Volterra2(μa, μb)`: `(μa·x)(t) · (μb·x)(t)` — n=1 (self-product;
  e.g. squared returns)

### 2. Plumbing — cache and kernels

```python
from tessera.expression import FunctionalCache

cache = FunctionalCache(mem_size=10_000, disk_dir="cache/feat")
y = cache.get_or_compute(measure_ema(24), var_id="x", x=x_arr)
```

Caches `(hash(measure), var_id, fill_warmup)` → result array. Two-tier
(memory LRU + optional disk). Subexpression sharing across many trees
in a GP population — typical hit rate climbs from ~30% (early gens)
to ~70-80% (late gens).

Underlying JIT kernels (auto-selected by `Measure.apply`):

| Measure shape | Backend | Cost |
|---|---|---|
| pure exponential (no atoms) | Numba recursive | O(N) |
| pure atomic | NumPy shift-and-accumulate | O(N · #atoms) |
| short general kernel (K ≤ 64) | Numba direct conv | O(N·K) |
| long general kernel (K > 64) | `scipy.signal.fftconvolve` | O(N log N) |

The recursive EMA is up to 706× faster than naive truncated convolution
at h=168 — and matches `pandas .ewm(halflife=h, adjust=False)` to
floating-point precision.

### 3. Search — tree, mutation, GP

```python
from tessera.expression import (
    # Tree: tagged-union Node types
    Var, Const, BinOp, UnOp, FunctionalOp, Node,
    BIN_OPS, UN_OPS,   # available pointwise ops
    complexity, depth, used_features, iter_subtrees, replace_at, evaluate,
    # Mutation: bounded, type-safe mutators
    MAX_DEPTH, MAX_COMPLEXITY,
    random_tree, mutate, validate_tree, OP_WEIGHTS,
    subtree_swap, subtree_crossover, constant_jitter,
    term_insert, term_delete, op_swap, measure_mutate,
    # GP: population-based search
    GP, GPConfig, Candidate, mse_loss, pareto_front,
)
```

**Tree.** Five Node types (Var, Const, BinOp, UnOp, FunctionalOp), all
frozen + hashable. `evaluate(node, env, cache)` walks the tree; cache
integration is automatic when provided.

**Mutation.** Seven operators: `subtree_swap`, `subtree_crossover`,
`constant_jitter`, `term_insert`, `term_delete`, `op_swap`,
`measure_mutate` (tessera-specific — mutates the measure inside a
FunctionalOp). `mutate(parents, rng, features)` is the weighted
dispatcher.

**GP.** `(μ+λ)` ES with tournament selection and Pareto-front elitism.
Reproducible by seed. Optional multiprocessing via `n_workers > 1` (see
the GPConfig docstring for honest perf characterisation — modest gains
on large problems; threading-with-nogil would be a bigger lever).

## Use cases by domain

`tessera.expression` was designed with time-series in mind, but the
abstractions generalise. Each row is something you could actually do:

| Domain | Problem | What primitives unlock |
|---|---|---|
| **Time series forecasting** | Predict y(t+h) from x(t-k) lags | Native fit |
| **Trading strategy discovery** | Find signal = f(price, volume, lags) | Native fit |
| **PDE identification** | Discover ∂u/∂t = f(∂²u/∂x², u, …) | Use measures on space + time grids |
| **Image filters** | Find best kernel for edge detection | 2D measures (planned tensor-product extension) |
| **Pharmacokinetics** | Drug concentration C(t) given dosing events | EMA/power-law for absorption + elimination |
| **Climate** | Multi-scale temperature dynamics | Hierarchical EMAs + diff at different scales |
| **Networking** | Queue-length dynamics + intervention | Cumulative sum + signed atomic |
| **Audio** | Time-frequency features | Wavelet-like custom densities (registerable) |

## Mathematical foundations

Reznikov, *Lecture Notes for Measure Theory* (MAA-5616, FSU 2019):

- **§3.3 Thm 99** — every measurable density induces a measure; this is
  why each `density_family` IS a measure.
- **§3.3 Example 102** — discrete infinite sums = integration against
  the counting measure; this is why `measure_signed_sum` naturally
  supports arbitrary lag positions and weights.
- **§3.3 Lemma 105** — `|∫ f dμ| ≤ ∫ |f| dμ`; absolute summability is
  the well-definedness criterion, enforced by
  `Measure.is_absolutely_summable()`.
- **§3.7 Thm 143 (Fubini)** — separable bilinear functionals decompose
  into iterated 1-D applies; this is the basis for `SeparableBilinear`
  and `Volterra2` decomposing into two cached `Measure.apply`s.

## Performance characteristics

On a typical 8-core Windows desktop with `koopman-env-ascii` Python:

- **`Measure.apply`** (single call, N=100 000):
  - Recursive EMA: ~0.3 ms
  - Direct conv: ~6-180 ms (depends on kernel length)
  - FFT conv: ~5 ms regardless of kernel length
- **`FunctionalCache.get_or_compute`**:
  - Hit: ~1-2 μs (dict lookup + array view)
  - Miss: same as the underlying `apply` call
- **GP** (pop=100, gens=30, N=5 000, 3 features): ~5 s
- **GP** (pop=200, gens=50, N=50 000, 10 features): ~30 s (sequential),
  ~25 s (n_workers=4 on Windows)

## Tests

117 tests across `tests/expression/`. Run with:

```bash
pytest tests/expression/
```

## What's planned

- **2-D measures** for image kernels and PDE discovery (`tensor product`
  of two 1-D measures, leveraging Fubini for the apply)
- **Time-varying kernels** K(s, t) — non-stationary functional analogues
- **Wavelet density family** as a registered kernel
- **GP threading with `nogil=True` numba kernels** — bigger speedup
  than process-pool MP on Windows
- **PySR adapter** — feed the discovered subexpressions as features to
  PySR for the pointwise composition layer (hybrid hierarchical search)

## See also

- Top-level [README](../../../README.md) — tessera as a whole
- [CHANGELOG](../../../CHANGELOG.md) — history
