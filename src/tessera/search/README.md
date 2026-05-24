# `tessera.search`

Search algorithms over `tessera.expression` trees. Each searcher returns
a Pareto front of `Candidate` objects in (complexity, train_loss) space;
all algorithms share the same scoring infrastructure (NaN precheck,
simplifier, const-opt polish) so they're directly comparable.

## Available searchers

| Class | Algorithm | When to use |
|---|---|---|
| `GP` | (μ+λ) ES with tournament selection + Pareto-front maintenance | Default. Good population diversity; benefits from subexpression cache; parallelisable. |
| `SimulatedAnnealing` | Single-state SA with Metropolis acceptance | Smooth losses with small budgets; provable convergence under log-cooling (Geman & Geman 1984); easier to debug a single trajectory than a population. |
| `RandomSearch` | i.i.d. random-tree sampling | Baseline — any directed searcher should beat this on a matched budget. |

## Shared infrastructure

| Symbol | Purpose |
|---|---|
| `Candidate` | (tree, train_loss, complexity, fitness, born_gen) frozen dataclass |
| `pareto_front(candidates)` | Non-dominated set in (cx, loss) space, sorted by cx ascending |
| `HallOfFame` | Per-complexity best-ever store; protects discoveries from mutation drift |
| `mse_loss(y_pred, y_true)` | Default loss; NaN-mask aware |
| `pnl_loss_hard / pnl_loss_smooth` | Trading-flavoured losses (the smooth variant is a Hamiltonian relaxation; both top-level picklable for multiprocess workers) |
| `_prediction_is_valid(y_pred, y_true, min_valid_frac)` | NaN-fraction precheck called before any user `loss_fn` |
| `_evaluate_tree(tree, env, y_true, cache, ..., loss_fn, ...)` | The scoring chokepoint used by every algorithm |
| `optimize_constants(tree, env, y_true, loss_fn, cache, ...)` | scipy-based Const-leaf polish; PySR-style |
| `mse_lower_bound / pareto_threshold` | Branch-and-bound primitives — combined with `tessera.expression.interval`, allow GP to skip candidates whose loss-lower-bound exceeds the incumbent Pareto front |

## Quick start

```python
import numpy as np
from tessera.search import GP, GPConfig, SimulatedAnnealing, SAConfig, RandomSearch, RSConfig

rng = np.random.default_rng(0)
n = 500
x = rng.standard_normal(n)
y = x * x + 0.5

# All three searchers share the same call shape:
gp_front = GP(GPConfig(pop_size=80, n_gens=20)).run({"x": x}, y, ["x"])
sa_front = SimulatedAnnealing(SAConfig(n_steps=2000)).run({"x": x}, y, ["x"])
rs_front = RandomSearch(RSConfig(n_trees=2000)).run({"x": x}, y, ["x"])

# Pareto fronts are mergeable (same Candidate type):
from tessera.search import pareto_front
merged = pareto_front(gp_front + sa_front + rs_front)
```

## Custom losses

All searchers accept `loss_fn=` in the constructor. Contract:

```python
def loss_fn(y_pred: np.ndarray, y_true: np.ndarray) -> float: ...
```

The NaN-fraction precheck happens BEFORE `loss_fn` is called, so loss
functions only need to handle shape broadcasting + their own arithmetic.
For multi-worker GP runs, the loss_fn must be picklable (top-level
function or `functools.partial`).

## Backwards compatibility

`from tessera.expression import GP, GPConfig, mse_loss, pareto_front`
keeps working — those symbols re-export from this submodule. New code
should import from `tessera.search` directly.

## Branch-and-bound pruning (optional)

Opt-in via `GPConfig(prune_by_lower_bound=True, n_workers=1)` (currently
MSE + serial path only). When enabled, each candidate is pre-checked
via interval arithmetic; if its provable loss-lower-bound exceeds the
Pareto-front loss at its complexity, the full evaluation is skipped.

```python
cfg = GPConfig(pop_size=120, n_gens=40, prune_by_lower_bound=True)
gp = GP(cfg)
front = gp.run(env, y, feature_names)
print(gp.prune_stats)   # {"n_pruned": ..., "n_evaluated": ...}
```

L1-norm bounds make this work for measure-theoretic operators
(LinearFunctional, SeparableBilinear, Volterra2, FunctionalOp2D) —
see `tessera.expression.interval`. Validated empirically on synthetic
benchmarks: tightness ratio median 0.47 on smooth targets.

## Tests

```bash
pytest tests/search/        # ~50 tests across the 3 searchers + HoF + bounds
```

Coverage:
- `test_imports.py` — public API, backwards-compat, Candidate sharing
- `test_sa.py` — Metropolis acceptance, cooling schedules, finds signal
- `test_random_search.py` — baseline runs + finds signal
- `test_compare.py` — all 3 on the same problem; merged Pareto front
- `test_hall_of_fame.py` — per-cx best-ever store; protection from drift
- `test_bounds.py` — interval arithmetic + MSE lower bound + Pareto threshold
- `test_losses_trading.py` — PnL hard/smooth losses (Hamiltonian relaxation)

## See also

- `docs/roadmap.md` — gap analysis vs PySR, reading list, QUBO/Ising
  direction for tessera.search
