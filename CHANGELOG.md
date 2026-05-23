# Changelog

All notable changes to `tessera` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`tessera.search` submodule** — extracts the search machinery from
  `tessera.expression.gp` into a dedicated submodule with a shared
  `Candidate` type, `pareto_front`, `mse_loss`, `_evaluate_tree`, and
  `optimize_constants`. Three searchers now share this infrastructure:
  - `GP` — population-based evolutionary search (was in expression.gp)
  - `SimulatedAnnealing` — single-state Metropolis-acceptance search
    with exponential/linear cooling, optional const-opt polish, and
    multi-restart support (new)
  - `RandomSearch` — i.i.d. random-tree baseline (new)
  All three return the same `Candidate` shape so Pareto fronts merge
  across algorithms (`pareto_front(gp_front + sa_front + rs_front)`
  is a single line). Backwards compatibility preserved: existing
  `from tessera.expression import GP, GPConfig, ...` keeps working
  via a re-export shim in `tessera.expression.gp`.

### Added (search submodule details)
- **`tessera.search.SimulatedAnnealing`** — Metropolis acceptance with
  `min(1, exp(-Δfitness/T))`, exponential or linear cooling, optional
  Const-leaf polish every K accepted moves, optional multi-restart.
  Provable convergence in probability under log-cooling
  (Geman & Geman 1984). Single-state search is easier to debug than
  a population.
- **`tessera.search.RandomSearch`** — sample N random trees, score,
  return Pareto front. Baseline for comparison; any directed searcher
  should beat it on a matched budget.

- **`tessera.koopman.LatentKoopman`** — Closed-form latent Koopman
  with time-delay embedding. Five-step identification:
  reduced-rank ridge OLS of one-step prediction operator β → SVD
  truncation → encoder E = V_k^T → latent OLS for K → OLS for
  decoder D. Single-matmul forecast at test time (`D · K^{h-1} · E ·
  past`). Separate from N4SID by having distinct E/K/D maps rather
  than tying via shared C. Supports `target_mode="delta"` for
  non-stationary / trending series with a per-coordinate
  mean-delta correction so constant slope is representable. 13
  tests in `tests/koopman/`. See [`docs/koopman.md`](docs/koopman.md).
- **`tessera.expression.tree`** — Five Node types (Var, Const, BinOp,
  UnOp, FunctionalOp) as frozen tagged-union dataclasses. Pointwise op
  tables (add/sub/mul/div/min/max, tanh/abs/sign/neg). Structural
  helpers: `complexity`, `depth`, `used_features`, `iter_subtrees`,
  `replace_at`. `evaluate(node, env, cache)` walker with automatic
  FunctionalCache integration.
- **`tessera.expression.mutation`** — `random_tree`, `random_measure`,
  `random_functional` for population init + mutation fresh material.
  Six classic mutation operators (`subtree_swap`, `subtree_crossover`,
  `constant_jitter`, `term_insert`, `term_delete`, `op_swap`) plus
  `measure_mutate` (tessera-specific: replace the measure inside a
  FunctionalOp). `mutate()` weighted dispatcher with retry-on-invalid.
  `validate_tree()` enforces depth/complexity/feature/constant caps.
- **`tessera.expression.gp`** — Population-based (μ+λ) GP loop with
  tournament selection, parsimony-weighted fitness, Pareto-front
  elitism, early-stop on plateau. `GPConfig` knobs, `Candidate` frozen
  dataclass, `GP.run(env, y_true, features)` returns the final Pareto
  front sorted ascending by complexity. `pareto_front()` and
  `mse_loss()` exposed as public utilities. `n_workers > 1` enables
  ProcessPoolExecutor multiprocessing (honest perf characterisation
  documented in GPConfig — modest gains on large problems; threading
  with `nogil=True` numba would be the bigger lever).
- **`tessera.expression` README** — full API map, primitive examples,
  use-cases-by-domain table, performance characteristics, mathematical
  foundations.

### Tests
117/117 passing across `tests/expression/`:
- 18 measure construction / kernel correctness
- 10 JIT backend routing + speedup smoke
- 13 cache memory / disk / LRU
- 12 functional bilinear / Volterra / cache-aware apply
- 25 tree Node types + evaluator + cache subexpression sharing
- 20 mutation operators + random tree generation + dispatcher
- 19 GP loop end-to-end + Pareto + reproducibility + multiprocessing

## [0.1.0] — 2026-05-24

### Added
- **`tessera.expression.measure`** — Lebesgue-decomposed signed measures on
  non-negative integer lags. `Measure(atoms, density_family, density_params,
  support_max)` with eager parameter validation.
- Density family registry with built-ins:
  `exponential` (halflife convention matching pandas),
  `exponential_e` (e-folding convention),
  `power_law` (long-memory; α > 1 required),
  `gaussian_half`, `rectangular`, `delta_minus_exp`.
- Convenience constructors:
  `measure_lag`, `measure_diff`, `measure_ema`, `measure_roll_mean`,
  `measure_power_law`, `measure_signed_sum`.
- **`tessera.expression._numba_kernels`** — JIT-accelerated hot paths:
  recursive EMA (O(N), exact pandas match), atomic shift-and-accumulate,
  direct conv (Numba), FFT conv (scipy).
- `Measure.apply(x, backend="auto")` routes to the optimal backend
  (706× speedup vs naive truncated convolution at h=168).
- **`tessera.expression.cache.FunctionalCache`** — two-tier (memory LRU +
  optional disk) cache keyed by `(hash(measure), var_id, fill_warmup)`.
  Subexpression sharing for GP search.
- **`tessera.expression.functional`** — n-ary wrappers:
  `LinearFunctional` (n=1, wraps Measure),
  `SeparableBilinear` (n=2, Fubini-decomposed: `(μa·x) · (μb·y)`),
  `Volterra2` (n=1, self-product `(μa·x) · (μb·x)` — captures squared returns,
  EMA cross-scale products, etc.).
- `apply_with_cache(functional, cache, var_ids, xs)` — cache-aware apply
  that memoizes each 1-D measure piece independently.

### Mathematical foundations
References to Reznikov's *Lecture Notes for Measure Theory* (MAA-5616, FSU
2019):
- §3.3 Thm 99 "Integral as a Measure" — each density family IS a measure.
- §3.3 Example 102 — discrete infinite sums = integrals against the counting
  measure (natural support for arbitrary signed-sum-of-lags kernels).
- §3.3 Lemma 105 — absolute summability is the well-definedness criterion
  (enforced by `Measure.is_absolutely_summable()`).
- §3.7 Thm 143 (Fubini) — separable bilinear functionals decompose into
  iterated 1-D applies; basis for the `SeparableBilinear` / `Volterra2`
  fast paths.

### Notes on extraction
Modules originally developed in a private research repo under
`src/lib/expression_layer/`. Extracted on 2026-05-24 to support sharing
across downstream projects (symbolic-chess, weather / PDE workbenches).
Original commits preserved at:
- `2b6080a` — Measure abstraction
- `2868da6` — Numba JIT kernels
- `fc10840` — FunctionalCache
- `3ad28a1` — n-ary Functional wrappers

### Tests
53/53 passing across `tests/expression/`:
- 18 measure construction / kernel correctness
- 10 JIT backend routing + correctness + speedup smoke
- 13 cache memory / disk / LRU / key uniqueness
- 12 functional bilinear / Volterra / cache-aware apply

[Unreleased]: https://github.com/davechendatascience/tessera/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davechendatascience/tessera/releases/tag/v0.1.0
