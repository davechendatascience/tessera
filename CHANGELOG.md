# Changelog

All notable changes to `tessera` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (GPU backend — Tier 1: end-to-end `evaluate` on JAX arrays)
- **`tessera.backend.array_module(x)`** — backend-polymorphic dispatch
  helper. Returns `jax.numpy` if `x` is a JAX array, else `numpy`. Used
  throughout the op tables and `Measure.apply` to let trees evaluate on
  whatever array library the inputs are in, without changing global
  state.
- **All `BIN_OP_FNS` / `UN_OP_FNS` are now backend-polymorphic.** Pass
  a JAX array in, get a JAX array out. Previously the indicator ops
  (`gt/lt/ge/le`, `step`) and the protected transcendentals
  (`sqrt/exp/log/pow`) hard-converted to numpy via `np.asarray(...)`;
  these now use `array_module(x).asarray(...)` to preserve the array
  library.
- **`Measure.apply(jax_array)` routes to a JAX path** that materializes
  the discrete kernel as a JAX array and runs `jnp.convolve` for the
  convolution. Bypasses the numba/FFT numpy-only fast-paths (recursive
  EMA, atomic shift-and-accumulate). Output is a JAX array on the same
  device as the input.
- **`evaluate(tree, env_jax)`** end-to-end works on JAX. `Var` resolution
  detects JAX inputs and returns JAX arrays; `_maybe_broadcast` is
  backend-polymorphic; `Const` leaves return Python floats that broadcast
  against either array library. Combined, this means **a tree can be
  evaluated on GPU just by passing a JAX env**.
- **33 new tests** in `tests/test_jax_backend_tier1.py` (skipped if jax
  unavailable). Cover: `array_module` dispatch; every bin/un op produces
  a JAX array on JAX inputs; numerical agreement with numpy path within
  float32 precision; `Measure.apply` JAX path matches numpy `backend="kernel"`
  exactly; full `evaluate(tree, env_jax)` round-trip for pointwise,
  indicator, transcendental, and Functional trees.
- **`notebooks/tessera_jax_tier1.ipynb`** — Tier 1 demo on Colab. Installs
  tessera + JAX-CUDA, builds a representative SR tree, times
  `evaluate` on numpy CPU vs JAX GPU at N=60K (MNIST scale). Pointwise
  trees see 5-50× speedup typical on Colab T4; measure-theoretic
  (Functional) trees see 10-100× because the JAX `convolve` saturates
  the GPU.

**Full test count: 424 passing** (was 392; +32 Tier 1 + 1 previously-skipped
backend test that now runs because JAX is available locally).

**What Tier 1 does NOT do:**
- GP search loop is still numpy-internal; only individual tree evaluations
  on JAX inputs run on GPU. Per-generation overhead dominates for small
  populations.
- No `jit`-compilation of trees (Tier 2).
- No batched-population evaluation via `vmap` (Tier 3).
- No `jax.grad`-based constant optimization (separate sub-milestone).

These are the next bottlenecks for the MNIST 95% target.

### Added (transcendental primitives — closes Feynman vocabulary gap)
- **`sqrt`, `exp`, `log`** added to `UN_OP_FNS` and **`pow`** added to
  `BIN_OP_FNS` in `tessera.expression.tree`. All use PySR-style
  protected semantics so the GP search cannot be NaN-poisoned:
    `sqrt(x)`   := `sqrt(|x|)`
    `log(x)`    := `log(max(|x|, 1e-12))`
    `exp(x)`    := `exp(clip(x, ±50))`
    `pow(a, b)` := `pow(max(|a|, 1e-12), clip(b, ±8))`, NaN → 0
- **Interval bounds** for all four added in
  `tessera.expression.interval`, so branch-and-bound pruning still
  works when the new ops appear in trees.
- **Simplifier folds** added in `simplify.core`:
    `log(exp(x)) → x`
    `exp(log(x)) → |x|`     (protected semantics)
    `sqrt(|x|)   → sqrt(x)` (drop redundant abs)
    `pow(x, 0)   → 1`
    `pow(x, 1)   → |x|`     (protected semantics)
    `pow(0, x)   → 0`
- **`op_swap` mutation** groups updated: `{mul, div, pow}` and
  `{sqrt, log, exp}` so the GP can swap between related ops.
- **18 tests** in `tests/expression/test_transcendentals.py` covering
  protected evaluation (no NaN/inf), interval-bound soundness, and
  the new simplifier folds.

**Motivation:** the Feynman subset benchmark exposed that tessera
could not represent `sqrt(x)`, `exp(x)`, or `log(x)` — equations
I.8.14 (Euclidean distance) and I.43.31 (Stokes-Einstein) failed
outright; I.6.20a (Gaussian) reached only rel=0.02 via a tanh-
indicator caricature. The new ops close this vocabulary gap.

**Effect on `benchmarks/results/feynman_subset.md`** (pop=200, gens=80):
- I.6.20a (Gaussian):    rel 0.02 → **0.0012** (~20× improvement)
- I.43.31 (Stokes):      rel 0.998 → **0.27** (was unfit; now partial)
- I.8.14 (Euclidean):    failed (constant) → rel=0.22 (uses `pow`)
- I.12.1 (μ*Nn):         exact, unchanged
- I.14.3 (m*g*z):        exact → rel=0.10 (search-space dilution;
                                            mitigated by larger budget)

Trade-off is documented: extra ops capture more forms but dilute
search for trivial ones. Larger pop/gens recovers most ground.

### Added (axis-semantic type system, scoped minimum)
- **`tessera.expression.axes`** — first step toward
  `invariance_in_sr.md`'s axis architecture, scoped as a minimum
  useful addition. Lives inside `tessera.expression` (not a
  top-level submodule) because it depends entirely on expression's
  tree types and doesn't introduce new mathematical primitives.
  Components:
    `Invariance` enum (TRANSLATION, CAUSAL_TRANSLATION, PERMUTATION,
                       CYCLIC, LOG_TRANSLATION, ROTATION, GRAPH, NONE)
    `Axis(name, size, invariance)` — declares one variable dimension
    `TypedVar(name, axes=(...,))` — Var + axis-tuple metadata
    `OperatorAxisRule` + `OPERATOR_RULES` table — per-operator
                                                  axis-compatibility rules
    `check_compatibility(tree, typed_env)` — non-enforcing checker
                                              that returns None or an
                                              error string
  Covers the common cases:
    Time series:        `Axis("time", N, CAUSAL_TRANSLATION)`
    Image:              two axes of TRANSLATION
    Multi-asset basket: time (CAUSAL_TRANSLATION) + asset (PERMUTATION)
  19 tests covering construction, all four invariance types, and
  per-operator compatibility (rejects LinearFunctional on permutation,
  FunctionalOp2D on 1-D, etc.).
- **`docs/framework_synthesis.md`** — maps every shipped tessera
  component to one of SEVEN roles in the Knuth-grounded perfect-info
  game framework. Answers "how do we integrate Knuth's work with our
  diverging implementations?" — the answer is the implementations
  aren't diverging; each fills one of seven slots. The document is
  the explicit mapping that makes that clear.

### Added (CPU/GPU backend abstraction)
- **`tessera.backend` module** — switchable CPU/GPU backend with a
  clean public API:
    `set_backend("numpy")` (default; fully functional)
    `set_backend("jax")` (skeleton; raises informative ImportError
                          if jax isn't installed)
    `current().asarray(x)` / `current().convolve(a, v)` etc.
  - `Backend` Protocol defining the minimum interface
  - `NumpyBackend` wraps existing numpy + numba paths
  - `JaxBackend` skeleton: `asarray`, `zeros`, `full_like`, `convolve`
    work when JAX is installed; tree-walker / measure-apply
    integration deferred to Tier 1 of the milestone
  - 12 tests covering both backends + switching API
  - Top-level re-exports from `tessera.__init__`: `set_backend`,
    `get_backend`, `current`, `Backend`, `NumpyBackend`, `JaxBackend`
- **`docs/milestones/gpu_backend.md`** — milestone tracking doc. The
  public API is committed; internal porting is broken into 4 tiers
  (measure-apply, tree eval + cache, batched eval, benchmarks)
  totaling ~8-10 days of focused work. Lists acceptance criteria,
  effort estimates, sequencing recommendations, and dependencies
  on other milestones.

### Added (reduce ops for invariance via aggregation)
- **`reduce_mean / reduce_max / reduce_sum / reduce_std`** — new
  unary ops in `UN_OP_FNS` that collapse an array to a scalar by
  reducing over all axes. Lets the GP DISCOVER the aggregation rule
  (mean-pool vs max-pool vs sum vs std) instead of having it
  hardcoded. Motivated by the MNIST validation experiment
  (`benchmarks/run_mnist_feature_discovery.py`): the 0.82 test
  accuracy capped because mean-pool is too crude; max-pool would
  capture digit-class structure better. Interval bounds: mean/max
  stay in [input.lo, input.hi]; sum is conservative ±∞ when input
  spans zero; std bounded by spread. Simplifier folds
  `reduce_X(Const(c))` → `Const(c)`. 20 new tests.

### Added (validation experiment)
- **`benchmarks/run_mnist_feature_discovery.py`** — runs the §12 first
  step from `invariance_in_sr.md`: MNIST 0-vs-rest classification
  using current (untyped) tessera + hardcoded mean-aggregation.
  Custom GP loop (per-image scoring; tessera's `GP.run` can't handle
  per-sample evaluation directly).

  **Result:** TEST accuracy 0.82 (chance 0.50) on 100 test samples,
  with pop=50/gens=15/200 train images at 14×14 downsampled. The GP
  discovered a horizontal Laplacian kernel `[+1, -2, +1]` wrapped in
  `step()` — i.e., a recognizable classical edge detector — from data
  in 14 seconds wall-clock. Kernel visualization saved to
  `benchmarks/results/mnist_discovered_kernel.png`.

  **Validates** the underlying claim of `invariance_in_sr.md`:
  tessera's measure-theoretic operators + simple aggregation can
  discover interpretable, translation-equivariant feature kernels.
  The axis-types architecture proposed in that note is worth
  building.

  **Caveat:** 0.82 is not CNN-competitive (CNNs reach ~0.99). The gap
  motivates the architectural follow-ups: discoverable aggregator
  operators (max-pool, etc.), larger budget, and the `tessera.axes`
  type system that lets the GP search BOTH the kernel and the
  pooling rule.

### Added (research notes)
- **`docs/research_notes/invariance_in_sr.md`** — invariance, sensor
  data, and axis-semantic SR. Argues for making **axis semantics a
  first-class search choice**: every variable carries not just a
  shape but an axis-type declaration (Translation, Permutation,
  Cyclic, LogTranslation, Rotation, Graph), and operators are
  constrained by axis-type compatibility. Three motivations:
  generality across sensor modalities (video, audio, multi-asset,
  point clouds), interpretability-with-invariance (the strongest
  unique claim for tessera vs CNNs and vanilla SR), and a clean
  GPU dispatch (each invariance group → its canonical GPU primitive).
  Sketches a new `tessera.axes` submodule and connects to the
  perfect-info game framework's |E_K| conjecture via Burnside-
  flavoured group-quotient compression. Includes a concrete first
  step: validate the underlying claim by running MNIST 0-vs-rest
  classification with `FunctionalOp2D` + hardcoded mean-aggregation
  on tessera's CURRENT (untyped) machinery, before investing in
  axis-types architecture.

### Added (measure algebra)
- **`Measure.compose(other)`** — measure convolution `μ * ν` returning
  a new canonical Measure. Maps to the operator-algebra identity
  $L_{\mu * \nu}(x) = L_\mu(L_\nu(x))$ from
  `docs/research_notes/measure_theory_and_perfect_info.md` §3.3.
  Implementation: `np.convolve` of the two discrete kernels, then
  sparsify into atoms. Tests cover: lag composition, diff*diff = 2nd
  diff, identity element (δ₀), zero element, commutativity,
  associativity, semantic equivalence with nested apply on the
  interior (NaN warmup), signed cancellation, and canonical-form output.
- **`collapse_functional_chain` mutation** — new GP mutation operator
  that finds `L_μ(L_ν(x))` patterns in a tree and replaces with
  `L_{μ*ν}(x)`. Strictly reduces complexity (typically by ~1 node) and
  exploits the measure-algebra equivalence the search couldn't
  previously discover via random mutation. Weight 0.05 in OP_WEIGHTS
  (the existing measure_mutate dropped to 0.07, measure_2d_mutate to 0.03).
  Skipped in `pointwise_only` mode. Tests verify pattern matching,
  complexity reduction, semantic preservation, and dispatcher
  integration.

### Changed
- **Measure canonicalisation at construction** — `Measure.__post_init__`
  now sorts atoms by lag, merges duplicates (summing weights), and
  drops near-zero atoms. Two semantically identical measures
  constructed in different atom orders now compare equal and have
  the same hash. Translates `docs/research_notes/measure_theory_and_perfect_info.md`
  §3.1 (Lebesgue decomposition uniqueness) into the actual `Measure`
  type. Downstream effect: FunctionalCache hits on
  mathematically-equivalent measures across mutations, search
  population dedup works on canonical measures. Backwards-compatible
  for all existing call sites; tests `test_measure_canonical.py`
  cover construction order, merging, zero-dropping, density
  preservation, and the cache-hit benefit.

### Added
- **FunctionalOp2D L1-norm interval bound** —
  `tessera.expression.interval.measure_2d_l1_norm` decomposes a
  Measure2D into atomic + separable-density parts and returns
  `Σ|atoms| + ||sep_t||_1 · ||sep_x||_1` (Fubini factorisation of
  the product kernel). `interval_evaluate` now bounds FunctionalOp2D
  output by `±||μ_2d||_1 · max(|x.lo|, |x.hi|)`. Closes the last
  conservative-±∞ case in the interval evaluator. Tests:
  test_interval_functional_2d_bounded + test_measure_2d_l1_norm.
- **`docs/research_notes/gpu_and_cv_via_sr.md`** — honest scoping
  document for "tessera → GPU → CV via SR-evolved architectures."
  Three-stage path (GPU backend → CV benchmarks → SR-as-NAS),
  realistic timelines (1-2 months / 1-2 / 2-3), with section 10
  specifically answering "does Knuth's framework work on GPU?" —
  yes with batched eval, equivalence-class collapse becomes more
  central than branch-and-bound pruning when GPU is available.
- **`docs/research_notes/measure_theory_and_perfect_info.md`** —
  theoretical companion to `fit_as_perfect_info_game.md`. Develops
  the argument that tessera's measure-theoretic operator algebra
  ADDS three things to the perfect-information framework: (1) a
  richer canonical-form structure via Lebesgue decomposition; (2)
  closed-form lower bounds via L1 norms; (3) tractable bilinear
  factorisation via Fubini. Grounded empirically in the
  step (a)/(b)/(c) benchmark results.
- **Empirical research benchmarks (steps a/b/c)**:
  - `benchmarks/run_equivalence_class_count.py` — enumerates all
    valid trees up to depth 3 with restricted pointwise grammar;
    computes |E_K| / |T_K| ratio under `simplify_canonical`.
    Result: ratio drops monotonically to 7.7% at cx=7
    (~92% of syntactic trees are equivalence-class duplicates).
  - `benchmarks/run_interval_bound_tightness.py` — samples 2000
    random trees on 3 workloads; reports tightness ratio
    (bound / actual_loss) distribution. Pre-step-(c): median 0.14
    on synthetic_xx, 47% trees unbounded.
  - L1-norm interval bounds (step c) on `LinearFunctional`,
    `SeparableBilinear`, `Volterra2`: extends
    `tessera.expression.interval` with
    `measure_l1_norm(m) = ∑|kernel|` and L1-bound-based interval
    semantics. Re-running tightness benchmark: median ratio
    0.14 → 0.47 on synthetic_xx (3.4× tighter); unbounded
    fraction 47% → 19%.

### Added
- **`tessera.expression.simplify` subpackage** — promoted the simplifier
  to its own submodule so multiple simplification strategies can grow
  as siblings:
  - `simplify` (rule-based folds; moved from `tree.py`)
  - `simplify_ac` — Associative-Commutative normalisation. Flattens
    nested `add`/`mul`/`min`/`max` chains, sorts children by
    `(complexity, str)`, rebuilds left-leaning. Result: `a + b` ≡
    `b + a`, `(a+b)+c` ≡ `a+(b+c)` ≡ `(c+b)+a` (canonical form).
    Per the perplexity research note (docs/research_notes/
    search_as_energy_min.md), parsimony was "distorted by arbitrary
    syntactic differences" without this; AC norm gives parsimony a
    fair semantic-equivalence-class basis.
  - `simplify_canonical = simplify ∘ simplify_ac` — recommended SR
    default. AC norm first so constants cluster, then rules fold
    them: `2 + x + 3 → 5 + x` in one canonical pass. Wired into
    GP / SA / RandomSearch as the default when `simplify_trees=True`.
- **`tessera.expression.interval`** — sound interval-arithmetic
  evaluation of Expr trees. Each pointwise op (`add`, `sub`, `mul`,
  `div`, `min`, `max`, `gt`/`lt`/`ge`/`le`, `neg`, `abs`, `tanh`,
  `sign`, `step`) has a closed-form interval semantics; tight bounds
  where possible (e.g. `gt(a, b)` is exactly 1 when `a.lo > b.hi`).
  `FunctionalOp` / `FunctionalOp2D` get conservative ±∞ (future:
  tighten via measure L1-norm bound). Used by the search submodule's
  lower-bound pruning.
- **`tessera.search.bounds`** — branch-and-bound infrastructure:
  - `mse_lower_bound(pred_lo, pred_hi, y_true)` — tight closed-form
    MSE lower bound: per-sample optimal pred is `clip(y_true,
    pred_lo, pred_hi)`; bound is mean squared distance to the clip.
  - `pareto_threshold(front, cx)` — loss a new candidate at
    complexity cx must beat to be Pareto-relevant.
- **`GPConfig.prune_by_lower_bound`** — opt-in branch-and-bound pruning
  in GP. When enabled (with `mse_loss` + `n_workers=1`), `_score()`
  computes the interval bound before full evaluation and skips
  candidates whose MSE lower bound exceeds the Pareto threshold.
  `GP.prune_stats` reports `n_pruned` / `n_evaluated`. Direct
  operationalisation of the "SR-for-fit as energy minimisation with
  full data information" framing (docs/research_notes/
  search_as_energy_min.md, validated by perplexity research as a
  "significant research opportunity").
- **`docs/research_notes/fit_as_perfect_info_game.md`** — independent
  research framework: SR-for-fit as a single-agent perfect-information
  game in the Knuth tradition. Develops the chess-game analogy from
  the user's 2026-05-24 session into a formal framework grounded in
  TAOCP Vol 4 combinatorial algorithms (backtracking, branch-and-bound,
  dancing links, BDD/ZDD). States open theoretical questions and
  connects to tessera's experiments. Future research base, not
  immediate implementation.

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
