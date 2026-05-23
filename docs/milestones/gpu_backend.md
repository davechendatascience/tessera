# Milestone: GPU backend (JAX) for tessera

**Status:** scaffold shipped 2026-05-25. Public API (`tessera.set_backend`,
`tessera.get_backend`, `tessera.current`) is the commitment. The JAX
backend's internals are partially implemented and partially deferred.

This document tracks the milestone's scope and the remaining work.

## What shipped today

- `tessera.backend` module with a `Backend` Protocol
- `NumpyBackend` (default, fully functional)
- `JaxBackend` skeleton:
  - imports `jax` lazily on first call
  - `asarray`, `zeros`, `full_like`, `convolve` work if JAX is installed
  - `is_available()` reports JAX presence
- `set_backend(name)` / `get_backend()` / `current()` public API
- 12 tests covering both backends + the switching API

What the API commits us to: tessera code that needs array operations
should call `tessera.current().asarray(x)` rather than `np.asarray(x)`.
That way the backend switch transparently changes the array library
underneath.

## What's NOT shipped yet

The scaffold is the interface; the FULL JAX support requires porting
the hot-path code. In dependency order:

### Tier 1 — measure-theoretic operators

These are the workhorses for SR-on-images. The current implementations
are in `tessera.expression.measure` (`Measure.apply`),
`tessera.expression._numba_kernels` (recursive EMA, atomic apply,
convolution), and `tessera.expression.functional` (LinearFunctional,
SeparableBilinear, Volterra2 wrappers).

**Required work:**

1. **`Measure.apply` backend dispatch.** Replace direct numpy calls
   with `tessera.current()` equivalents. The recursive-EMA fast path
   (Numba-JITted) needs a JAX equivalent (or fall back to FFT conv
   when on JAX). ~150 LOC.
2. **`Measure2D.apply` backend dispatch.** Same idea for 2-D
   measures. The atomic part is straightforward; the separable
   density needs `jax.numpy` convolution. ~80 LOC.
3. **Functional wrappers**. `LinearFunctional`, `SeparableBilinear`,
   `Volterra2` are thin wrappers — they just call `Measure.apply`.
   Should work transparently once (1) and (2) are done. ~0 LOC.

### Tier 2 — tree evaluation + cache

The tree walker (`tessera.expression.tree.evaluate`) and
`FunctionalCache` are the next layer.

**Required work:**

4. **`evaluate(tree, env)` array-type-aware.** Currently the walker
   uses `np.isscalar`, `np.asarray`, `np.where`, etc. These mostly
   work via JAX's `__array_function__` protocol but `np.where` may
   need explicit `jax.numpy.where`. ~30 LOC of audit + fixes.
5. **`FunctionalCache`** — cached values should be on the backend's
   array type. Adding `backend.asarray(...)` at cache insert is
   sufficient. ~20 LOC.

### Tier 3 — search-loop batching (the actual GPU win)

The big speedup comes from EVALUATING MANY CANDIDATES IN PARALLEL on
the GPU. Today's GP loop evaluates candidates one-by-one.

**Required work:**

6. **Batched-tree-eval API**. `evaluate_batch(trees, env)` that
   `jax.vmap`s over candidates. Each candidate's measures parameterise
   a small kernel; the vmap batches across candidates. ~200 LOC,
   non-trivial design (handles trees with different STRUCTURES — needs
   either uniform structure constraints or a slow-path per-tree fallback).
7. **`GP.run` batched mode**. Opt-in: `GPConfig(use_batched_eval=True)`.
   Falls back to per-candidate if any candidate's structure can't be
   vmapped. ~100 LOC.

### Tier 4 — diagnostics + benchmarks

8. **Re-run the BTC and MNIST benchmarks on JAX** to quantify the
   speedup. Expected: 10-100× on candidate-evaluation cost depending
   on dataset size and GPU vs CPU.
9. **Document the speedup**: a `benchmarks/results/jax_speedup.md`
   that shows the CPU-vs-GPU comparison on shared problems.

## Effort estimate

| Tier | LOC | Days |
|---|---|---|
| 1 — Measure apply on JAX | 230 | 2-3 |
| 2 — tree eval + cache | 50 | 1 |
| 3 — batched-tree-eval | 300 | 4-5 |
| 4 — benchmarks + docs | 100 | 1-2 |
| **Total** | **~700** | **~8-10** |

This is ~2 focused weeks of work, materially smaller than the original
"6+ months" estimate in `gpu_and_cv_via_sr.md`. Why the difference?
The earlier estimate assumed building both the GPU backend AND the
CV-specific NAS layer (SR-as-NAS). The milestone here is JUST the
backend — the NAS work is a separate downstream milestone.

## Sequencing recommendations

In order of impact-per-effort:

1. **Tier 1 first.** Without measure-apply on JAX, no other tier
   matters. Stage-1 of the full GPU direction.
2. **Tier 4 (incremental).** As soon as Tier 1 works, run BTC PnL
   on JAX. Even without batched eval, the per-candidate convolution
   cost should drop, especially for long-window measures.
3. **Tier 2 + Tier 3** in parallel. Tree-walker auditing is small
   work that gates the bigger Tier 3 effort.
4. **Tier 3 final**: the headline GPU win. Re-run all benchmarks.

## Dependencies on other milestones

- **`tessera.axes` (per `invariance_in_sr.md`)** would benefit from
  the JAX backend but doesn't strictly require it. Axes can ship on
  CPU first; JAX backend extends it later.
- **`WeightedIndicatorSum` primitive (deferred from BTC closure
  research)** is orthogonal — can ship before or after GPU.
- **Equality saturation (deferred Exp 4 from `search_as_energy_min.md`)**
  is orthogonal.

## Acceptance criteria for "GPU backend shipped"

1. `set_backend("jax")` succeeds on a machine with JAX installed
2. All existing benchmarks (`run_*.py` scripts) work under both
   backends without code changes
3. `benchmarks/results/jax_speedup.md` shows a measured speedup on
   at least one benchmark
4. CI runs the full test suite under both backends

Until all four are met, the milestone is *partial*. The scaffold
shipped today meets only criterion 1 (and only for `set_backend`'s
public API; the underlying measure-apply still falls back to numpy).

## Public-API stability

The `tessera.set_backend` / `tessera.current` API is committed: future
work won't change the function signatures. Internal modules
(`tessera.expression.*`) may grow `backend.current()` calls but
won't change their public APIs.

This means user code written today (e.g., `tessera.set_backend("jax")`)
will *work* once the GPU backend ships; it just won't be faster than
numpy in the interim because the operators haven't been ported.

## Changelog
- 2026-05-25: initial document. Scaffold shipped; tiers 1-4 deferred.
