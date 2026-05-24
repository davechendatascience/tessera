# Tessera planned work — open roadmap

**Status:** all items in this file are *not yet shipped*. Done items have been moved out (see "Recently shipped" section at the end for pointers). Research-direction material has been moved to `docs/research/`.

Items here are flagged as:
- **○ PLANNED** — committed; effort estimated
- **▷ IN PROGRESS** — actively being built
- **× DEFERRED** — was planned, explicitly postponed (with reason)

## 1. Gap analysis vs PySR (remaining open items)

After the GP / search / GPU work shipped in May 2026, here's what tessera *still* doesn't match PySR on:

| Feature | PySR | Tessera | Status | Effort |
|---|---|---|---|---|
| **ε-lexicase / loss-biased selection** | yes (`LossSelect`) | tournament only | ○ PLANNED | ~80 LOC |
| **Multi-population + migration** | P populations of N each, top migrate every K gens | single pop | ○ PLANNED | medium |
| **Annealed mutation temperature** | weights decay over gens | fixed weights | ○ PLANNED | ~30 LOC |
| **Adaptive complexity penalty** | `parsimony` adjusts via population stats | fixed | ○ PLANNED | small |
| **Constraint system** (max complexity per op, etc.) | yes | partial | ○ PLANNED | medium |
| **`jax.grad` constant optimisation** | n/a (Julia BFGS) | scipy Nelder-Mead | ○ PLANNED | ~half day |
| **2D Measure recursive EMA on JAX** (via `lax.scan`) | n/a | kernel-conv only | × DEFERRED — kernel-conv works for MNIST features | half day |

The top three to ship next, in priority order:

### 1.1 ε-lexicase selection ○ PLANNED

Tessera's `_tournament` picks the best of K random candidates by `fitness` (loss + parsimony·cx). PySR uses stronger strategies:

- **`LossSelect`**: weighted by `loss_function` exponent — candidates with very low loss have higher probability of being picked as parent. Heavier exploitation pressure.
- **ε-lexicase selection**: treat each TRAIN sample as a test case. A candidate "passes" sample t if its prediction is within ε of the median for that sample. Best-passing candidates win. Preserves diversity — candidates that nail HARD samples (which most others miss) are kept even if their average loss is mediocre.

**Why it matters:** plain tournament converges fast to a local minimum; lexicase explores wider for longer. For PnL+flip-style losses where the optimum involves complex sign-flip structure, the diversity helps.

**Effort:** ~80 LOC for both. Could be added as `selection_method` GPConfig option.

### 1.2 Annealed mutation temperature ○ PLANNED

PySR's mutation operators have weights that change over generations. Early: more exploration (large `subtree_swap` and `term_insert`); late: more exploitation (more `constant_jitter` and `do_simplification`). Implemented via a temperature parameter `T(gen)` that decays.

**Why it matters:** fixed `OP_WEIGHTS` in tessera means a 50-gen run behaves the same on gen 1 as gen 49. Many real searches benefit from "settling down" toward refinement.

**Effort:** ~30 LOC. Multiply the weights by a per-gen schedule.

## 2. Scalable upgrades from the high-dim SR research note

These items were promoted from `docs/research/high_dim_symbolic_regression.md` §5 on 2026-05-24. The research doc surveys why high-dim SR is hard and proposes five scalable directions; the two cheapest are promoted here to ship first.

### 2.1 Sparsity-bias `random_tree` for high-K Var spaces ○ PLANNED

**Origin:** [`docs/research/high_dim_symbolic_regression.md`](../research/high_dim_symbolic_regression.md) §5.5.

**What:** add a `max_distinct_vars` (or `var_sparsity`) parameter to `random_tree(...)` and the mutation operators. Bias generation toward trees that reference a small subset of the available variables (e.g., ≤ 10 of 784 pixels), rather than uniform-random across all features.

**Why now:** highest leverage-per-effort of the five §5 directions. For high-K inputs the effective search space shrinks from `K^depth` to `C(K, s) × small_tree_space` — at K=784, s=10, this is ~10^21× compression. Cheap to implement; gives a clean A/B against the current uniform-random baseline; lets us measure whether scaling SR search reach matters on MNIST at all.

**Effort:** ~1 day. Add the parameter; update `random_tree` + the relevant mutators; tests; one MNIST notebook run with the new flag enabled at sparsity=10.

**Acceptance criterion:** the flag exists, defaults to no constraint, has a test that confirms `sparsity=10` produces trees with ≤ 10 distinct Var leaves. MNIST K=10 run with `sparsity=10` reaches ≥ 73% test accuracy (3pt over the 71% baseline). If accuracy is unchanged or worse, we've ruled out sparsity as a useful lever — also a clean negative result.

### 2.2 Default-on GPU-parallel B&B for MSE workloads ○ PLANNED

**Origin:** [`docs/research/high_dim_symbolic_regression.md`](../research/high_dim_symbolic_regression.md) §5.1; companion to `docs/research/fit_as_perfect_info_game.md` §12.

**What:** (a) flip `GPConfig.prune_by_lower_bound` default from False to True when `loss_fn is mse_loss`; (b) report pruning rate alongside `hit_rate` in the GP's per-generation log; (c) vmap the interval-bound check across the population so it runs as one batched GPU call instead of K scalar calls.

**Why now:** the perfect-info-game framing's most-underused lever. The existing scalar B&B path was opt-in because the per-tree bound check was slow; the GPU-batched version removes that cost. Combined with the existing Tier-3 batched eval, every generation gets a "free" pre-filter.

**Effort:** ~2 days total. Day 1: scalar default-on + log integration. Day 2: GPU-batched interval evaluation via vmap over the tree's `interval_evaluate` walk (the bound is a 2-element [lo, hi] per node; vmap over a population's interval-trees is the right shape for jit).

**Acceptance criterion:** with `prune_by_lower_bound=True` (default for MSE), the GP run on a Feynman benchmark equation prunes ≥ 30% of candidates per generation (median across 5 seeds) without changing the Pareto front it returns. Wall-clock per generation drops by ≥ 20% over the same problem without pruning.

## 3. Reading list

(Reference material, not status-flagged.)

### 3.1 GP / SR foundations

- Koza, *Genetic Programming* (1992) — original GP textbook
- Poli, Langdon, McPhee, *A Field Guide to Genetic Programming* (2008) — free PDF
- Cranmer, *Interpretable Machine Learning for Science with PySR/SymbolicRegression.jl* (2023, arxiv 2305.01582) — PySR design paper

### 3.2 Const optimisation in SR

- Topchy & Punch, *Faster Genetic Programming based on Local Gradient Search* (2001) — original local-search-inside-GP paper
- Kommenda et al., *Parameter identification for symbolic regression using nonlinear least squares* (2020) — Levenberg-Marquardt for SR
- Eureqa (Schmidt & Lipson 2009) — the original const-opt-in-SR system

### 3.3 PySR specifically

- PySR docs: https://astroautomata.com/PySR/
- `SymbolicRegression.jl` source: https://github.com/MilesCranmer/SymbolicRegression.jl

### 3.4 Multi-objective / Pareto-front methods

- Deb, *Multi-Objective Optimization Using Evolutionary Algorithms* (2001) — NSGA-II origin
- La Cava, Helmuth, Spector, Moore, *A Probabilistic and Multi-Objective Analysis of Lexicase Selection and ε-Lexicase Selection* (2019)

### 3.5 Modern data-driven SR

- Petersen et al., *Deep Symbolic Regression* (2021) — RL-style SR
- Mundhenk et al., *Symbolic regression via deep reinforcement learning enhanced GP seeding* (2021)
- Biggio et al., *Neural Symbolic Regression that Scales* (2021)

## 4. Long-term: tessera as a research workbench

Beyond the SR engine itself, tessera's measure-theoretic vocabulary is useful for:
- **PDE discovery** (already shown in `benchmarks/run_heat_equation_discovery.py`)
- **Multi-asset signal extraction** (cross-section × time as 2D)
- **Aggtrade-microstructure** (time × trade-size-bin as 2D)
- **Hamiltonian / Lagrangian discovery** (energy functions of multi-dimensional state — adds to the SR-on-physics literature alongside AI Feynman)

These are 6-month directions. The roadmap above covers the next 2-4 weeks.

## Recently shipped (pointers, not detail)

Done items previously listed here have been moved to:

| Item | Status | Where |
|---|---|---|
| `jax.grad` constant optimisation (`optimize_constants_jax`) | ✓ DONE | `tessera.search.const_opt`; GPConfig.optimize_constants_method='jax_adam' |
| Trigonometric primitives (`sin`, `cos`) | ✓ DONE | `tessera.expression.tree.UN_OP_FNS`; from `docs/research/benchmark_score_improvement.md` §4.1 |
| Hall of Fame (per-cx best-ever store) | ✓ DONE | `tessera.search.HallOfFame`; see `src/tessera/search/README.md` |
| Algebraic simplifier | ✓ DONE | `tessera.expression.simplify` + `simplify_ac` |
| Const-opt (Nelder-Mead, every K gens) | ✓ DONE | `tessera.search.optimize_constants` |
| `sqrt`, `exp`, `log`, `pow` primitives | ✓ DONE | `BIN_OP_FNS` / `UN_OP_FNS`; see CHANGELOG |
| Threshold indicators (`gt/lt/ge/le/step`) | ✓ DONE | `BIN_OP_FNS` / `UN_OP_FNS` |
| GPU backend Tier 1+2+3 (JAX) | ✓ DONE | `docs/shipped/gpu_backend.md` |
| Branch-and-bound pruning (interval arithmetic) | ✓ DONE | `tessera.search.bounds`; opt-in via `prune_by_lower_bound=True` |
| Axis-semantic type system (scoped minimum) | ✓ DONE | `tessera.expression.axes` |
| 2D Measure on JAX | ✓ DONE | `Measure2D._apply_jax`; see CHANGELOG |

For the Hamiltonian / Ising / QUBO research direction (was section 3 of this file), see `docs/research/hamiltonian_ising_for_sr.md`.

## Changelog
- 2026-05-24: docs reorganised into shipped/planned/research. This file now contains only open items. Done items moved to "Recently shipped" pointers; Hamiltonian/Ising research extracted to `docs/research/hamiltonian_ising_for_sr.md`.
- 2026-05-24: initial document. Synthesis of PySR study + BTC 1h benchmarks + user request for QUBO/Ising direction.
