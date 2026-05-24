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

### 1.3 `jax.grad` constant optimisation ○ PLANNED

`optimize_constants` is currently scipy's Nelder-Mead, called serially per tree. Replacing with `jax.grad + optax.adam` would:

- Speed: ~10-50× per-tree on GPU (Adam vs derivative-free Nelder-Mead)
- Batchability: vmap const-opt across the population, fusing with the Tier-3 batched evaluation

This is the largest remaining GP wall-clock improvement after the Tier-1/2/3 GPU port.

**Effort:** ~half day, contained. Needs to handle the smooth/non-smooth loss distinction: PnL+flip is non-smooth so Adam may not help there; for MSE it's a clear win.

## 2. Reading list

(Reference material, not status-flagged.)

### 2.1 GP / SR foundations

- Koza, *Genetic Programming* (1992) — original GP textbook
- Poli, Langdon, McPhee, *A Field Guide to Genetic Programming* (2008) — free PDF
- Cranmer, *Interpretable Machine Learning for Science with PySR/SymbolicRegression.jl* (2023, arxiv 2305.01582) — PySR design paper

### 2.2 Const optimisation in SR

- Topchy & Punch, *Faster Genetic Programming based on Local Gradient Search* (2001) — original local-search-inside-GP paper
- Kommenda et al., *Parameter identification for symbolic regression using nonlinear least squares* (2020) — Levenberg-Marquardt for SR
- Eureqa (Schmidt & Lipson 2009) — the original const-opt-in-SR system

### 2.3 PySR specifically

- PySR docs: https://astroautomata.com/PySR/
- `SymbolicRegression.jl` source: https://github.com/MilesCranmer/SymbolicRegression.jl

### 2.4 Multi-objective / Pareto-front methods

- Deb, *Multi-Objective Optimization Using Evolutionary Algorithms* (2001) — NSGA-II origin
- La Cava, Helmuth, Spector, Moore, *A Probabilistic and Multi-Objective Analysis of Lexicase Selection and ε-Lexicase Selection* (2019)

### 2.5 Modern data-driven SR

- Petersen et al., *Deep Symbolic Regression* (2021) — RL-style SR
- Mundhenk et al., *Symbolic regression via deep reinforcement learning enhanced GP seeding* (2021)
- Biggio et al., *Neural Symbolic Regression that Scales* (2021)

## 3. Long-term: tessera as a research workbench

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
