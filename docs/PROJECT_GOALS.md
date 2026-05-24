# Project goals — tessera

**Established 2026-05-25.** Living document. Updated when priorities shift.

## Goals in priority order

### 1. Primary — multi-modal SR workbench

**Tessera as a general SR library that works well across modalities.**

Tessera should be a *library others can use*, with clean APIs, working
examples, and standard benchmarks demonstrating value across time
series, dynamical systems, PDE discovery, images, multi-asset
panels, and other structured sensor data.

**Success metrics:**
- Each submodule has a README + an executable example
- Standard SR benchmarks (Feynman dataset, SRBench-style problems)
  ship with the library and pass
- Installation is a one-liner: `pip install git+...` from GitHub
  for now. PyPI publishing deferred until a clear release milestone;
  `pytessera` (the available distribution name) is reserved when we
  ship — import name stays `tessera`, same pattern as scikit-learn.
- A new user can clone the repo, run a benchmark, and understand
  what tessera does within 30 minutes

**Non-goals at this level:**
- Beating PySR / Operon on every benchmark — being COMPETITIVE +
  having distinct advantages (measure-theoretic ops, axis system)
  is enough
- Mass adoption — polish is for the user who finds tessera
  through research, not for marketing

### 2. Secondary — theoretical contribution (Knuth framework)

**Tessera as the first SR engine to operationalize perfect-info
game + branch-and-bound + equivalence-class collapse + axis-aware
search.**

Build out the theoretical claims with empirical validation. Where
the workbench (goal 1) provides general utility, the theoretical
contribution (goal 2) provides academic and intellectual depth.

**Success metrics:**
- Formal write-up of the framework as a publishable paper
- Empirical validation of the |E_K| / |T_K| conjecture at larger
  grammar / complexity scale
- Equality saturation experiment showing the bound on what
  algebraic-equivalence simplification can achieve
- The four research notes (`search_as_energy_min`,
  `fit_as_perfect_info_game`, `measure_theory_and_perfect_info`,
  `invariance_in_sr`) consolidated into a coherent argument

**Subordinate to goal 1:** theory work supports the workbench but
doesn't block it. If we have to choose between polishing the
library and writing a paper, polish first.

### 3. Tertiary — invariance + interpretability for CV / sensor data

**Tessera's unique pitch realised at the demonstration level:
explainable formulas that preserve invariance for structured sensor
data, including computer vision.**

This is the most ambitious and most distinct claim. Workable on CPU
for small problems (MNIST 0-vs-rest = 0.82 acc on 200 samples in 14s).
Needs the GPU backend + axis enforcement for full-dataset / CIFAR-class
problems.

**Concrete target (set 2026-05-25):** tessera reaches **≥ 95% accuracy
on the full MNIST dataset (10-class, all 60k training samples), with
the JAX backend on Colab.** This is the visible payoff that bundles
all three goals together — it demonstrates the workbench (Goal 1)
scales, the framework (Goal 2) supports CV-grade problems, and the
invariance machinery (Goal 3) produces an interpretable model that
beats vanilla SR while approaching CNN performance.

**Success metrics for this target:**
- 95% test accuracy on full MNIST 10-class, held-out test split
- Discovered model is interpretable (a tree the user can read)
- Runs on Colab GPU end-to-end via git install + `jax[cuda12_pip]`
- Wall-clock < 10 minutes per training run

**Sub-targets along the way:**
- GPU backend Tier 1+ shipped (per `docs/shipped/gpu_backend.md`)
- Axis enforcement in `random_tree` / `mutate` so the search
  respects invariance declarations
- The axes submodule fuses data representation with SR seamlessly —
  `TypedVar` is the bridge: declare your data's invariance once,
  and the search respects it automatically
- Multi-modal sensor benchmark beyond images (e.g., audio,
  multi-channel time series) using the same axes machinery

**Vision for axes (set 2026-05-25):** the axes submodule should
allow data representation fused with SR seamlessly. A user
shouldn't have to choose between "flatten my data for SR" and
"keep its structure for invariance." The axes type system is the
bridge that makes a single `TypedVar` work for both — declare the
invariance once, and the rest of the pipeline (search, evaluation,
GPU dispatch) respects it.

**Subordinate to goals 1 and 2:** this is the visible payoff; the
workbench and theory are the foundations that make it work.

### 4. Explicit non-goal — applied tool for any one user

**Tessera is not a private research tool.** Features should be added
because they fit the workbench / theory / CV agenda, not because
"the user needs this for one specific thing." If a feature is
narrowly applied (e.g., a trading-specific loss for a closed
problem), it lives in `benchmarks/` not in the library.

## How this ordering changes day-to-day priorities

When picking the next task, ask: does this serve goal 1, 2, or 3?
If multiple, prefer the lower-numbered goal. If none, defer.

**Goal-1 tasks** (workbench polish):
- Per-submodule READMEs that aren't placeholders
- Working `from tessera import ...` examples in the top README
- Feynman dataset benchmark runner
- Documentation of the canonical use cases
- PyPI publishing DEFERRED. `pytessera` is reserved as the future
  distribution name (import name stays `tessera`). Not blocking
  Goal 1 — git install works fine for the workbench claim.

**Goal-2 tasks** (theory):
- Equivalence-class enumeration at larger scale
- Equality saturation prototype (egg / snake-egg)
- |E_K| / |T_K| measurement on the FULL grammar (not just restricted)
- Paper draft

**Goal-3 tasks** (CV invariance):
- GPU backend Tier 1
- Axis enforcement in `random_tree`
- `wrap_in_reduce` mutation (addresses MNIST aggregation gap)
- CIFAR-10 SR benchmark

## What this ordering means for OPEN research notes

The four research notes documented future directions. Mapping them
to this priority:

| Note | Serves goal | Status under this ordering |
|---|---|---|
| `search_as_energy_min` | 2 | active research target |
| `fit_as_perfect_info_game` | 2 | foundational doc, write-up needed |
| `measure_theory_and_perfect_info` | 2 | core theoretical claim, paper-ready |
| `gpu_and_cv_via_sr` | 3 | deferred until goals 1+2 mature |
| `invariance_in_sr` | 3 (also touches 1) | axes module just shipped |

## Process commitments

- **Every commit should reference its goal**, e.g., `[g1]`, `[g2]`,
  `[g3]` in the commit message subject. Untagged commits should be
  rare and explainable.
- **Every research note opens with which goal it serves.**
- **`PROJECT_GOALS.md` gets updated when the ordering changes.**
  Currently 1 > 2 > 3. If that flips (e.g., GPU work becomes
  urgent), the doc gets revised.

## First concrete step under this ordering

Goal 1 (workbench) starts with **knowing what we have**. The
synthesis doc maps shipped pieces to roles; the polishing pass
needs a complementary inventory of what's *user-facing* (READMEs,
examples, install instructions, ergonomics).

Proposed next task: audit + polish the user-facing surface of
tessera. Specifically:
- Top-level README: install + 5-line example for each submodule
- Per-submodule README: usage + worked example + tests reference
- A single canonical "what is tessera" doc
- Identify any half-finished modules and mark them clearly

This is ~1-2 hours of focused work and immediately serves Goal 1.

## Changelog
- 2026-05-25: initial ordering (1: workbench, 2: theory, 3: CV).
  User confirmed `1 first, 2 second, 3 third. Perfect order in our
  goal ordering.`
