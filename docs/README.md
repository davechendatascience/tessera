# Tessera documentation index

This directory is organised by **status**, not by topic. Three subdirs:

| Subdir | What lives here |
|---|---|
| [`shipped/`](shipped/) | Designs for features that ARE in the library now. Read these to understand how a shipped piece works. |
| [`planned/`](planned/) | Committed-to-build items that are NOT in the library yet. Each item has an explicit status flag (open / in progress / blocked). |
| [`research/`](research/) | Open-ended exploration. Not committed to ship. Foundational ideas, conjectures, scoping documents. |

Top-level files (`README.md`, `PROJECT_GOALS.md`, `process.md`) span all three.

**Adding to the docs?** Read [`process.md`](process.md) first. It defines the lifecycle that moves an idea from `research/` → `planned/` → `shipped/`. Each transition is a small mechanical edit, not a rewrite.

## Status legend

Inside any individual doc, items are flagged with:

- **✓ DONE** — implemented, tested, released
- **▷ IN PROGRESS** — actively being built; expect a working commit within days
- **○ PLANNED** — committed-to-build; not started; effort estimated
- **? RESEARCH** — exploration; outcome uncertain; may or may not lead to implementation
- **× DEFERRED** — was planned, explicitly postponed (with reason)

## Where to start

| If you want to... | Read |
|---|---|
| Know what tessera IS and what it's for | [`../README.md`](../README.md) (top-level) + [`PROJECT_GOALS.md`](PROJECT_GOALS.md) |
| Understand how the shipped pieces fit together | [`shipped/framework_synthesis.md`](shipped/framework_synthesis.md) |
| Use the GPU backend | [`shipped/gpu_backend.md`](shipped/gpu_backend.md) |
| Use the Koopman module | [`shipped/koopman.md`](shipped/koopman.md) |
| See what's planned next | [`planned/roadmap.md`](planned/roadmap.md) |
| Read the theoretical framework | [`research/fit_as_perfect_info_game.md`](research/fit_as_perfect_info_game.md) |
| Understand the search-as-energy-min direction | [`research/search_as_energy_min.md`](research/search_as_energy_min.md) |
| See the measure-theory + perfect-info bridge | [`research/measure_theory_and_perfect_info.md`](research/measure_theory_and_perfect_info.md) |
| Read about axis-aware SR / invariance | [`research/invariance_in_sr.md`](research/invariance_in_sr.md) |
| See the GPU + CV scoping | [`research/gpu_and_cv_via_sr.md`](research/gpu_and_cv_via_sr.md) |
| Understand the high-dim SR problem + tessera's research position | [`research/high_dim_symbolic_regression.md`](research/high_dim_symbolic_regression.md) |
| Diagnose Feynman failures + plan score-improvement on existing benchmarks | [`research/benchmark_score_improvement.md`](research/benchmark_score_improvement.md) |
| Explore SR for inverse kinematics with simulation as ground truth | [`research/sr_for_inverse_kinematics.md`](research/sr_for_inverse_kinematics.md) |
| Think about network-SR architecture + deterministic budget allocation | [`research/network_sr_and_budget_allocation.md`](research/network_sr_and_budget_allocation.md) |
| Engage with Bowery's MDL/AIT critique of MSE-based SR selection | [`research/algorithmic_information_for_sr.md`](research/algorithmic_information_for_sr.md) |
| Estimate per-benchmark difficulty + the climb-then-simplify path problem | [`research/benchmark_difficulty_and_climb_then_simplify.md`](research/benchmark_difficulty_and_climb_then_simplify.md) |
| Compute analytical Δloss for symbolic mutations (Knuth + FMM + boosting connections) | [`research/analytical_delta_loss.md`](research/analytical_delta_loss.md) |
| Read the rationale for the Dancing Links side track + implementation-budget tracking | [`research/dancing_links_for_sr.md`](research/dancing_links_for_sr.md) |
| Map applied-math sample-complexity theorems (Boullé-Townsend, SINDy, RandNLA) to tessera sub-problems | [`research/randomized_recovery_bounds_for_sr.md`](research/randomized_recovery_bounds_for_sr.md) |
| **READ FIRST** — closing synthesis: three-layer framework, four levers, class taxonomy, convergence point, composite-dynamics frontier | [`research/from_data_to_mechanism.md`](research/from_data_to_mechanism.md) |

## Organisational principles

1. **Status before topic.** A doc lives in `shipped/`, `planned/`, or `research/` first, and only then is organised by topic within that bucket. This forces every doc to declare its maturity.
2. **One doc, one status.** A doc that's "half done, half open" is a code smell — split it into a shipped portion (what works today) and a research/planned portion (what's still open).
3. **No mixed `roadmap.md`.** Done items live in `shipped/` (or the changelog). The roadmap is *only* open items.
4. **Cross-references use full paths.** Internal links say `docs/shipped/foo.md`, not `foo.md`, so moving a file later doesn't silently break references.
