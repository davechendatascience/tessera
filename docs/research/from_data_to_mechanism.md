# Research note: from data to mechanism — convergence point + composite-dynamics frontier

**Status:** ? RESEARCH (synthesis). Captures the trajectory of work across 2026-05-23 → 2026-05-26 culminating in the canonical heat-equation discovery and the articulated frontier beyond unit-dynamics SR. Future readers should treat this as the *reading-order anchor* for the multiple research notes shipped during that period.

**Provenance:** user (2026-05-26): *"so I'm wondering if our fixes perform well on all Feynman benchmarks? Is tuning on benchmark beneficial for all of data fitting?"* The Feynman A/B check passed; user then asked for a closing synthesis: *"do a closing research note that synthesizes the full session's findings."*

This note is the synthesis. It ties together: the three-layer framework (data / equation / mechanism), the four levers of unit-dynamics SR, the natural-overfit reframe, the methodological discipline that emerged for legitimate tuning, the convergence point reached, and the composite-dynamics frontier that's the natural next direction.

---

## 1. The trajectory in one sentence

We started exploring search/scoring optimizations for tessera and ended having articulated *what tessera fundamentally is* — a discoverer of unit dynamics where mechanism (not just data fit) can be operationally measured, with a methodological discipline for distinguishing legitimate tuning from benchmark-overfit.

## 2. The three-layer framework

The most important conceptual clarification from the session. SR work commonly conflates two layers; the user's reframing surfaced a third.

| Layer | What it is | What perfectly-fit means |
|---|---|---|
| 1. Data | Observed (x_i, y_i) pairs | Numerical record of process samples |
| 2. Equation | Symbolic f̂ such that f̂(x_i) ≈ y_i | A token sequence that reproduces data |
| 3. Mechanism | The causal process that generated the data | The actual physics / causality |

The "perfect information data fitting game" framework as previously articulated (in `fit_as_perfect_info_game.md`) was about the layer 1→2 problem. With unlimited compute and finite vocabulary V, the GP CAN enumerate all V-expressible equations and pick the best fit. That game has clean structure.

But **layer 2 → 3 is the gap that matters scientifically**, and it isn't covered by the perfect-info game. An equation doesn't carry its derivation with it. `diff_t(U)` and `α · Laplacian(U)` can be indistinguishable as data-fitters while being mechanistically very different. The equation alone doesn't tell you which is mechanism and which is tautology.

The Kepler / Newton analogy made this concrete:
- Kepler: 3 phenomenological equations, mathematically perfect on data
- Newton: 1 mechanistic law, derives Kepler's 3 as special cases, ALSO predicts tides, falling apples, comets

From the equations alone, Kepler-fit and Newton-fit are equally consistent with planetary data. The distinguisher is out-of-distribution predictive reach — exactly the test the user named.

## 3. The natural-overfit reframe

The session's pivotal conceptual move. The user (2026-05-26):

> *"I think the overfit natural sense is, we can't explain something, so we make tweaks in our argument to make the data fit for the sake of fitting. The argument itself may not make sense if the information is truely observable."*

This is a much sharper notion than statistical overfit (TRAIN low / TEST high). It's *epistemic overfit*:

| Type | Symptoms | Mechanism status |
|---|---|---|
| Statistical overfit | TRAIN < TEST | Could be either |
| Natural-sense overfit | Fits data via tweaks that don't correspond to mechanism | Provably NOT mechanism |

Historical instances of natural-sense overfit:
- Epicycles in Ptolemaic astronomy: each addition improved the fit; heliocentric truth was simpler
- Aether properties: added to explain why aether couldn't be detected
- Phlogiston refinements: decades of tweaks until oxygen replaced the whole framework

In each case, the fits got BETTER over time while the framework was wrong. Adding more parameters to a wrong model is natural-sense overfit even when there's no TRAIN/TEST split to flag it.

**For SR specifically**, the heat-equation result demonstrated this concretely:

```
(M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U) / reduce_max(U))
```

The Laplacian template / reduce_max wrapper. On TRAIN: train_loss/oracle ≈ 1.0. On TEST: test_loss/oracle ≈ 30. The reduce_max divisor is a TRAIN-specific scalar; it has no mechanistic meaning. The tree is a natural-sense overfit — argument tweaks (the divisor) to make data fit, while making the mechanism non-portable.

This reframe is the key insight that changed our diagnostic and our intervention strategy.

## 4. The four levers of unit-dynamics SR

Empirically validated across the heat equation thread. Each lever moves the needle; none alone is sufficient; combined they enable (but don't guarantee) reliable mechanism discovery.

### 4.1 Vocabulary completeness

The primitive set determines what's representable. Tessera's earlier benchmarks (IK Run 1) showed: missing `atan2` makes IK structurally unsolvable. Adding `atan2/acos/asin` raised the ceiling. Each vocabulary expansion is benchmark-class-specific but justifiable independently (these are real mathematical operators).

For heat equation: `measure_2d_laplacian_5pt` is a factory primitive that exists but isn't auto-sampled by `random_tree`'s atom generator. The vocabulary has the template but the search reaches it via the more general 3-atom Measure2D path. Both modes work; the factory is shortcuts, the atom generation is honest.

### 4.2 Scoring (parsimony, MDL, simplification)

The trade between fit quality and representation length. The polynomial simplifier we shipped (`tessera.expression.simplify.polynomial`) closed about half the cx gap on §2.3 polish output by canonicalizing additive polynomial forms.

But scoring alone has limits. The user's observation:

> *"the problem with parsimony is we trade too much of the loss with simplicity, if it's 90% information encapsulation and we compressed the data by 10 fold. This is good. But we flip the script, we start out with the minimal ops trying to decrease the loss this way."*

The Pareto front based on `loss + parsimony · cx` doesn't track the true MDL trade. A more principled scoring would be description-length-based: model_bits + residual_bits. But even MDL doesn't kill natural-overfit tautologies — they're MDL-efficient too. Scoring is downstream of search; you can't score what doesn't exist.

### 4.3 Search bias (mutation operator weights)

The biggest single ROI move of the session: lowering the sampling weight of `reduce_*` operators in `random_tree` by 10×. Five lines of code. Justification independent of any benchmark — reductions collapse arrays to trajectory-specific scalars; per-sample regression cannot use those.

Empirical result: the GP's natural-overfit shape (`template / reduce_max(...)`) went from probability ~17% to ~17% (didn't drop dramatically), BUT the canonical mechanism form (`Const · template`) went from 0% to 8% (single-trajectory) and to 33% under multi-trajectory training.

The mechanism CAN be found if the search isn't being pulled toward overfit-shaped tweaks.

### 4.4 Training data structure (multi-trajectory)

Single-trajectory training admits trajectory-specific tricks. Multi-trajectory training (K=3 different ICs, same α) punishes them — `reduce_max(traj_1) ≠ reduce_max(traj_2)`, so any candidate using such a scalar can't fit multiple trajectories at once.

Result on heat equation:

```
multi-traj seed 2026, pop=240 gens=100:
  (M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U) * 0.049883)
  TRAIN/oracle = 1.00, TEST/oracle = 1.00, cx = 4
```

The canonical textbook form. α extracted to 0.2% accuracy via the multiplicative coefficient. Smallest possible cx for "constant times Laplacian." This is the cleanest possible Class C result — and it required all four levers stacked.

## 5. The class taxonomy

A useful diagnostic that emerged from the paired-diagnostic experiment. For a unit-dynamics target like heat equation, GP outputs fall into three structurally distinct classes:

### Class A — Generic diff-style tautology

Examples: `M2D[1·(0,0) + -1·(1,0)](U)` (= `U[t,x] − U[t+1,x]`).
- 2-atom Measure2D, no reductions
- Train ≈ Test ≈ 2× oracle (stable)
- Generalizes (structurally generic)
- **The "diff_t-style" fit — generic backwards-difference**

This is the safe default; it doesn't capture mechanism but it doesn't catastrophically fail either.

### Class B — Template with trajectory-specific wrapping

Examples: `(Laplacian(U) / reduce_max(...))`.
- Contains the correct mechanism template
- Wrapped in reduce_* operators that compute trajectory-specific scalars
- Train ≈ 1× oracle (great), Test ≈ 4-32× oracle (catastrophic)
- **The natural-overfit signature — argument tweaks around real mechanism**

The most pernicious failure mode. Looks great on TRAIN; selection-by-TRAIN picks it; it fails OOS.

### Class C — Clean canonical mechanism

Examples: `(Laplacian(U) * 0.049883)` or `(Laplacian(U) / 20.0639)`.
- Mechanism template + clean coefficient
- Train ≈ Test ≈ oracle
- The textbook form a physicist would write
- **Mechanism captured**

The goal state. Generalizes by construction.

### How the class taxonomy reframes the discovery question

The question isn't "did we find a low-loss tree" but "what CLASS did we find?" A Class B tree with train_loss/oracle = 1.04 is *worse* than a Class A tree with train_loss/oracle = 2.0 — because Class A generalizes and Class B doesn't.

This is why TRAIN/TEST split is the load-bearing diagnostic. Without it, Class A and Class B look indistinguishable.

## 6. The methodological discipline for legitimate tuning

The Feynman A/B check made this explicit. When is benchmark tuning beneficial for general data fitting (vs benchmark-overfit)? Three tests:

### 6.1 Independent justification

The argument for the tuning should not reference the benchmark. The reduce_* downweight argument is:

> *"Reductions collapse arrays to trajectory-specific scalars. Per-sample regression cannot use trajectory-specific scalars. Therefore reductions shouldn't be in the default sampling distribution for per-sample SR."*

This argument doesn't reference heat equation. It follows from per-sample regression semantics. **The tuning expresses a general principle.**

By contrast, "add Laplacian_5pt as a one-pick factory primitive" would reference heat equation directly. That move was explicitly rejected during the session.

### 6.2 Held-out empirical test

The tuning should not harm benchmarks it wasn't designed for. Feynman A/B confirmed: the reduce_* downweight has at most modest effects across 8 Feynman equations, with 1 clear win (Stokes) and 0 catastrophic losses. The principle generalized.

If we'd tuned by adding factory Laplacian, the Feynman A/B would have shown Laplacian-irrelevant equations (polynomial, transcendental) didn't benefit — exposing the tuning as benchmark-specific.

### 6.3 Mechanism preservation across contexts

The tuning should change PRIOR (sampling probability), not AVAILABILITY (existence). `UN_OP_WEIGHTS["reduce_max"] = 1.0` restores uniform sampling for benchmarks that genuinely need reductions (trading indicators, trajectory summaries). The capability is preserved; only the default prior changed.

**Reversible defaults are safer than removed capabilities.**

### 6.4 What FAILS these tests (and why we avoided them)

| Pattern | Example | Test it fails |
|---|---|---|
| Factory primitives matching the target | "Add Laplacian_5pt one-pick" for heat eq | 6.1 (smuggles knowledge) |
| Tuning a knob to maximize held-out metric | "Set parsimony=0.003 because it maxes Feynman" | 6.2 (Goodhart's Law) |
| Disabling mutation operators entirely | "Remove all reduce_*" | 6.3 (forecloses future) |
| Re-running until results look good | "Got 0/3 Class C, try seed 9999" | 6.1 (selection bias, not principle) |

## 7. The convergence point reached

We've established:

1. **Unit dynamics ARE discoverable by tessera under the right conditions.** The heat equation Class C result demonstrates this on held-out data with canonical-form output (`α · Laplacian(U)`).

2. **Reliability is ~33% even with all four levers tuned.** The four levers (vocabulary, scoring, search-bias, training-structure) each move the needle; combined they make discovery POSSIBLE but still high-variance.

3. **The diagnostic discipline (TRAIN/TEST split + class taxonomy) distinguishes mechanism from natural-overfit.** Without it, Class A and Class B look the same.

4. **The methodological discipline (three tuning tests) distinguishes legitimate from benchmark-overfit tuning.** The Feynman A/B verified our tuning passes.

5. **The information-vs-optimizer ceiling distinction** clarifies what improvements are buying us. The recent ships (polynomial simplifier, sufficient-stats polish, reduce_* downweight) all improved optimizer-ceiling, not information-ceiling. Information ceiling improvements require more data sources (interventional, multi-modal, multi-trajectory) — which the session also explored.

This is the natural endpoint for unit-dynamics SR. Further improvements would be incremental (stronger reduce_* downweight, more sophisticated grammars, better initialization) — diminishing returns within the same architectural envelope.

## 8. The composite-dynamics frontier

The natural next direction, identified by the user:

> *"How do we do dynamics composition with unit dynamics? Like how do we mix heat equation with naiver stokes? Is it structurally meaningful?"*

Yes, it's structurally meaningful — most real-world physical systems are coupled multi-mechanism. But it's a *different architecture* from tessera-as-current:

### 8.1 What composite dynamics requires

| Requirement | Unit dynamics SR (current) | Composite dynamics SR (frontier) |
|---|---|---|
| Output structure | Single expression tree | Coupled equation system (multi-output) |
| Operator combination | Random tree composition | Operator algebra (linear superposition, commutators, composition) |
| Physics constraints | None enforced | Conservation laws (energy, momentum, mass) as priors |
| Coupling discovery | Implicit (just another tree node) | First-class object (the coupling IS the discovery) |
| Search space | All V-expressible trees | All V-expressible *systems* — dramatically larger |

### 8.2 Examples of composite systems

- **Convection-diffusion**: `∂u/∂t = α∇²u − v·∇u` (heat + advection)
- **Reaction-diffusion / Turing patterns**: `∂u/∂t = D∇²u + R(u, v)` (diffusion + nonlinear reaction)
- **Boussinesq thermal convection**: T-field couples to flow-field via buoyancy term
- **Maxwell's equations**: 4 coupled PDEs (E, B fields) with curl coupling
- **Lotka-Volterra predator-prey**: coupled ODEs

In each case, multiple unit mechanisms interact via specific COUPLING STRUCTURES that are themselves the scientifically interesting objects.

### 8.3 Why this is a new project, not an extension

Tessera's architecture is built around single-tree outputs and free composition of primitives. The shifts required:

- **Multi-output GP**: each candidate produces multiple expressions
- **Operator algebra primitives**: linear combination, commutator, composition as first-class operations
- **Constraint solvers**: conservation laws as hard constraints in the search
- **Cross-equation cohesion**: coupling terms must be consistent across the system

These aren't config flags or new operators; they're architectural changes. The closest published prior: SINDy-PDE for sparse multi-equation discovery (handles linear-in-parameters case); neural operators with multi-physics layers (different methodology entirely). Neither is mature.

### 8.4 The honest scoping

Tessera-current is a *unit dynamics discoverer*. It's done; the heat equation Class C demonstrates it; the methodological framework documents it. The composite-dynamics frontier is a *new project* with a different architectural starting point.

This is a natural scoping decision, not a failure. Most published SR systems handle unit dynamics and have nothing to say about composite. Tessera being good at unit dynamics with a principled methodology is a complete product.

## 9. Lifecycle artifacts produced during the session

For reading order, the canonical sequence:

| # | Artifact | What to read it for |
|---|---|---|
| 1 | `docs/research/analytical_delta_loss.md` | The "calculus of loss impact" research note that initiated the §2.3 chain |
| 2 | `docs/research/dancing_links_for_sr.md` | The DLX side-track research note; pattern-fluency for future SR ideas |
| 3 | `docs/research/randomized_recovery_bounds_for_sr.md` | The applied-math sample-complexity catalog (Boullé-Townsend etc.) |
| 4 | `benchmarks/results/heat_equation_sample_complexity.md` | The first calibration of theory vs empirical |
| 5 | `benchmarks/results/heat_equation_mode1_parsimony_zero.md` | The clean Mode-1-falsification experiment |
| 6 | `benchmarks/results/heat_equation_traintest_computescale.md` | The paired diagnostic — found Class B, narrowed problem to search bias |
| 7 | **CHANGELOG entry for `[g2] random_tree: downweight reduce_* ops 10x`** | The 5-LOC fix that delivered first Class C |
| 8 | `benchmarks/results/heat_equation_multitrajectory.md` | The closing empirical anchor — canonical mechanism at cx=4 |
| 9 | `benchmarks/results/feynman_reduce_downweight_ab.md` | The generalization verification — fix doesn't harm Feynman |
| 10 | **This note** | The synthesis |

For someone joining the project, the reading order is: §1-3 establish the framework, §4-6 set up the empirical work, §7-9 deliver and verify, §10 (this note) ties it together.

## 10. What this note explicitly does NOT claim

To prevent over-interpretation:

- **NOT that tessera is reliable at unit-dynamics SR.** It's ~33% at best with all four levers. Reliability requires further work.
- **NOT that the methodological discipline solves benchmark-overfit in general.** The three tests are necessary, not sufficient. Other failure modes exist.
- **NOT that composite-dynamics SR is impossible.** It's a different architectural problem, and there's active research in adjacent areas (SINDy-PDE, neural operators). We just didn't solve it during this session.
- **NOT that this synthesis is complete.** Future work will identify gaps in the framework. This is a checkpoint, not an end state.

## 11. Falsification anchors

If future work shows:

- The reduce_* downweight harms a non-trivial fraction of benchmarks we haven't tested → the "general principle" claim weakens
- The class taxonomy doesn't generalize to non-PDE benchmarks → the diagnostic framework needs broadening
- Class C reliability stays at ~33% across all interventions → the unit-dynamics ceiling claim is validated
- Class C reliability rises to >80% with a single new intervention → the four-levers framing was incomplete

The synthesis is a working framework, not a closed theory. Each claim has a falsification path.

## 12. Closing reflection

The session traced the path from *"how do we speed up the search?"* (the analytical Δloss question that opened §2.3) to *"what does it MEAN to find the right answer?"* (the natural-overfit reframe that closed the heat equation thread).

The arc — search efficiency → mechanism discovery → generalization tests → composite dynamics frontier — isn't an arbitrary sequence. Each step exposed a deeper question than the previous step had addressed. Each answer narrowed the next problem.

The end-state isn't "tessera is fixed." It's *"tessera is a unit-dynamics SR discoverer with a methodological framework for what it can and cannot do, and the natural frontier beyond it is clearly named."* That's a complete artifact, not a perpetual work-in-progress.

The forward direction (composite dynamics) is a real research project. The work that's done can stand on its own.

## Changelog

- 2026-05-26: initial synthesis. Captures the four levers of unit-dynamics SR, the natural-overfit reframe, the class taxonomy (A/B/C), the methodological discipline for legitimate tuning, the convergence point reached, and the composite-dynamics frontier. Provenance: user request to *"do a closing research note that synthesizes the full session's findings."*
