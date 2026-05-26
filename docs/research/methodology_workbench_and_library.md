# Research note: methodology workbench + canonical-system library

**Status:** STAGE 0 design fix. Architecture, canonical-system list, signature framework, and curriculum specified before implementation. This note is the design contract for all subsequent workbench / library / identification work.

**Provenance:** user (2026-05-26), after a long discussion working through (a) SCMs don't solve most SR benchmarks, (b) deep learning overparameterizes, (c) SR's loss surface is unstable with new structures, (d) the field is methodologically under-developed (PySR paper observation), (e) Kepler didn't discover laws from data alone — he had geometric priors, the right reference frame, and discovered invariances first:

> "What we have to do is, do the engineering that actually evolves into a library that can identify which system it is given what we generate from the workbench. ... We should only choose the systems that provide enough information first, then we slowly add in systems that provide partial information and we use common sense to fill in the gap. We build this common sense into tessera by first learning to fit all the different systems accurately."

This note fixes the design before code so that the implementation stages don't drift.

---

## 1. The thesis in one paragraph

SR-as-rediscovery is exhausted: Feynman benchmarks fit Newton from Newton-generated data, IK rediscovers Spong & Vidyasagar 1989, the 1h BTC closure says nothing more remains. The field's actual deficit is **methodology**, not search algorithms. What tessera should become is a *system-identification library* with a workbench-evaluated methodology stack: given data, classify the system into a canonical class, then apply class-aware SR with library priors. Common sense — i.e., the discriminative prior that lets scientists pattern-match new data against known templates — is built up by first fitting every canonical system in the workbench until tessera has a library of anchors. New data gets matched against the library and identified before search. This synthesizes SINDy's fixed-library projection, AI-Feynman's pipeline, DreamCoder's library learning, and classical system-identification theory into one integrated framework. Nobody has done this for SR; the integration is the contribution.

## 2. Three layers and the boundary between them

| Layer | What it does | Status in tessera today |
|---|---|---|
| **Workbench** | Controlled simulators with declared metadata (dimensions, symmetries, mode count, smoothness class, info-richness knobs). Ground truth is known | Partially exists: heat eq generator, FHN/Lorenz scripts; no unified API, no metadata schema |
| **Library** | Canonical anchors — per-system best-fit expressions with their signatures, parameter ranges, sample-complexity thresholds | Does not exist |
| **Identification pipeline** | Data in → signature extraction → nearest-anchor match → class-aware SR with anchor as prior → fitted expression + residual diagnostic | Does not exist |

The honest deliverable is the right-hand column moving from "does not exist" to "shipped and evaluated." The benchmarks tessera has run so far become *methodology-evaluation cases* under this new framing.

## 3. Why this isn't another over-claimed "AI scientist"

Honest scope, set deliberately:

- **Will deliver:** identification of which canonical-class an unknown dataset belongs to (high confidence within the library); parameter recovery within the identified class; diagnosis of partial-information shortfalls (e.g., "you need 3× more samples or 2 more state observables to disambiguate Lorenz from Rossler"); mode/regime decomposition for multi-modal data; rigorous methodology evaluation on a controlled workbench.
- **Won't deliver:** discovery of novel physics outside the library; replacement of scientist judgment in choosing what to measure; success on systems with infinite-dimensional dynamics, deep non-stationarity, or anthropic feedback.

"Automated scientific discovery" is the over-claim trap. **System identification + library extension** is the honest scope. This is what most "discovery" in real science actually is — recognizing a new dataset as a variant of a known system, then refining the parameters.

## 4. The canonical-system list (Phase 1)

Information-rich, well-understood systems where ground truth is known and the SR target is tractable. The list is deliberately small and curated; expansion happens after the framework is proven.

| ID | System | Dynamics | Why include it |
|---|---|---|---|
| `harmonic_1d` | 1D simple harmonic oscillator | $\ddot x = -\omega^2 x$ | Trivial baseline; tests dim analysis + conservation (energy) |
| `damped_harmonic_1d` | Damped harmonic oscillator | $\ddot x = -\omega^2 x - \gamma \dot x$ | Adds dissipation; tests model selection between conservative vs dissipative classes |
| `vdp` | Van der Pol oscillator | $\ddot x = \mu(1-x^2)\dot x - x$ | Limit cycle; nonlinear damping; tests mode detection (limit-cycle vs fixed-point) |
| `lorenz63` | Lorenz-63 | $\dot x=\sigma(y-x),\ \dot y = x(\rho-z)-y,\ \dot z = xy-\beta z$ | Chaotic; tests multi-equation SR with shared state; already a benchmark |
| `fhn` | FitzHugh-Nagumo | Excitable membrane | Two timescales; tests multiscale detection |
| `heat_1d` | 1D heat equation | $\partial_t u = \alpha \partial_x^2 u$ | PDE; already canonical in our portfolio; multi-trajectory anchor |
| `burgers_1d` | 1D Burgers' equation | $\partial_t u + u\partial_x u = \nu\partial_x^2 u$ | Nonlinear PDE with shock formation; tests advection + diffusion |
| `linear_pendulum` | Small-angle pendulum | $\ddot\theta = -(g/L)\theta$ | Dimensional analysis discriminator from `harmonic_1d` |
| `nonlinear_pendulum` | Full pendulum | $\ddot\theta = -(g/L)\sin\theta$ | Tests trig vocabulary; conservation of energy with non-quadratic potential |
| `kepler` | Kepler problem | Inverse-square central force | Tests rotational symmetry, conservation (angular momentum, energy) |

Ten systems. Each has well-documented analytical structure; each tests a different methodology component (smoothness, modes, conservation, multi-equation coupling, PDE-vs-ODE, trig vocabulary, multiscale). Crucially **they span the kinds of distinctions we want the identification pipeline to make**, not just the union of "famous systems."

## 5. The workbench API contract

```python
# src/tessera/workbench/systems.py

@dataclass
class CanonicalSystem:
    id: str                          # "lorenz63", etc.
    domain: Literal["ode", "pde", "sde", "algebraic"]
    dynamics_doc: str                # plain-English description
    state_dim: int                   # latent state dimensionality
    observable_dim: int              # what we get to measure
    parameters: dict[str, tuple[float, float]]  # name -> (min, max)
    symmetries: list[str]            # e.g. ["time_translation", "rotation_so2"]
    conservation_laws: list[str]     # e.g. ["energy", "angular_momentum"]
    smoothness_class: Literal["analytic", "c_infty", "c_2", "lipschitz", "general"]
    mode_count: int                  # >1 if multi-modal
    info_min: InformationRequirements  # filled in by sample-complexity study

    def generate(self, *, params, ic, t_max, dt, noise=0.0, seed=...) -> Trajectory: ...
    def signature(self, trajectory) -> Signature: ...  # callable from data alone
```

Every entry in the library has all these fields. Signature extraction is callable both from the system (ground truth) and from data alone (estimated) — the gap between the two is one of the things we measure.

## 6. Signature extraction — the discriminative layer

These are the *data-derivable* properties that identify which canonical system a trajectory came from. Computed without knowing the ground truth.

| Signature | What it measures | Method |
|---|---|---|
| Smoothness exponent | Hölder regularity of trajectory | Wavelet leaders / structure-function scaling |
| Mode count | Number of attractors / regimes | GMM on phase-space points; changepoint statistics on residuals |
| Effective dimensionality | Manifold dimension of state | Correlation dimension; Takens embedding + false-nearest-neighbors |
| Symmetry tests | Invariance under candidate group actions | Compute trajectory under candidate transformation; test for orbit closure |
| Conservation tests | Existence of conserved scalar | Search low-order polynomial $Q(x)$ minimizing $\text{Var}(Q)$ along trajectory; report best $Q$ + residual |
| Spectral content | Power spectrum / dominant frequencies | Welch periodogram; spike sharpness |
| Determinism vs stochasticity | Predictability of next-step from past | k-NN forecasting residual vs surrogate-data null |
| Lyapunov estimate | Chaos vs regularity | Rosenstein / Kantz algorithm |

These signatures are the **feature vector** for system identification. Distance in signature space is the matching metric. The library anchors store reference signatures; new data's signature is compared.

This is also where the **multi-objective** lever lives: the methodology stack scores candidates on (data fit, parsimony, smoothness, conservation satisfaction, symmetry equivariance, mode coherence). Each comes from a signature with a deviation measure. Pareto-front across these is the multi-objective Pareto the user identified as the right framing.

## 7. The information-sufficiency framework

Identifiability theory says you cannot identify some systems from *any* finite data, regardless of how good the algorithm is. The library is built around this, not in spite of it.

Each canonical system gets an **information requirements** record:

```python
@dataclass
class InformationRequirements:
    min_samples: int                 # empirically calibrated
    min_trajectories: int            # for multi-traj-required systems
    observable_subset: list[str]     # which state vars are sufficient
    noise_max: float                 # max noise level at which ID succeeds
    excitation_requirements: list[str]  # e.g., "non-equilibrium IC required"
    identifiability_proof: str | None # citation if formal
```

These are calibrated by **sample-complexity studies on the workbench itself** — exactly the experimental form we already have for heat equation. The deliverable is a calibrated table: for each canonical system, what's the minimum data to identify it under the library framework?

**Curriculum implication.** Phase A of the project includes only systems with `info_min` met by abundant data + full observation. Phase B introduces systems where partial observation requires the library-prior step (i.e., where common sense matters). Phase C introduces noisy or under-determined data where the framework must say "I cannot identify this with confidence."

## 8. The identification pipeline (Phase 2 contract)

```python
def identify(data: Trajectory, library: Library) -> IdentificationResult:
    # Step 1: extract signature from data alone
    sig = compute_signature(data)

    # Step 2: rank library anchors by signature distance
    ranked = library.rank_by_distance(sig)

    # Step 3: check information sufficiency
    sufficient = [a for a in ranked if data_meets_requirements(data, a.info_min)]
    if not sufficient:
        return IdentificationResult(verdict="insufficient_information",
                                    shortfall=describe_gap(data, ranked[0].info_min))

    # Step 4: class-aware SR using top anchors as priors
    candidates = []
    for anchor in sufficient[:K]:
        fit = sr_with_prior(data, anchor.expression, anchor.parameters)
        candidates.append((anchor, fit))

    # Step 5: multi-objective Pareto over (fit, parsimony, smoothness,
    #         conservation, symmetry, mode coherence)
    pareto = multi_objective_select(candidates)

    return IdentificationResult(
        verdict="identified", anchor=pareto.best.anchor,
        fit=pareto.best.fit, residual_diagnostic=...)
```

What this delivers in deployment: a user with experimental data calls `tessera.workbench.identify(data)` and gets back either (a) "this is a Lorenz-type system with σ≈10.1, ρ≈28.2, β≈8/3, residual matches expected noise" or (b) "this is closest to Van der Pol but signature doesn't match cleanly; consider whether you need additional state observables or sample density."

## 9. Multi-objective replaces single-loss

Current GP scoring collapses everything into `loss + parsimony·complexity`. The user is correct that this is the wrong framing. The honest scoring is multi-objective:

| Objective | What it measures |
|---|---|
| Fit | $\|y - \hat f(x)\|$ on data |
| Parsimony | Tree complexity |
| Smoothness | Hölder regularity of $\hat f$ |
| Invariance satisfaction | Equivariance error under declared symmetry group |
| Conservation satisfaction | Variance of declared conserved quantity along trajectory |
| Mode coherence | Per-mode fit quality after mode detection |
| Counterfactual stability (C5) | Held-out-IC ranking |

Each candidate gets a vector score; the Pareto frontier across all objectives is what we report. A natural-overfit candidate may have low fit + low parsimony but fail smoothness, conservation, or counterfactual stability. **Most natural-overfit failure modes violate at least one structural objective.** The multi-objective framing is the right way to catch them mechanically.

This formalizes what we already partly do (parsimony + counterfactual selection from C5) and adds the dimensions the workbench will populate (smoothness, invariance, conservation, modes).

## 10. Why the "common sense via library" intuition is right

The Kepler reference the user invoked is exactly right and points at the mechanism:

| Kepler's tool | Library-learning analogue |
|---|---|
| Closed-curve prior (he assumed orbits were ellipses, parabolae, etc.) | Anchor expressions in library — initialization not from scratch |
| Heliocentric reframe | Coordinate-system search step (axis module) |
| Mars chosen for high eccentricity | Test-case selection: choose data where the signature is most discriminative |
| Equal-areas-law discovered first | Conservation laws as signatures discriminate before form fitting |
| Iterative residual diagnostics | Per-mode fit + residual diagnostic loop |

**Common sense is not pre-programmed wisdom — it's accumulated library entries plus signature-distance-based pattern matching.** A scientist looking at planetary orbit data doesn't search the space of all functions; they pattern-match against known closed curves. Our system should do the same.

This is DreamCoder's central insight applied to SR: solved problems become reusable primitives; solving rate accelerates as the library grows.

## 11. Stages and deliverables

| Stage | Deliverable | Time |
|---|---|---|
| 0 | **This note** — design contract | (now) |
| 1 | `src/tessera/workbench/systems.py` — 10 canonical systems with metadata; generators tested | 2-3 weeks |
| 2 | `src/tessera/workbench/signatures.py` — 8 signature extractors with unit tests | 2-3 weeks |
| 3 | Information-sufficiency calibration — sample-complexity study for all 10 systems | 2-3 weeks |
| 4 | Library construction — fit each canonical with current methodology; store anchors with signatures and info_min | 2-3 weeks |
| 5 | Identification benchmark — held-out trajectories from each canonical, measure identification accuracy + confusion matrix | 1-2 weeks |
| 6 | Multi-objective scoring — replace single-loss with vector scoring; Pareto-front selection | 2-3 weeks |
| 7 | Partial-information curriculum — re-run identification with degraded data (sparse sampling, partial state, noise); produce degradation curves | 2-3 weeks |
| 8 | Methodology paper — workbench-evaluated quantitative ablations of each component | 1 month |
| 9 | First real-domain application — aggtrade SDE identification (the BTC bridge) | TBD after Stages 1-7 |

Total core build: 4-6 months. Stage 8 is the publishable methodology contribution. Stage 9 is the trading-domain bridge that motivated this in the first place.

## 12. Risks and limits

To be explicit about where this might fail:

- **Signature distance might not be discriminative enough.** Two distinct systems can produce nearly-identical signatures on short trajectories. Mitigation: include conservation tests and Lyapunov estimates which are more system-specific than aggregate statistics.
- **Library priors might dominate.** Strong anchor priors could prevent discovery of variants. Mitigation: tunable prior strength + measured ablation against no-prior baseline.
- **The canonical list is small.** Real scientific data may not match any of the 10. Honest answer: the framework reports "no good match" — that itself is useful, and the library is extensible.
- **Multi-objective complicates evaluation.** Pareto frontiers don't give single rankings. Mitigation: report dominated/non-dominated counts and let the user pick the trade-off.

## 13. What this does NOT replace

- Existing benchmark scripts stay — they become *methodology test cases*, not project goals.
- C5 counterfactual selection stays — it becomes one of the multi-objective dimensions.
- The PnL-loss / fee-aware loss work for the trading bridge stays — it's domain-specific application on top of the framework.
- The CAS simplification work stays — it operates at the expression-cleanup layer below the methodology layer.

The framework is additive. It reorganizes how the pieces fit together, but doesn't deprecate any of them.

## 14. Open design questions

1. **Library representation**: store anchors as parsed trees, sympy expressions, or both? (Both seems right; trees for tessera-native search, sympy for human-readable identification output.)
2. **Signature distance metric**: weighted Euclidean? Mahalanobis? Mixture-aware? (Defer — calibrate empirically in Stage 5.)
3. **Anchor count per system**: one canonical form, or multiple equivalent representations? (Multiple — different parameterizations of the same dynamics.)
4. **PDE workbench representation**: PDE as discretized ODE on a grid, or PDE-native primitives? (Both, with explicit conversion. Heat eq already has both forms in our portfolio.)
5. **How to handle the BTC aggtrade case which is SDE, not deterministic?** Stage 9 will need an SDE branch of the workbench. The signature framework extends naturally (drift + diffusion as separate identifiable pieces) but the library anchors are TBD.

These get pinned down in subsequent notes during the relevant stages — not now.

## 15. Closing

The deliverable is the framework, not any single experiment. The honest project framing: **tessera becomes the first SR library that tells you what kind of system you're looking at and identifies it within a curated catalog, with workbench-evaluated methodology contributions at each step.** The trading-domain bridge (aggtrade SDE identification) is the eventual stress test — the framework is the contribution.

This note is the design contract. Subsequent implementation must reference back to this for the canonical-system list, the signature catalog, the curriculum, and the multi-objective scoring contract.
