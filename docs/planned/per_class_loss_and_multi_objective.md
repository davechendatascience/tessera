# Research note: per-class loss families + multi-objective Pareto

**Status update (2026-05-26 → `docs/planned/`):** Moved from `docs/research/`. Stage 6 multi-objective scoring not yet shipped; `ScoreVector` interface is fixed by §7. Stage 2 signatures (the prerequisite measurement layer) **SHIPPED** in `src/tessera/workbench/signatures/`; they serve both Stage 5 identification and Stage 6 scoring readings per §5 (bidirectional reading).

**Status:** STAGE 2.5 design fix. Formalizes the per-model-class loss families and the resulting multi-objective Pareto structure that the workbench / library / identification pipeline needs.

**Provenance:** user (2026-05-26): *"do we need different loss for different model class types? How would that affect our Pareto front?"* Surfaced while scoping Stage 2 signatures.

**Sees forward to:** Stage 6 (multi-objective scoring). The Stage 0 design note's §9 sketched the multi-objective idea; this note pins it down and ties it to the per-class loss decision so that Stage 2 / Stage 5 / Stage 6 design is coherent.

**See also:** Stage 0 [`methodology_workbench_and_library.md`](./methodology_workbench_and_library.md), Stage 0.5 [`model_class_taxonomy.md`](./model_class_taxonomy.md).

---

## 1. Thesis in one paragraph

Different model classes require structurally different losses, not just MSE-with-different-targets. For ALGEBRAIC, MSE on $(y, \hat y)$ is sufficient. For ODE/PDE, **one-step finite-difference MSE** and **trajectory rollout MSE** measure different things and yield different fits — the temporal-differencing tautology (Class A natural overfit) wins on one-step and fails on rollout, which is precisely why we've been picking it up. Each model class also admits *structural* losses beyond fit accuracy: conservation violation, invariance violation, smoothness penalty, BC respect. These structural losses turn the Pareto front from 2D `(cx, fit)` into a multi-dimensional object whose non-fit axes catch natural-overfit *mechanically* — Class B candidates are dominated on at least one structural axis even when they tie on fit and complexity.

## 2. Loss families per model class

| Model class | Primary loss | Optional rollout | Structural losses (Pareto axes) |
|---|---|---|---|
| ALGEBRAIC | $\mathrm{MSE}(y, \hat y)$ | — (no time) | Smoothness penalty (if declared analytic); NLL with noise model when added |
| DISCRETE_MAP | One-step MSE on $(y_{n+1}, \hat y_{n+1})$ | $k$-step rollout MSE | (Same as ODE downstream) |
| ODE | One-step MSE on $(\dot x, \hat f(x))$ | Rollout error from IC | Conservation violation, invariance violation, smoothness |
| PDE | One-step MSE on $(\partial_t u, \hat f(u, \partial_x u, \ldots))$ | Rollout error from IC | Same as ODE plus BC respect, shock-handling |

Three observations:

- **The finite-difference vs rollout distinction is real**. One-step MSE has truncation-error structure (cf. Stage 0.5 §2); rollout reveals true predictive quality. They produce different fits on the same data.
- **Structural losses are not loss-of-fit**. They're penalties for violating declared system properties. They have units of "constraint deviation," not "data residual."
- **Algebraic doesn't have rollout** — pure functions have no time structure to integrate. This is one place where the model class fundamentally changes the loss menu.

## 3. The one-step vs rollout distinction — the load-bearing case

Heat equation, our most-studied PDE. The Class A natural-overfit tautology is:

$$\text{M2D}[1\cdot(0,0) + -1\cdot(1,0)](U) \approx -\partial_t U$$

i.e., the candidate computes $U(t,x) - U(t+1, x)$ — pure temporal differencing.

- **One-step MSE**: this candidate has *machine-precision-zero* loss. It is the target via algebraic restatement.
- **Rollout from IC**: this candidate cannot integrate forward at all — it requires knowing $U(t+1)$ to predict $\partial_t U(t)$. Rollout MSE is effectively infinite.

So one-step picks Class A; rollout rejects it. We've been using one-step. **We have been measuring the wrong thing.** Class C (the actual heat-equation mechanism $\alpha \cdot \nabla^2 U$) ties on one-step but dominates on rollout because it can actually predict forward.

This is the mechanism by which the existing GP scoring fails to distinguish mechanism from tautology. Multi-objective Pareto fixes it structurally: rollout is a *separate Pareto axis*, not a different choice of single loss.

## 4. Multi-objective Pareto structure

The current `(cx, train_loss)` Pareto becomes per-class:

| Model class | Pareto axes |
|---|---|
| ALGEBRAIC | (cx, MSE) + (smoothness penalty) |
| ODE | (cx, one-step MSE) + (rollout error, conservation violation, invariance violation, smoothness) |
| PDE | (cx, one-step MSE) + (rollout error, conservation violation, smoothness, BC respect) |

Three to four dimensions for dynamical-system classes, two to three for algebraic. The structural-loss dimensions are *populated by signature deviations* (next section).

## 5. The bidirectional reading of signatures

This is the architectural insight that lets Stages 2, 5, and 6 share machinery.

A **Signature** (the Stage 2 deliverable) measures a property from data. The same measurement serves two purposes:

- **Identification reading** (Stage 5): the unknown trajectory's signature is matched against library anchors. Signature distance = identification confidence.
- **Scoring reading** (Stage 6): a candidate fit's *predicted* signature deviation from the declared system signature = Pareto penalty.

Example: conservation test on a Kepler trajectory. The Stage 2 signature finds energy + angular momentum are conserved with variance ratio < 1e-4. Stage 5 uses this to confirm "yes, this is a Kepler-class system." Stage 6 uses it on a candidate fit: simulate the candidate forward, measure how much energy drifts, penalize. Same machinery, two readings.

**Implication for Stage 2 design**: Signatures must be callable on *generated* trajectories from candidate fits, not only on observed data. Stage 6 needs them to evaluate candidates. The current Stage 2 design (Section 6 of Stage 0 note) is already compatible — signatures take a Trajectory, no special source restriction — but we should write tests that exercise both readings.

## 6. How natural-overfit gets eliminated mechanically

The Class A/B taxonomy (from `from_data_to_mechanism.md`) is unstable on a 2D Pareto because Class A often ties Class C on $(cx, \text{one-step loss})$ and the GP tournament can't distinguish them. Multi-objective Pareto resolves this **automatically**:

- **Class A — temporal-differencing tautology**: dominated on rollout (infinite rollout error). Falls off Pareto.
- **Class B — template-overfit (e.g., dissipation-mimicking forms)**: dominated on conservation when system is conservative, or on rollout when fit is local-only. Falls off Pareto.
- **Class C — clean mechanism**: ties on cx with Class A/B candidates, ties on one-step loss, but wins on rollout + conservation + invariance.

C5 (counterfactual eval) does this as a post-hoc selector. Multi-objective Pareto does it during the search. Both should coexist; multi-objective at scoring catches most cases cheaply, C5 catches the residual at selection.

## 7. Implementation contract — Stage 6 (forward)

When Stage 6 ships:

```python
@dataclass(frozen=True)
class ScoreVector:
    cx: int
    one_step_mse: float
    rollout_mse: Optional[float]              # ODE/PDE only
    conservation_violation: Optional[float]   # systems with declared conservation
    invariance_violation: Optional[float]     # systems with declared symmetries
    smoothness_penalty: Optional[float]       # smooth-class systems
    bc_violation: Optional[float]             # PDE only
```

Pareto front is computed across all non-None axes; the model class determines which axes are populated. The GP `loss_fn` interface extends to return `ScoreVector` instead of a single float. Tournament + selection uses dominance.

Stage 6 is **not Stage 2**. But Stage 2 signatures are the inputs to Stage 6 scoring. The two are coupled but ship in order: signatures first, then multi-objective scoring built on top.

## 8. Implications for Stage 2 (no design change)

The 11 signatures specified in Stage 0 §6 + Stage 0.5 stay as-is. No additions or removals motivated by this note.

What *is* clarified:

- Signatures must run on *any* Trajectory (observed, ground-truth, or candidate-generated). This is already the design; we just commit to it explicitly so Stage 6 can call them on simulated candidate trajectories.
- The two-tier split (model-class classifier in Tier A, within-class in Tier B) from earlier Stage 2 discussion still holds.
- The metadata-driven symmetry + conservation tests (structured outputs rather than scalars) become *more* important because Stage 6 needs structured penalties, not scalar feature values.

## 9. Implications for Stage 5 (refined pipeline)

The Stage 5 pipeline updates from `model_class_taxonomy.md` §6 grow one more step:

```
data in
  Step 1: classify model class (Tier A signatures)
  Step 2: SELECT LOSS FAMILY per model class       ← NEW from this note
  Step 3: extract within-class signatures (Tier B)
  Step 4: match against library anchors of same model class
  Step 5: class-aware SR with anchor as prior + class-appropriate loss
  Step 6: multi-objective Pareto across class-appropriate ScoreVector axes
  Step 7 (optional): if multiple model classes plausible, report Pareto
                     across model classes
```

Step 2 is new. It picks one-step vs rollout vs algebraic-MSE based on the inferred class. Step 5 uses the selected loss during SR. Step 6 applies the full ScoreVector for selection.

## 10. Explicitly deferred

- **Bayesian / NLL losses** for noisy data: needed once SDE class (Stage 9) is added; not now.
- **Adaptive loss weighting** for Pareto scalarization (Pareto reports a frontier; if you need a single answer you need a weighting policy): defer to Stage 6 ablation.
- **Implicit-equation loss** $|F(y, \dot y, \ldots)|$ for IMPLICIT class: defer with the implicit-equation deferral from Stage 0.5.
- **Stochastic-rollout loss**: meaningful only with SDE; deferred with SDE.

## 11. Open questions for Stage 6 (not now)

- **How to weight Pareto axes**: do we report the full frontier, or scalarize via a declared weighting?
- **Rollout horizon**: $k$-step vs full-trajectory; per-class default?
- **Numerical jacobian** for adjoint-style rollout differentiation if Stage 6 wants gradient-based const-opt
- **Multi-trajectory rollout aggregation**: mean over training trajectories, or worst-case?

These become decisions when Stage 6 lands. Stage 2 (next) doesn't need them.

## 12. Verdict

The loss-per-class decision **does not change Stage 2 design**. It does change Stage 5 (adds Step 2 = loss selection) and motivates Stage 6 multi-objective. The architectural insight — signatures used twice (identify + score) — tightens project coupling in a structurally good way: building Stage 2 well delivers two-thirds of Stage 6 for free.

Stage 2 proceeds as designed. Stage 6 lands after Stage 5 (identification pipeline) but is now scoped concretely enough that its `ScoreVector` interface is fixed by this note.
