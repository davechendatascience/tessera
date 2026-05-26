# Research note: model class as a first-class concept in SR

**Status update (2026-05-26 → `docs/planned/`):** Moved from `docs/research/`. Stage 1.5 schema retrofit **SHIPPED** (ModelClass enum, `model_class` field per system, `target_form()` method, algebraic canonical added, PDE grid metadata populated). Design contract continues to inform ongoing Stage 5 identification pipeline + Stage 6 multi-objective scoring.

**Status:** STAGE 0.5 design fix. Extends the Stage 0 workbench design note (`methodology_workbench_and_library.md`) with explicit model-class taxonomy. This note is the design contract for the Stage 1 update — code changes must reference back to it.

**Provenance:** user (2026-05-26), surfaced while reviewing Stage 1 tests:

> "Against a pure function that describes the data and a pde that describes the data, how do we know which is which? ... One representation is denoting y' and y'' simultaneously, and the other is denoting y. These are completely different and we don't know which representation to use beforehand. ... That's probably not a very good way to do math, not knowing if you are just fitting a function or whether you are solving a pde. Those are different things, but we mix them together."

The conflation is real, and we have been making it. This note fixes the design.

---

## 1. The conflation we have been making

Given a dataset, SR tools (tessera included) typically produce a "fitted expression" without first asking *what mathematical object the dataset came from*. The expression is a string of operators; what that string *means* depends entirely on how the human prepared the target.

Concrete examples from our own portfolio:

| System | What we did | What this implicitly assumed |
|---|---|---|
| Heat equation | Pre-computed `dt_U = U(t+1) - U(t)`, asked SR for `f(U, ∂_x U, ...)` | The model class is PDE first-order-in-time |
| Lorenz | Pre-computed `dx/dt`, asked SR for `f(x, y, z)` | The model class is ODE first-order in state |
| Feynman | Handed `(x_1, ..., x_n)`, asked for `y` | The model class is algebraic (pure function) |

In all three cases the **model class was pre-decided by data preparation**, not discovered by SR. The fitted expression is then interpreted *as if* it were the right kind of object — but if the human chose the wrong class, the expression is still produced and looks fine. SR has no mechanism to flag "this dataset doesn't actually fit this class of model."

This is what "mixing them together" looks like in practice. **A mathematician handed a data table first asks "what kind of object generated this?" — the kind determines what we solve for.** SR currently skips this step.

## 2. The ModelClass taxonomy

The four model classes in scope for the workbench + library project:

| ID | Name | Form | Target form for SR | Example canonical |
|---|---|---|---|---|
| `algebraic` | Pure function | $y = f(x)$ | $y$ given features $x$ | Feynman equations |
| `discrete_map` | Iterated map | $y_{n+1} = f(y_n)$ | $y_{n+1}$ given $y_n$ | Logistic map |
| `ode` | Ordinary differential equation | $\dot x = f(x, t)$ | $\dot x$ given state $x$ | Lorenz, harmonic, Kepler |
| `pde` | Partial differential equation, first-order in time | $\partial_t u = f(u, \partial_x u, \partial_x^2 u, \ldots)$ | $\partial_t u$ given $u$ and spatial neighbors | Heat, Burgers |

These are the four classes the workbench will span in Phase A. Higher-order ODEs (e.g., $\ddot x = -\omega^2 x$) are encoded as `ode` on a higher-dimensional state — the canonical reduction (state $= (x, \dot x)$, first-order in this state) is what we represent. The mathematical hierarchy of derivatives becomes a choice of state-space representation, not a separate class.

**Each ModelClass defines a `target_form` map**: given a `Trajectory` from the workbench, how do we transform it into SR-ready `(features, target)` pairs?

```python
class ModelClass(Enum):
    ALGEBRAIC = "algebraic"
    DISCRETE_MAP = "discrete_map"
    ODE = "ode"
    PDE = "pde"

def target_form(model_class, trajectory) -> tuple[features, target]:
    """Transform a raw trajectory into the SR-ready form for this class."""
    if model_class is ModelClass.ALGEBRAIC:
        # features = inputs as-is; target = y. No time/order assumption.
        return trajectory.inputs, trajectory.y
    if model_class is ModelClass.DISCRETE_MAP:
        # features = y[:-1]; target = y[1:]
        return trajectory.state[:-1], trajectory.state[1:]
    if model_class is ModelClass.ODE:
        # features = state; target = numerical derivative of state
        return trajectory.state[:-1], finite_diff(trajectory.state, trajectory.t)
    if model_class is ModelClass.PDE:
        # features = (state at t, spatial neighbors); target = dt of state
        return spatial_stencil(trajectory.state), time_diff(trajectory.state)
```

This is the only place model-class-dependent reshaping lives. The downstream SR run is then model-class-agnostic — it just sees `(features, target)` pairs.

## 3. The multiple-representation problem (y vs y' vs y'')

The user's sharpest observation: given a trajectory $y(t)$, multiple model classes can describe it, and they give materially different fits.

Take $y(t) = \cos(\omega t)$ for concreteness:

| Model class | Fitted form | Parsimony | What it captures |
|---|---|---|---|
| `algebraic` (with input = $t$) | $y = \cos(\omega t)$ | Low cx, requires trig vocabulary | The trajectory in time |
| `ode` first-order on $(y)$ alone | $\dot y = -\omega \sqrt{A^2 - y^2}$ | Higher cx; needs sqrt + nonlinear | Not clean — under-determined |
| `ode` first-order on $(y, \dot y)$ | $\ddot y = -\omega^2 y$ | Lowest cx — pure polynomial | The dynamics, in canonical form |

Three different "right answers" for the same data, depending on what we say we're solving for. None is incorrect; they describe different aspects.

**This is not a defect — it is the actual structure of the problem.** Real mathematicians make this choice consciously: "I will model this as a second-order ODE in $y$" or "I will model this as a function of time." The choice carries epistemic weight (it commits to a story about generating mechanism) and it should be explicit, not hidden in data-prep code.

The workbench's response: each canonical system declares its **canonical model class** (the most-natural representation for that dynamics), but the *identification pipeline* should be able to fit multiple model classes on the same data and report them — with their relative parsimony / fit quality / residual diagnostic. The library's anchor entries store the canonical form, but Stage 5 evaluation can compare model-class hypotheses on new data.

This is the structural answer to "how do we know which is which": **we try each, score each, and report the Pareto front across model classes.** The discrimination is empirical (which class explains the data best at low complexity), informed by signatures (the Stage 2 extractors that test for time/space structure), and ultimately reported as a comparative claim — not a single fitted expression.

## 4. Model vs Solution vs Fit (terminology cleanup)

We have been conflating three distinct objects under the word "expression":

| Term | What it is | Example |
|---|---|---|
| **Model** | The structural equation specifying the dynamics. Belongs to a `ModelClass` | $\dot x = \sigma(y - x)$ — Lorenz's first equation |
| **Solution** | A trajectory satisfying the model under given initial conditions | The chaotic curve through $(x_0, y_0, z_0)$ |
| **Fit** | An expression minimizing loss on observed data; corresponds to a hypothesized Model | The tessera-discovered Pareto candidate |

Going forward, the workbench / library / identification code uses these terms precisely:

- `tessera.workbench.Trajectory` represents a **solution** (specific dynamics under specific parameters and ICs).
- `tessera.workbench.CanonicalSystem` represents a **model** (the structural equation + its declared metadata).
- `tessera.search.GP.run(...)` outputs **fits** (expressions minimizing loss).
- The library entry for each canonical system stores the **fitted form** of its canonical **model**, distinct from any specific **solution** trajectory.

We do *not* introduce a `Solution` class. `Trajectory` is the existing data carrier for solutions and is sufficient. We *do* add a `Model` concept (next section).

## 5. Schema updates to the workbench (Stage 1 contract)

The Stage 1 implementation needs three additions to be ModelClass-aware.

### 5.1 Add `ModelClass` enum and `model_class` field on `CanonicalSystem`

```python
class ModelClass(str, Enum):
    ALGEBRAIC = "algebraic"
    DISCRETE_MAP = "discrete_map"
    ODE = "ode"
    PDE = "pde"

@dataclass
class CanonicalSystem(ABC):
    ...
    model_class: ModelClass    # NEW — first-class
    canonical_target_form: str # NEW — short description, e.g. "dx/dt = f(state)"
```

The existing `domain` field (`"ode" | "pde" | "sde" | "algebraic"`) is **superseded by `model_class`**. The two largely overlap — they were the same idea expressed twice with slightly different vocabulary. Stage 1 keeps `domain` as a backward-compat alias derived from `model_class` and deprecates it in code comments.

### 5.2 Add at least one algebraic canonical system

The workbench currently has nine ODE/PDE systems and zero algebraic systems. To span the discrimination the identification pipeline must learn, we need at least one. Stage 1 update will add:

| ID | Form | Why |
|---|---|---|
| `algebraic_feynman_gaussian` | $y = \exp(-\theta^2 / 2)$ | Standard Feynman target; pure function of iid inputs; no time |

Other Feynman-style algebraic systems can be added in Stage 7 expansion; one is sufficient to make the model-class discrimination a real test.

### 5.3 Add `target_form(trajectory)` method to `CanonicalSystem`

This is the model-class-specific data preparation. Each subclass overrides to produce `(features, target)` arrays ready for SR:

```python
class CanonicalSystem(ABC):
    @abstractmethod
    def target_form(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
        """Transform a trajectory into (features, target) for SR.
        ALGEBRAIC: (inputs, y)
        ODE: (state[:-1], finite_diff_state)
        PDE: (stencil_features, dt_field)
        """
```

The Stage 4 library construction uses this method when fitting each canonical with abundant data — no hand-coded data prep in the per-system fit scripts.

### 5.4 Add `time_step`, `grid_resolution` to `InformationRequirements` for PDEs

The discrete-vs-continuum gap from the previous discussion: PDE identification at fixed grid has a truncation-error floor that doesn't shrink with sample count. The information requirements must record this:

```python
@dataclass
class InformationRequirements:
    ...
    min_dt: float | None = None        # only meaningful for ODE/PDE
    min_dx: float | None = None        # only meaningful for PDE
    grid_floor: dict | None = None     # PDE-specific resolution constraints
```

`grid_floor` is the right place to record "below this resolution, identification is impossible because truncation error dominates noise."

## 6. Updates to the identification pipeline (Stage 5 contract)

The pipeline grows a model-class identification step *before* signature distance matching:

```
data in
  →  Step 1: classify model class (algebraic / discrete_map / ode / pde)
      via signature tests (autocorrelation, stencil locality, smoothness)
  →  Step 2: extract within-class signature
  →  Step 3: match against library anchors of the same model class
  →  Step 4: class-aware SR with anchor as prior
  →  Step 5: multi-objective Pareto across (fit, parsimony, smoothness,
              invariance, conservation, mode coherence)
  →  Step 6 (optional): if Step 4 fails or produces high-residual fit,
              try a different model class and compare. Report the
              Pareto front across model classes.
```

Step 1 is new. It is *cheap* — autocorrelation and stencil-locality tests are O(N) — and rules out incorrect model classes before any SR is run, saving the bulk of the compute budget for the right class. It is also the discrimination the user is asking about: "how do we know which is which" gets answered by these signature tests.

Step 6 is the explicit handling of the multiple-representation problem. When the data admits multiple model classes (e.g., a periodic trajectory can be modeled as algebraic-in-time or as second-order-ODE), the pipeline reports both fits and lets the user / downstream consumer see the Pareto comparison.

## 7. Explicitly deferred — not in scope now

To keep this note actionable rather than expansive, we explicitly defer:

| Capability | Why deferred | Earliest stage |
|---|---|---|
| Conditional / piecewise / `if`-trees | Vocabulary expansion is separate from model-class taxonomy | Stage 7+ |
| Multi-mode / hybrid systems | Mode detection itself is a Stage 7 curriculum question | Stage 7+ |
| Implicit equations $F(y, \dot y) = 0$ | SR engine cannot natively express implicit constraints | Stage 8+ |
| SDE — stochastic dynamics | Drift + diffusion identification is methodologically different | Stage 9 (BTC bridge) |
| Recursive / inductive definitions | Out of scope for SR; belongs to program synthesis | Indefinite |
| Symbolic constants (π, 1/2, e) | Quality-of-fit issue, not model-class taxonomy | Stage 6+ |

The user (2026-05-26) explicitly chose ModelClass as first-class and elected to defer the others. This list is the contract on what stays out of Stage 1.

## 8. Implementation contract for Stage 1 update

Concrete deliverables in this round:

1. **Add `ModelClass` enum** in `src/tessera/workbench/types.py`. Four values: `ALGEBRAIC`, `DISCRETE_MAP`, `ODE`, `PDE`.
2. **Add `model_class` class attribute** to `CanonicalSystem`. Each of the 10 existing systems sets it. Deprecate `domain` (keep as alias for one transition window; doc-string says superseded).
3. **Add `canonical_target_form`** as a one-line documentation field on each system.
4. **Add at least one algebraic canonical** (`algebraic_feynman_gaussian` or similar). The 10 become 11.
5. **Add `target_form(traj)` method** as abstract on `CanonicalSystem`; implement for each of the 11 systems.
6. **Add `min_dt`, `min_dx`, `grid_floor`** to `InformationRequirements`. Populate for PDEs.
7. **Update tests** so they exercise `target_form` per system (each transformation produces shapes consistent with the declared model class).
8. **Update Stage 0 design note** (`methodology_workbench_and_library.md`) with a forward-reference to this note + a one-line table revision in Section 4.

After Stage 1 update: all 11 systems generate trajectories, declare their model class, and can produce SR-ready data via `target_form()`. Stage 2 (signatures) can then build the model-class discrimination tests on top of this contract.

## 9. Why this is worth doing now

Two reasons it has to happen before more code lands:

1. **The schema change touches every system**. Adding `model_class` later means editing 10+ classes and the tests. Doing it now is a single sweep.
2. **The conflation is conceptually load-bearing**. Without an explicit ModelClass, the identification pipeline's "which system is this?" question is malformed — we cannot answer it if "system" doesn't distinguish algebraic from PDE.

The Stage 0 note correctly identified the workbench / library / identification pipeline as the project. This Stage 0.5 note fixes the type system underneath it. Without this fix, Stage 5 identification is technically incoherent.

## 10. Open design questions for later notes

Marked but not resolved:

- **Higher-order ODE representation**: should we expose the underlying second-order form (e.g., $\ddot x = -\omega^2 x$) as a *display* representation distinct from the canonical first-order state-space reduction? Useful for human readability of library entries. Not blocking; deferred.
- **Discrete maps vs sampled ODEs**: a discrete-map fit and a coarsely-sampled ODE can be observationally identical. Distinguishing them requires sub-sample dynamics (multiple time resolutions). Stage 7+ curriculum.
- **Algebraic with time as one input** vs **ODE on state**: a trajectory $y(t)$ admits both fits as noted in §3. The Pareto report (§6 Step 6) handles this but the *recommendation* rule (which fit to default to) is open. Likely "prefer ODE form when autocorrelation in observed sequence indicates dynamics, prefer algebraic when inputs are iid in time."

These get resolved in subsequent notes as we hit them empirically.
