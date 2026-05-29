# Research note: C7 — coordinate-discovery prepass

**Status:** ? RESEARCH. Conjecture C7 proposed 2026-05-29. Empirical
A/B on Feynman: see Result section below (updated post-run).

**Provenance:** user direction during framework synthesis discussion
(2026-05-29):

> *"does the topological system identification uses coordinate transforms?"*

Followed by request to implement as an experimental subtask and A/B it
against the existing decompose v2 prepass.

## 1. The conjecture in one sentence

A meaningful fraction of physical relationships look like power-law
products in a TRANSFORMED target space, where the transform is one of
a small physics-motivated library `{identity, log_abs, sqrt_abs, square,
inverse}`. Detecting the transform AND the power-law jointly should
catch forms that aren't power-law in raw or log space but ARE in some
other coordinate.

## 2. Architectural framing

C7 extends the detect-then-seed pattern from the production decompose
prepass (`tessera.search.decompose`):

Production (already shipped):
- `detect_power_law(env, y)` — log|y| ≈ a₀ + Σ aᵢ log|xᵢ|
- `detect_exp_wrapper(env, y)` — apply log to y, then power-law
- `power_law_seed()` — orchestrator, tries power-law first, exp-wrapper
  as fallback

Experimental (C7 — this module):
- `detect_coord_discovery_seed(env, y)` — try each target transform,
  pick the highest-R² one, wrap the inverse transform around the
  resulting power-law seed.

The library:
| transform | y → y_t | inverse used on seed tree |
|---|---|---|
| identity | y | (none) |
| log_abs | log\|y\| | exp(seed) |
| sqrt_abs | √\|y\| | seed*seed |
| square | y² | sqrt(seed) |
| inverse | 1/y | 1/seed |

## 3. What this connects to in the literature

The closest precedent is **Champion, Lusch, Kutz, Brunton 2019**
"Data-driven discovery of coordinates and governing equations" (PNAS).
Their argument: SR-style equation discovery often fails on real data
because the data isn't in the right coordinates. The right architecture
is to discover coordinates AND equations *jointly* — they use an
autoencoder for the coordinate side, SINDy for the equation side.

C7 is a discrete, lightweight version of the same principle: instead
of learning continuous coordinates with a neural network, try a small
fixed library of transforms and pick the one that makes the existing
detector fire most cleanly. Less expressive but interpretable and
cheap.

Adjacent: Koopman operator theory (Mezić 2005, Brunton-Brunton-Proctor-
Kutz 2017) lifts nonlinear dynamics to a higher-dimensional linear space
via observables. The "observables" are coordinate transforms; the
"linear dynamics" in lifted space is what we'd discover. C7's transform
library is a tiny Koopman observable basis.

## 4. Pre-analysis: expected outcome on Feynman

**Honest expectation: neutral or modest improvement, because the
existing power-law detector already handles fractional exponents.**

Specifically: in log-log space, the transforms {identity, sqrt_abs,
square, inverse} are all related by linear rescaling of the regression
target. For a target y and feature x:
- identity:  log|y| = a + b·log|x|
- sqrt_abs:  log(√|y|) = 0.5·log|y| → a' + b'·log|x| (constants/slopes scaled by 0.5)
- square:    log(y²) = 2·log|y| → scaled by 2
- inverse:   log(1/|y|) = -log|y| → negated

The R² of the regression is invariant under these rescalings —
identical to identity's R². The detector picks integer-exponent
representations or fractional-exponent representations depending on
the natural form, but the *detection rate* is unchanged.

Only `log_abs` is a genuinely different transform — and that's exactly
what the existing exp-wrapper detector already does.

Therefore C7 on Feynman should reproduce decompose v2's +10 exact
count, not improve on it. If empirically true, that validates the
architecture (C7 *does* generalize decompose) but tells us C7 adds no
empirical value on Feynman specifically.

Where C7 *would* matter empirically is on benchmarks where the natural
target coordinate isn't already covered by power-law + exp-wrapper:
- Real-data forms with additive offsets (e.g., I.40.1's `n₀·exp(-mgx/(kT))`
  has `log y = log n₀ - mgx/(kT)` — additive constant + product. Identity
  rejects; log_abs rejects because the log_abs detector requires uniform
  sign of log|y| and the additive constant violates that.)
- Time-series convolution forms where the target coordinate is delay-
  embedded (currently handled by sufficient_stats polish, not the
  decompose prepass).
- Power-law forms with negative bases (currently rejected by both
  power-law and exp-wrapper because they require uniform-sign features).

These are the cases C7 would help if extended with more sophisticated
transforms (Box-Cox, polynomial-in-y, etc.). The current library is
too small to materially help on Feynman.

## 5. MVP design

Implemented in `src/tessera/experimental/coordinate_discovery.py`:

```python
def detect_coord_discovery_seed(
    env, y, *, r2_threshold=0.99, margin_over_identity=0.0,
    skip_identity=False, round_exponents=True,
) -> Optional[CoordDiscoveryResult]:
```

Protocol:
1. For each transform φ in TARGET_TRANSFORMS, apply to y → y_t.
2. Run `detect_power_law(env, y_t, r2_threshold=0.0)` to get the R².
3. Pick the transform with the highest R² above `r2_threshold`,
   optionally requiring `margin_over_identity` above identity's R².
4. Build seed tree: inverse_transform(power_law_tree(fit)).

Returns a `CoordDiscoveryResult` with the chosen transform name, the
inner power-law fit, the assembled seed tree, and the per-transform
R² for diagnostics.

Per the experimental discipline, production code does NOT import from
`tessera.experimental`. C7's seed is passed to the GP via the public
`precomputed_seed_trees` config option (added to GPConfig 2026-05-29).

## 6. Validation

NEW benchmarks/run_feynman_coord_discovery_ab.py — A/B Feynman 30-eq
with C7 OFF (no prepass) vs C7 ON (precomputed_seed_trees populated by
C7). Production prepass disabled on both arms to isolate C7's effect.

NEW benchmarks/results/feynman_coord_discovery_ab.md — result tables.

(Result text added below post-A/B.)

### A/B Result (2026-05-29)

Headline: `benchmarks/results/feynman_coord_discovery_ab.md`.

| arm | exact | partial | failed |
|---|---|---|---|
| C7=OFF | 10 | 13 | 7 |
| C7=ON  | **20** | 6 | 4 |

**+10 exact, 0 regressions. Identical to decompose v2's +10 exact.**

Transitions: 7 partial→exact, 3 failed→exact, 20 same. The 10 promoted
equations are exactly the same equations decompose v2 promotes
(Coulomb ×2, q1·q2/r², ½kx², Larmor, qvB/p, Stokes-Einstein, κv²/(nσ),
sound speed √(γpr/ρ), I.12.2 Coulomb, I.6.20 Gaussian).

**Pre-analysis prediction held: C7 ≡ decompose v2 on Feynman.**

The empirical equivalence with decompose v2 is the strongest possible
evidence for the in-log-log-space-linear-equivalence argument:
- identity, sqrt_abs, square, inverse: all give the same R² as
  identity (the underlying power-law regression target is rescaled,
  but the R² is invariant under rescaling)
- log_abs: genuinely different, catches the same Gaussian forms
  exp_wrapper does

Per the smoke test in §5 (in code review):
- 16/30 Feynman targets chose identity transform (the cleanly-power-law cases)
- 2/30 chose log_abs transform (I.6.20a, I.6.20 — both Gaussian)
- 12/30 chose no transform (R² < 0.99 on all)
- Total 18/30 with a chosen transform — exactly matches the count of
  detections from decompose v2 (power-law + exp-wrapper).

### Verdict against criteria

**Graduation criterion**: "at least +1 exact transition that decompose
v2 alone does not produce, with 0 regressions on currently-exact
equations." — **NOT MET**. C7 produces +10 exact (same as decompose
v2), 0 regressions, but 0 *new* transitions beyond decompose v2.

**Removal criterion**: "0 new exact transitions AND no improvement on
seed tree complexity AND no improvement on real-data benchmarks." —
**NOT FULLY MET YET**. The first two conditions hold on Feynman, but
the real-data condition is untested.

**Decision**: stay experimental. C7 architecturally subsumes decompose
v2 (correct generalization) but adds no empirical value on Feynman
because the existing detectors already cover the same coordinate
space. The architectural insight (target-space transforms as a first-
class operation in detect-then-seed) is preserved as a research
artifact for future benchmarks where the natural target coordinate
isn't already covered by power-law + exp-wrapper. Concrete next
benchmarks where C7 might empirically pay off:

1. Forms with additive offsets inside an exp/log wrapper (I.40.1
   `n₀·exp(-mgx/(kT))` — additive log(n₀) term breaks the inner
   power-law structure). Would need an *additive-offset detector* on
   top of C7's transform library.
2. Real-data benchmarks where the natural target is e.g. Box-Cox of
   the raw observable. CAMELS uses log(Q) explicitly; weather uses
   dT/dt directly; both have an implicit coordinate choice that C7's
   library could in principle search over.

These are speculative — the result is "C7 is architecturally correct
but empirically equivalent to decompose v2 on the one benchmark
tested." Future evaluation needed before graduation or removal.

### Graduation criterion

**On Feynman**: at least +1 exact transition that decompose v2 alone
does not produce, with 0 regressions on currently-exact equations.

OR

**On weather PDE or CAMELS**: produces a seed tree that escapes the
persistence trap (CAMELS) or beats the diffusion oracle (weather)
where the baseline detectors stay silent.

### Removal criterion

**On Feynman**: 0 new exact transitions AND no improvement on seed
tree complexity (no integer-exponent simplification benefit).

AND

**On real-data benchmarks**: produces no improvement over the existing
prepass.

## 7. What this note explicitly does NOT claim

- NOT that C7 is the right way to do coordinate discovery in general.
  The library is small and physics-motivated; a Champion-style joint
  autoencoder + SR would be a deeper instantiation.
- NOT that the result will be positive. Pre-analysis predicts neutral
  on Feynman because the transforms degenerate to identity in log-log.
- NOT that the experimental module will graduate to production. That
  decision waits for empirical evidence per the §6 discipline.

## 8. Connection to the broader framework

C7 makes explicit what tessera was doing implicitly: every detector
performs a coordinate transform before applying its structural test.
- Power-law detector: (x, y) → (log|x|, log|y|), test linearity
- Exp-wrapper detector: y → log|y|, then power-law
- CAMELS polish: P → delay-embedding {P_lag_k}, then linear regression
- Heat equation: T → ∇²T via Measure2D, then linear regression

C7 unifies the target-space transforms (the y → φ(y) step) into a
single configurable prepass. If validated, the architectural pattern
graduates: every future detector should ask "what coordinate makes my
structural test fire most cleanly?" before applying the test.

If falsified on Feynman but successful on real-data benchmarks, the
result tells us the unification is the right abstraction for *some*
domains but Feynman's natural coordinates are already too well-covered
by the existing detectors to benefit.

## Changelog

- 2026-05-29: initial note. Conjecture proposed, pre-analysis predicts
  neutral outcome on Feynman per the in-log-log-space equivalence
  argument. Implementation in `src/tessera/experimental/coordinate_discovery.py`.
  A/B verifier: `benchmarks/run_feynman_coord_discovery_ab.py`.
