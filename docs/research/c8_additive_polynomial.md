# Research note: C8 — additive polynomial structure detector

**Status:** ? RESEARCH. Conjecture C8 proposed 2026-05-29 as the
multiplicative-↔-additive complement of decompose v2. A/B verifier
running concurrently with this note's initial draft; result section
appended post-A/B.

**Provenance:** discussion 2026-05-29 about what remains uncovered
after C7's empirical neutrality on Feynman. The cross-experiment
pattern revealed that decompose v2 (power-law products + exp-wrappers)
catches the MULTIPLICATIVE algebraic class. C8 targets the ADDITIVE
class: sums of monomials.

## 1. The conjecture in one sentence

A meaningful fraction of physical relationships are **sums of
multiplicative terms** (dot products, mixed-power polynomials,
additive Lagrangians) that the existing power-law detector cannot
catch because log of a sum doesn't factor. Detecting these via
direct polynomial OLS in raw feature space, and seeding the GP with
the top-N significant terms, should produce a measurable improvement
beyond decompose v2.

## 2. Where C8 fits

The existing prepass detectors target progressively richer structural
classes:

| Layer | Catches | Module | Status |
|---|---|---|---|
| Multiplicative power-law | `C · ∏ xᵢ^{aᵢ}` | `tessera.search.decompose.power_law` | shipped |
| Multiplicative exp-wrap | `±exp(C · ∏ xᵢ^{aᵢ})` | `tessera.search.decompose.exp_wrapper` | shipped |
| **Additive polynomial** | **`Σ_α c_α · monomial_α(x)`** | **`tessera.experimental.additive_polynomial`** | **C8 (this note)** |

Together, decompose v2 + C8 cover most low-order algebraic structures
seen in physics: products of powers, exp-wrapped products, and sums
of products.

## 3. Pre-analysis: what should C8 catch on Feynman

Smoke test (`detect_additive_polynomial(env, y, max_degree=3,
r2_threshold=0.0)` reporting R²) on all 30 Feynman targets reveals
C8 reaches R² ≥ 0.99 on:

**Genuine polynomials** (C8 fit IS the true form):
- **I.11.19** `x1·y1 + x2·y2 + x3·y3` — dot product. R²=1.000.
- I.14.4 `½kx²` — already exact via decompose; C8 also exact.
- I.14.3 `m·g·z` — same.

**Polynomial approximations of non-polynomial forms** (C8 fit is a
Taylor-style approximation; GP must refine):
- I.6.20a `exp(-θ²/2)` — polynomial of θ around 0 approximates the
  Gaussian; C8 D=3 R²=0.997.
- I.6.20 `exp(-(θ/σ)²/2)` — similar; D=3 R²=0.978, D=4 R²=0.994.
- I.8.14 `√((x2-x1)² + (y2-y1)²)` — D=3 R²=0.939 (degree-2 polynomial
  in differences captures the squared inside), D=4 R²=0.986.
- I.12.11 `q·(Ef + B·v·sin θ)` — additive structure with trig
  factor; D=3 R²=0.985, D=4 R²=0.999.
- I.15.3t `(x − u·t)/√(1−u²/c²)` — Lorentz, small-v approximation;
  D=3 R²=1.000.
- I.16.6 `(u+v)/(1+uv/c²)` — same; D=3 R²=1.000.
- I.18.4 `(m1·r1 + m2·r2)/(m1+m2)` — rational, but polynomial
  approximates locally; D=4 R²=1.000.
- I.24.6 `½m(ω² + ω₀²)x²` — sum of squares × multiplicative factor;
  D=3 R²=0.999.
- I.27.6 `d1·d2/(d1+d2)` — rational; D=3 R²=1.000.
- I.30.3 — Bragg pattern; D=3 R²=0.998.
- I.40.1 `n0·exp(-mgx/(kT))` — polynomial approximation of the
  exponential; D=3 R²=0.994.

## 4. The honest uncertainty

The key empirical question: do polynomial-approximation seeds **help
the GP find the TRUE non-polynomial form**, or do they **mislead**
the search by anchoring it in the wrong functional class?

Three plausible outcomes:

1. **Genuine polynomials win**: C8 catches I.11.19, gets it exact,
   no regression elsewhere. +1 exact.
2. **Approximations also win**: the polynomial seed acts as a
   stepping stone — the GP discovers the right wrapper (exp, sqrt,
   trig) by perturbing the polynomial seed. +3-5 exact.
3. **Approximations mislead**: the polynomial seed dominates the
   Pareto front at a low loss; the GP can't simplify it to the right
   form. ~0 new exact; possibly even some regressions (Class B-like
   behavior where the polynomial overfits TRAIN).

The cross-experiment pattern from the existing basket (C1, C3, C4,
C6) suggests scoring-layer interventions don't reliably help. But C8
is a SEED-INJECTION at the SEARCH-LAYER (not a scoring change), which
historically has been the productive layer (decompose v2 went +10
this way).

So the prior probability favors outcome 1 or 2.

## 5. MVP design

Implemented in `src/tessera/experimental/additive_polynomial.py`:

```python
def detect_additive_polynomial(env, y, *,
    max_degree=3, r2_threshold=0.99, top_n=8,
    coef_threshold=1e-6, max_basis_size=500,
) -> Optional[AdditivePolynomialFit]: ...
```

Protocol:
1. Enumerate monomials of total degree 1..D in the variables.
2. Bail if basis size > max_basis_size (signals too high dim/degree).
3. Build design matrix X (one column per monomial) + intercept column.
4. Solve OLS via `np.linalg.lstsq`.
5. Compute R²; reject if < r2_threshold.
6. Filter to top-N coefficients by magnitude (above coef_threshold).
7. Return AdditivePolynomialFit.

Tree builder `build_additive_polynomial_tree` constructs
`Σ_α c_α · ∏ xᵢ^{eᵢα}` using:
- e=1: bare Var
- e=2: Var*Var (smaller cx than pow(Var, 2))
- e=3: Var*Var*Var
- e≥4: pow(Var, Const)

Coefficient handling: c=1 → omit multiplier; c=-1 → unary neg
wrapper; otherwise BinOp("mul", Const, monomial). Sum of terms is
left-folded BinOp("add", ...).

Per the experimental discipline, production code does NOT import
from `tessera.experimental.additive_polynomial`. The A/B runner
(`benchmarks/run_feynman_additive_poly_ab.py`) computes the C8 seed
externally and injects via `GPConfig.precomputed_seed_trees`.

## 6. Validation criteria

**Graduation**: A/B at pop=400, gens=120, seed=2026 must produce:
- At least +2 NEW exact transitions beyond decompose v2 alone
- 0 regressions on currently-exact equations (decompose v2's 20 exacts)
- At least 1 of the new exacts is a known additive form

**Removal**: 0 new exact transitions OR ≥ 1 regression that doesn't
trace to selection-layer noise.

## 7. A/B Result (2026-05-29)

Headline: `benchmarks/results/feynman_additive_poly_ab.md`.

| arm | exact | partial | failed |
|---|---|---|---|
| OFF (decompose v2) | 20 | 6 | 4 |
| ON  (decompose v2 + C8) | **21** | 5 | 4 |

**+1 exact, 0 regressions.**

The single transition: **I.11.19 (`x1·y1 + x2·y2 + x3·y3`, dot
product)** OFF partial (rel=0.045, cx=15) → ON exact (rel=0.000, cx=11).
This is the only genuine polynomial form in the Feynman 30-eq subset
— the C8 seed IS the correct expression, and the GP refined the
coefficients via const-opt.

**No other transitions.** All other equations where C8's pre-analysis
showed R² ≥ 0.99 — Gaussians (I.6.20a, I.6.20), distance (I.8.14),
Lorentz forms (I.15.3t, I.16.6), center of mass (I.18.4), sum-of-
squares (I.24.6), Boltzmann (I.40.1), Bragg pattern (I.30.3) — were
**polynomial APPROXIMATIONS** of the true non-polynomial forms. The
GP couldn't refine these seeds into the true forms (sqrt, exp, sin,
1/x).

Why the approximation seeds didn't help: the polynomial-seed loss
was much higher than the true-form loss decompose v2 already
discovered (e.g., decompose v2 finds I.6.20a Gaussian exact via
exp-wrapper, loss ~0; C8's polynomial Taylor expansion has irreducible
truncation error). The polynomial seed couldn't enter the Pareto front
once decompose v2's exp-wrapper seed was present. Selection-layer
filtering correctly killed the inferior seed.

### Verdict against criteria

**Graduation criterion** ("≥ +2 new exact transitions"): NOT MET.
Only +1 transition (I.11.19).

**Removal criterion** ("0 new transitions OR ≥ 1 regression"):
NOT MET. +1 transition (better than zero), 0 regressions.

**Decision**: C8 stays experimental as **PARTIAL VALIDATION**.

The narrow positive result (genuine polynomial detection) is real
and demonstrates the architecture works. The broader hypothesis
(polynomial-approximation seeds bootstrap discovery of true non-
polynomial forms) is **falsified for Feynman**: the GP cannot
Taylor-expand backwards from an approximation seed into a different
functional class.

### What this teaches about the framework

A cross-experiment lesson: **seed-injection works when the seed IS
the correct form; it doesn't work when the seed is an approximation
of the correct form**.

This is consistent with the existing pattern: decompose v2's success
came from seeds that ARE power-laws/exp-wrappers, not from seeds
that APPROXIMATE other functional classes. C8's success on I.11.19
came from a seed that IS the polynomial.

For seeding to help approximation cases, we'd need either:
1. A way for the seed to "spread" the GP search into a larger
   functional neighborhood (e.g., wrapping the polynomial seed in
   randomly-sampled transcendental wrappers as initial population
   members)
2. A different detect-then-seed conjecture: detect the *outer
   wrapper* (exp, sqrt, log) first, then fit the inside polynomially

The latter is the architectural insight for a possible C9. For now,
C8 is documented as the narrow polynomial detector.

### Comparison to the broader basket

| Conjecture | Status | Lesson |
|---|---|---|
| C1 ABC scoring | Falsified | Scoring-layer interventions don't help |
| C3 MDL scoring | Falsified | Calibration math right, effect below noise |
| C4 causal axes | Partial | Necessary but not sufficient |
| C5 counterfactual | **Validated → Graduated 2026-05-29** | Selection layer is productive |
| C6 adaptive search | Null | Validated-as-predicted (no effect) |
| C7 coord-discovery | Neutral on Feynman | Architecturally redundant w/ decompose v2 |
| **C8 additive poly** | **Partial validation** | **Seed-as-correct-form works; seed-as-approximation doesn't** |

C8 fits the "narrow positive, hypothesis-falsified" pattern of C4
(causal axes). The implementation is correct; the conjecture's broad
form doesn't hold; the narrow form holds reliably.

## 8. Connection to the broader framework

C8 closes the multiplicative-vs-additive gap in the detect-then-seed
prepass family. After C7 turned out architecturally redundant with
decompose v2 (linearly equivalent in log-log space), C8 is the
genuinely new structural class to target.

The conjecture stays consistent with the framework's pattern:
- **Detect** specific algebraic structure
- **Seed** the GP with the discovered form
- **Let selection decide** if the seed is right

C8 also opens a path to higher-order detectors (C9 might be Box-Cox
target transform, C10 might be additive-inside-exp, etc.) that
together build a structural taxonomy of algebraic forms.

## Changelog

- 2026-05-29: initial note. Conjecture proposed; pre-analysis run
  on all 30 Feynman targets shows C8 R²≥0.99 on 18/30 at D=3 and
  25/30 at D=4. A/B verifier kicked off in background.
