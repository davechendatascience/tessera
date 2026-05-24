# Research note: measure-theoretic operator algebra in the perfect-information game

**Status:** open research direction. Theoretical companion to
`fit_as_perfect_info_game.md`, grounded empirically in the
`benchmarks/run_equivalence_class_count.py` and
`benchmarks/run_interval_bound_tightness.py` results from 2026-05-25.

## 1. The question

The Knuth-grounded fit-as-perfect-info-game framework (companion doc)
treats SR as single-agent search over an expression grammar. That
grammar is usually some flavour of *polynomial-trigonometric algebra*
— operators are +, −, ×, ÷, sin, cos, exp, log, tanh, abs, sign. All
**pointwise**. Eureqa, PySR, AI Feynman, Operon all live here.

Tessera is different. Its grammar includes **integral-transform
operators** — `LinearFunctional`, `SeparableBilinear`, `Volterra2`,
`FunctionalOp2D` — each parameterised by one or more signed measures.
These are not pointwise; they encode *temporal structure* (or
spatial-temporal structure in 2D) that polynomial-trig ops can't
express.

**The question this note develops**: what does the measure-theoretic
operator algebra add to the perfect-information game framework
beyond what the polynomial-trig algebra gives?

The answer is concrete and empirically validated: it adds **a richer
canonical-form structure** (Lebesgue decomposition + Fubini) and
**closed-form lower bounds via L1 norms** that the polynomial-trig
algebra doesn't natively support. Both directly tighten the
branch-and-bound search the framework prescribes.

## 2. What the standard polynomial-trig algebra offers

In the polynomial-trig setting, the algebraic structure available is:

- **Ring operations** (+, −, ×) with their identities
- **Field operations** (+ × ÷) for invertible elements
- **Special-function algebra**: trig identities (sin² + cos² = 1),
  log/exp identities, etc.
- **Commutativity / associativity** of + and ×

These yield:

- A finite term-rewriting system (e.g. SymPy's `simplify`)
- Polynomial normal forms via Gröbner bases (for the polynomial
  subalgebra)
- Per-operator constant folding (fold `exp(0) → 1`, `sin(0) → 0`, etc.)

Equivalence classes here are well-studied. The empirical evidence
from `run_equivalence_class_count.py` shows that under tessera's
pointwise subset, the ratio |E_K| / |T_K| drops to ~8% at cx=7 — most
syntactic trees are equivalence-class duplicates. The polynomial-trig
algebra collapses *a lot* of the syntactic space.

## 3. What the measure-theoretic operator algebra adds

Tessera's grammar extends the pointwise polynomial-trig vocabulary
with three families of operators, each parameterised by signed
measures on non-negative lags:

| Operator | Form | Parameter |
|---|---|---|
| `LinearFunctional` | $(L_\mu x)(t) = \sum_k \kappa_\mu[k] \, x(t-k)$ | measure $\mu$ |
| `SeparableBilinear` | $(B_{\mu_a, \mu_b}(x, y))(t) = L_{\mu_a}(x)(t) \cdot L_{\mu_b}(y)(t)$ | measures $\mu_a, \mu_b$ |
| `Volterra2` | $(V_{\mu_a, \mu_b}(x))(t) = L_{\mu_a}(x)(t) \cdot L_{\mu_b}(x)(t)$ | measures $\mu_a, \mu_b$ |

Each measure decomposes (Lebesgue) into atomic + absolutely continuous
parts:

$$\mu = \sum_i w_i \, \delta_{\text{lag}_i} + \int_0^{S} \kappa(s) \, ds$$

This decomposition is the foundation of three structural advantages
the standard algebra doesn't have:

### 3.1 Canonical form via Lebesgue decomposition

A measure's atomic + density decomposition is **unique** (Royden 1988,
§11.5; Reznikov MAA-5616 §5.1, Theorem). Two measures equal iff their
atomic parts are equal AND their density parts are equal almost
everywhere.

**Implication for canonical-form normalisation**: a search over
measures collapses naturally over the atomic-part lattice (sort atoms
by lag, merge duplicates) and the density-family enumerator (one
canonical per family + parameter set). Tessera's `mutation.py`
already implements parts of this — `measure_mutate` operates on the
canonical atomic list — but the search-space-size analysis hasn't
been formalised.

**Open question**: under what term-rewriting system on measures does
the equivalence-class count |E_K^{measure}| reduce? Conjecture: for
atomic-only measures of complexity ≤ k, the count grows polynomially
(not exponentially) because the atomic-part lattice has efficient
enumeration via sorted-lag canonicalisation.

### 3.2 Fubini's theorem ↔ separable bilinear

Fubini's theorem (Reznikov §3.7, Thm 143) says: for a 2D measure
that decomposes as a product $\mu_a \otimes \mu_b$, the double
integral factors:

$$\iint f(s, \tau) \, d(\mu_a \otimes \mu_b)(s, \tau) = \int \left( \int f(s, \tau) \, d\mu_b(\tau) \right) d\mu_a(s)$$

For tessera, this is operationally exact: any 2D kernel $K(s, \tau)$
that factors as $\kappa_a(s) \cdot \kappa_b(\tau)$ can be applied via
two 1D convolutions (one per axis), not one 2D convolution.
`SeparableBilinear` is the explicit operator for this factorisation.

**Implication for search**: the cost of evaluating a candidate drops
from $O(N \cdot K_s K_\tau)$ to $O(N \cdot (K_s + K_\tau))$. On a
typical bilinear (window 24 × 168), that's a 50× speed-up. The
*search budget* recovered by this factorisation is real, and it has
no direct analogue in the polynomial-trig algebra.

### 3.3 L1-norm bounds (validated empirically by step c)

For any signed measure $\mu$ and bounded input $x \in [\text{lo}, \text{hi}]$:

$$\| L_\mu(x) \|_\infty \leq \|\mu\|_1 \cdot \max(|\text{lo}|, |\text{hi}|)$$

This is Hölder's inequality on the convolution. For
`SeparableBilinear` and `Volterra2`, the bound compounds via interval
multiplication.

The empirical step (c) result: median tightness ratio on
synthetic_xx jumped from 0.14 to **0.47** when these bounds replaced
the conservative $\pm\infty$ default. The L1-norm bound is the
**tessera-distinct branch-and-bound tool** that the polynomial-trig
algebra doesn't have.

In Knuth's branch-and-bound framework (TAOCP Vol 4B §7.2.2), the
"lower-bound function" is the entire ball-game — without it you have
no pruning. The L1-norm bound is tessera's specific instantiation.

## 4. What this changes about the perfect-information framing

The companion doc (`fit_as_perfect_info_game.md`) treats SR as a
single-agent perfect-information game with three structural
properties (F1: perfect info, F2: decomposable evaluation, F3:
algebraic equivalence). Adding measure-theoretic operators sharpens
each:

| Property | Polynomial-trig algebra | + Measure-theoretic operators |
|---|---|---|
| (F1) Perfect info | data ⇒ loss is deterministic | unchanged |
| (F2) Decomposable eval | loss factors per-sample | + measure-application factors as convolution (atomic shift-sum + density convolution) |
| (F3) Algebraic equivalence | polynomial / trig identities | + Lebesgue-decomposition uniqueness + Fubini factorisation + measure-algebra identities (μ * ν = ν * μ, etc.) |

**Key claim**: the measure-theoretic operator algebra gives the
perfect-information framework a richer **bound-derivation toolkit**.
Where polynomial-trig SR has to lean on interval arithmetic + per-op
monotonicity, tessera also has:

- $L_1$ bounds (norm of the measure)
- $L_\infty$ bounds (max of the kernel)
- Spectral bounds (eigenvalues of the measure operator under
  convolution)
- Approximation bounds (truncation of a long-tail measure to a finite
  support)

Each gives a different lower bound on $|L_\mu(x)|$ for bounded $x$. A
sophisticated branch-and-bound search would compute the tightest of
several (per Knuth's TAOCP 7.2.2 discussion of bound combination).

## 5. Empirical evidence

The three benchmarks shipped with this note quantify the claim:

### Step (a): equivalence-class collapse

`benchmarks/run_equivalence_class_count.py`: enumerates all valid
trees up to depth 3 with restricted grammar (4 leaves, all pointwise
ops). Reports |E_K| / |T_K| per complexity.

Result: ratio drops monotonically from 1.0 at cx=1 to 0.077 at cx=7.
**The simplifier collapses ~92% of trees at moderate complexity.**

This was on pointwise-only grammar. The analogous experiment on
tessera's full grammar (with FunctionalOp) hasn't been run because
the measure parameter space is continuous — but the polynomial
collapse on the atomic-part lattice should be even sharper, since
many distinct atomic sequences represent the same measure
(commutativity of summing atoms, mergeability of same-lag atoms).

**Open empirical question**: at cx=K for K ∈ {5, 7, 10}, what's the
ratio when LinearFunctional is included with restricted measures
(e.g., atomic only, lags ∈ {1, 2, 6, 24, 168})?

### Step (b): interval-bound tightness on real data

`benchmarks/run_interval_bound_tightness.py`: samples 2000 random
trees on three workloads, compares interval-derived MSE lower bound
vs actual MSE. Reports the distribution of `bound / actual` ratio.

Result (initial, pointwise grammar with conservative ±∞ on
FunctionalOp):
- synthetic_xx: median ratio 0.14, **47% of trees had unbounded
  intervals** (the FunctionalOp gap)
- synthetic_multi: median 0.0004, 49% unbounded
- synthetic_sin: median 0.00, 57% unbounded

Without step (c), the perfect-info game framework's branch-and-bound
prescription stalls because half the trees have no actionable bound.

### Step (c): L1-norm bounds close the gap

`tessera.expression.interval` extended: `LinearFunctional`,
`SeparableBilinear`, `Volterra2` all get bounds derived from
`measure_l1_norm(m) = ∑_k |κ[k]|`.

Re-running step (b) with the new bounds:
- synthetic_xx: median ratio 0.14 → **0.47** (3.4× improvement);
  unbounded 47% → 19%
- synthetic_multi: median 0.0004 → 0.11; unbounded 49% → 27%
- synthetic_sin: median 0.00 → 0.001; unbounded 57% → 41%

**The L1-norm bound IS load-bearing.** Roughly half the previously-
unbounded trees now have actionable bounds. The pruning hook (Exp 2)
goes from "useful on the pointwise-only subset" to "useful across
~75% of the tree distribution".

## 6. Tessera-distinct conjectures and open questions

Building on the framework:

### Conjecture (sharper than the companion doc's general form)

For tessera's full grammar including FunctionalOp at complexity ≤ K
with measures restricted to atomic-only with lags from a fixed grid:

> The branch-and-bound search using L1-norm + Lebesgue-decomposition-
> based bounds visits at most $O(|E_K^\text{atomic}| \cdot \log K)$
> distinct candidates to find the optimum, where $|E_K^\text{atomic}|$
> is the number of equivalence classes under atomic-part
> canonicalisation.

The log factor would come from the bound-tightening hierarchy: at
each level of the search tree, the bound rules out half the remaining
candidates. This is in the spirit of Knuth's branch-and-bound analysis
in TAOCP 7.2.2.

### Open questions

1. **What is $|E_K^\text{atomic}|$?** Step (a) gives a baseline for
   pointwise-only. The functional addition should reduce the ratio
   further (more identities → more collapse). Tessera-side experiment:
   enumerate atomic-measure-restricted trees + count canonical forms.

2. **Is the L1-norm bound the tightest sound bound for $L_\mu(x)$?**
   For atomic measures, yes (trivially achievable on extreme inputs).
   For density measures, possibly not — a $L_2$-norm bound paired with
   Cauchy-Schwarz might be tighter for smooth densities.

3. **How does FunctionalOp2D's L1 norm generalise?** A 2D measure on
   space-time has a natural L1 norm $\sum_{s,\tau} |\kappa(s, \tau)|$.
   Implementing this in `interval_evaluate` would close the last
   conservative-±∞ gap.

4. **Can the Pareto-front pruning be measure-aware?** Currently it
   compares loss at a fixed complexity. A measure-aware variant
   would also compare "measure complexity" — does this candidate's
   measure subsume (in some partial order) the incumbent's? If yes,
   the candidate is dominated even before evaluation.

5. **Term-rewriting for measure algebra:** can equality saturation
   (the deferred Exp 4 from `search_as_energy_min.md`) be applied to
   measure-algebra identities, not just term-rewriting? E.g.,
   $L_\mu(L_\nu(x)) = L_{\mu * \nu}(x)$ where $*$ is measure
   convolution. This is a non-trivial extension of e-graphs from
   purely-syntactic rewriting to algebraic-structure-aware rewriting.

## 7. Connection to tessera's other research directions

| Research direction | Connection to this note |
|---|---|
| Equality saturation (Exp 4, `search_as_energy_min.md`) | Could be extended with measure-algebra rules, not just term-rewriting |
| Knuth Vol 4B branch-and-bound (`fit_as_perfect_info_game.md`) | L1-norm bound is tessera's specific implementation; measure-theoretic identities are the equivalence-class structure |
| Hall of Fame (shipped) | Per-cx incumbent for measure-augmented trees works unchanged; cx is unchanged when augmenting with FunctionalOp |
| `WeightedIndicatorSum` primitive (planned) | Would extend the measure algebra with a new primitive; question of how it affects $|E_K|$ |
| FunctionalOp2D bound (deferred) | The last conservative-±∞ gap; closing it is mechanical given step (c) |

## 8. The big-picture argument

The standard SR literature has two stylistic camps:

- **Empirical** (most GP-based SR): benchmark-driven, "our algorithm
  beat theirs on SRBench"
- **Symbolic-computation** (AI Feynman, Eureqa): leverage CAS tools
  (SymPy, dimensional analysis) to attack physics problems

Neither has a *measure-theoretic* angle. The measure-theoretic
operator algebra in tessera is genuinely new vocabulary in the SR
context. It gives:

1. A richer canonical-form space (Lebesgue decomposition)
2. Closed-form lower bounds (L1 norm)
3. Tractable bilinear factorisation (Fubini)
4. A natural algebra of identities (measure convolution, etc.)

The perfect-information game framework is a *unifying lens* for these
advantages: each maps to a concrete branch-and-bound exploit.

**Practical implication**: tessera's distinct value proposition vs PySR
/ AI Feynman / Operon isn't *just* "we have functional operators."
It's "the measure-theoretic structure gives the search machinery
provably-sound bounds and canonical forms that polynomial-trig SR
can't access." That's a research-paper-shaped claim worth defending
empirically.

## 9. Future tessera work along this line

In priority order:

1. **Close the FunctionalOp2D bound** (~50 LOC). The 2D L1 norm is
   $\sum_{s,\tau} |\kappa(s, \tau)|$. Trivial to compute on a
   `Measure2D` instance. Closes the last unbounded case.

2. **Measure-aware equality saturation** — extend the deferred Exp 4
   (egg/egglog wrapper) with measure-algebra rewrite rules:
   - Commutativity of convolution: $L_{\mu * \nu} = L_{\nu * \mu}$
   - $L_\mu(L_\nu(x)) \equiv L_{\mu * \nu}(x)$
   - Linearity: $L_\mu(\alpha x + \beta y) = \alpha L_\mu(x) + \beta L_\mu(y)$

3. **Measure-restricted equivalence-class count** — extend
   `run_equivalence_class_count.py` to include LinearFunctional with
   small atomic-measure parameter sets. Empirically measure the
   collapse rate when functional operators are included.

4. **Tighter density bounds** — for measures with a density part, the
   L1 norm may not be the tightest bound. $L_2$ + Cauchy-Schwarz, or
   spectral norms via Plancherel, may give sharper inequalities.

## Changelog
- 2026-05-25: initial document. Theoretical companion to
  fit_as_perfect_info_game.md grounded in steps (a)/(b)/(c) empirical
  results.
