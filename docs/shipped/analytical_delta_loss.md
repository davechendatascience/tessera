# Research note: analytical Δloss for symbolic mutations

**Status update (2026-05-26 → `docs/shipped/`):** Moved from `docs/research/`. Sufficient-statistics polish **SHIPPED** in `src/tessera/search/sufficient_stats.py`; A/B benchmark recorded in `benchmarks/results/feynman_sufficient_stats.md`. The narrow analytical Δloss for polynomial-basis mutations is delivered; the broader research direction (Δloss for general mutation classes) remains open but is not blocking.

**Status:** ? RESEARCH. Live thinking; partial answers exist in the literature, restricted to mutation classes that don't cover general SR but DO cover useful subspaces.

**Provenance:** user (2026-05-24): *"Is there a way to do the calculus of loss impact of incremental moves? Since these are all symbols. We can calculate the analytic impact we can save compute in some way to make it utterly O(1)? Is there any research on this? I would imagine Knuth would have something similar."*

The question is sharp. Standard SR evaluates each candidate in O(N · tree_size) — a real bottleneck for large N (MNIST, IK, financial data). If we could compute `Δloss = loss(tree') − loss(tree)` from the *symbolic structure* of the mutation without re-evaluation, the cost asymmetry that drives the perfect-info-game framing (cheap mutation, expensive eval, per `fit_as_perfect_info_game.md` §4) would partially collapse.

This doc names what's possible, what isn't, and what's worth implementing.

---

## 1. Three regimes, with sharp distinctions

The user's "utterly O(1)" goal is achievable in **specific cases**, partial in some cases, and impossible in general. Three regimes:

### Regime A — incremental partial evaluation (well-known)

When a mutation changes only ONE subtree, the rest of the tree's intermediate values are unchanged. If we *cached* those intermediate values, we re-evaluate only from the mutation point upward.

For a tree of size T with a mutation at depth d touching a subtree of size s:
- Naive full re-eval: O(N · T)
- Partial re-eval: O(N · s) — typically `s ≪ T`

**Where tessera stands**: partially shipped via `tessera.expression.cache.FunctionalCache`. The current cache memoises `(measure, var_id, fill_warmup) → array` at the FunctionalOp level. It does NOT do per-node memoisation across the whole tree, nor does it have a dirty-flag propagation for mutations.

**Cost to extend**: meaningful but contained. A per-node cache with dirty-flag invalidation is ~200 LOC. Pays off mostly when tree_size ≫ subtree_size, which is the late-GP regime.

**Not the user's deeper question**: this is still O(N · s), not O(1).

### Regime B — sufficient-statistic precomputation (the actually-tractable analytical case)

For MSE specifically (which is the loss tessera uses on Feynman, IK, MNIST), the loss decomposes:

```
Δloss = loss(tree') − loss(tree)
      = (1/N) Σ (tree'(x_i) − y_i)² − (1/N) Σ (tree(x_i) − y_i)²
      = (1/N) Σ [2·(tree(x_i) − y_i)·δ(x_i) + δ(x_i)²]
```

where `δ(x) = tree'(x) − tree(x)` is the *difference function* of the mutation.

If we precompute `residual_i = tree(x_i) − y_i` (O(N) once, when we set up the current best candidate), then:

```
Δloss = (2/N) Σ residual_i · δ(x_i)   ← "first-order" term
      + (1/N) Σ δ(x_i)²                ← "curvature" term
```

The first term is the **inner product** of the residual vector and the mutation's delta. The second is the L2 norm of the delta. Both involve δ evaluated at each x_i — so in general they're O(N · subtree_size). Same as Regime A.

**The interesting case**: if δ lives in a fixed *basis* — polynomial monomials, a precomputed feature bank, or a low-dimensional functional space — we can precompute basis-feature moments and basis-residual correlations *once*, then evaluate any basis mutation in O(basis_size²) **independent of N**.

For polynomial mutations `δ(x) = Σ c_k · x^k`:
- Precompute once (O(N · max_degree)): `M_k = Σ x_i^k`, `R_k = Σ residual_i · x_i^k`
- Then `Δloss = (2/N) Σ_k c_k R_k + (1/N) Σ_{k,j} c_k c_j M_{k+j}` — O(degree²)

For mutations in any fixed feature space (e.g., a bank of basis functions `φ_k(x)`):
- Precompute `M_{k,j} = Σ φ_k(x_i) φ_j(x_i)`, `R_k = Σ residual_i φ_k(x_i)`
- Evaluate any linear-combination mutation in O(basis_size²)

This is the **sufficient-statistic regime**. It's O(1) in N (the dominant cost), O(basis_size²) per evaluation.

**Where this fits real SR work:**

- **Boosting-style additive mutations** (`tree' = tree + δ`) where δ is a small basis function: this IS gradient boosting (Friedman, *Greedy Function Approximation*, 2001). The basis is implicit; each round picks the δ that minimises Δloss locally.

- **Linear-in-parameters models** with discoverable structure (e.g., RBF regression on a fixed bank of kernels): same precomputation pattern. STLSQ (Sequential Thresholded Least Squares — the SINDy algorithm, Brunton et al. 2016) is sufficient-statistic-based.

- **Sparse linear regression** with feature selection: each "feature add" mutation evaluates in O(1) given precomputed Gram matrices. This is the LASSO/OMP family.

### Regime C — fully analytical Δloss for general symbolic mutations

The user's "utterly O(1) from symbols alone" — independent of N, basis, ANY mutation form.

**Not possible in general.** Proof sketch: an arbitrary symbolic mutation can produce a difference function δ(x) whose value at each x is a non-trivial function of x. Computing `Σ residual · δ` requires *knowing* δ at each x — which is O(N) at minimum unless δ is in a precomputed basis.

The only mutations that fit "truly O(1)" are:
- **Constant mutations** (changing a Const leaf by a delta): Δloss depends only on the residual at the leaf's evaluation positions. If the Const appears at exactly one tree position, δ(x) = const · indicator function of tree path; O(1) given precomputed residual sums.
- **Identity-preserving rewrites** (X·1 = X, X+0 = X): Δloss = 0 trivially.
- **Algebraic equivalences caught by the simplifier**: Δloss = 0 (the mutation didn't change the function semantically).

These are interesting but limited. Most SR mutations *change the function meaningfully*; they don't fit regime C.

## 2. What Knuth has on this

The user wondered if Knuth has something. Direct matches are partial:

**TAOCP Vol 4B §7.2.2 (Backtracking)**: the *Dancing Links* algorithm (Knuth, *Dancing Links*, arXiv:cs/0011047, 2000) maintains incremental state for exact-cover problems. When a column is "covered," only the affected rows are updated; restoring is O(1) per undo. This is exactly Regime A — partial re-eval via dirty-flag propagation. **The pattern is reusable for SR's incremental mutation eval.**

**TAOCP Vol 4B alpha-beta pruning** (Knuth & Moore, *An Analysis of Alpha-Beta Pruning*, 1975): the cutoff condition is "incremental relative to current bound" — we don't fully evaluate a position when its bound already exceeds the incumbent. This is **bound-based skipping**, not analytical Δloss; but it's the right intuition for "don't evaluate if you don't need to." Tessera's `prune_by_lower_bound` mode is the direct descendant.

**Outside Knuth, the right reference for Regime B** is the *fast multipole method* (Greengard & Rokhlin, *A Fast Algorithm for Particle Simulations*, J. Comput. Phys. 1987). For N-particle interactions, naive is O(N²); FMM achieves O(N) by **exploiting algebraic structure of the interaction kernel** to precompute multipole expansions. The structural analog for SR: if mutations live in a basis, precompute the basis × residual sufficient statistics, then evaluate any basis mutation cheaply.

**For Regime C (general analytical Δloss), no clean published result.** The closest is **influence functions** in statistics (Koh & Liang, *Understanding Black-box Predictions via Influence Functions*, ICML 2017) — but these compute the effect of *removing one sample* on the model fit, not the effect of changing the *model* on the loss. Adjacent but not the same.

## 3. What's been published in SR specifically

A few closely-adjacent threads:

- **STLSQ / SINDy** (Brunton et al., *Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems*, PNAS 2016): SR for ODEs via sparse regression in a precomputed basis. Each candidate's loss evaluates O(1) in N because the basis is precomputed; this is Regime B in action.

- **PySR's `optimize_constants`**: uses BFGS on the const leaves — gradient information flows analytically (Regime C-like, but only for the const subspace, not for structural mutations).

- **DSR / risk-seeking policy gradient** (Petersen et al., 2021): the policy gradient is analytical w.r.t. the policy parameters, but each *candidate evaluation* is still O(N).

- **No paper I'm aware of computes Regime-B-style structural-mutation Δloss analytically in a GP context.** The pattern exists in adjacent fields (FMM, gradient boosting, SINDy) but hasn't been ported into the SR mutation operator.

This is a research gap — and a tessera-shaped one.

## 4. What tessera could exploit

Concrete experiments, ordered by implementation cost:

### 4.1 Residual-aware mutations (cheap, immediate value)

For the GP loop, maintain `residual[i]` for the current best candidate. When proposing a mutation, prefer mutations that *fit the residual* — i.e., bias `random_tree` to generate δ(x) shapes that have high inner product with the residual.

This converts the GP into a *gradient-boosting-like* process: each generation adds a structure that reduces the current residual the most. Cost per mutation: O(N · subtree_size), same as full re-eval, but the SELECTION is now residual-aware.

Concrete: change `random_tree` to optionally accept a `residual` argument; bias the op weights toward operators correlated with the residual's signed pattern.

**Effort**: 1-2 days. **Risk**: low; falls back gracefully to the current uniform-random mutation if `residual` is None.

### 4.2 Polynomial-basis sufficient statistics (Regime B implementation)

> **Promoted to PLANNED → IN PROGRESS (2026-05-24)**: see [`docs/planned/roadmap.md`](../planned/roadmap.md) §2.3. First phase: `PolynomialMoments` foundation + tests.

For benchmarks where the target is well-approximated in a polynomial basis (most of Feynman, much of MNIST features), precompute `M_k`, `R_k` once. Mutations that ADD a polynomial term evaluate in O(degree) instead of O(N).

This requires a NEW kind of mutation operator: `add_polynomial_term(tree, x_var, coefficient, degree)`. The mutation's δ is `coefficient · x^degree`; its loss impact is `(2/N) coefficient · R_degree + (1/N) coefficient² · M_{2·degree}` — O(1) in N.

**Effort**: 3-5 days. The mutation operator itself is small; the orchestration (precompute moments at GP start; refresh residuals when the best candidate changes; integrate with the existing search loop) is the work.

### 4.3 Tighter FunctionalCache with dirty-flag propagation (Regime A polish)

Extend the existing `FunctionalCache` to per-node memoization with explicit invalidation. When a mutation modifies subtree at position p, only entries on the path from p to the root get invalidated; everything else stays valid.

**Effort**: 2 days. The path-from-p-to-root depth is typically much less than full tree size, so the amortised cost drops significantly.

**Caveat**: this only helps if the SAME tree gets evaluated MULTIPLE times with small mutations between. In the current GP, each mutation produces a brand-new tree object; subexpression sharing across the population *across* generations isn't exploited. Combined with §4.1 (residual-aware mutations), it could become meaningful.

### 4.4 Hybrid: GP for STRUCTURE, sufficient-stat for COEFFICIENTS

The cleanest exploitation of regime B: let the GP discover only the STRUCTURAL TEMPLATE of the model (which operators in which arrangement); fit COEFFICIENTS via sufficient-statistic-based least squares.

For trees of the form `Σ_k c_k · φ_k(x)` (linear-in-parameters):
- GP searches `{φ_1, ..., φ_K}` (structural)
- Coefficients `c_k` solved by `(Φᵀ Φ)⁻¹ Φᵀ y` — closed-form, O(K²) per least-squares, K³ for the invert

This is exactly what STLSQ does for SINDy. The tree-search lives at the structural level; the numerical fitting is closed-form sufficient-statistic-based.

**Effort**: 1 week. Requires significant restructuring of the GP loop; the payoff is potentially large for benchmarks where linear-in-parameters is a good prior (most physics, some image features).

## 5. The honest verdict

The user's "utterly O(1)" goal is achievable for **restricted mutation classes** but not in general. The achievable cases are:

| Mutation class | Cost per mutation | How |
|---|---|---|
| Const-leaf perturbation | O(1) given precomputed leaf evals | Track residual contribution per leaf |
| Polynomial basis add | O(degree) | Precompute moments + residual-moments |
| Linear-in-parameters add | O(K²) | Precompute Gram matrix Φᵀ Φ |
| Algebraic equivalence | 0 (no change) | Simplifier catches |
| Arbitrary structural mutation | O(N · subtree_size) lower bound | No precomputation possible without knowing the mutation's form |

The "Knuth-shaped answer" is **Regime A (Dancing Links incremental state)** + **Regime B (FMM-style precomputation)**. Tessera ships partial Regime A; Regime B is the unexplored direction.

## 6. Concrete next experiments

If the user directs:

| Item | Regime | Effort | Expected impact |
|---|---|---|---|
| Residual-aware mutation bias (§4.1) | A+ | 1-2 days | Modest; helps GP focus on uncaptured signal |
| Polynomial-basis sufficient stats (§4.2) | B | 3-5 days | Large for polynomial-friendly benchmarks (Feynman) |
| Per-node FunctionalCache with dirty-flag (§4.3) | A | 2 days | Minor; only helps repeat-evaluation workflows |
| Linear-in-parameters hybrid GP (§4.4) | B | 1 week | Large; enables SINDy-style discovery cleanly |

The cheapest test of the analytical-Δloss idea is **§4.1**: residual-aware mutations don't require precomputation, just biasing the random-tree distribution by the current residual's correlation pattern. A clean A/B test on the Feynman benchmark (with vs without residual bias) would falsify quickly.

## 7. Connection to existing notes

- `fit_as_perfect_info_game.md` §4: the cost asymmetry (cheap mutation, expensive eval) is what makes this question matter. If Regime B closes some of the gap, the asymmetry becomes 'cheap mutation, cheap-in-N eval for basis mutations, expensive eval for arbitrary mutations'.

- `benchmark_difficulty_and_climb_then_simplify.md` §3.3: the "structural lookahead" open problem is the user's earlier framing of the same question. This note RESOLVES that question's "lookahead" aspect: lookahead through ARBITRARY mutations is impossible analytically (Regime C); lookahead through BASIS mutations is O(1) given precomputation (Regime B).

- `network_sr_and_budget_allocation.md` §4: deterministic admissible search under budget — Regime B's precomputation IS an admissible heuristic for the bound check. Direct synergy with that note's framing.

## 8. The thing this note doesn't claim

Regime C — fully analytical Δloss for arbitrary structural mutations — is impossible in general by the same argument that Kolmogorov complexity is uncomputable. The user's intuition pointed at something real but bounded; the bound is exactly Regimes A and B.

For specific problem classes (Feynman polynomials, IK basis functions, MNIST features), the bound holds in our favour: Regime B is fully achievable and underexplored.

## Changelog

- 2026-05-24: initial document. Provenance: user's question about analytical Δloss for symbolic mutations. Identifies three regimes (incremental partial eval, sufficient-statistic precomputation, fully analytical). Names Knuth's Dancing Links + alpha-beta + FMM as the closest analogs. Proposes four concrete experiments ordered by implementation cost, with §4.1 (residual-aware mutations) as the cheapest test of the analytical-Δloss thread.
