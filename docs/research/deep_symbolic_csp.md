# Deep symbolic csp: stacking fails, decomposition is the way

**One line:** A gradient-free *deep symbolic network* built by stacking
`csp_sr` layers (`discover_deep` / `discover_boosted`) does **not** robustly
reach deep compositional structure — it overfits catastrophically and never
beats a single deeper enumeration. The wall is broken by **decomposition
(top-down), not stacking (bottom-up).** This note records the negative
result, *why* it happens, and the recommended path.

Status: `discover_deep` / `discover_boosted` implemented in
`src/tessera/experimental/csp_sr.py`; benchmark `benchmarks/run_deep_csp.py`
(results `benchmarks/results/deep_csp.md`). Recommended path NOT yet built.
Date: 2026-05-30.

---

## 1. What was built

`discover_deep(env, y, depth, width, cfg, lr, dense)` — a DNN-shaped
symbolic net, gradient-free:
- layer 0 = the inputs; each later layer has `width` nodes;
- every node is a free-form `csp_sr` expression over a pool of prior nodes,
  fit to the **running residual** (symbolic gradient boosting — each hidden
  node gets a target, so no backprop / no credit-assignment-by-gradient);
- `bounded` connectivity (default): a layer reads the previous layer +
  inputs (skip), pool stays `O(n_inputs + width)`, per-layer dictionary is
  constant-size, cost is depth-linear. A deep node still depends on every
  earlier node *transitively, through composition* `f(g(h(x)))` — depth
  replaces flat connectivity, which is what keeps the search space bounded.
- `dense` connectivity: a layer reads ALL prior nodes; pool grows by
  `width`/layer, dictionary grows polynomially → the search re-explodes.

The combined model composes to one self-contained tessera `Expr`.
`discover_deep(width=1, dense=True)` ≈ the older `discover_boosted`.

The motivation was a user request: "a DNN where each node's input is an
arbitrary symbolic combination of all nodes from previous layers." The
bounded form is the tractable realization of that idea.

## 2. The negative result (held-out)

`run_deep_csp.py`, four noise-free analytic targets, scored on a held-out
1/3 split (depth=6, width=3, vocab `[neg,sqrt,tanh,add,sub,mul]`):

| target | regime | single (size2) | single (size5) | deep lr=1.0 | deep lr=0.3 |
|---|---|---|---|---|---|
| `x0x1+x2x3` | shallow | **1.000** | 1.000 | 1.0 | 1.000 |
| `(x0x1+x2x3)^2` | deep/tractable | 0.394 | **1.000** | −1.5e6 | −1.045 |
| `sqrt((x0x1+x2x3)^2+(x4x5+x6x7)^2)` | deep/intractable | 0.398 | **0.429** | −383 | 0.291 |
| `sqrt((x0x1x2)^2+(x3x4x5)^2)` | deep/intractable | 0.725 | 0.925 | 0.887 | **0.936** |

- **Train R² lies.** At `lr=1.0`, depth drives *train* R² to ~0.95 while
  *held-out* R² is −1.5e6 (in-distribution, on noise-free data).
- **Shrinkage doesn't reliably fix it.** `lr=0.3` is unstable and
  data-dependent (`(x0x1+x2x3)^2` → −1.0 on one draw, +0.82 on another) and
  non-monotonic in `lr`.
- **Single-layer wins 3/4.** Deep wins once (`norm_triple6`, 0.936 vs 0.925)
  — within the instability noise. No regime where stacking robustly wins.

## 3. Why stacking fails (diagnosis)

Confirmed by an independent review of the engine. Three coupled causes:

**(a) Boosting solves the wrong problem.** Gradient boosting reaches the
*additive closure* `f ≈ Σ_L h_L`. The wall targets are *compositional*
(`sqrt(1−v²/c²)`, `sqrt((·)²+(·)²)`), reachable additively only via a
high-order series each finite layer approximates with high-variance terms.
The feature-augmentation trick tries to recover composition, but each layer
fits its node to the **raw residual**, not to "the subexpression the outer
function needs" — there is no reason the residual-optimal intermediate
equals the `v²/c²` the `sqrt` wants inside. **Additive credit is
structurally mismatched to compositional targets.**

**(b) The −1.5e6 blow-up is a conditioning pathology, not classic
variance.** Augmenting with `h0 = f̂(x)` and letting a later layer build
`mul(h0,h0)` makes dictionary columns that are smooth nonlinear functions of
a *fitted* quantity. Off-train, `h0` is slightly wrong and `h0²` amplifies
that error super-linearly; the near-collinear high-variance columns drive
least-squares to large opposite-sign coefficients that cancel on train and
diverge off-train. This is the SINDy/STLSQ ill-conditioning failure,
*manufactured* by augmentation. Compounding it: `_beam_search` ranks by
**train R²** with no coefficient-norm / condition-number / held-out guard,
so it actively prefers the exploding solution. Shrinkage is implicit ridge
at the wrong layer — hence the non-monotonicity.

**(c) Stacking is strictly less expressive per unit of credit than a single
enumeration.** Single-layer searches all const-free trees of a size with a
*clean* target; the stack searches a constrained compositional family with a
*shifting, noisy* residual target. Below the enumeration cap the stack is
dominated; above it, its compositional reach is real but its credit signal
is too weak to exploit. No-win is structural, not bad luck.

**Verdict: deep structure requires decomposition (top-down — split the
target into sub-targets with *known* identities), not stacking (bottom-up —
hope useful intermediates emerge from residual-fitting).** Stop investing in
`discover_deep`/`discover_boosted` as a route to depth.

## 4. Recommended path (decomposition + bounded const-refine)

`csp_sr` is an excellent *shallow* solver (it exactly recovers size-≤4
forms). The win is to wrap it in a top-down decomposition driver, reusing
machinery that already exists in the tree but is **not** wired into
`csp_sr` today:
- `coordinate_discovery.TARGET_TRANSFORMS` / `INVERSE_TRANSFORMS` — an
  outer-op peeler (`sqrt`, `square`, `log`, `inv`, …).
- `tessera/search/decompose.py:detect_power_law` — multiplicative test.
- `diff_sr.make_core_eval` + `make_refiner` — batched, vmapped Adam
  constant-refinement over a **fixed** opcode structure (NOT diff_eml:
  structure stays discrete/enumerated; gradients tune ≤2 scalars).
- `additive_polynomial.py` (monomial OLS), `iter_subtrees` (subtree mining).

Ranked plan:

1. **Conditioning guard — do this first (~50 LOC, helps everything).** In
   `_beam_search`/`_omp`: ridge-regularize the least-squares, reject
   candidate subsets whose Gram condition number / coefficient norm explodes,
   and select on **held-out** R² rather than train. Success: the −1.5e6
   regime becomes impossible (worst-case held-out R² ≥ 0) with no `lr` knob.

2. **A — separability-directed divide & conquer (highest payoff).** Detect
   additive / multiplicative separability via *finite differences on the
   data* (mixed second difference ≈ 0 ⇒ split; same on `log|y|` ⇒ product).
   When no clean split exists, *peel an outer unary wrapper* (reuse
   `TARGET_TRANSFORMS`) and recurse. Recurse `discover()` on the low-var
   leaves; reassemble with `BinOp`/`UnOp`/`_substitute`. Success criterion:
   recover `sqrt((x0x1+x2x3)²+(x4x5+x6x7)²)` to **held-out R² > 0.9999**
   (square-peel → additive split → two size-3 csp calls → sqrt-wrap), with no
   `lr` and no catastrophic regime. ~250–350 LOC, reuses 3 modules.

3. **D — skeleton enumeration + minimal const-refine (cheapest real win).**
   Promote the existing 1-template `_FREE_CONST` refine to the main path:
   enumerate const-free skeletons with ≤2 free-constant placeholders, refine
   them with the *existing* batched `diff_sr` refiner. Closes the
   embedded-constant half of the Feynman wall (`sqrt(1−v²/c²)` via θ→−1/c²,
   `exp(−x²/2)`, `sin(2.3x)`). Success: `sqrt(1−v²/c²)` to rel < 1e-8 on
   held-out, one refined `Const`. ~200 LOC of glue (infra is done). This is
   the **minimal, defensible** gradient signal — structure stays enumerated;
   it does not reintroduce differentiable structure search.

4. **B — subexpression promotion (poor-man's ADFs).** Mine high-value
   const-free subexprs, promote them to atomic leaves, re-enumerate at the
   same `max_size` (effective depth shrinks). Composes with A. Const-free
   blocks evaluate identically on train/test, so this *cannot* cause the
   augmentation overfit. ~150–200 LOC.

5. **C — best-first / A* guided enumeration (research bet, last).** For a
   genuinely non-decomposable hard core: replace breadth-first-by-size with
   promise-ordered search (residual-correlation heuristic). Caveat: the
   heuristic is not admissible (jointly-predictive-but-marginally-uncorrelated
   terms — the Lorenz-`y` lesson), so keep it soft/beam-like with partial-
   correlation diversity injections.

**Caution:** A/B/C lean on finite-difference structure tests; razor-sharp on
noise-free targets, noise-sensitive on real data. Set tolerances from a
bootstrap of `y`'s noise floor and prefer held-out statistics, or the same
overfit returns in subtler form.

## 5. Relation to prior work

Continues `differentiable_eml_jax.md` (the diff_eml / relaxation arc) and
[[project_csp_sr]]. The decomposition program is the AI-Feynman insight
(its edge is *pre-search decomposition*, not search) ported to a CSP
dictionary + sparse-linear leaf solver, with the bounded const-refine from
`diff_sr` as the only — and minimal — gradient signal.
