# 3-DoF planar IK benchmark

**Status:** first tessera robotics benchmark (per `docs/research/sr_for_inverse_kinematics.md` §3).

**Setup:** 3 revolute joints, L1 = L2 = L3 = 1; q1 ∈ (-3.141592653589793, 3.141592653589793), q2 ∈ (0.0, 3.141592653589793) (elbow-up), q3 ∈ (-3.141592653589793, 3.141592653589793).

**Train:** 4000 samples · **Test:** 1000 samples · **Seed:** 2026

**GP config:** pop=300, gens=60, pointwise_only=True, trig + sqrt + jax.grad const-opt available.

## Headline

**Tier D** — D — failed (trig vocab insufficient or budget too small)

- **Exact** (test rel < 0.01): 0 / 3
- **Partial** (0.01 ≤ rel < 0.1): 0 / 3
- **Failed** (rel ≥ 0.1): 3 / 3

## Per-joint results

| Joint | cx | Train rel | Test rel | Runtime (s) |
|---|---|---|---|---|
| q1 | 24 | 0.3449 | 0.3375 | 16.8 |
| q2 | 16 | 0.7626 | 0.7794 | 5.7 |
| q3 | 26 | 0.3087 | 0.2981 | 7.2 |

## Discovered formulas

### q1
- cx=24, train_rel=0.3449, test_rel=0.3375
  ```
  ((0.35845 * (y - (((th < y) < (th * x)) - th))) - (cos(max(th, (1.07559 - min(x, y)))) <= x))
  ```

### q2
- cx=16, train_rel=0.7626, test_rel=0.7794
  ```
  ((0.893364 + cos(pow(((0.147459 * x) + (x * y)), 0.498169))) - (-0.0639149 * th))
  ```

### q3
- cx=26, train_rel=0.3087, test_rel=0.2981
  ```
  ((0.163792 * th) + (tanh(((th * exp(x)) - ((tanh(th) - (y + y)) < th))) - (y >= (th - (-0.0554804 <= y)))))
  ```

## Analytical IK (reference)

```
x_w  = x_ee − cos(θ_ee)
y_w  = y_ee − sin(θ_ee)
r²   = x_w² + y_w²
q2   = acos((r² − 2) / 2)            # elbow-up branch
q1   = atan2(y_w, x_w) − atan2(sin(q2), 1 + cos(q2))
q3   = θ_ee − q1 − q2
```

Note: the analytical form uses `atan2` and `acos`. As of 2026-05-24
tessera ships these primitives (see roadmap.md §2.3 "Recently
shipped"). The benchmark above ran WITH them available.

## Empirical run history

| Run | Vocab | Best results | Tier |
|---|---|---|---|
| 1 (2026-05-24) | sin/cos/sqrt only | q1=0.33, q2=0.73, q3=0.35 | D |
| 2 (2026-05-24, this file) | + atan2/acos/asin | q1=0.34, q2=0.78, q3=0.30 | D |

**Critical finding:** adding the right primitives DID NOT move the
tier. Inspecting the discovered trees (above): they don't USE atan2,
acos, or asin. The GP found compositions of the existing operators
(cos, pow, tanh, exp, comparisons) just like in run 1.

**Diagnosis: search-space-explosion, not vocabulary.**

With ~30 ops in the alphabet, uniform random sampling gives each new
op ~3% probability per tree slot. Composing the IK formula
(e.g., `atan2(y_w, x_w) - atan2(sin(q2), 1 + cos(q2))` for q1) requires
*two* specific `atan2` calls in a ~10-node sub-tree. Random-walk
probability of this composition is roughly `(0.03)² ≈ 0.1%` per tree.
With pop=300 × gens=60 = ~18,000 trees seen, expected number of trees
containing the right `atan2(y, x) - atan2(...)` skeleton is ~18 —
before considering the rest of the structure has to be correct too.

So the GP "had access" to the right primitives but never tried them
in the right composition. This is exactly the "combinatorial
explosion in tree space" failure mode catalogued in
`high_dim_symbolic_regression.md` §3 (theoretical analysis #1).

## Interpretation (revised after run 2)

Tier A = the unit-architecture validation: per-joint SR works
Tier B = atan2 gap is real and needs to be addressed
Tier C = SR finds approximations but not the exact form
Tier D = **EITHER trig vocab insufficient OR search-space explosion**

Run 1 was tier D for vocab reasons. Run 2 is tier D for the **second**
reason — the search-space-explosion category. This is a structurally
different failure mode and points to a different next step.

**Two candidate fixes**, in increasing order of investment:

1. **Op-weight scheduling**: bias `OP_WEIGHTS` toward the new ops
   during the first ~30% of generations, then anneal toward uniform.
   Forces exploration through atan2/acos before parsimony pressure
   discards them. ~30 LOC; analogous to PySR's annealed mutation
   temperature (roadmap.md §1.2, already planned).

2. **Template-based mutations**: explicit `template_atan2_composition`
   mutation that wraps two existing subtrees as `atan2(a, b)`. This
   is `benchmark_score_improvement.md §4.2` "Structural-template
   mutations" — was research-only; this benchmark result promotes it
   in priority.

Or a higher-budget run (pop=1000, gens=200) — but at ~30× the compute
of this run, that's a different kind of investment.

## See also

- `docs/research/sr_for_inverse_kinematics.md` — full research note
- `docs/research/high_dim_symbolic_regression.md` §6 — unit-architecture framing
- `docs/planned/roadmap.md` §2.3 — this benchmark's promotion entry