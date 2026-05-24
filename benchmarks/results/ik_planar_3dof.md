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
| q1 | 27 | 0.3277 | 0.3345 | 7.9 |
| q2 | 20 | 0.6929 | 0.7310 | 4.3 |
| q3 | 23 | 0.3590 | 0.3504 | 6.1 |

## Discovered formulas

### q1
- cx=27, train_rel=0.3277, test_rel=0.3345
  ```
  (((th <= y) + (((0.39335 / sign(y)) - (th / -2.03344)) - (x >= (0.351783 * (y - 2.81382))))) - (min(x, y) >= cos(th)))
  ```

### q2
- cx=20, train_rel=0.6929, test_rel=0.7310
  ```
  sqrt(((((0.974465 + cos(x)) + cos(y)) + cos(y)) + cos(pow(max(th, x), (-0.061499 < x)))))
  ```

### q3
- cx=23, train_rel=0.3590, test_rel=0.3504
  ```
  (((x / (1.15325 <= th)) + ((((0.531127 - (th - y)) < 0.000106857) - 0.943976) - (th / -2.62233))) - (0.168294 * x))
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

Note: the analytical form uses `atan2` and `acos`, which
tessera does NOT currently have. Per `sr_for_inverse_kinematics.md` §7's predictions:
  - q3 = θ − q1 − q2 (pure subtraction; SR should find this)
  - q2 involves `acos`; SR may approximate via algebraic forms
  - q1 involves two `atan2` calls; without that primitive,     expect partial-fit at best

## Interpretation

Tier A = the unit-architecture validation: per-joint SR works.
Tier B = the atan2 gap is real and needs to be addressed (next
         iteration: add atan2 as a primitive).
Tier C = SR finds approximations but not the exact form.
Tier D = trig vocabulary or search budget is insufficient.

## See also

- `docs/research/sr_for_inverse_kinematics.md` — full research note
- `docs/research/high_dim_symbolic_regression.md` §6 — unit-architecture framing
- `docs/planned/roadmap.md` §2.3 — this benchmark's promotion entry