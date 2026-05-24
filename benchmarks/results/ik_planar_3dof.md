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
| q1 | 25 | 0.3392 | 0.3539 | 11.3 |
| q2 | 32 | 0.8276 | 0.8345 | 5.9 |
| q3 | 21 | 0.3333 | 0.3271 | 8.8 |

## Discovered formulas

### q1
- cx=25, train_rel=0.3392, test_rel=0.3539
  ```
  (-0.347337 + min(pow(0.548289, x), atan2(((th + (y - (th > y))) - (((-0.0388495 + th) * pow(2.19286, x)) > y)), x)))
  ```

### q2
- cx=32, train_rel=0.8276, test_rel=0.8345
  ```
  (1.17406 + ((pow(x, th) + (((-0.044717 + th) / (th - 0.502743)) <= y)) < (th - pow((y > atan2(reduce_sum(th), (y <= 2.02515))), sign(abs((y > x)))))))
  ```

### q3
- cx=21, train_rel=0.3333, test_rel=0.3271
  ```
  (((y <= th) + atan2(((th + x) - 2.36356), 0.963209)) - (min((1.21211 + x), (x + y)) >= th))
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

Note: as of 2026-05-24 tessera ships `atan2`/`acos`/`asin` primitives.
The benchmark above ran WITH them available + climb-then-anneal
parsimony schedule active (low parsimony in early gens to allow
high-cx exploration; anneal to normal pressure).

## Empirical run history (three runs as of 2026-05-24)

| Run | Vocab | Parsimony | Best result | Tier |
|---|---|---|---|---|
| 1 | sin/cos/sqrt only | static 0.005 | q1=0.33, q2=0.73, q3=0.35 | D |
| 2 | + atan2/acos/asin | static 0.005 | q1=0.34, q2=0.78, q3=0.30 | D |
| 3 | + climb-then-anneal | sched: 0.0001→0.005 | q1=0.35, q2=0.83, q3=0.33 | D |

**Nuanced verdict on Run 3 (the schedule did work, but the score
didn't move):**

- **The "vocab present but unused" failure mode IS partially fixed.**
  Run 2 trees used NO atan2 anywhere. Run 3 trees show atan2 in
  q1 (`atan2(((th + (y - (th > y))) - ...), x)`) and q3
  (`atan2(((th + x) - 2.36356), 0.963209)`). The schedule's
  intended effect — exploring atan2 compositions during the
  low-parsimony climb phase — is empirically working.
- **Score didn't improve.** q1 went 0.34 → 0.35; q3 went 0.30 → 0.33.
  q2 actually REGRESSED 0.78 → 0.83 with cx jumping 20 → 32.
  The GP is now using atan2 in the WRONG composition; the anneal
  phase didn't drive it toward the right structure.
- **q2 still no `acos`** — the schedule alone can't make the GP
  try a structurally-distinct subtree replacement.

**Diagnostic split (failure-mode taxonomy):**

| Mode | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Vocab gap | D | (closed) | (closed) |
| Vocab unused | (gap unproven) | D | (closed) |
| Wrong composition with right vocab | n/a | n/a | **D (new mode)** |

Three runs, three failure modes, three different root causes.
The "right composition" mode is what's blocking now — the GP can
use atan2 but doesn't compose it as `atan2(y_w, x_w) - atan2(sin(q2),
1+cos(q2))`. That's the EXACT shape benchmark_score_improvement.md
§4.2 (template mutations) was designed to target.

## Interpretation

Tier A = the unit-architecture validation: per-joint SR works.
Tier B = atan2 gap is real and needs to be addressed.
Tier C = SR finds approximations but not the exact form.
Tier D = could be ANY of: trig vocab, vocab-unused, or wrong-composition.
         The discovered trees + run history disambiguate WHICH Tier-D mode.

After Run 3, the remaining failure mode is "wrong composition with right
vocab." The next experiment is **template-based mutations** that wrap
existing subtrees as `atan2(a, b)` / `acos(...)` directly — forces the
GP to try the structurally-correct shapes, not just include the ops
randomly.

## See also

- `docs/research/sr_for_inverse_kinematics.md` — full research note
- `docs/research/high_dim_symbolic_regression.md` §6 — unit-architecture framing
- `docs/planned/roadmap.md` §2.3 — this benchmark's promotion entry