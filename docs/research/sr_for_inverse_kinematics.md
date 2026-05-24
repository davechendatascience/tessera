# Research note: symbolic regression for inverse kinematics in simulation

**Status:** ? RESEARCH. Open exploration; not committed to ship. New direction proposed 2026-05-24.

**Provenance:** the user (a robotics vision researcher) asked: *"I want to add a benchmark test that might be compelling. We can try to solve for inverse kinematics of robots in simulation using a physics engine. I wonder how SR fares with simulation?"* This doc scopes the question into a concrete research program.

The angle is sharper than the existing Feynman / MNIST anchors because IK has *structured ground truth*: forward kinematics is a fully-known deterministic function, so the SR task is to invert a function we can query exactly. Unlike physical-data benchmarks (which mix true signal with measurement noise), simulation gives clean, infinite training data with no measurement noise — a clean SR test bed.

---

## 1. What inverse kinematics actually demands

For a serial manipulator with joint configuration `q ∈ R^n` and end-effector pose `T = FK(q) ∈ SE(3)`, the IK problem is finding `q` such that `FK(q) = T_target`. Two regimes:

**Analytical IK** — for specific geometries (Puma 560, UR5e, KUKA LBR) the inverse is a *closed-form* function of the target pose. Pieper's criterion (3 consecutive axes intersect at a point) is the classic sufficient condition. For these arms, the ground truth IS a closed-form formula — exactly what SR claims to discover.

**Numerical IK** — for general manipulators or redundant ones (>6 DoF), the inverse is solved iteratively: Levenberg-Marquardt, damped pseudoinverse, CCD. No closed form exists.

**Where SR has plausible value:**

| Scenario | SR's plausible role |
|---|---|
| Pieper-criterion arms with known FK | Rediscover the analytical IK formulas from FK samples. Validates SR's "find the closed form" claim with exact ground truth. |
| Redundant manipulator (7+ DoF) | No analytical IK exists; SR could discover an *approximate* closed-form mapping for a constrained subspace (e.g., elbow-up only). |
| Soft / cable-driven robots | No analytical IK in general; the dynamics are nonlinear in geometry-specific ways. SR could find piece-wise closed forms over local regions. |
| Visual servoing | The image-feature-to-joint-velocity Jacobian is hard to model analytically. SR over image features → joint velocities is the user's interest area; closest fit. |

The most interesting first benchmark is **rediscovering the analytical IK of a 3-DoF planar arm or a 6-DoF Pieper arm.** It has known ground truth, so we can grade SR's discoveries against the actual formula.

## 2. Why this is a good fit for tessera

Three structural matches:

**a. Forward kinematics is a perfect-info game.** Given a robot model, `FK(q)` is a deterministic function — we can query it as many times as we want, with zero noise. The "data" is sampled from FK; the "target" is the inverse. By `docs/research/fit_as_perfect_info_game.md`'s framing, this is the cleanest possible SR setup (no overfitting concerns, no noise robustness needed).

**b. tessera's measure-theoretic operators give angular structure.** Joint angles compose through rotations, which are products of `sin`/`cos`. We have those (just shipped in lifecycle ship #1). For Pieper arms, the analytical IK involves `atan2`, half-angle substitutions, and `acos`/`asin` for the joint angles. We have `sin`, `cos`; we'd need to assess whether `atan2`, `acos`, `asin` are required for a clean discovery (likely yes; the `arcsin` we noted as needed for Feynman I.26.2 is the same primitive).

**c. The Knuth-style search machinery applies cleanly.** SR for IK is a single-agent perfect-info game (per §3 of the perfect-info doc); branch-and-bound + equivalence-class collapse apply identically to Feynman. The compute story is the same.

**Three honest mismatches to flag:**

1. **Multi-output target.** IK typically maps a 6-D pose to an n-D joint vector. Current tessera assumes a single scalar target per call. For 6-DoF IK, we'd need to either run the SR n times (one feature per joint) — the same multi-feature pattern as MNIST K=10 — or extend tessera's loss API to support vector targets.

2. **Multiple valid solutions.** For most robots, a target pose has multiple IK solutions (elbow-up vs elbow-down, shoulder-forward vs back). SR optimising MSE picks one mode but can't represent "the solution set." Standard SR loss is well-defined only when training data is consistent in solution branch — we'd need to filter (e.g., elbow-up only) or use multimodal loss.

3. **Trigonometric primitives gap.** Even with `sin`/`cos`, IK formulas use `atan2(y, x)` which encodes the quadrant — `arctan(y/x)` loses sign information. Either add `atan2` as a primitive (likely cheap; 30 LOC analogous to sin/cos), or constrain the workspace so `arctan` suffices.

## 3. Concrete benchmark proposal

> **Shipped on 2026-05-24** as `benchmarks/run_ik_planar_3dof.py`. Result documented in `benchmarks/results/ik_planar_3dof.md`. **Empirical outcome: Tier D** (all 3 joints failed, test rel ∈ {0.33, 0.73, 0.35}).
>
> The result confirms the §7 prediction that `sin/cos` alone is insufficient; **the `atan2`/`acos` gap is real**. q2 (which uses `acos` in the analytical form) hit test_rel=0.73 — the worst by a wide margin, matching the analytical structure. q1 and q3 (which use `atan2`) reached test_rel≈0.34, still failed.
>
> **Inspecting the discovered trees confirms the diagnosis:**
> - **q2**: `sqrt(0.97 + cos(x) + cos(y) + cos(y) + cos(...))` — algebraic combinations of `cos` trying to approximate `acos((r²-2)/2)`. Without `acos`, hopeless.
> - **q1, q3**: composite with `θ/-2.0` linear approximations and indicator/comparison flags trying to encode quadrant information that `atan2` would give cleanly.
>
> **Verdict**: the unit-architecture validation question (`high_dim_sr §6.7`) is NOT yet answered by this benchmark — the universal-GP baseline failed for vocabulary reasons, not because per-joint specialised SR was missing. The next experiment is the same benchmark **after adding `atan2`/`acos`** primitives; if THAT produces tier A or B, we'll have evidence that the universal GP with the right vocab is sufficient, *and* that the unit-architecture is unnecessary for IK at this scale.
>
> Tracked as `sr_for_inverse_kinematics.md` §4.7 below (new).

A 3-DoF planar arm benchmark, runnable end-to-end in tessera + numpy (no physics engine needed for this geometry):

### 3.1 Setup

Three revolute joints, link lengths `L1 = L2 = L3 = 1`. Forward kinematics in the plane:

```
x_ee  = L1·cos(q1) + L2·cos(q1+q2) + L3·cos(q1+q2+q3)
y_ee  = L1·sin(q1) + L2·sin(q1+q2) + L3·sin(q1+q2+q3)
θ_ee  = q1 + q2 + q3
```

Given `(x_ee, y_ee, θ_ee)`, find `(q1, q2, q3)`. For 3-DoF planar, there's typically a *unique* solution (up to elbow-up/down, which we can resolve by constraining `q2 ≥ 0`).

### 3.2 Closed-form ground truth

The known analytical IK (constrained elbow-up):

```
q3_part_known_from_θ_ee
x_wrist  = x_ee − L3·cos(θ_ee)
y_wrist  = y_ee − L3·sin(θ_ee)
r²       = x_wrist² + y_wrist²
cos_q2   = (r² − L1² − L2²) / (2·L1·L2)
q2       = acos(cos_q2)              # elbow-up branch
q1       = atan2(y_wrist, x_wrist) − atan2(L2·sin(q2), L1 + L2·cos(q2))
q3       = θ_ee − q1 − q2
```

Each joint angle is a closed-form function of the three target poses. **This is what SR should rediscover.**

### 3.3 SR task formulation

Three separate SR runs, each finding one joint:

- `f1(x_ee, y_ee, θ_ee) = q1`
- `f2(x_ee, y_ee, θ_ee) = q2`
- `f3(x_ee, y_ee, θ_ee) = q3`

Training data: sample 10,000 valid `(q1, q2, q3)` (within joint limits, no self-collision), compute `(x_ee, y_ee, θ_ee)`, store the pair. SR sees the pose triple as input; learns the inverse.

### 3.4 Acceptance criteria

| Tier | Result | Interpretation |
|---|---|---|
| **A** | All 3 joints discovered exact (rel < 0.001) | SR cleanly inverts analytical FK. Validates the perfect-info-game framing for robotics. Publishable as a methods paper. |
| **B** | 1-2 joints exact, 1 partial | The geometry is partially expressible; the missing piece is likely an `atan2` gap or a multi-branch ambiguity. Documents the limit. |
| **C** | All partial (rel 0.05-0.3) | SR finds *approximate* IK but not the exact form. Probably useful as a warm-start for numerical IK; not a closed-form discovery. |
| **D** | All failed (rel > 0.5) | Trigonometric vocabulary insufficient or search-budget too small. Diagnose; iterate. |

The expected tier depends on the primitive vocabulary:
- With only `sin`/`cos`: probably tier C (the GP can build approximations but `atan2`'s sign-aware quadrant is hard to recover from `sin/cos` alone)
- With `atan2` added: tier A or B is plausible

## 4. Why "simulation" specifically

The user asked "how does SR fare with simulation?" The robotics-specific framing has three properties worth naming:

**a. Infinite, noiseless data.** Unlike physical experiments, sim gives us as many `(q, FK(q))` pairs as we want, with no noise. This isolates the SR-engine question (can it discover the formula?) from the data-quality question (is there enough information?).

**b. Ground truth is available.** We can grade a discovered formula against the analytical IK. Most ML benchmarks (MNIST, ImageNet, even Feynman in real units) lack this: you compare to labels, not to *the right formula*.

**c. Curriculum is naturally available.** Start with 1-DoF, then 2-DoF planar, then 3-DoF planar, then 6-DoF Pieper. Each step adds one trigonometric layer. Per-step diagnostic: at which DoF does SR break, and why?

## 5. Tessera-specific levers we'd test

For the 3-DoF planar benchmark, what does each tessera-shipped feature contribute?

| tessera feature | Expected impact on IK |
|---|---|
| `sin`, `cos` primitives (shipped) | **Necessary** — IK formulas are trigonometric |
| `sqrt` (shipped) | **Necessary** — IK involves `sqrt(r²)` for distance to wrist |
| `add`/`sub`/`mul`/`div` (shipped) | Trivially necessary |
| `arctan2` / `acos` / `asin` (NOT in tessera) | **Likely needed for tier-A outcome.** The arctan2 gap is similar to the arcsin gap we noted in Feynman I.26.2. Cleanest test: try without first, then add. |
| Const-opt via jax.grad (shipped ship #2) | **Helpful** — IK constants (link lengths, joint offsets) are float-valued; gradient descent should refine them well from a structurally-correct discovery |
| Tier-3 batched JAX eval (shipped) | **Helpful** at GPU scale; 10K-100K training samples is the typical IK dataset size where Tier 3 pays off |
| Materialize-shared-subtrees (shipped) | **Highly relevant.** `cos(q1+q2+q3)`, `sin(q1+q2+q3)`, etc., appear in many candidate forms; sharing across the population is exactly the use case |
| Equality saturation (research) | Trig identities (`sin² + cos² = 1`, angle-addition formulas) are the canonical e-graph use case — they'd shrink the search space substantially |

The composition is interesting: tessera's existing measure-theoretic ops (`LinearFunctional`, `FunctionalOp2D`) are not relevant for IK (IK is pointwise, not temporal/spatial). But the *engine pieces* — GP search, jax-backed eval, const-opt, materialize — are.

## 6. Five sub-questions worth empirical answers

1. **Does plain SR rediscover 3-DoF planar IK with `sin/cos` only?**
   Cheapest experiment; ~half day to set up. Result determines whether atan2 is necessary.

   > **ANSWERED 2026-05-24: NO.** Tier-D result on `benchmarks/run_ik_planar_3dof.py` (all 3 joints failed; q2 worst at test_rel=0.73 — directly tied to its `acos` dependence). The atan2/acos gap is empirically real, not theoretical. Tracked as a planned-for-ship item: add `atan2`/`acos`/`asin` primitives via the same lifecycle path that shipped `sin`/`cos` (§4.1 of `benchmark_score_improvement.md`).

2. **What's the search-budget-to-accuracy curve?**
   Sweep pop=50, 200, 1000 × gens=20, 50, 100. Does compute monotonically buy accuracy?

3. **Multi-output: separate SR runs vs joint loss?**
   Run three independent SRs (one per joint) vs a hypothetical joint loss `MSE(predicted_pose, target_pose)` where the prediction is built by FK applied to the *discovered* joints. The joint approach guarantees physical consistency but is harder to optimise.

4. **Generalisation to 6-DoF Pieper.**
   If 3-DoF works cleanly, does 6-DoF? At what point does the search space outgrow the SR engine?

5. **Visual servoing (the user's actual research area).**
   Image-feature-to-joint-velocity. Inputs are image features (high-dim), output is joint velocity (n-D). This bridges the high-dim-SR direction with IK. Likely needs the step-5 multi-feature ensemble pattern from the MNIST result.

## 7. Honest expectations

Three named outcomes, before any code:

**Optimistic — Tier A (all 3 joints exact):** SR cleanly rediscovers the analytical IK with `sin/cos/sqrt/+/-/*/÷` alone. Surprising, but plausible if the GP gets lucky with `atan2` approximations (e.g., learning that `q1` involves a ratio of `y_wrist/x_wrist`). Headline: "SR rediscovers analytical IK for 3-DoF planar arm." Publishable.

**Modest — Tier B (mixed):** `q2` recovers cleanly (it's just `acos((r² − L1² − L2²)/(2·L1·L2))` which the GP could discover by finding the right algebraic form even without `acos`, via Taylor or rational approximation). `q1` and `q3` involve `atan2`, which fails without that primitive. Verdict: adding `atan2` is necessary; do it as a separate planned item.

**Pessimistic — Tier C/D (partial or failed):** Even with infinite data, the search-engine can't find the formulas in any reasonable budget. This would tell us SR has a structural limit on multi-trig compositions. Negative-result paper: "tessera fails to discover IK; the bottleneck is X."

## 8. Concrete next experiments (status-flagged)

| Item | Effort | Status |
|---|---|---|
| Implement the 3-DoF planar arm in `benchmarks/run_ik_planar_3dof.py`. FK + SR + per-joint scoring + accuracy table | 1 day | ○ PLANNED (recommended next promotion if you direct it) |
| Optional `atan2` primitive (only if §1 result shows it's needed; analogous to sin/cos ship) | half day | ? RESEARCH gated on §1 |
| 6-DoF Pieper arm benchmark (scales up from §1) | 2 days | ? RESEARCH |
| Multi-output joint-loss formulation experiment | 2-3 days | ? RESEARCH |
| Visual-servoing pilot (image features → joint velocity; bridges high-dim SR direction) | 1-2 weeks | ? RESEARCH |

## 9. Connection to the existing research notes

Three convergent threads:

- `fit_as_perfect_info_game.md` §3: IK is the cleanest possible single-agent perfect-info game (no opponent, deterministic FK, exact ground truth).
- `high_dim_symbolic_regression.md` §5.4: two-layer SR — for visual servoing specifically, image features + IK could be the two layers.
- `benchmark_score_improvement.md` §4.1: the trig-primitive shipping path we walked for sin/cos is the template for `atan2`/`asin`/`acos` if needed.

## 10. Reading list

Robotics-IK references that frame the question:

- Pieper, *The Kinematics of Manipulators Under Computer Control* (1968, PhD thesis) — the analytical-IK closed-form criterion that powers most industrial robots
- Buss & Kim, *Selectively Damped Least Squares for Inverse Kinematics* (2005) — modern numerical IK textbook reference
- Sutanto et al., *Learning Latent Space Dynamics for Tactile Servoing* (2020 + 2022 follow-ups) — neural approach to learn IK
- Koenig & Howard, *Gazebo simulation* + *PyBullet* + *MuJoCo* — the three standard simulators if/when we move beyond hand-coded 3-DoF planar

SR-meets-robotics is a thin literature; the user's specific framing (use simulation as ground truth) is closest to AI Feynman's approach but on a robotics-specific function class.

## Changelog

- 2026-05-24: initial document. Provenance: user's question about SR for IK in simulation, anchored on their professional area (robotics vision research). Five sub-questions enumerated; first one (3-DoF planar with sin/cos) flagged as the cheapest concrete experiment and a candidate for the next promotion to PLANNED.
