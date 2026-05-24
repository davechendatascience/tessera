"""3-DoF planar arm inverse kinematics — first robotics benchmark for tessera.

Per `docs/research/sr_for_inverse_kinematics.md` §3 (now ▷ IN PROGRESS
via `docs/planned/roadmap.md` §2.3). This is the first concrete
validation of the unit-architecture question raised in
`docs/research/high_dim_symbolic_regression.md` §6.

Setup
-----
Three revolute joints, all link lengths = 1. Forward kinematics:

    x_ee  = cos(q1) + cos(q1+q2) + cos(q1+q2+q3)
    y_ee  = sin(q1) + sin(q1+q2) + sin(q1+q2+q3)
    θ_ee  = q1 + q2 + q3

Given the target pose (x_ee, y_ee, θ_ee), find the joint angles.

Closed-form analytical IK (elbow-up branch):
    x_w  = x_ee − cos(θ_ee)             # wrist position
    y_w  = y_ee − sin(θ_ee)
    r²   = x_w² + y_w²
    cos_q2 = (r² − 2) / 2                # since L1=L2=L3=1
    q2   = acos(cos_q2)                  # elbow-up
    q1   = atan2(y_w, x_w) − atan2(sin(q2), 1 + cos(q2))
    q3   = θ_ee − q1 − q2

Three independent SR runs, one per joint angle. Acceptance tiered
A→D as documented in `sr_for_inverse_kinematics.md` §3.4.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig, climb_then_anneal_parsimony
from tessera.expression.tree import complexity


OUT_DIR = Path(__file__).parent / "results"
OUT_REPORT = OUT_DIR / "ik_planar_3dof.md"

# Constants
L1 = L2 = L3 = 1.0
N_TRAIN = 4000
N_TEST = 1000
SEED = 2026

# Joint limits (radians). Elbow-up constraint: q2 ∈ [0, π].
# q1 ∈ [-π, π], q3 ∈ [-π, π].
Q1_LIMITS = (-np.pi, np.pi)
Q2_LIMITS = (0.0, np.pi)
Q3_LIMITS = (-np.pi, np.pi)


def sample_workspace(n: int, seed: int):
    """Sample n valid (q1, q2, q3) triples within joint limits, compute
    the corresponding (x_ee, y_ee, θ_ee). Returns the data matrix and
    target joint array."""
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(*Q1_LIMITS, n).astype(np.float32)
    q2 = rng.uniform(*Q2_LIMITS, n).astype(np.float32)
    q3 = rng.uniform(*Q3_LIMITS, n).astype(np.float32)

    # Forward kinematics
    s1 = np.sin(q1); c1 = np.cos(q1)
    s12 = np.sin(q1 + q2); c12 = np.cos(q1 + q2)
    s123 = np.sin(q1 + q2 + q3); c123 = np.cos(q1 + q2 + q3)
    x_ee = (L1 * c1 + L2 * c12 + L3 * c123).astype(np.float32)
    y_ee = (L1 * s1 + L2 * s12 + L3 * s123).astype(np.float32)
    th_ee = (q1 + q2 + q3).astype(np.float32)
    return (x_ee, y_ee, th_ee), (q1, q2, q3)


def run_sr_for_joint(joint_name: str, X: dict, y: np.ndarray, *,
                     pop_size: int = 300, n_gens: int = 60,
                     parsimony: float = 0.005, seed: int = 0,
                     verbose: bool = True):
    """Run tessera GP to discover f(x_ee, y_ee, θ_ee) → q_i.

    Uses pointwise_only + sin/cos/sqrt + jax.grad const-opt (the
    ship-#1 and ship-#2 work). No FunctionalOp; this is pure pointwise
    SR with trig.
    """
    cfg = GPConfig(
        pop_size=pop_size,
        n_gens=n_gens,
        init_max_depth=4,
        parsimony=parsimony,
        seed=seed,
        pointwise_only=True,
        verbose=verbose,
        optimize_constants_every=5,
        optimize_constants_method="Nelder-Mead",  # scipy path (jax_adam
        # requires use_jax_population_eval which has different overhead
        # tradeoffs at this small pop size; scipy is fine here)
        optimize_constants_maxiter=40,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(X, y, feature_names=list(X.keys()))
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    return dict(
        joint=joint_name,
        runtime=runtime,
        best_tree=str(best.tree),
        best_cx=best.complexity,
        best_train_loss=float(best.train_loss),
    )


def evaluate_on_test(tree_str: str, X_test: dict, y_test: np.ndarray):
    """Evaluate a discovered tree on the test set. We re-parse from
    string by reconstructing through tessera's GP scoring path; simpler
    to just look up the Candidate.tree by complexity match. For this
    benchmark we'll cheat: re-run a tiny GP on the SAME data to get the
    tree, OR refactor to keep the tree object around.

    Cleaner path: return the Candidate directly from run_sr_for_joint.
    """
    raise NotImplementedError("see updated run_sr_for_joint that returns the tree")


def run_sr_for_joint_with_tree(joint_name, X_train, y_train, X_test, y_test, *,
                                pop_size=300, n_gens=60, parsimony=0.005,
                                seed=0, verbose=True):
    """Version of run_sr_for_joint that returns the tessera tree object
    so we can evaluate on test directly via evaluate()."""
    from tessera.expression.tree import evaluate

    cfg = GPConfig(
        pop_size=pop_size, n_gens=n_gens, init_max_depth=4,
        parsimony=parsimony, seed=seed,
        # Climb-then-anneal: keeps parsimony at 0.0001 for the first 30%
        # of gens (lets the GP explore high-cx atan2/acos compositions),
        # then anneals to the configured `parsimony` value. Tests the
        # climb-then-simplify hypothesis from
        # docs/research/benchmark_difficulty_and_climb_then_simplify.md §2.
        parsimony_schedule=climb_then_anneal_parsimony(
            climb_until=0.3, climb_value=0.0001, final_value=parsimony,
        ),
        pointwise_only=True, verbose=verbose,
        optimize_constants_every=5,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=40,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(X_train, y_train, feature_names=list(X_train.keys()))
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)

    # Test eval
    try:
        y_pred = np.asarray(evaluate(best.tree, X_test), dtype=np.float64)
        if y_pred.ndim == 0:
            y_pred = np.full(len(y_test), float(y_pred))
        valid = np.isfinite(y_pred)
        if valid.any():
            mse_test = float(np.mean((y_pred[valid] - y_test[valid]) ** 2))
            var_test = float(np.var(y_test[valid]))
            rel_test = mse_test / max(var_test, 1e-12)
        else:
            mse_test = float("inf")
            rel_test = float("inf")
    except Exception as e:
        mse_test = float("inf")
        rel_test = float("inf")

    return dict(
        joint=joint_name,
        runtime=runtime,
        tree=best.tree,
        tree_str=str(best.tree),
        cx=best.complexity,
        train_loss=float(best.train_loss),
        train_rel=float(best.train_loss / max(np.var(y_train), 1e-12)),
        test_mse=mse_test,
        test_rel=rel_test,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 3-DoF planar IK benchmark ===")
    print(f"Train: {N_TRAIN} samples; Test: {N_TEST} samples; seed={SEED}\n")

    # Generate data
    (xtr, ytr, thtr), (q1tr, q2tr, q3tr) = sample_workspace(N_TRAIN, SEED)
    (xte, yte, thte), (q1te, q2te, q3te) = sample_workspace(N_TEST, SEED + 1)

    X_train = {"x": xtr, "y": ytr, "th": thtr}
    X_test = {"x": xte, "y": yte, "th": thte}
    targets_train = {"q1": q1tr, "q2": q2tr, "q3": q3tr}
    targets_test = {"q1": q1te, "q2": q2te, "q3": q3te}

    # Run SR for each joint
    results = []
    for joint in ["q1", "q2", "q3"]:
        print(f"\n--- {joint} ---")
        r = run_sr_for_joint_with_tree(
            joint, X_train, targets_train[joint], X_test, targets_test[joint],
            pop_size=300, n_gens=60, parsimony=0.005,
            seed=SEED, verbose=False,
        )
        print(f"  {joint}: cx={r['cx']} train_rel={r['train_rel']:.4f} "
              f"test_rel={r['test_rel']:.4f}  ({r['runtime']:.1f}s)")
        results.append(r)

    # Tier verdict
    rel_exact = 0.01
    rel_partial = 0.10
    n_exact = sum(1 for r in results if r["test_rel"] < rel_exact)
    n_partial = sum(1 for r in results
                    if rel_exact <= r["test_rel"] < rel_partial)
    n_failed = sum(1 for r in results if r["test_rel"] >= rel_partial)
    if n_exact == 3:
        tier = "A — all 3 joints exact (clean SR rediscovery of analytical IK)"
    elif n_exact >= 1 and n_failed <= 1:
        tier = "B — mixed (likely atan2 gap on the q1/q3 joints)"
    elif n_partial >= 2:
        tier = "C — partial fit (approximations, not exact form)"
    else:
        tier = "D — failed (trig vocab insufficient or budget too small)"

    # Report
    L = ["# 3-DoF planar IK benchmark",
         "",
         "**Status:** first tessera robotics benchmark (per "
         "`docs/research/sr_for_inverse_kinematics.md` §3).",
         "",
         f"**Setup:** 3 revolute joints, L1 = L2 = L3 = 1; "
         f"q1 ∈ {Q1_LIMITS}, q2 ∈ {Q2_LIMITS} (elbow-up), "
         f"q3 ∈ {Q3_LIMITS}.",
         f"",
         f"**Train:** {N_TRAIN} samples · **Test:** {N_TEST} samples · "
         f"**Seed:** {SEED}",
         "",
         "**GP config:** pop=300, gens=60, pointwise_only=True, "
         "trig + sqrt + jax.grad const-opt available.",
         "",
         "## Headline",
         "",
         f"**Tier {tier[:1]}** — {tier}",
         "",
         f"- **Exact** (test rel < {rel_exact}): {n_exact} / 3",
         f"- **Partial** ({rel_exact} ≤ rel < {rel_partial}): {n_partial} / 3",
         f"- **Failed** (rel ≥ {rel_partial}): {n_failed} / 3",
         "",
         "## Per-joint results",
         "",
         "| Joint | cx | Train rel | Test rel | Runtime (s) |",
         "|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['joint']} | {r['cx']} | {r['train_rel']:.4f} | "
                 f"{r['test_rel']:.4f} | {r['runtime']:.1f} |")

    L += ["", "## Discovered formulas", ""]
    for r in results:
        L.append(f"### {r['joint']}")
        L.append(f"- cx={r['cx']}, train_rel={r['train_rel']:.4f}, "
                 f"test_rel={r['test_rel']:.4f}")
        tree_ascii = r["tree_str"].encode("ascii", "replace").decode("ascii")
        L.append("  ```")
        L.append(f"  {tree_ascii[:400]}{'...' if len(tree_ascii) > 400 else ''}")
        L.append("  ```")
        L.append("")

    L += ["## Analytical IK (reference)",
          "",
          "```",
          "x_w  = x_ee − cos(θ_ee)",
          "y_w  = y_ee − sin(θ_ee)",
          "r²   = x_w² + y_w²",
          "q2   = acos((r² − 2) / 2)            # elbow-up branch",
          "q1   = atan2(y_w, x_w) − atan2(sin(q2), 1 + cos(q2))",
          "q3   = θ_ee − q1 − q2",
          "```",
          "",
          "Note: the analytical form uses `atan2` and `acos`, which",
          "tessera does NOT currently have. Per "
          "`sr_for_inverse_kinematics.md` §7's predictions:",
          "  - q3 = θ − q1 − q2 (pure subtraction; SR should find this)",
          "  - q2 involves `acos`; SR may approximate via algebraic forms",
          "  - q1 involves two `atan2` calls; without that primitive, "
          "    expect partial-fit at best",
          "",
          "## Interpretation",
          "",
          "Tier A = the unit-architecture validation: per-joint SR works.",
          "Tier B = the atan2 gap is real and needs to be addressed (next",
          "         iteration: add atan2 as a primitive).",
          "Tier C = SR finds approximations but not the exact form.",
          "Tier D = trig vocabulary or search budget is insufficient.",
          "",
          "## See also",
          "",
          "- `docs/research/sr_for_inverse_kinematics.md` — full research note",
          "- `docs/research/high_dim_symbolic_regression.md` §6 — unit-architecture framing",
          "- `docs/planned/roadmap.md` §2.3 — this benchmark's promotion entry"]

    OUT_REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {OUT_REPORT}")
    print(f"\nHeadline: Tier {tier[:1]} — {tier}")
    print(f"  {n_exact} exact / {n_partial} partial / {n_failed} failed")


if __name__ == "__main__":
    main()
