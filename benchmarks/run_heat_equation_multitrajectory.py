"""Multi-trajectory training for heat equation — final empirical anchor.

Hypothesis: training on K trajectories (different ICs, same α)
simultaneously should make Class B (template / reduce_*) self-defeat,
because reduce_*(traj_1) ≠ reduce_*(traj_2) — a single divisor can't
fit all trajectories. The GP is forced toward Class C (clean
`Const · Template`), which is trajectory-invariant.

Implementation
--------------
Stack K trajectories along the time axis:
  U_stacked shape = (K · T, X)
  dt_U computed within each trajectory; boundary rows set to 0

The K · T - K boundary rows where dt_U crosses trajectory boundaries
have target=0; the GP's prediction there is small (Measure2D outputs
of stacked data near the boundary give small/zero values). Loss
contribution from boundaries is sub-1% of total; doesn't materially
affect the search.

Compared against single-trajectory baseline at MATCHED sample count
(single trajectory of K · T timesteps). Both share the post-fix
random_tree with reduce_* downweighted 10×.

Expected outcome
----------------
- Class C (clean `Const · Laplacian`) rate: rises from 1/12 to ≥3/12
- Class B (template / reduce_*): rate drops; reduce_* arguments
  return different scalars per trajectory, can't fit all
- Class A (generic diff): unchanged; trajectory-invariant
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression import (
    GP, GPConfig,
    FunctionalOp2D, iter_subtrees,
)
from tessera.expression.measure_2d import measure_2d_laplacian_5pt
from tessera.expression.tree import BinOp, Const, evaluate as eval_tree

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_traintest_computescale import simulate_heat_with_ic  # noqa: E402


# ----------------------------------------------------------------------
# Classification: A / B / C
# ----------------------------------------------------------------------

def is_laplacian_atom(measure_2d) -> bool:
    """Check if a Measure2D has the 5-point Laplacian atom pattern:
    weights (1, -2, 1) at offsets (0,-1), (0,0), (0,+1) or similar."""
    if not hasattr(measure_2d, "atoms"):
        return False
    atoms = measure_2d.atoms
    if len(atoms) != 3:
        return False
    weights = sorted(round(float(a.weight), 4) for a in atoms)
    return weights == [-2.0, 1.0, 1.0]


def tree_has_reduce(node) -> bool:
    """Walk tree; return True if any UnOp is a reduce_* operator."""
    from tessera.expression.tree import UnOp
    for sub in iter_subtrees(node):
        if isinstance(sub, UnOp) and sub.op.startswith("reduce_"):
            return True
    return False


def tree_has_laplacian(node) -> bool:
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            if is_laplacian_atom(sub.measure_2d):
                return True
    return False


def classify_tree(node, train_loss, test_loss, oracle_train, oracle_test):
    """Classify into A / B / C / degenerate."""
    has_lap = tree_has_laplacian(node)
    has_red = tree_has_reduce(node)
    train_ratio = train_loss / oracle_train
    test_ratio = test_loss / oracle_test if oracle_test > 0 else float("inf")
    if not has_lap and train_ratio > 5.0:
        return "degenerate"
    if has_lap and not has_red:
        # Possibly Class C — need to check test/train ratio
        if train_ratio < 1.5 and test_ratio < 1.5:
            return "C"
        return "C-partial"
    if has_lap and has_red:
        return "B"
    return "A"


# ----------------------------------------------------------------------
# Multi-trajectory stacker
# ----------------------------------------------------------------------

def build_multi_trajectory(K: int, T_per: int, X: int, alpha: float,
                           ic_seeds: list[int], noise_std: float = 0.002):
    """Stack K trajectories along time axis. Each has its own IC.

    Returns:
        U_stack: shape (K*T_per, X) — stacked field
        dt_U_stack: shape (K*T_per, X) — target, with boundary rows set to 0
    """
    trajectories = []
    for ic_seed in ic_seeds[:K]:
        traj = simulate_heat_with_ic(
            T=T_per, X=X, alpha=alpha,
            noise_std=noise_std, ic_seed=ic_seed, sim_seed=0,
        )
        trajectories.append(traj)

    U_stack = np.vstack(trajectories)  # (K*T_per, X)
    dt_U_stack = np.zeros_like(U_stack)
    # Compute dt_U within each trajectory; leave boundary rows = 0.
    for k in range(K):
        start = k * T_per
        end = start + T_per - 1
        dt_U_stack[start:end] = U_stack[start+1:end+1] - U_stack[start:end]
    # The row at end of each trajectory (e.g., index T_per-1) is correctly 0;
    # the row at start of each NEXT trajectory begins a new trajectory.
    return U_stack, dt_U_stack


# ----------------------------------------------------------------------
# Run one config
# ----------------------------------------------------------------------

def run_one(U, dt_U, seed, pop, gens):
    target_var = float(np.var(dt_U[1:-1, 1:-1]))
    parsimony = max(target_var * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=pop, n_gens=gens,
        init_max_depth=4, parsimony=parsimony,
        early_stop_patience=gens, seed=seed,
        enable_2d=True, fill_warmup=0.0, verbose=False,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run({"U": U}, dt_U, feature_names=["U"])
    rt = time.time() - t0
    best = min(front, key=lambda c: c.train_loss)
    return dict(
        seed=seed, train_loss=best.train_loss,
        best_cx=best.complexity, runtime=rt,
        tree=best.tree, front=front,
    )


def evaluate_tree_on(tree, U, dt_U):
    try:
        pred = eval_tree(tree, {"U": U}, fill_warmup=0.0)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != dt_U.shape or not np.isfinite(pred).all():
            return float("inf")
        interior = (slice(1, -1), slice(1, -1))
        return float(np.mean((pred[interior] - dt_U[interior]) ** 2))
    except Exception:
        return float("inf")


def oracle_loss_on(U, dt_U, alpha=0.05):
    lap = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    return float(np.mean((alpha * lap[interior] - dt_U[interior]) ** 2))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3, help="Number of TRAIN trajectories")
    p.add_argument("--T-per", type=int, default=200,
                   help="Timesteps per trajectory")
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--pop", type=int, default=240)
    p.add_argument("--gens", type=int, default=100)
    args = p.parse_args(argv)

    print("=== Multi-trajectory training experiment ===")
    print(f"K={args.K} TRAIN trajectories, T_per={args.T_per}, X={args.X}")
    print(f"GP: pop={args.pop} gens={args.gens}, {args.seeds} seeds per config")
    print()

    # Build multi-trajectory TRAIN: K trajectories, ic_seeds 100, 200, 300
    multi_ic_seeds = [100, 200, 300, 400, 500][:args.K]
    print(f"Multi-traj IC seeds: {multi_ic_seeds}")
    U_multi, dt_U_multi = build_multi_trajectory(
        K=args.K, T_per=args.T_per, X=args.X,
        alpha=args.alpha, ic_seeds=multi_ic_seeds,
    )
    print(f"Multi-traj shape: U={U_multi.shape}")

    # Build single-trajectory baseline: ONE trajectory of K*T_per timesteps
    print(f"Single-traj baseline: T={args.K * args.T_per}, ic_seed=100")
    U_single = simulate_heat_with_ic(
        T=args.K * args.T_per, X=args.X, alpha=args.alpha,
        noise_std=0.002, ic_seed=100, sim_seed=0,
    )
    dt_U_single = np.zeros_like(U_single)
    dt_U_single[:-1] = U_single[1:] - U_single[:-1]

    # TEST trajectory (held-out IC for evaluation in BOTH configs)
    print(f"TEST trajectory: ic_seed=999 (held-out)")
    U_test = simulate_heat_with_ic(
        T=args.T_per, X=args.X, alpha=args.alpha,
        noise_std=0.002, ic_seed=999, sim_seed=1,
    )
    dt_U_test = np.zeros_like(U_test)
    dt_U_test[:-1] = U_test[1:] - U_test[:-1]

    oracle_multi = oracle_loss_on(U_multi, dt_U_multi, args.alpha)
    oracle_single = oracle_loss_on(U_single, dt_U_single, args.alpha)
    oracle_test = oracle_loss_on(U_test, dt_U_test, args.alpha)
    print(f"Oracle MULTI: {oracle_multi:.4g}")
    print(f"Oracle SINGLE: {oracle_single:.4g}")
    print(f"Oracle TEST: {oracle_test:.4g}")
    print()

    # Run both configs
    results = {"multi": [], "single": []}
    for label, U, dt_U, oracle_train in [
        ("single", U_single, dt_U_single, oracle_single),
        ("multi", U_multi, dt_U_multi, oracle_multi),
    ]:
        print(f"--- {label.upper()} trajectory ---")
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            r = run_one(U, dt_U, seed, args.pop, args.gens)
            # Evaluate on TEST
            test_loss = evaluate_tree_on(r["tree"], U_test, dt_U_test)
            r["test_loss"] = test_loss
            r["train_ratio"] = r["train_loss"] / oracle_train
            r["test_ratio"] = test_loss / oracle_test if oracle_test > 0 else float("inf")
            r["classification"] = classify_tree(
                r["tree"], r["train_loss"], test_loss,
                oracle_train, oracle_test,
            )
            r["tree_str"] = str(r["tree"])
            print(f"  seed={seed}  train/oracle={r['train_ratio']:.2f}  "
                  f"test/oracle={r['test_ratio']:.2f}  cx={r['best_cx']}  "
                  f"class={r['classification']}  ({r['runtime']:.1f}s)")
            results[label].append(r)
        print()

    total_rt = sum(r["runtime"] for label in results for r in results[label])
    print(f"Total wall-clock: {total_rt:.1f}s")

    # Write report
    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_multitrajectory.md"
    write_report(results, args, oracle_multi, oracle_single, oracle_test, total_rt, out_path)
    return 0


def write_report(results, args, oracle_multi, oracle_single, oracle_test, total_rt, out_path):
    L = ["# Multi-trajectory training — heat equation", ""]
    L.append("Tests whether training on K trajectories simultaneously punishes")
    L.append("Class B (template/reduce_*) overfit and raises Class C (clean")
    L.append("`Const · Template`) discovery rate.")
    L.append("")
    L.append(f"**Setup:** K={args.K} TRAIN trajectories of T={args.T_per} each, X={args.X}, α={args.alpha}. "
             f"Single-trajectory baseline uses T={args.K * args.T_per} (matched sample count). "
             f"All evaluations on shared held-out TEST trajectory (ic_seed=999).")
    L.append("")
    L.append(f"**Oracles:** MULTI={oracle_multi:.4g}, SINGLE={oracle_single:.4g}, TEST={oracle_test:.4g}")
    L.append(f"**Wall-clock:** {total_rt:.1f}s")
    L.append("")

    # Class counts
    def count_classes(runs):
        counts = {"A": 0, "B": 0, "C": 0, "C-partial": 0, "degenerate": 0}
        for r in runs:
            counts[r["classification"]] = counts.get(r["classification"], 0) + 1
        return counts

    single_classes = count_classes(results["single"])
    multi_classes = count_classes(results["multi"])
    n_single = len(results["single"])
    n_multi = len(results["multi"])

    L.append("## Class taxonomy comparison")
    L.append("")
    L.append("| Class | Single-traj | Multi-traj | Interpretation |")
    L.append("|---|---|---|---|")
    L.append(f"| **C** (clean `Const · Laplacian`) | {single_classes['C']}/{n_single} | {multi_classes['C']}/{n_multi} | Mechanism captured exactly |")
    L.append(f"| C-partial (template + Const, some imperfection) | {single_classes['C-partial']}/{n_single} | {multi_classes['C-partial']}/{n_multi} | Near-mechanism |")
    L.append(f"| **B** (template / reduce_*) | {single_classes['B']}/{n_single} | {multi_classes['B']}/{n_multi} | Natural-overfit shape |")
    L.append(f"| A (generic diff) | {single_classes['A']}/{n_single} | {multi_classes['A']}/{n_multi} | Tautology-flavoured |")
    L.append(f"| degenerate (predict-zero) | {single_classes['degenerate']}/{n_single} | {multi_classes['degenerate']}/{n_multi} | Search didn't converge |")
    L.append("")

    L.append("## Verdict")
    L.append("")
    delta_c = (multi_classes['C'] + multi_classes['C-partial']) - (single_classes['C'] + single_classes['C-partial'])
    delta_b = single_classes['B'] - multi_classes['B']
    if delta_c > 0 and delta_b > 0:
        L.append(f"**Multi-trajectory training works as predicted.** Class C rate ")
        L.append(f"rose by {delta_c}; Class B rate fell by {delta_b}. Multi-IC TRAIN ")
        L.append(f"prevents trajectory-specific reductions from succeeding.")
    elif delta_c > 0:
        L.append(f"**Partial success.** Class C rose by {delta_c} (good) but Class B ")
        L.append(f"didn't drop ({delta_b}). The multi-traj pressure helps mechanism ")
        L.append(f"discovery but doesn't fully eliminate reduce_* shortcuts.")
    elif delta_b > 0:
        L.append(f"**Mixed result.** Class B rate fell by {delta_b}, but Class C didn't ")
        L.append(f"rise ({delta_c}). Multi-traj punishes overfit but doesn't help GP find ")
        L.append(f"clean mechanism — search may need additional bias.")
    else:
        L.append(f"**No improvement.** Multi-trajectory training didn't help here. ")
        L.append(f"The 5-LOC reduce_* downweight may already be doing most of the work, ")
        L.append(f"OR our sample size is too small to see the effect.")
    L.append("")

    # Per-seed details for both
    for label in ["single", "multi"]:
        L.append(f"## Per-seed details — {label}")
        L.append("")
        L.append("| seed | train/oracle | test/oracle | cx | class | tree |")
        L.append("|---|---|---|---|---|---|")
        for r in results[label]:
            tree_str = r["tree_str"].encode("ascii", "replace").decode("ascii")
            if len(tree_str) > 80:
                tree_str = tree_str[:77] + "..."
            L.append(f"| {r['seed']} | {r['train_ratio']:.2f} | "
                     f"{r['test_ratio']:.2f} | {r['best_cx']} | "
                     f"{r['classification']} | `{tree_str}` |")
        L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_heat_equation_multitrajectory.py")
    L.append("```")
    L.append("")
    L.append(f"Wall-clock ~{int(total_rt)}s at default settings.")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
