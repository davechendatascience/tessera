"""MVP / Conjecture C5: counterfactual evaluation as post-hoc selector.

Tests the prediction from `docs/research/c5_counterfactual_eval_analysis.md`:
post-hoc counterfactual ranking can reliably identify Class C
candidates from a Pareto front, even though it doesn't affect search.

Approach
--------
1. Run baseline GP on heat equation discovery (~5 seeds)
2. Take the FULL Pareto front from each run (not just best-by-train)
3. Score each front candidate on a set of counterfactual perturbations
4. Check: does counterfactual ranking correctly identify Class C
   candidates? Does it rank Class B last?

This is a post-hoc selection experiment, not a search experiment.

Predicted outcome (per pre-analysis)
------------------------------------
- Class C trees: counterfactual ratios near 1× oracle on IC CFs,
  higher on α-changed CF (because α is fixed in the tree)
- Class B trees: catastrophic counterfactual ratios on IC CFs
  (already established by paired-diagnostic experiment)
- Class A trees: moderate counterfactual ratios; structurally
  generic so they "fit" the perturbations similarly to baseline

Discriminator: median counterfactual ratio. Class C < Class A < Class B.
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
from tessera.expression.tree import BinOp, Const, UnOp, evaluate as eval_tree
from tessera.experimental.counterfactual_eval import (
    generate_heat_eq_counterfactuals,
    score_counterfactual,
)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_traintest_computescale import simulate_heat_with_ic  # noqa: E402


# Classification — same as previous experiments
def is_laplacian_atom(measure_2d) -> bool:
    if not hasattr(measure_2d, "atoms"):
        return False
    atoms = measure_2d.atoms
    if len(atoms) != 3:
        return False
    weights = sorted(round(float(a.weight), 4) for a in atoms)
    return weights == [-2.0, 1.0, 1.0]


def tree_has_reduce(node) -> bool:
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


def classify_candidate(cand, oracle_train, oracle_test, test_loss):
    has_lap = tree_has_laplacian(cand.tree)
    has_red = tree_has_reduce(cand.tree)
    train_ratio = cand.train_loss / max(oracle_train, 1e-30)
    test_ratio = test_loss / max(oracle_test, 1e-30) if oracle_test > 0 else float("inf")
    if not has_lap and train_ratio > 5.0:
        return "degenerate"
    if has_lap and not has_red:
        if train_ratio < 1.5 and test_ratio < 1.5:
            return "C"
        return "C-partial"
    if has_lap and has_red:
        return "B"
    return "A"


def oracle_loss_on(U, dt_U, alpha=0.05):
    lap = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    return float(np.mean((alpha * lap[interior] - dt_U[interior]) ** 2))


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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pop", type=int, default=240)
    p.add_argument("--gens", type=int, default=100)
    args = p.parse_args(argv)

    print("=== MVP / C5: counterfactual evaluation as post-hoc selector ===")
    print(f"T={args.T} X={args.X} seeds={args.seeds} pop={args.pop} gens={args.gens}")
    print()
    print("Method:")
    print("  1. Run baseline GP, get full Pareto front")
    print("  2. Score each front candidate on 5 counterfactual perturbations")
    print("  3. Check if counterfactual ranking identifies Class C correctly")
    print()

    # Generate counterfactuals ONCE (shared across all seeds)
    print("Generating counterfactual set...")
    counterfactuals = generate_heat_eq_counterfactuals(
        T=args.T, X=args.X, alpha_base=0.05, noise_std_base=0.002,
    )
    for cf in counterfactuals:
        print(f"  {cf.name}: U.shape={cf.U.shape}  α={cf.alpha}  oracle_mse={cf.oracle_mse:.4g}")
    print()

    # Build TRAIN + TEST trajectories
    U_train = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                    noise_std=0.002, ic_seed=100, sim_seed=0)
    U_test = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                   noise_std=0.002, ic_seed=999, sim_seed=2)
    dt_U_train = np.zeros_like(U_train)
    dt_U_train[:-1] = U_train[1:] - U_train[:-1]
    dt_U_test = np.zeros_like(U_test)
    dt_U_test[:-1] = U_test[1:] - U_test[:-1]

    oracle_train = oracle_loss_on(U_train, dt_U_train)
    oracle_test = oracle_loss_on(U_test, dt_U_test)
    print(f"Oracle TRAIN={oracle_train:.4g} TEST={oracle_test:.4g}")
    print()

    # Run baseline GP across seeds; collect Pareto fronts
    all_results = []
    target_var = float(np.var(dt_U_train[1:-1, 1:-1]))
    parsimony = max(target_var * 0.001, 1e-9)

    t_start = time.time()
    for seed_idx in range(args.seeds):
        seed = 2026 + seed_idx
        cfg = GPConfig(
            pop_size=args.pop, n_gens=args.gens,
            init_max_depth=4, parsimony=parsimony,
            early_stop_patience=args.gens, seed=seed,
            enable_2d=True, fill_warmup=0.0, verbose=False,
        )
        gp = GP(cfg)
        t0 = time.time()
        front = gp.run({"U": U_train}, dt_U_train, feature_names=["U"])
        rt = time.time() - t0

        # Classify each front candidate
        front_data = []
        for cand in front:
            test_loss = evaluate_tree_on(cand.tree, U_test, dt_U_test)
            cls = classify_candidate(cand, oracle_train, oracle_test, test_loss)
            # Counterfactual score
            cf_score = score_counterfactual(cand.tree, counterfactuals)
            front_data.append({
                "cand": cand,
                "test_loss": test_loss,
                "test_ratio": test_loss / oracle_test if oracle_test > 0 else float("inf"),
                "train_ratio": cand.train_loss / oracle_train if oracle_train > 0 else float("inf"),
                "class": cls,
                "cf_score": cf_score,
            })

        all_results.append({
            "seed": seed,
            "front": front_data,
            "runtime": rt,
        })

        # Print summary
        best_by_train = min(front_data, key=lambda d: d["cand"].train_loss)
        best_by_cf = min(front_data, key=lambda d: d["cf_score"]["median_ratio"])
        print(f"seed={seed}  front_size={len(front_data)}  "
              f"best_by_train: cls={best_by_train['class']} cx={best_by_train['cand'].complexity} "
              f"train/o={best_by_train['train_ratio']:.2f}  "
              f"best_by_cf: cls={best_by_cf['class']} cx={best_by_cf['cand'].complexity} "
              f"cf_median={best_by_cf['cf_score']['median_ratio']:.2f}  ({rt:.1f}s)")

    total_rt = time.time() - t_start
    print(f"\nTotal wall-clock: {total_rt:.1f}s")

    # Write report
    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_counterfactual_mvp_c5.md"
    write_report(all_results, counterfactuals, args, total_rt, out_path)
    return 0


def write_report(all_results, counterfactuals, args, total_rt, out_path):
    L = ["# MVP / C5: counterfactual evaluation as post-hoc selector", ""]
    L.append("Empirical test of conjecture C5 (post-hoc ranking version),")
    L.append("following the theoretical pre-analysis in")
    L.append("`docs/research/c5_counterfactual_eval_analysis.md`.")
    L.append("")
    L.append("**Pre-analysis prediction:** counterfactual ranking can")
    L.append("reliably identify Class C from a Pareto front; Class B")
    L.append("scores worst due to TRAIN-specific reduce_* failing on")
    L.append("counterfactual perturbations.")
    L.append("")
    L.append(f"**Setup:** T={args.T}, X={args.X}, {args.seeds} baseline seeds,")
    L.append(f"pop={args.pop}, gens={args.gens}, post-hoc CF analysis.")
    L.append(f"Wall-clock: {total_rt:.1f}s")
    L.append("")
    L.append("## Counterfactual set")
    L.append("")
    for cf in counterfactuals:
        L.append(f"- **{cf.name}**: α={cf.alpha}, oracle_mse={cf.oracle_mse:.4g}")
    L.append("")

    # Per-seed best-by-train vs best-by-cf comparison
    L.append("## Best-by-train vs best-by-counterfactual per seed")
    L.append("")
    L.append("| seed | best-by-train class | best-by-train train/o | best-by-cf class | best-by-cf cf_median | best-by-cf train/o |")
    L.append("|---|---|---|---|---|---|")
    n_train_picks_c = 0
    n_cf_picks_c = 0
    n_train_picks_b = 0
    n_cf_picks_b = 0
    n_seeds_with_c_in_front = 0
    cf_picked_c_when_available = 0

    for r in all_results:
        front = r["front"]
        best_by_train = min(front, key=lambda d: d["cand"].train_loss)
        best_by_cf = min(front, key=lambda d: d["cf_score"]["median_ratio"])

        n_train_picks_c += (best_by_train["class"] == "C")
        n_cf_picks_c += (best_by_cf["class"] == "C")
        n_train_picks_b += (best_by_train["class"] == "B")
        n_cf_picks_b += (best_by_cf["class"] == "B")

        # Did the front contain Class C?
        has_c = any(d["class"] == "C" for d in front)
        if has_c:
            n_seeds_with_c_in_front += 1
            if best_by_cf["class"] == "C":
                cf_picked_c_when_available += 1

        L.append(f"| {r['seed']} | {best_by_train['class']} | "
                 f"{best_by_train['train_ratio']:.2f} | "
                 f"{best_by_cf['class']} | "
                 f"{best_by_cf['cf_score']['median_ratio']:.2f} | "
                 f"{best_by_cf['train_ratio']:.2f} |")
    L.append("")

    L.append("## Selection summary")
    L.append("")
    n_seeds = len(all_results)
    L.append(f"- best-by-train picks Class C: {n_train_picks_c}/{n_seeds}")
    L.append(f"- best-by-cf picks Class C: {n_cf_picks_c}/{n_seeds}")
    L.append(f"- best-by-train picks Class B (overfit): {n_train_picks_b}/{n_seeds}")
    L.append(f"- best-by-cf picks Class B: {n_cf_picks_b}/{n_seeds}")
    L.append(f"- seeds with Class C in front at all: {n_seeds_with_c_in_front}/{n_seeds}")
    L.append(f"- when Class C is in the front, best-by-cf picks it: "
             f"{cf_picked_c_when_available}/{n_seeds_with_c_in_front}")
    L.append("")

    # Verdict
    L.append("## Verdict against pre-analysis prediction")
    L.append("")
    if n_seeds_with_c_in_front == 0:
        L.append("**INCONCLUSIVE.** No seed produced a Class C candidate;")
        L.append("can't evaluate whether CF ranking would have picked it.")
    elif cf_picked_c_when_available == n_seeds_with_c_in_front:
        L.append("**CONJECTURE VALIDATED.** Whenever Class C exists in the")
        L.append("front, post-hoc counterfactual ranking correctly identifies it.")
        L.append(f"CF picked Class C in {cf_picked_c_when_available}/{n_seeds_with_c_in_front}")
        L.append("eligible seeds.")
    elif cf_picked_c_when_available > 0:
        L.append("**PARTIAL.** CF ranking sometimes identifies Class C but not")
        L.append(f"reliably ({cf_picked_c_when_available}/{n_seeds_with_c_in_front} eligible).")
    else:
        L.append("**FALSIFIED.** CF ranking fails to identify Class C even when")
        L.append("it exists in the front. Counterfactual evaluation doesn't add")
        L.append("selection signal beyond cx + train_loss.")
    L.append("")

    # Class B suppression
    L.append("**Class B suppression:** best-by-train picked Class B in")
    L.append(f"{n_train_picks_b}/{n_seeds} seeds; best-by-cf picked Class B in")
    L.append(f"{n_cf_picks_b}/{n_seeds} seeds.")
    if n_train_picks_b > n_cf_picks_b:
        L.append("CF ranking REDUCES Class B selection — useful diagnostic.")
    elif n_train_picks_b == n_cf_picks_b:
        L.append("CF ranking equivalent to train-loss for Class B avoidance.")
    L.append("")

    # Per-seed details with full front + cf scores
    L.append("## Per-seed Pareto fronts with counterfactual scores")
    L.append("")
    for r in all_results:
        L.append(f"### seed={r['seed']}  ({r['runtime']:.1f}s)")
        L.append("")
        L.append("| cx | class | train/o | test/o | cf_median | cf_mean | cf_max |")
        L.append("|---|---|---|---|---|---|---|")
        for d in r["front"]:
            cand = d["cand"]
            cf = d["cf_score"]
            L.append(f"| {cand.complexity} | {d['class']} | "
                     f"{d['train_ratio']:.2f} | {d['test_ratio']:.2f} | "
                     f"{cf['median_ratio']:.2f} | {cf['mean_ratio']:.2f} | "
                     f"{cf['max_ratio']:.2f} |")
        L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_heat_equation_counterfactual_mvp_c5.py --seeds 5")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
