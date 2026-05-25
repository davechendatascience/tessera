"""MVP 7.1 / Conjecture C1-refined: ABC scoring on held-out trajectory.

Empirical test of the conjecture from
`docs/research/process_discovery_sr.md` §6.2 + §7.1:

> "ABC-style summary-statistics scoring evaluated on a held-out slice
> or alternate trajectory suppresses Class B (template/reduce_*
> natural overfit) more than pointwise MSE on the same held-out data."

Three-mode comparison on heat equation discovery:

  Mode A (baseline):  score = MSE_train
  Mode B (hold MSE):  score = MSE_train + β · MSE_hold
  Mode C (hold ABC):  score = MSE_train + β · ABC_dist_hold

If Mode C produces MORE Class C and FEWER Class B than Mode B at the
same compute budget, the ABC term adds genuine information beyond
held-out CV. If Mode C ≈ Mode B, ABC offers no signal beyond CV.
If both B and C degrade Class C discovery vs A, the conjecture is
falsified.

Setup: 3 trajectories with different ICs (TRAIN ic_seed=100, HOLD
ic_seed=200, TEST ic_seed=999). Three modes × 5 seeds × pop=240
gens=100. Wall-clock target ~3-5 minutes.
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
from tessera.experimental.abc_scoring import GPWithHoldout

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_traintest_computescale import simulate_heat_with_ic  # noqa: E402


# ----------------------------------------------------------------------
# Classification (reused from multitrajectory experiment)
# ----------------------------------------------------------------------

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


def classify_tree(node, train_loss, test_loss, oracle_train, oracle_test):
    has_lap = tree_has_laplacian(node)
    has_red = tree_has_reduce(node)
    train_ratio = train_loss / max(oracle_train, 1e-30)
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


# ----------------------------------------------------------------------
# Per-mode runner
# ----------------------------------------------------------------------

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


def run_one_mode(
    mode_name: str,
    U_train, dt_U_train,
    U_hold, dt_U_hold,
    U_test, dt_U_test,
    seed: int, pop: int, gens: int,
    beta_abc: float, beta_hold_mse: float,
) -> dict:
    target_var = float(np.var(dt_U_train[1:-1, 1:-1]))
    parsimony = max(target_var * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=pop, n_gens=gens,
        init_max_depth=4, parsimony=parsimony,
        early_stop_patience=gens,  # disabled
        seed=seed, enable_2d=True, fill_warmup=0.0, verbose=False,
    )

    if beta_abc == 0.0 and beta_hold_mse == 0.0:
        gp = GP(cfg)
    else:
        gp = GPWithHoldout(
            cfg,
            hold_env={"U": U_hold},
            hold_y_true=dt_U_hold,
            beta_abc=beta_abc,
            beta_hold_mse=beta_hold_mse,
        )

    t0 = time.time()
    front = gp.run({"U": U_train}, dt_U_train, feature_names=["U"])
    rt = time.time() - t0
    best = min(front, key=lambda c: c.train_loss)

    test_loss = evaluate_tree_on(best.tree, U_test, dt_U_test)
    oracle_train = oracle_loss_on(U_train, dt_U_train)
    oracle_test = oracle_loss_on(U_test, dt_U_test)
    train_ratio = best.train_loss / oracle_train if oracle_train > 0 else float("inf")
    test_ratio = test_loss / oracle_test if oracle_test > 0 else float("inf")
    cls = classify_tree(best.tree, best.train_loss, test_loss,
                        oracle_train, oracle_test)
    return dict(
        mode=mode_name, seed=seed,
        train_loss=best.train_loss, test_loss=test_loss,
        train_ratio=train_ratio, test_ratio=test_ratio,
        best_cx=best.complexity,
        classification=cls,
        front_size=len(front),
        runtime=rt,
        tree_str=str(best.tree),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pop", type=int, default=240)
    p.add_argument("--gens", type=int, default=100)
    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight on hold-out term (β in score = train + β · hold)")
    args = p.parse_args(argv)

    print("=== MVP 7.1 / C1-refined: ABC scoring on held-out ===")
    print(f"T={args.T} X={args.X} seeds={args.seeds} pop={args.pop} gens={args.gens}")
    print(f"β = {args.beta}")
    print()

    # Simulate trajectories
    print("Simulating TRAIN (ic_seed=100), HOLD (ic_seed=200), TEST (ic_seed=999)...")
    U_train = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                    noise_std=0.002, ic_seed=100, sim_seed=0)
    U_hold = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                   noise_std=0.002, ic_seed=200, sim_seed=1)
    U_test = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                   noise_std=0.002, ic_seed=999, sim_seed=2)

    def make_dt_U(U):
        dt = np.zeros_like(U)
        dt[:-1] = U[1:] - U[:-1]
        return dt

    dt_U_train = make_dt_U(U_train)
    dt_U_hold = make_dt_U(U_hold)
    dt_U_test = make_dt_U(U_test)

    oracle_train = oracle_loss_on(U_train, dt_U_train)
    oracle_hold = oracle_loss_on(U_hold, dt_U_hold)
    oracle_test = oracle_loss_on(U_test, dt_U_test)
    print(f"Oracle TRAIN={oracle_train:.4g} HOLD={oracle_hold:.4g} TEST={oracle_test:.4g}")
    print()

    # The three modes
    modes = [
        ("A_baseline", 0.0, 0.0),
        ("B_hold_mse", 0.0, args.beta),
        ("C_hold_abc", args.beta, 0.0),
    ]
    # Note: beta for B and C are individually applied — Mode B uses
    # β · MSE_hold, Mode C uses β · ABC_dist_hold. Both with the same β.

    results = []
    t_start = time.time()
    for mode_name, beta_abc, beta_hold_mse in modes:
        print(f"--- Mode {mode_name} (β_abc={beta_abc}, β_hold_mse={beta_hold_mse}) ---")
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            r = run_one_mode(
                mode_name=mode_name,
                U_train=U_train, dt_U_train=dt_U_train,
                U_hold=U_hold, dt_U_hold=dt_U_hold,
                U_test=U_test, dt_U_test=dt_U_test,
                seed=seed, pop=args.pop, gens=args.gens,
                beta_abc=beta_abc, beta_hold_mse=beta_hold_mse,
            )
            results.append(r)
            print(f"  seed={seed}  train/o={r['train_ratio']:.2f}  "
                  f"test/o={r['test_ratio']:.2f}  cx={r['best_cx']}  "
                  f"class={r['classification']}  ({r['runtime']:.1f}s)")
        print()

    total_rt = time.time() - t_start
    print(f"Total wall-clock: {total_rt:.1f}s")

    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_abc_mvp71.md"
    write_report(results, modes, args, oracle_train, oracle_test, total_rt, out_path)
    return 0


def write_report(results, modes, args, oracle_train, oracle_test, total_rt, out_path):
    L = ["# MVP 7.1 / C1-refined: ABC scoring on held-out trajectory", ""]
    L.append("Empirical test of conjecture C1-refined from")
    L.append("`docs/research/process_discovery_sr.md`. First occupant of")
    L.append("`tessera.experimental`.")
    L.append("")
    L.append("**The conjecture:** ABC-style summary-statistics scoring on a")
    L.append("held-out trajectory suppresses Class B (template/reduce_*)")
    L.append("MORE than pointwise MSE on the same held-out data.")
    L.append("")
    L.append(f"**Setup:** T={args.T}, X={args.X}, α=0.05. Three trajectories")
    L.append(f"(TRAIN ic_seed=100, HOLD ic_seed=200, TEST ic_seed=999).")
    L.append(f"pop={args.pop}, gens={args.gens}, {args.seeds} seeds/mode, β={args.beta}.")
    L.append(f"Wall-clock: {total_rt:.1f}s")
    L.append("")
    L.append("## Mode definitions")
    L.append("")
    L.append("- **Mode A (baseline)**: fitness = train_loss + parsimony · cx")
    L.append("- **Mode B (hold MSE)**: fitness += β · MSE(tree(U_hold), dt_U_hold)")
    L.append("- **Mode C (hold ABC)**: fitness += β · ABC_distance(summary_stats(dt_U_hold), summary_stats(tree(U_hold)))")
    L.append("")
    L.append("ABC summary statistics: var_total, mean_abs, acf_time_lag1,")
    L.append("acf_space_lag1, spatial_mean_var, temporal_mean_var (all on")
    L.append("interior, NaN-robust).")
    L.append("")

    # Class counts per mode
    def count_classes(mode_name):
        runs = [r for r in results if r["mode"] == mode_name]
        counts = {"A": 0, "B": 0, "C": 0, "C-partial": 0, "degenerate": 0}
        for r in runs:
            counts[r["classification"]] = counts.get(r["classification"], 0) + 1
        return counts, len(runs)

    L.append("## Class distribution comparison")
    L.append("")
    L.append("| Mode | A (generic diff) | B (template/reduce_*) | C-partial | **C** (clean mechanism) | degenerate |")
    L.append("|---|---|---|---|---|---|")
    for mode_name, _, _ in modes:
        counts, n = count_classes(mode_name)
        L.append(f"| {mode_name} | {counts['A']}/{n} | {counts['B']}/{n} | "
                 f"{counts['C-partial']}/{n} | **{counts['C']}/{n}** | "
                 f"{counts['degenerate']}/{n} |")
    L.append("")

    # Median test ratios
    L.append("## Median ratios across seeds")
    L.append("")
    L.append("| Mode | median train/oracle | median test/oracle | median cx |")
    L.append("|---|---|---|---|")
    for mode_name, _, _ in modes:
        runs = [r for r in results if r["mode"] == mode_name]
        med_train = float(np.median([r["train_ratio"] for r in runs]))
        med_test = float(np.median([r["test_ratio"] if np.isfinite(r["test_ratio"]) else 1e9
                                    for r in runs]))
        med_cx = float(np.median([r["best_cx"] for r in runs]))
        L.append(f"| {mode_name} | {med_train:.2f} | {med_test:.2f} | {med_cx:.0f} |")
    L.append("")

    # Verdict
    counts_A, n_A = count_classes("A_baseline")
    counts_B, n_B = count_classes("B_hold_mse")
    counts_C, n_C = count_classes("C_hold_abc")

    L.append("## Verdict")
    L.append("")
    # Determine outcome
    c_count_a = counts_A["C"] + counts_A["C-partial"]
    c_count_b = counts_B["C"] + counts_B["C-partial"]
    c_count_c = counts_C["C"] + counts_C["C-partial"]
    b_count_a = counts_A["B"]
    b_count_b = counts_B["B"]
    b_count_c = counts_C["B"]

    L.append(f"- Class C (+ C-partial) discoveries: A={c_count_a}/{n_A}, B={c_count_b}/{n_B}, C={c_count_c}/{n_C}")
    L.append(f"- Class B (natural-overfit) occurrences: A={b_count_a}/{n_A}, B={b_count_b}/{n_B}, C={b_count_c}/{n_C}")
    L.append("")
    if c_count_c > c_count_b and c_count_c > c_count_a:
        L.append("**CONJECTURE SUPPORTED.** ABC mode produces more clean-mechanism")
        L.append("candidates than either baseline or held-out MSE.")
    elif c_count_b > c_count_a and abs(c_count_c - c_count_b) <= 1:
        L.append("**ABC TIES WITH HELD-OUT CV.** Adding the held-out term helps")
        L.append("(both B and C exceed baseline A) but ABC offers no clear signal")
        L.append("beyond pointwise CV. C1-refined as ABC-specific is not validated;")
        L.append("the broader 'use held-out data in scoring' principle is.")
    elif c_count_a >= c_count_b and c_count_a >= c_count_c:
        L.append("**HOLD-OUT SCORING DEGRADES DISCOVERY.** Baseline beats both")
        L.append("held-out modes. Either β is wrong, the hold-out is too dissimilar,")
        L.append("or the conjecture is falsified at this scale.")
    else:
        L.append("**MIXED RESULT.** Some signal but not the predicted ordering.")
        L.append(f"Counts: A={c_count_a}, B={c_count_b}, C={c_count_c}. May")
        L.append(f"need more seeds (current N={args.seeds}) for robust comparison.")
    L.append("")

    L.append("## Per-seed details")
    L.append("")
    L.append("| Mode | seed | train/oracle | test/oracle | cx | class | tree (truncated) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        tree_str = r["tree_str"].encode("ascii", "replace").decode("ascii")
        if len(tree_str) > 70:
            tree_str = tree_str[:67] + "..."
        L.append(f"| {r['mode']} | {r['seed']} | {r['train_ratio']:.2f} | "
                 f"{r['test_ratio']:.2f} | {r['best_cx']} | "
                 f"{r['classification']} | `{tree_str}` |")
    L.append("")

    L.append("## Methodological caveat")
    L.append("")
    L.append(f"N={args.seeds} seeds/mode is small. Sampling variance can flip")
    L.append("verdicts. For statistically robust comparison, would need N=20+")
    L.append("per mode — ~12× wall-clock. Current result is directional, not")
    L.append("conclusive.")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_heat_equation_abc_mvp71.py --seeds 5")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
