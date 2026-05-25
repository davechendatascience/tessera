"""Paired diagnostic: TRAIN/TEST split + compute scaling on heat eq.

Two diagnostics in one experiment:

  (a) **TRAIN/TEST split**: simulate two trajectories with different
      initial conditions (same α). Fit GP on TRAIN; evaluate best
      Pareto candidate on TEST. If TRAIN ≈ TEST, mechanism captured.
      If TRAIN < TEST, real overfit.

  (b) **Compute scaling**: vary (pop_size × n_gens) across 4 levels
      from ~1500 to ~24000 total candidate evaluations. Fixed N.
      If best_loss/oracle improves smoothly with compute, the gap
      is search-efficiency. If flat, the search has a fundamental
      reach limit at this benchmark.

Outcome diagnostic table:

  TRAIN/TEST agreement   Compute scaling     Interpretation
  --------------------   -----------------   ------------------------
  TRAIN ≈ TEST           Improves            Underfit, fixable with
                                              more search
  TRAIN ≈ TEST           Flat                Fundamental search limit
                                              at this architecture
  TRAIN < TEST           Either              Overfit; need regularization
                                              or smaller class
  TRAIN ≈ TEST ≈ oracle  N/A                 Solved

Wall-clock target: ~5-10 minutes for the full sweep.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression import GP, GPConfig
from tessera.expression.measure_2d import measure_2d_laplacian_5pt
from tessera.expression.tree import evaluate as eval_tree


# ----------------------------------------------------------------------
# Simulator with randomised initial conditions
# ----------------------------------------------------------------------

def simulate_heat_with_ic(
    T: int, X: int, alpha: float, noise_std: float,
    ic_seed: int, sim_seed: int = 0, amplitude: float = 10.0,
) -> np.ndarray:
    """Same physics as run_heat_equation_discovery.simulate_heat_1d
    but with a randomised initial condition. Used to construct
    TRAIN/TEST trajectory pairs that share α but differ in IC."""
    rng_ic = np.random.default_rng(ic_seed)
    rng_sim = np.random.default_rng(sim_seed)
    U = np.zeros((T, X), dtype=np.float64)
    xs = np.arange(X) - X / 2

    n_bumps = int(rng_ic.integers(2, 4))
    for _ in range(n_bumps):
        center = rng_ic.uniform(-X / 4, X / 4)
        width = rng_ic.uniform(3.0, 7.0)
        amp = amplitude * rng_ic.uniform(0.4, 1.0)
        U[0] += amp * np.exp(-((xs - center) ** 2) / (2 * width ** 2))

    for t in range(1, T):
        prev = U[t - 1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        U[t] = prev + alpha * lap + noise_std * rng_sim.standard_normal(X)
    return U


def oracle_loss_on(U: np.ndarray, dt_U: np.ndarray, alpha: float = 0.05) -> float:
    """The benchmark's oracle: α · Laplacian. Returns MSE."""
    lap = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    return float(np.mean((alpha * lap[interior] - dt_U[interior]) ** 2))


def evaluate_tree_on(tree, U: np.ndarray, dt_U: np.ndarray) -> float:
    """Evaluate a GP candidate's tree on a (U, dt_U) pair. Returns MSE
    on the interior. Returns +inf on numerical failure."""
    try:
        pred = eval_tree(tree, {"U": U}, fill_warmup=0.0)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != dt_U.shape or not np.isfinite(pred).all():
            return float("inf")
        interior = (slice(1, -1), slice(1, -1))
        return float(np.mean((pred[interior] - dt_U[interior]) ** 2))
    except Exception:
        return float("inf")


# ----------------------------------------------------------------------
# Per-config runner
# ----------------------------------------------------------------------

def run_one(
    U_train, dt_U_train, U_test, dt_U_test,
    pop: int, gens: int, seed: int,
) -> dict:
    """Fit GP on TRAIN, evaluate best Pareto candidate on TEST."""
    target_var_train = float(np.var(dt_U_train[1:-1, 1:-1]))
    parsimony = max(target_var_train * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=pop,
        n_gens=gens,
        init_max_depth=4,
        parsimony=parsimony,
        early_stop_patience=gens,  # disable early stop
        seed=seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=False,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run({"U": U_train}, dt_U_train, feature_names=["U"])
    rt = time.time() - t0

    # Best by train loss
    best = min(front, key=lambda c: c.train_loss)
    # Evaluate the best ON TEST
    test_loss_best = evaluate_tree_on(best.tree, U_test, dt_U_test)

    # Also: evaluate EVERY front candidate on test, find the one with best test loss
    front_with_test = []
    for c in front:
        tl = evaluate_tree_on(c.tree, U_test, dt_U_test)
        front_with_test.append((c, tl))
    # Best by test loss (lower is better)
    best_by_test = min(front_with_test, key=lambda ct: ct[1])

    return dict(
        pop=pop, gens=gens, seed=seed,
        budget=pop * gens,
        train_loss_best=best.train_loss,
        test_loss_best=test_loss_best,
        best_cx=best.complexity,
        # The candidate that minimises TEST loss (may be different)
        best_by_test_train_loss=best_by_test[0].train_loss,
        best_by_test_test_loss=best_by_test[1],
        best_by_test_cx=best_by_test[0].complexity,
        front_size=len(front),
        runtime=rt,
        best_tree=str(best.tree),
    )


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------

DEFAULT_BUDGETS = (
    (60, 25),    # 1500 evaluations
    (120, 50),   # 6000
    (240, 100),  # 24000
    (360, 120),  # 43200
)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budgets", type=str, default=None,
                   help="Comma-sep pop:gens pairs, e.g. 60:25,120:50")
    args = p.parse_args(argv)

    budgets = DEFAULT_BUDGETS
    if args.budgets:
        budgets = tuple(
            tuple(map(int, pair.split(":")))
            for pair in args.budgets.split(",")
        )

    print("=== Heat eq paired diagnostic: TRAIN/TEST + compute scaling ===")
    print(f"T={args.T} X={args.X} α={args.alpha}")
    print(f"Budgets: {budgets}")
    print(f"Seeds per budget: {args.seeds}")

    # Simulate ONE train + ONE test trajectory pair (different IC, same α).
    print(f"\nSimulating TRAIN trajectory (ic_seed=100)...")
    U_train = simulate_heat_with_ic(
        T=args.T, X=args.X, alpha=args.alpha,
        noise_std=0.002, ic_seed=100, sim_seed=0,
    )
    dt_U_train = np.zeros_like(U_train)
    dt_U_train[:-1] = U_train[1:] - U_train[:-1]

    print(f"Simulating TEST trajectory (ic_seed=200, different IC, same α)...")
    U_test = simulate_heat_with_ic(
        T=args.T, X=args.X, alpha=args.alpha,
        noise_std=0.002, ic_seed=200, sim_seed=1,
    )
    dt_U_test = np.zeros_like(U_test)
    dt_U_test[:-1] = U_test[1:] - U_test[:-1]

    oracle_train = oracle_loss_on(U_train, dt_U_train, args.alpha)
    oracle_test = oracle_loss_on(U_test, dt_U_test, args.alpha)
    var_train = float(np.var(dt_U_train[1:-1, 1:-1]))
    var_test = float(np.var(dt_U_test[1:-1, 1:-1]))
    print(f"TRAIN: var={var_train:.4g}  oracle={oracle_train:.4g}")
    print(f"TEST:  var={var_test:.4g}  oracle={oracle_test:.4g}")
    print()

    # Sweep
    all_runs = []
    t_start = time.time()
    for pop, gens in budgets:
        budget = pop * gens
        print(f"--- pop={pop}, gens={gens} (budget={budget}) ---")
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            r = run_one(U_train, dt_U_train, U_test, dt_U_test,
                        pop=pop, gens=gens, seed=seed)
            all_runs.append(r)
            train_ratio = r["train_loss_best"] / oracle_train
            test_ratio = r["test_loss_best"] / oracle_test
            test_best_ratio = r["best_by_test_test_loss"] / oracle_test
            print(f"  seed={seed} "
                  f"train_loss/oracle={train_ratio:.2f} "
                  f"test_loss(best_train)/oracle={test_ratio:.2f} "
                  f"test_loss(best_test)/oracle={test_best_ratio:.2f} "
                  f"cx={r['best_cx']} ({r['runtime']:.1f}s)")
        print()

    total_rt = time.time() - t_start
    print(f"Total wall-clock: {total_rt:.1f}s")

    # Write report
    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_traintest_computescale.md"
    write_report(all_runs, budgets, args, oracle_train, oracle_test,
                 var_train, var_test, total_rt, out_path)
    return 0


def write_report(all_runs, budgets, args, oracle_train, oracle_test,
                 var_train, var_test, total_rt, out_path):
    L = ["# Heat eq paired diagnostic — TRAIN/TEST split + compute scaling", ""]
    L.append("Two diagnostics in one experiment:")
    L.append("- TRAIN/TEST split: fit on trajectory A, evaluate on B "
             "(different IC, same α). Detects overfit if TRAIN < TEST.")
    L.append("- Compute scaling: vary budget at fixed N. Detects "
             "underfit (improves with compute) vs fundamental search "
             "limit (flat in compute).")
    L.append("")
    L.append(f"**Setup:** T={args.T}, X={args.X}, α={args.alpha}. "
             f"TRAIN ic_seed=100, TEST ic_seed=200 (different bump "
             f"locations/amplitudes/counts).")
    L.append("")
    L.append(f"**TRAIN:** var={var_train:.4g}  oracle={oracle_train:.4g}")
    L.append(f"**TEST:**  var={var_test:.4g}  oracle={oracle_test:.4g}")
    L.append("")
    L.append(f"**Wall-clock:** {total_rt:.1f}s")
    L.append("")
    L.append("## Compute scaling table (medians across seeds)")
    L.append("")
    L.append("| pop × gens | budget | median train/oracle | median test/oracle | median best-by-test/oracle | median cx |")
    L.append("|---|---|---|---|---|---|")

    train_medians = []
    test_medians = []
    best_test_medians = []
    budget_values = []

    for pop, gens in budgets:
        budget = pop * gens
        runs = [r for r in all_runs if r["pop"] == pop and r["gens"] == gens]
        med_train = float(np.median([r["train_loss_best"] for r in runs]))
        med_test = float(np.median([r["test_loss_best"] for r in runs]))
        med_best_test = float(np.median([r["best_by_test_test_loss"] for r in runs]))
        med_cx = float(np.median([r["best_cx"] for r in runs]))
        train_medians.append(med_train / oracle_train)
        test_medians.append(med_test / oracle_test)
        best_test_medians.append(med_best_test / oracle_test)
        budget_values.append(budget)
        L.append(f"| {pop}×{gens} | {budget} | "
                 f"{med_train/oracle_train:.2f} | "
                 f"{med_test/oracle_test:.2f} | "
                 f"{med_best_test/oracle_test:.2f} | "
                 f"{med_cx:.0f} |")

    L.append("")
    L.append("Notes:")
    L.append("- `train/oracle`: how the best-by-train-loss candidate scores on TRAIN")
    L.append("- `test/oracle`: how that SAME candidate scores on TEST (same tree, new data)")
    L.append("- `best-by-test/oracle`: the LOWEST test loss achieved by ANY tree in the front (selection-with-test-knowledge)")
    L.append("")

    # Verdict
    L.append("## Verdict")
    L.append("")
    # Compute the diagnostic
    median_train_ratio = float(np.median(train_medians))
    median_test_ratio = float(np.median(test_medians))
    diff_train_test = float(np.median(test_medians)) - float(np.median(train_medians))

    train_low = train_medians[0]
    train_high = train_medians[-1]
    compute_factor = budget_values[-1] / budget_values[0]
    improvement_factor = train_low / max(train_high, 1e-12)

    L.append(f"- Median TRAIN/oracle across all budgets: **{median_train_ratio:.2f}**")
    L.append(f"- Median TEST/oracle across all budgets: **{median_test_ratio:.2f}**")
    L.append(f"- TRAIN/TEST gap (test - train): **{diff_train_test:+.2f}**")
    L.append(f"- Compute scaling: {compute_factor:.1f}× compute → "
             f"{train_low:.2f}× → {train_high:.2f}× train/oracle  "
             f"(improvement factor: {improvement_factor:.2f}×)")
    L.append("")

    # Decision tree
    overfit_threshold = 0.5  # gap > 50% of oracle = clear overfit
    compute_improvement_threshold = 1.15  # >15% improvement = "scales with compute"

    if diff_train_test > overfit_threshold:
        L.append(f"**Diagnosis: REAL OVERFIT.** TEST loss is {diff_train_test:.2f}× oracle ")
        L.append(f"higher than TRAIN. The GP is fitting trajectory-specific noise that ")
        L.append(f"doesn't transfer. Mitigation: regularization (lower max_depth, ")
        L.append(f"higher parsimony) OR smaller hypothesis class OR more diverse training data.")
    elif improvement_factor >= compute_improvement_threshold:
        L.append(f"**Diagnosis: UNDERFIT, FIXABLE WITH COMPUTE.** Train and test ratios ")
        L.append(f"agree closely (gap = {diff_train_test:+.2f}); more compute improves ")
        L.append(f"train/oracle ({improvement_factor:.2f}×). The mechanism is being ")
        L.append(f"approached as budget grows. Worth investing in search efficiency ")
        L.append(f"(ε-lexicase, annealed mutation temperature, etc.).")
    else:
        L.append(f"**Diagnosis: SEARCH-LIMITED, COMPUTE DOESN'T HELP.** Train and test ")
        L.append(f"ratios agree (gap = {diff_train_test:+.2f}) but compute scaling shows ")
        L.append(f"only {improvement_factor:.2f}× improvement across {compute_factor:.0f}× more compute. ")
        L.append(f"Beyond a certain budget, the search hits a structural limit. ")
        L.append(f"Need either: (a) different architecture (Mode 2 / derivation grammars), ")
        L.append(f"(b) different methodology (distributive observation), or (c) acknowledge ")
        L.append(f"this benchmark class is out of reach for current tessera.")
    L.append("")

    # Per-seed details
    L.append("## Per-seed details")
    L.append("")
    L.append("| budget | seed | train_loss | test_loss | train/oracle | test/oracle | cx | runtime | best tree (truncated) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in all_runs:
        tree_str = r["best_tree"].encode("ascii", "replace").decode("ascii")
        if len(tree_str) > 60:
            tree_str = tree_str[:57] + "..."
        L.append(f"| {r['pop']}×{r['gens']} | {r['seed']} | "
                 f"{r['train_loss_best']:.3g} | {r['test_loss_best']:.3g} | "
                 f"{r['train_loss_best']/oracle_train:.2f} | "
                 f"{r['test_loss_best']/oracle_test:.2f} | "
                 f"{r['best_cx']} | {r['runtime']:.1f}s | "
                 f"`{tree_str}` |")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_heat_equation_traintest_computescale.py")
    L.append("```")
    L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
