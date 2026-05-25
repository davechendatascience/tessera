"""Heat-equation sample-complexity calibration.

Implements §6 of `docs/research/randomized_recovery_bounds_for_sr.md`:
extend the heat-equation discovery benchmark with a sample-count
sweep, measure success-rate-vs-N over multiple seeds, and produce a
report comparing tessera's empirical sample complexity against the
oracle achievability and against Boullé-Townsend-style scaling
expectations.

Setup
-----
Simulate ONE long heat-equation trajectory (T_max=400, X=32). At each
target N, use the first N_t timesteps of that trajectory as the GP's
training data (same physics, different observation budget). For each
N, run the GP across multiple seeds and record:

  - "accuracy success":  best_loss / oracle_loss < threshold
  - "structural success": Pareto front contains Laplacian_5pt operator
  - best loss / oracle ratio

Output
------
benchmarks/results/heat_equation_sample_complexity.md — markdown
report with the sample-complexity curve and verdict.

Wall-clock: ~5-10 minutes for the default (5 trajectory lengths × 3
seeds × pop=60 × gens=25 × enable_2d=True).
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
from tessera.expression.measure_2d import (
    measure_2d_laplacian_5pt, measure_2d_diff_t,
)

# Reuse the existing simulator (benchmarks/ is not a package, so add to path).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_discovery import simulate_heat_1d  # noqa: E402


# ----------------------------------------------------------------------
# Sample-count config
# ----------------------------------------------------------------------

DEFAULT_T_VALUES = (25, 50, 100, 200, 400)
"""Trajectory lengths to sweep. Sample count at each T is roughly
(T-2) * (X-2) interior points."""

DEFAULT_X = 32
DEFAULT_N_SEEDS = 3
DEFAULT_ALPHA = 0.05
DEFAULT_AMPLITUDE = 10.0
DEFAULT_NOISE_STD = 0.002

SUCCESS_RATIO = 2.0
"""Accuracy success: best_loss / oracle_loss < SUCCESS_RATIO. Loose
threshold; we're testing whether the GP found Laplacian-LIKE structure,
not whether it perfectly matched coefficients."""


# ----------------------------------------------------------------------
# Structural detection
# ----------------------------------------------------------------------

def tree_contains_laplacian(node) -> bool:
    """Check whether any subtree applies the 5-point Laplacian
    Measure2D. Structural recovery indicator."""
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            name = str(sub.measure_2d)
            if "laplacian" in name.lower() or "lap" in name.lower():
                return True
    return False


def tree_extracted_alpha(node, target_alpha: float = DEFAULT_ALPHA,
                        tolerance: float = 0.5) -> float | None:
    """If the tree has shape `Const * Laplacian(U)` or similar,
    extract the Const and return it. Returns None if no clean match.

    Uses a loose tolerance — we report the extracted value; the user
    judges whether it's close to α."""
    from tessera.expression.tree import BinOp, Const
    for sub in iter_subtrees(node):
        if (isinstance(sub, BinOp) and sub.op == "mul"
                and isinstance(sub.a, Const)
                and isinstance(sub.b, FunctionalOp2D)
                and "lap" in str(sub.b.measure_2d).lower()):
            return float(sub.a.value)
        if (isinstance(sub, BinOp) and sub.op == "mul"
                and isinstance(sub.b, Const)
                and isinstance(sub.a, FunctionalOp2D)
                and "lap" in str(sub.a.measure_2d).lower()):
            return float(sub.b.value)
    return None


# ----------------------------------------------------------------------
# Per-(T, seed) runner
# ----------------------------------------------------------------------

def run_one(
    U_full: np.ndarray,
    dt_U_full: np.ndarray,
    oracle_loss_full: float,
    T_use: int,
    seed: int,
    pop: int = 120,
    gens: int = 40,
) -> dict:
    """Run GP on the first T_use timesteps of the trajectory."""
    U = U_full[:T_use]
    dt_U = dt_U_full[:T_use]

    # Interior target variance (for parsimony scaling).
    interior = (slice(1, -1), slice(1, -1))
    target_var = float(np.var(dt_U[interior]))

    # Oracle on the truncated trajectory.
    lap_U = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    oracle_loss = float(
        np.mean((DEFAULT_ALPHA * lap_U[interior] - dt_U[interior]) ** 2)
    )

    parsimony = max(target_var * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=pop,
        n_gens=gens,
        init_max_depth=4,
        parsimony=parsimony,
        # Higher patience — heat-equation discovery needs the GP to
        # explore Measure2D mutations, which doesn't happen if it
        # early-stops at the predict-zero fixed point. Effectively
        # disable patience-based early stop.
        early_stop_patience=gens,
        seed=seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=False,
    )
    gp = GP(cfg)
    env = {"U": U}
    t0 = time.time()
    front = gp.run(env, dt_U, feature_names=["U"])
    rt = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)

    # Structural: any tree in front contains Laplacian?
    has_laplacian = any(tree_contains_laplacian(c.tree) for c in front)
    # If we can extract an α-coefficient, report it.
    alphas = [tree_extracted_alpha(c.tree) for c in front]
    alphas = [a for a in alphas if a is not None]

    accuracy_success = best.train_loss < SUCCESS_RATIO * oracle_loss

    n_samples_interior = (T_use - 2) * (U.shape[1] - 2)

    return dict(
        T=T_use,
        seed=seed,
        n_samples=n_samples_interior,
        target_var=target_var,
        oracle_loss=oracle_loss,
        best_loss=best.train_loss,
        best_cx=best.complexity,
        loss_ratio=best.train_loss / oracle_loss if oracle_loss > 0 else np.nan,
        accuracy_success=accuracy_success,
        structural_success=has_laplacian,
        extracted_alphas=alphas,
        front_size=len(front),
        runtime=rt,
    )


# ----------------------------------------------------------------------
# Sweep + report
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T-values", type=int, nargs="+", default=list(DEFAULT_T_VALUES),
                   help="Trajectory lengths to sweep.")
    p.add_argument("--X", type=int, default=DEFAULT_X)
    p.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                   help="Number of GP seeds per T value.")
    p.add_argument("--pop", type=int, default=120)
    p.add_argument("--gens", type=int, default=40)
    args = p.parse_args(argv)

    T_max = max(args.T_values)
    print(f"=== Heat equation sample-complexity calibration ===")
    print(f"T_values={args.T_values}  X={args.X}  seeds={args.seeds}")
    print(f"pop={args.pop}  gens={args.gens}")
    print(f"Simulating one trajectory at T_max={T_max}...")

    # Simulate ONE long trajectory; use prefixes for smaller N values.
    U_full = simulate_heat_1d(
        T=T_max, X=args.X, alpha=DEFAULT_ALPHA, seed=0,
        amplitude=DEFAULT_AMPLITUDE, noise_std=DEFAULT_NOISE_STD,
    )
    dt_U_full = np.zeros_like(U_full)
    dt_U_full[:-1] = U_full[1:] - U_full[:-1]

    # Oracle loss on full trajectory (reference).
    interior_full = (slice(1, -1), slice(1, -1))
    lap_full = measure_2d_laplacian_5pt().apply(U_full, fill_warmup=0.0)
    oracle_loss_full = float(np.mean(
        (DEFAULT_ALPHA * lap_full[interior_full] - dt_U_full[interior_full]) ** 2
    ))
    print(f"Oracle loss on full trajectory: {oracle_loss_full:.4g}")
    print()

    # Run sweep.
    all_runs = []
    t_start = time.time()
    for T_use in args.T_values:
        per_T = []
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            print(f"  T={T_use}  seed={seed}  ...", end="", flush=True)
            r = run_one(U_full, dt_U_full, oracle_loss_full,
                        T_use=T_use, seed=seed,
                        pop=args.pop, gens=args.gens)
            per_T.append(r)
            print(f"  best_cx={r['best_cx']}  "
                  f"loss/oracle={r['loss_ratio']:.2f}  "
                  f"struct={r['structural_success']}  "
                  f"acc={r['accuracy_success']}  "
                  f"({r['runtime']:.1f}s)")
        all_runs.extend(per_T)

    total_rt = time.time() - t_start
    print(f"\nTotal wall-clock: {total_rt:.1f}s")

    # Aggregate per T.
    out = Path(__file__).resolve().parent / "results" / "heat_equation_sample_complexity.md"
    write_report(all_runs, args, total_rt, oracle_loss_full, out)
    return 0


def write_report(all_runs, args, total_rt, oracle_loss_full, out_path):
    # Per-T aggregation
    per_T = {}
    for r in all_runs:
        per_T.setdefault(r["T"], []).append(r)

    L = ["# Heat-equation sample-complexity calibration", ""]
    L.append("Implements §6 of "
             "`docs/research/randomized_recovery_bounds_for_sr.md`.")
    L.append("")
    L.append(f"**Setup:** 1-D heat equation on X={args.X} grid, "
             f"α=0.05, Dirichlet BCs, noise σ=0.002. Initial bump "
             f"amplitude 10.0. One trajectory simulated at T_max="
             f"{max(args.T_values)}; smaller-N runs use prefix of "
             f"that trajectory.")
    L.append("")
    L.append(f"**GP config:** pop={args.pop}, gens={args.gens}, "
             f"enable_2d=True, parsimony auto-scaled to 0.1% of "
             f"target variance per N. {args.seeds} GP seeds per N.")
    L.append("")
    L.append(f"**Wall-clock:** {total_rt:.1f}s.")
    L.append("")
    L.append("## Acceptance criteria")
    L.append("")
    L.append(f"- **Accuracy success**: best Pareto loss < {SUCCESS_RATIO}× oracle loss")
    L.append("- **Structural success**: any Pareto tree contains the 5-point Laplacian operator")
    L.append("")
    L.append("## Sweep results")
    L.append("")
    L.append("| T | N samples | Accuracy success rate | Structural success rate | "
             "Median loss / oracle | Median best cx | "
             "α-extracted (if found) |")
    L.append("|---|---|---|---|---|---|---|")

    n_T = len(per_T)
    accuracy_rates = []
    structural_rates = []
    sample_counts = []

    for T_use in sorted(per_T.keys()):
        runs = per_T[T_use]
        n = runs[0]["n_samples"]
        sample_counts.append(n)
        acc_rate = sum(r["accuracy_success"] for r in runs) / len(runs)
        struct_rate = sum(r["structural_success"] for r in runs) / len(runs)
        accuracy_rates.append(acc_rate)
        structural_rates.append(struct_rate)
        med_ratio = float(np.median([r["loss_ratio"] for r in runs]))
        med_cx = float(np.median([r["best_cx"] for r in runs]))
        alphas_found = [a for r in runs for a in r["extracted_alphas"]]
        alpha_str = (
            f"{np.median(alphas_found):.4f}±{np.std(alphas_found):.4f}"
            if alphas_found else "—"
        )
        L.append(f"| {T_use} | {n} | "
                 f"{acc_rate:.0%} ({sum(r['accuracy_success'] for r in runs)}/"
                 f"{len(runs)}) | "
                 f"{struct_rate:.0%} ({sum(r['structural_success'] for r in runs)}/"
                 f"{len(runs)}) | "
                 f"{med_ratio:.2f} | "
                 f"{med_cx:.0f} | "
                 f"{alpha_str} |")

    L.append("")
    L.append("## Verdict")
    L.append("")
    # Determine verdict heuristic
    smallest_N_with_high_accuracy = None
    smallest_N_with_high_structural = None
    for T_use, acc_rate, struct_rate, n in zip(
            sorted(per_T.keys()), accuracy_rates, structural_rates, sample_counts):
        if smallest_N_with_high_accuracy is None and acc_rate >= 0.66:
            smallest_N_with_high_accuracy = n
        if smallest_N_with_high_structural is None and struct_rate >= 0.66:
            smallest_N_with_high_structural = n

    if smallest_N_with_high_accuracy is None:
        L.append("- **Accuracy success not reached** even at the largest N tested.")
        L.append("  Either the GP is search-bottlenecked (not sample-bottlenecked),")
        L.append("  or the parsimony schedule is preventing recovery, or the budget")
        L.append("  (pop × gens) is too small.")
    else:
        L.append(f"- **Accuracy success reached at N ≈ {smallest_N_with_high_accuracy}** "
                 f"(≥ 2/3 seeds achieve loss < {SUCCESS_RATIO}× oracle).")

    if smallest_N_with_high_structural is None:
        L.append("- **Structural success not reached** — Pareto fronts don't contain")
        L.append("  the Laplacian operator. The GP may be finding equivalent forms")
        L.append("  (`diff_t` shifts) that match accuracy without exposing the physics.")
    else:
        L.append(f"- **Structural success reached at N ≈ {smallest_N_with_high_structural}** "
                 f"— Pareto front contains the 5-point Laplacian.")

    L.append("")
    L.append("## Reading the curve in the language of recovery bounds")
    L.append("")
    L.append("Boullé-Townsend-family theorems predict, for an unrestricted")
    L.append("smooth Green's function in 1-D, **polynomial-in-(1/ε) sample**")
    L.append("**complexity** with **logarithmic** dimension penalty. Translation")
    L.append("for our setting:")
    L.append("")
    L.append("- The unrestricted bound is on input-output PDE-solution pairs; we")
    L.append("  have one trajectory (one IC) but many spacetime samples from it.")
    L.append("  Direct transfer of the constant factor isn't possible.")
    L.append("- The SHAPE the theorem predicts — success rises smoothly with N,")
    L.append("  with diminishing returns at large N — is testable.")
    L.append("- Tessera's vocabulary commitment (Measure2D includes Laplacian_5pt")
    L.append("  as a primitive) is a STRONG prior that target ∈ V. If the GP")
    L.append("  successfully uses this prior, sample complexity should be BETTER")
    L.append("  than the unrestricted bound — the vocab-restriction advantage")
    L.append("  framed in §4 of the recovery-bounds research note.")
    L.append("")
    L.append("**What this experiment tells us specifically:**")
    L.append("")
    if smallest_N_with_high_structural is not None and smallest_N_with_high_structural <= 1500:
        L.append("- Tessera achieves structural recovery at modest N. This is")
        L.append("  consistent with vocab-restriction advantage being real:")
        L.append("  having Laplacian_5pt as a primitive lets the GP commit to")
        L.append("  the right structure quickly.")
    elif smallest_N_with_high_structural is not None:
        L.append("- Tessera achieves structural recovery only at larger N. The")
        L.append("  vocabulary prior helps but search overhead is non-trivial;")
        L.append("  combinatorial GP cost dominates query cost in this regime.")
    else:
        L.append("- Tessera fails structural recovery within the tested N range.")
        L.append("  Either the budget is too small, or the GP can't reliably")
        L.append("  select the Laplacian primitive even when it's available.")
        L.append("  This would be a real surprise; would require deeper analysis.")
    L.append("")
    L.append("## Open follow-on questions")
    L.append("")
    L.append("- **What if we DON'T have Laplacian_5pt as a primitive?** Run the")
    L.append("  same sweep with the Measure2D vocabulary restricted to differential")
    L.append("  building blocks (∂/∂x, ∂/∂t) only. The expected effect: the GP")
    L.append("  has to COMPOSE the Laplacian, taking more samples to find it.")
    L.append("  This would quantify the vocab-restriction advantage directly.")
    L.append("- **Noise scaling.** Repeat at noise_std ∈ {0, 0.002, 0.02, 0.2}.")
    L.append("  Boullé-Townsend-family bounds degrade with noise; we should see")
    L.append("  the success curve shift right as noise grows.")
    L.append("- **Budget vs samples decoupled.** Vary pop and gens independently")
    L.append("  to separate sample-bottleneck from search-bottleneck.")
    L.append("")
    L.append("## Per-seed details")
    L.append("")
    L.append("| T | seed | N | best cx | loss / oracle | structural? | accuracy? | runtime |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in all_runs:
        L.append(f"| {r['T']} | {r['seed']} | {r['n_samples']} | "
                 f"{r['best_cx']} | {r['loss_ratio']:.2f} | "
                 f"{'✓' if r['structural_success'] else '✗'} | "
                 f"{'✓' if r['accuracy_success'] else '✗'} | "
                 f"{r['runtime']:.1f}s |")
    L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
