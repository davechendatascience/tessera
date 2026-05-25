"""Minimal Mode-1 experiment: parsimony=0 on heat equation discovery.

Hypothesis: with no parsimony pressure, the GP's Pareto front should
contain BOTH the cx=2 `diff_t(U)` shape (which fits data via near-
tautology) AND the cx≥4 `α·Laplacian(U)` shape (the mechanistic
answer). The current parsimony-driven scoring suppresses the latter
because the former ties on accuracy at lower cx.

Outcome interpretation:

  - Front contains BOTH shapes → Mode 1 minimal works. Removing
    parsimony exposes the mechanism candidate. The downstream user
    chooses which Pareto point matches their task.

  - Front contains ONLY diff_t shape → Search itself doesn't reach
    Laplacian compositions even without parsimony pressure. Mode 2
    (derivation grammars) is needed; Mode 1 alone isn't sufficient.

  - Front contains many Laplacian-flavoured candidates but no clean
    `Const · Laplacian(U)` → vocabulary is being used but search
    doesn't find the canonical form. Edge case; informative.

Setup: ~6000 spacetime samples (T=200, X=32). 5 GP seeds at
parsimony=0 vs the existing parsimony= var_y * 0.001 baseline. Same
budget (pop=120, gens=40).

Wall-clock target: ~3 minutes.
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
from tessera.expression.tree import BinOp, Const

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_discovery import simulate_heat_1d  # noqa: E402


# ----------------------------------------------------------------------
# Tree structural detection
# ----------------------------------------------------------------------

def measure_op_name(node) -> str:
    """Return a short identifier for the dominant Measure2D in a tree."""
    names = []
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            n = str(sub.measure_2d).lower()
            if "laplacian" in n or "lap" in n:
                names.append("laplacian")
            elif "diff_t" in n or "difft" in n:
                names.append("diff_t")
            elif "diff_x" in n or "diffx" in n:
                names.append("diff_x")
            else:
                names.append(n.split("(")[0])
    return ",".join(names) if names else "(none)"


def has_laplacian(node) -> bool:
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            if "lap" in str(sub.measure_2d).lower():
                return True
    return False


def has_diff_t(node) -> bool:
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            n = str(sub.measure_2d).lower()
            if "diff_t" in n or "difft" in n:
                return True
    return False


def extracted_alpha(node) -> float | None:
    for sub in iter_subtrees(node):
        if (isinstance(sub, BinOp) and sub.op == "mul"
                and isinstance(sub.b, FunctionalOp2D)
                and "lap" in str(sub.b.measure_2d).lower()
                and isinstance(sub.a, Const)):
            return float(sub.a.value)
        if (isinstance(sub, BinOp) and sub.op == "mul"
                and isinstance(sub.a, FunctionalOp2D)
                and "lap" in str(sub.a.measure_2d).lower()
                and isinstance(sub.b, Const)):
            return float(sub.b.value)
    return None


# ----------------------------------------------------------------------
# One GP run with given parsimony
# ----------------------------------------------------------------------

def run_one(U, dt_U, parsimony, seed, pop=120, gens=40):
    target_var = float(np.var(dt_U[1:-1, 1:-1]))
    cfg = GPConfig(
        pop_size=pop,
        n_gens=gens,
        init_max_depth=4,
        parsimony=parsimony,
        early_stop_patience=gens,  # disable
        seed=seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=False,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run({"U": U}, dt_U, feature_names=["U"])
    rt = time.time() - t0

    front_summary = []
    for c in front:
        front_summary.append(dict(
            cx=c.complexity,
            loss=c.train_loss,
            has_lap=has_laplacian(c.tree),
            has_diff_t=has_diff_t(c.tree),
            ops=measure_op_name(c.tree),
            extracted_alpha=extracted_alpha(c.tree),
            tree=str(c.tree),
        ))
    return dict(
        parsimony=parsimony,
        seed=seed,
        target_var=target_var,
        runtime=rt,
        front=front_summary,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pop", type=int, default=120)
    p.add_argument("--gens", type=int, default=40)
    args = p.parse_args(argv)

    print("=== Minimal Mode-1 experiment: parsimony=0 vs default ===")
    print(f"T={args.T}  X={args.X}  seeds={args.seeds}  pop={args.pop}  gens={args.gens}")

    U = simulate_heat_1d(T=args.T, X=args.X, alpha=0.05, seed=0,
                         amplitude=10.0, noise_std=0.002)
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    target_var = float(np.var(dt_U[1:-1, 1:-1]))

    # Oracle loss
    lap_U = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    oracle_loss = float(np.mean((0.05 * lap_U[interior] - dt_U[interior]) ** 2))
    print(f"Target var: {target_var:.4g}  Oracle loss: {oracle_loss:.4g}")
    print()

    # Default parsimony (the baseline from sample-complexity calibration)
    default_parsimony = max(target_var * 0.001, 1e-9)

    settings = [
        ("default", default_parsimony),
        ("zero", 0.0),
    ]

    all_runs = {label: [] for label, _ in settings}

    for label, parsimony in settings:
        print(f"--- parsimony = {label} ({parsimony:.4g}) ---")
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            r = run_one(U, dt_U, parsimony, seed,
                        pop=args.pop, gens=args.gens)
            all_runs[label].append(r)
            # Summarize this seed's front
            n_with_lap = sum(1 for c in r["front"] if c["has_lap"])
            n_with_diff_t = sum(1 for c in r["front"] if c["has_diff_t"])
            best_with_lap = min((c for c in r["front"] if c["has_lap"]),
                                key=lambda c: c["loss"], default=None)
            best_overall = min(r["front"], key=lambda c: c["loss"])
            print(f"  seed={seed} front_size={len(r['front'])} "
                  f"with_lap={n_with_lap} with_diff_t={n_with_diff_t} "
                  f"best_overall_loss={best_overall['loss']:.4g} "
                  f"cx={best_overall['cx']} "
                  f"ops={best_overall['ops']}", end="")
            if best_with_lap is not None:
                print(f" | best_lap_loss={best_with_lap['loss']:.4g} "
                      f"cx={best_with_lap['cx']} "
                      f"α={best_with_lap['extracted_alpha']}", end="")
            print(f" ({r['runtime']:.1f}s)")
        print()

    # Write the report
    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_mode1_parsimony_zero.md"
    write_report(all_runs, settings, oracle_loss, target_var, args, out_path)
    return 0


def write_report(all_runs, settings, oracle_loss, target_var, args, out_path):
    L = ["# Heat equation Mode-1 minimal experiment — parsimony=0 vs default", ""]
    L.append("Tests whether removing parsimony pressure exposes the")
    L.append("α·Laplacian candidate on the Pareto front alongside diff_t.")
    L.append("")
    L.append(f"**Setup:** T={args.T}, X={args.X}, {args.seeds} seeds each, "
             f"pop={args.pop}, gens={args.gens}. Oracle loss = {oracle_loss:.4g}.")
    L.append("")

    # Per-setting summary
    for label, parsimony in settings:
        runs = all_runs[label]
        n_seeds = len(runs)
        n_with_lap = sum(
            1 for r in runs if any(c["has_lap"] for c in r["front"])
        )
        n_with_diff_t = sum(
            1 for r in runs if any(c["has_diff_t"] for c in r["front"])
        )
        n_with_both = sum(
            1 for r in runs
            if any(c["has_lap"] for c in r["front"])
            and any(c["has_diff_t"] for c in r["front"])
        )
        best_lap_losses = []
        for r in runs:
            lap_cands = [c for c in r["front"] if c["has_lap"]]
            if lap_cands:
                best_lap_losses.append(
                    min(c["loss"] for c in lap_cands)
                )

        L.append(f"## parsimony = {label} ({parsimony:.4g})")
        L.append("")
        L.append(f"- Seeds with Laplacian in front: **{n_with_lap}/{n_seeds}**")
        L.append(f"- Seeds with diff_t in front: **{n_with_diff_t}/{n_seeds}**")
        L.append(f"- Seeds with BOTH: **{n_with_both}/{n_seeds}**")
        if best_lap_losses:
            L.append(f"- Median best-Laplacian-candidate loss / oracle: "
                     f"**{np.median(best_lap_losses)/oracle_loss:.2f}×**")
        L.append("")

        # Per-seed detail
        L.append("### Per-seed Pareto fronts")
        L.append("")
        for r in runs:
            L.append(f"**seed={r['seed']}**  (runtime {r['runtime']:.1f}s)")
            L.append("")
            L.append("| cx | loss | loss/oracle | ops | tree |")
            L.append("|---|---|---|---|---|")
            for c in r["front"]:
                ratio = c["loss"] / oracle_loss if oracle_loss > 0 else float("nan")
                tree_str = c["tree"].encode("ascii", "replace").decode("ascii")
                if len(tree_str) > 80:
                    tree_str = tree_str[:77] + "..."
                lap_marker = " 🅛" if c["has_lap"] else ""
                dt_marker = " 🅓" if c["has_diff_t"] else ""
                L.append(f"| {c['cx']} | {c['loss']:.4g} | "
                         f"{ratio:.2f} | {c['ops']}{lap_marker}{dt_marker} | "
                         f"`{tree_str}` |")
            L.append("")

    L.append("## Verdict")
    L.append("")
    # Compare default vs zero
    default_runs = all_runs["default"]
    zero_runs = all_runs["zero"]
    default_lap = sum(
        1 for r in default_runs if any(c["has_lap"] for c in r["front"])
    )
    zero_lap = sum(
        1 for r in zero_runs if any(c["has_lap"] for c in r["front"])
    )

    if zero_lap > default_lap:
        L.append(f"**Removing parsimony EXPOSED the Laplacian candidate.**")
        L.append(f"Default parsimony: {default_lap}/{len(default_runs)} seeds had it.")
        L.append(f"parsimony=0: {zero_lap}/{len(zero_runs)} seeds had it.")
        L.append("")
        L.append("Mode 1 minimal works for this benchmark — the mechanism")
        L.append("candidate WAS in the search but was being parsimony-suppressed.")
        L.append("Downstream user can pick from a parsimony-free front.")
    elif zero_lap == default_lap == 0:
        L.append(f"**Laplacian not found at either parsimony setting** "
                 f"(0/{len(zero_runs)} for both).")
        L.append("")
        L.append("Mode 1 alone is insufficient — the issue isn't scoring,")
        L.append("it's that search doesn't reach Laplacian compositions in")
        L.append("the given budget. Mode 2 (derivation grammars) is needed,")
        L.append("OR a different vocabulary curation move (e.g., remove")
        L.append("diff_t to force composition).")
    elif zero_lap == default_lap:
        L.append(f"**No change between parsimony=0 and default** "
                 f"({zero_lap}/{len(zero_runs)} for both).")
        L.append("")
        L.append("Suggests parsimony at this small value isn't the binding")
        L.append("constraint; either way the GP finds (or doesn't find) the")
        L.append("Laplacian. If non-zero, parsimony at this level is mostly")
        L.append("decorative. Search structure is the actual lever.")
    else:
        L.append(f"**Removing parsimony REDUCED Laplacian discovery** "
                 f"({default_lap} → {zero_lap}/{len(default_runs)} seeds). "
                 f"Surprising; would need re-runs to verify.")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
