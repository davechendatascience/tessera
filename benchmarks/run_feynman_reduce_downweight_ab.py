"""A/B test the reduce_* downweight on Feynman subset.

The reduce_* mutation downweight (10x, from yesterday's heat eq fix)
is the only default-on behavior change. This benchmark checks whether
the change generalizes harmlessly to the broader Feynman suite, or
harms benchmarks that might benefit from reduce_* operators.

Hypothesis (testable): for closed-form math targets (Feynman), reduce_*
operators are useless — they collapse arrays to scalars, which can't
fit per-sample targets. The downweight should be either NEUTRAL (GP
wasn't using them anyway) or POSITIVE (less wasted exploration).

If the downweight HURTS Feynman, that's a real generalization concern
that warrants making it config-gated rather than default.

Setup: 8 Feynman equations, GP with pop=200 gens=60, 3 seeds each.
Run twice: once with UN_OP_WEIGHTS uniform, once with reduce_* × 0.1.
Compare best train MSE per equation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig
from tessera.expression import mutation as _mutation

# Reuse the Feynman subset definitions
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_feynman_subset import SUBSET  # noqa: E402


def set_reduce_weight(weight: float) -> dict[str, float]:
    """Set UN_OP_WEIGHTS for all reduce_* ops. Returns the prior dict."""
    prior = dict(_mutation.UN_OP_WEIGHTS)
    for op in _mutation.UN_OPS:
        if op.startswith("reduce_"):
            _mutation.UN_OP_WEIGHTS[op] = weight
    return prior


def restore_un_op_weights(prior: dict[str, float]) -> None:
    _mutation.UN_OP_WEIGHTS.clear()
    _mutation.UN_OP_WEIGHTS.update(prior)


def run_one(env, y, feature_names, seed, pop=200, gens=60):
    var_y = float(np.var(y))
    parsimony = max(var_y * 0.005, 1e-4)
    cfg = GPConfig(
        pop_size=pop,
        n_gens=gens,
        init_max_depth=4,
        parsimony=parsimony,
        early_stop_patience=20,
        seed=seed,
        pointwise_only=True,
        verbose=False,
        optimize_constants_every=3,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    rt = time.time() - t0
    best = min(front, key=lambda c: c.train_loss)
    return dict(
        seed=seed,
        best_loss=best.train_loss,
        best_cx=best.complexity,
        runtime=rt,
        rel_to_var=best.train_loss / var_y if var_y > 0 else float("nan"),
        tree=str(best.tree),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--pop", type=int, default=150)
    p.add_argument("--gens", type=int, default=50)
    args = p.parse_args(argv)

    print("=== Feynman A/B: reduce_* uniform vs downweighted ===")
    print(f"{len(SUBSET)} equations × 2 modes × {args.seeds} seeds")
    print(f"pop={args.pop} gens={args.gens}")
    print()

    results = []
    t_start = time.time()

    for name, formula, sampler in SUBSET:
        env, y = sampler()
        feature_names = list(env.keys())
        print(f"[{name}] {formula}")

        per_mode = {}
        for mode_name, weight in [("downweight_0.1", 0.1), ("uniform_1.0", 1.0)]:
            prior = set_reduce_weight(weight)
            try:
                seed_results = []
                for seed_idx in range(args.seeds):
                    seed = 2026 + seed_idx
                    r = run_one(env, y, feature_names, seed,
                                pop=args.pop, gens=args.gens)
                    seed_results.append(r)
                per_mode[mode_name] = seed_results
            finally:
                restore_un_op_weights(prior)

            best_losses = [r["best_loss"] for r in seed_results]
            rels = [r["rel_to_var"] for r in seed_results]
            cxs = [r["best_cx"] for r in seed_results]
            rts = [r["runtime"] for r in seed_results]
            print(f"  {mode_name:>16}: "
                  f"loss(med)={np.median(best_losses):.4g}  "
                  f"rel(med)={np.median(rels):.4f}  "
                  f"cx(med)={np.median(cxs):.0f}  "
                  f"rt(med)={np.median(rts):.1f}s")
        results.append(dict(name=name, formula=formula, per_mode=per_mode))
        print()

    total_rt = time.time() - t_start
    print(f"Total wall-clock: {total_rt:.1f}s")

    out_path = Path(__file__).resolve().parent / "results" / "feynman_reduce_downweight_ab.md"
    write_report(results, args, total_rt, out_path)
    return 0


def write_report(results, args, total_rt, out_path):
    L = ["# Feynman A/B: reduce_* uniform vs downweighted", ""]
    L.append("Tests whether the reduce_* mutation downweight (shipped 2026-05-26")
    L.append("for heat equation discovery) generalizes harmlessly to Feynman")
    L.append("benchmarks, or harms equations that might benefit from reductions.")
    L.append("")
    L.append(f"**Setup:** 8 Feynman equations × 2 modes × {args.seeds} seeds. "
             f"pop={args.pop}, gens={args.gens}, optimize_constants every 3 gens.")
    L.append("")
    L.append(f"**Wall-clock:** {total_rt:.1f}s")
    L.append("")

    L.append("## Per-equation comparison (medians across seeds)")
    L.append("")
    L.append("| # | Equation | True formula | rel (downweight) | rel (uniform) | Δrel | cx (downweight) | cx (uniform) | Winner |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    wins_down = 0
    wins_unif = 0
    ties = 0

    for i, r in enumerate(results, 1):
        down_runs = r["per_mode"]["downweight_0.1"]
        unif_runs = r["per_mode"]["uniform_1.0"]
        rel_down = float(np.median([x["rel_to_var"] for x in down_runs]))
        rel_unif = float(np.median([x["rel_to_var"] for x in unif_runs]))
        cx_down = float(np.median([x["best_cx"] for x in down_runs]))
        cx_unif = float(np.median([x["best_cx"] for x in unif_runs]))
        delta = rel_down - rel_unif

        if rel_down < rel_unif * 0.95:
            winner = "**downweight**"
            wins_down += 1
        elif rel_unif < rel_down * 0.95:
            winner = "uniform"
            wins_unif += 1
        else:
            winner = "tie"
            ties += 1

        L.append(f"| {i} | {r['name']} | `{r['formula']}` | "
                 f"{rel_down:.4f} | {rel_unif:.4f} | "
                 f"{delta:+.4f} | {cx_down:.0f} | {cx_unif:.0f} | "
                 f"{winner} |")

    L.append("")

    L.append("## Verdict")
    L.append("")
    L.append(f"- **Downweight wins** (rel < uniform - 5%): {wins_down} / {len(results)}")
    L.append(f"- **Uniform wins** (rel < downweight - 5%): {wins_unif} / {len(results)}")
    L.append(f"- **Ties** (within 5%): {ties} / {len(results)}")
    L.append("")

    if wins_unif == 0:
        L.append("**No benchmark is harmed by the downweight.** Reduce_* operators")
        L.append("are useless for closed-form math targets (Feynman); the downweight")
        L.append("either improves or doesn't change outcomes. Safe to keep as default.")
    elif wins_unif <= 2:
        L.append(f"**Modest concern:** {wins_unif} benchmark(s) prefer uniform reduce_*.")
        L.append("Examine which equations and whether the GP was using reduce_* in a")
        L.append("meaningful way (vs just incidental).")
    else:
        L.append(f"**Real concern:** {wins_unif} benchmarks prefer uniform reduce_*.")
        L.append("The default downweight may be benchmark-specific to heat equation;")
        L.append("consider making it config-gated rather than default.")
    L.append("")

    L.append("## Per-seed details")
    L.append("")
    L.append("| Equation | mode | seed | best_loss | rel | cx | tree (truncated) | runtime |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        for mode_name, runs in r["per_mode"].items():
            for run in runs:
                tree_str = run["tree"].encode("ascii", "replace").decode("ascii")
                if len(tree_str) > 60:
                    tree_str = tree_str[:57] + "..."
                L.append(f"| {r['name']} | {mode_name} | {run['seed']} | "
                         f"{run['best_loss']:.4g} | {run['rel_to_var']:.4f} | "
                         f"{run['best_cx']} | `{tree_str}` | "
                         f"{run['runtime']:.1f}s |")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_feynman_reduce_downweight_ab.py")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
