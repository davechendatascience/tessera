"""Feynman A/B: scipy Nelder-Mead vs scipy BFGS const-opt.

UPDATED 2026-05-28: replaces the earlier NM-vs-jax_adam A/B after the
Colab T4 run showed jax_adam doesn't help on Feynman scale (small
data + heterogeneous trees → JIT compile overhead dominates). On
Colab the user reran with BFGS swapped in and saw genuine partial→
exact transitions; this local-run version reproduces that with the
full 30-equation SUBSET and the baseline pop=400/gens=120 budget.

Both methods are scipy-side; no JAX involvement. BFGS uses finite-
difference gradients via scipy.optimize.minimize. For smooth MSE
losses on small (3-5 dim) constant subspaces, BFGS converges in
~10-30 iters vs NM's ~30-50, with no JIT overhead.

Hypothesis (corrected): switching const-opt from Nelder-Mead
(gradient-free, default) to BFGS (quasi-Newton with finite-diff
gradients) closes the partial-tier gap on a meaningful fraction
of equations, particularly when constant precision is the bottleneck
rather than structural search.

Reuses SUBSET from run_feynman_extended.py (30 equations).

Pass criteria:
  - At least 2 equations move "partial" → "exact" (rel < 0.01)
  - No more than 1 regression on currently-exact equations
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig

# Reuse the 30-equation SUBSET from the extended Feynman runner
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_feynman_extended import SUBSET  # noqa: E402


def run_with_method(name, sampler, method: str, n_gens: int, pop_size: int):
    """Run one equation with the specified const-opt method.

    Method options:
      - "Nelder-Mead" — scipy default; gradient-free simplex
      - "BFGS" — scipy quasi-Newton with finite-diff gradient
      - "Powell" — scipy gradient-free, line-search based
    """
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    cfg = GPConfig(
        pop_size=pop_size,
        n_gens=n_gens,
        init_max_depth=5,
        parsimony=max(var_y * 0.005, 1e-4),
        early_stop_patience=25,
        seed=2026,
        pointwise_only=True,
        verbose=False,
        optimize_constants_every=3,
        optimize_constants_method=method,
        optimize_constants_maxiter=30,
        use_jax_population_eval=False,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")
    return dict(
        name=name, method=method, n_vars=len(feature_names),
        runtime=runtime, front_size=len(front),
        best_cx=best.complexity, best_loss=best.train_loss, best_rel=rel,
        best_tree=str(best.tree), var_y=var_y,
    )


def classify_verdict(rel: float) -> str:
    if rel < 0.01:
        return "exact"
    if rel < 0.20:
        return "partial"
    return "failed"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_gens", type=int, default=120)
    p.add_argument("--pop_size", type=int, default=400)
    p.add_argument("--quick", action="store_true",
                   help="Reduce pop/gens for fast smoke test")
    args = p.parse_args(argv)

    if args.quick:
        args.n_gens = 30
        args.pop_size = 100

    print(f"=== Feynman const-opt A/B (n_gens={args.n_gens}, pop={args.pop_size}) ===")
    print(f"Comparing Nelder-Mead (default) vs BFGS on {len(SUBSET)} equations\n")

    t_start = time.time()
    pairs = []
    for i, (name, formula, sampler, expected) in enumerate(SUBSET):
        print(f"\n[{i+1}/{len(SUBSET)}] {name}: {formula[:50]}")
        try:
            r_nm = run_with_method(name, sampler, "Nelder-Mead",
                                   args.n_gens, args.pop_size)
            print(f"  NM:    cx={r_nm['best_cx']:3d} rel={r_nm['best_rel']:.4f} "
                  f"({classify_verdict(r_nm['best_rel'])}) in {r_nm['runtime']:.1f}s")
        except Exception as e:
            print(f"  NM: FAILED — {e}")
            r_nm = {"name": name, "method": "Nelder-Mead", "best_rel": float("inf"),
                    "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        try:
            r_bfgs = run_with_method(name, sampler, "BFGS",
                                     args.n_gens, args.pop_size)
            print(f"  BFGS:  cx={r_bfgs['best_cx']:3d} rel={r_bfgs['best_rel']:.4f} "
                  f"({classify_verdict(r_bfgs['best_rel'])}) in {r_bfgs['runtime']:.1f}s")
        except Exception as e:
            print(f"  BFGS: FAILED — {e}")
            r_bfgs = {"name": name, "method": "BFGS", "best_rel": float("inf"),
                      "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        pairs.append((r_nm, r_bfgs))

    elapsed_total = time.time() - t_start
    print(f"\nTotal wall-clock: {elapsed_total:.1f}s")

    # Tally verdicts
    nm_verdicts = [classify_verdict(r["best_rel"]) for r, _ in pairs if "error" not in r]
    bfgs_verdicts = [classify_verdict(r["best_rel"]) for _, r in pairs if "error" not in r]

    def tally(vs):
        from collections import Counter
        c = Counter(vs)
        return {"exact": c.get("exact", 0), "partial": c.get("partial", 0),
                "failed": c.get("failed", 0)}
    nm_tally = tally(nm_verdicts)
    bfgs_tally = tally(bfgs_verdicts)

    print(f"\n=== Summary ({len(pairs)} eqs) ===")
    print(f"Nelder-Mead: {nm_tally['exact']} exact / {nm_tally['partial']} partial / {nm_tally['failed']} failed")
    print(f"BFGS:        {bfgs_tally['exact']} exact / {bfgs_tally['partial']} partial / {bfgs_tally['failed']} failed")

    # Transitions
    moves = {"partial→exact": 0, "failed→partial": 0, "failed→exact": 0,
             "exact→partial": 0, "exact→failed": 0, "partial→failed": 0,
             "same": 0}
    for nm, bfgs in pairs:
        if "error" in nm or "error" in bfgs:
            continue
        v_nm = classify_verdict(nm["best_rel"])
        v_bfgs = classify_verdict(bfgs["best_rel"])
        key = f"{v_nm}→{v_bfgs}" if v_nm != v_bfgs else "same"
        moves[key] = moves.get(key, 0) + 1
    print(f"\nTransitions (NM → BFGS):")
    for k, v in moves.items():
        if v > 0:
            print(f"  {k}: {v}")

    out_path = HERE / "results" / "feynman_constopt_ab.md"
    write_report(pairs, args, elapsed_total, nm_tally, bfgs_tally, moves, out_path)
    return 0


def write_report(pairs, args, runtime, nm_tally, bfgs_tally, moves, out_path):
    L = []
    L.append("# Feynman A/B: scipy Nelder-Mead vs scipy BFGS const-opt")
    L.append("")
    L.append("**Per-equation A/B comparison of constant optimization methods on")
    L.append("the 30-equation Feynman subset.**")
    L.append("")
    L.append(f"**GP config**: pop_size={args.pop_size}, n_gens={args.n_gens}, ")
    L.append(f"init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, ")
    L.append(f"pointwise_only=True, seed=2026")
    L.append("")
    L.append(f"**Total wall-clock**: {runtime:.1f}s")
    L.append("")

    L.append("## Headline tally")
    L.append("")
    L.append("| Method | Exact (rel<0.01) | Partial (rel<0.20) | Failed |")
    L.append("|---|---|---|---|")
    L.append(f"| Nelder-Mead | {nm_tally['exact']} | {nm_tally['partial']} | {nm_tally['failed']} |")
    L.append(f"| BFGS | {bfgs_tally['exact']} | {bfgs_tally['partial']} | {bfgs_tally['failed']} |")
    L.append("")

    delta_exact = bfgs_tally['exact'] - nm_tally['exact']
    if delta_exact > 0:
        L.append(f"**BFGS wins on exact-count by +{delta_exact}**. This is the score-moving step.")
    elif delta_exact < 0:
        L.append(f"**BFGS loses on exact-count by {delta_exact}**. Investigate.")
    else:
        L.append("**Same exact count.** Look at the transitions table for finer detail.")
    L.append("")

    L.append("## Transitions (NM → BFGS)")
    L.append("")
    L.append("| Transition | Count |")
    L.append("|---|---|")
    for k in ["partial→exact", "failed→partial", "failed→exact",
              "exact→partial", "exact→failed", "partial→failed", "same"]:
        if k in moves:
            L.append(f"| {k} | {moves[k]} |")
    L.append("")

    L.append("## Per-equation results")
    L.append("")
    L.append("| Eq | NM cx | NM rel | NM verdict | BFGS cx | BFGS rel | BFGS verdict | Δ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm, bfgs in pairs:
        if "error" in nm:
            nm_str = "ERROR"
            nm_v = "—"
            nm_cx = "—"
        else:
            nm_str = f"{nm['best_rel']:.4f}"
            nm_v = classify_verdict(nm["best_rel"])
            nm_cx = str(nm["best_cx"])
        if "error" in bfgs:
            bfgs_str = "ERROR"
            bfgs_v = "—"
            bfgs_cx = "—"
        else:
            bfgs_str = f"{bfgs['best_rel']:.4f}"
            bfgs_v = classify_verdict(bfgs["best_rel"])
            bfgs_cx = str(bfgs["best_cx"])
        delta = ""
        if "error" not in nm and "error" not in bfgs:
            if bfgs["best_rel"] < nm["best_rel"] * 0.5:
                delta = "**BFGS much better**"
            elif bfgs["best_rel"] < nm["best_rel"] * 0.95:
                delta = "BFGS better"
            elif bfgs["best_rel"] > nm["best_rel"] * 2.0:
                delta = "NM much better"
            elif bfgs["best_rel"] > nm["best_rel"] * 1.05:
                delta = "NM better"
            else:
                delta = "tie"
        name = nm.get("name", "?")
        L.append(f"| {name} | {nm_cx} | {nm_str} | {nm_v} | {bfgs_cx} | {bfgs_str} | {bfgs_v} | {delta} |")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_feynman_constopt_ab.py --n_gens {args.n_gens} --pop_size {args.pop_size}")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
