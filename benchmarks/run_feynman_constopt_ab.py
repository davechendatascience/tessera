"""Feynman A/B: scipy Nelder-Mead vs JAX-Adam const-opt.

Per the diagnostic finding (`feynman_signature_diagnostic.md`), Feynman
is uniformly algebraic and pure-pointwise with MSE loss — exactly the
applicability conditions for tessera's JAX autodiff const-opt path
(~10-50× faster per tree per the const_opt.py docstring).

Hypothesis: switching const-opt from Nelder-Mead (gradient-free,
default) to jax_adam (autodiff, available since the §1.3 ship) closes
the partial-tier gap on a meaningful fraction of equations:
  - Equations where SR finds the right structure but partial loss
    because of imprecise constants → jax_adam converges further
  - Equations where structure isn't found → no change expected

Reuses SUBSET from run_feynman_extended.py (30 equations).

Pass criteria for a successful upgrade:
  - At least 2 equations move "partial" → "exact" (rel < 0.01)
  - No regression on currently-exact equations
  - Total wall-clock stays comparable (jax_adam should be faster per
    candidate; whether overall runtime drops depends on convergence)
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
    """Run one equation with the specified const-opt method."""
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    use_jax = (method == "jax_adam")

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
        optimize_constants_maxiter=50 if use_jax else 30,
        use_jax_population_eval=use_jax,
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
    print(f"Comparing Nelder-Mead (default) vs jax_adam on {len(SUBSET)} equations\n")

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
            r_jax = run_with_method(name, sampler, "jax_adam",
                                    args.n_gens, args.pop_size)
            print(f"  JAX:   cx={r_jax['best_cx']:3d} rel={r_jax['best_rel']:.4f} "
                  f"({classify_verdict(r_jax['best_rel'])}) in {r_jax['runtime']:.1f}s")
        except Exception as e:
            print(f"  JAX: FAILED — {e}")
            r_jax = {"name": name, "method": "jax_adam", "best_rel": float("inf"),
                     "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        pairs.append((r_nm, r_jax))

    elapsed_total = time.time() - t_start
    print(f"\nTotal wall-clock: {elapsed_total:.1f}s")

    # Tally verdicts
    nm_verdicts = [classify_verdict(r["best_rel"]) for r, _ in pairs if "error" not in r]
    jax_verdicts = [classify_verdict(r["best_rel"]) for _, r in pairs if "error" not in r]

    def tally(vs):
        from collections import Counter
        c = Counter(vs)
        return {"exact": c.get("exact", 0), "partial": c.get("partial", 0),
                "failed": c.get("failed", 0)}
    nm_tally = tally(nm_verdicts)
    jax_tally = tally(jax_verdicts)

    print(f"\n=== Summary ({len(pairs)} eqs) ===")
    print(f"Nelder-Mead: {nm_tally['exact']} exact / {nm_tally['partial']} partial / {nm_tally['failed']} failed")
    print(f"JAX-Adam:    {jax_tally['exact']} exact / {jax_tally['partial']} partial / {jax_tally['failed']} failed")

    # Transitions
    moves = {"partial→exact": 0, "failed→partial": 0, "failed→exact": 0,
             "exact→partial": 0, "exact→failed": 0, "partial→failed": 0,
             "same": 0}
    for nm, jax in pairs:
        if "error" in nm or "error" in jax:
            continue
        v_nm = classify_verdict(nm["best_rel"])
        v_jax = classify_verdict(jax["best_rel"])
        key = f"{v_nm}→{v_jax}" if v_nm != v_jax else "same"
        moves[key] = moves.get(key, 0) + 1
    print(f"\nTransitions (NM → JAX):")
    for k, v in moves.items():
        if v > 0:
            print(f"  {k}: {v}")

    out_path = HERE / "results" / "feynman_constopt_ab.md"
    write_report(pairs, args, elapsed_total, nm_tally, jax_tally, moves, out_path)
    return 0


def write_report(pairs, args, runtime, nm_tally, jax_tally, moves, out_path):
    L = []
    L.append("# Feynman A/B: scipy Nelder-Mead vs JAX-Adam const-opt")
    L.append("")
    L.append("**Per-equation A/B comparison of constant optimization methods on")
    L.append("the 30-equation Feynman subset.**")
    L.append("")
    L.append(f"**GP config**: pop_size={args.pop_size}, n_gens={args.n_gens}, ")
    L.append(f"init_max_depth=5, optimize_constants_every=3, pointwise_only=True, seed=2026")
    L.append("")
    L.append(f"**Total wall-clock**: {runtime:.1f}s")
    L.append("")

    L.append("## Headline tally")
    L.append("")
    L.append("| Method | Exact (rel<0.01) | Partial (rel<0.20) | Failed |")
    L.append("|---|---|---|---|")
    L.append(f"| Nelder-Mead | {nm_tally['exact']} | {nm_tally['partial']} | {nm_tally['failed']} |")
    L.append(f"| JAX-Adam | {jax_tally['exact']} | {jax_tally['partial']} | {jax_tally['failed']} |")
    L.append("")

    delta_exact = jax_tally['exact'] - nm_tally['exact']
    if delta_exact > 0:
        L.append(f"**JAX-Adam wins on exact-count by +{delta_exact}**. This is the score-moving step.")
    elif delta_exact < 0:
        L.append(f"**JAX-Adam loses on exact-count by {delta_exact}**. Investigate.")
    else:
        L.append("**Same exact count.** Look at the transitions table for finer detail.")
    L.append("")

    L.append("## Transitions (NM → JAX)")
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
    L.append("| Eq | Formula | NM rel | NM verdict | JAX rel | JAX verdict | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for nm, jax in pairs:
        if "error" in nm:
            nm_str = "ERROR"
            nm_v = "—"
        else:
            nm_str = f"{nm['best_rel']:.4f}"
            nm_v = classify_verdict(nm["best_rel"])
        if "error" in jax:
            jax_str = "ERROR"
            jax_v = "—"
        else:
            jax_str = f"{jax['best_rel']:.4f}"
            jax_v = classify_verdict(jax["best_rel"])
        delta = ""
        if "error" not in nm and "error" not in jax:
            if jax["best_rel"] < nm["best_rel"] * 0.5:
                delta = "**JAX much better**"
            elif jax["best_rel"] < nm["best_rel"] * 0.95:
                delta = "JAX better"
            elif jax["best_rel"] > nm["best_rel"] * 2.0:
                delta = "NM much better"
            elif jax["best_rel"] > nm["best_rel"] * 1.05:
                delta = "NM better"
            else:
                delta = "tie"
        # Use a shorter eq name for the table — strip the formula if too long
        name = nm.get("name", "?")
        formula = (nm.get("formula") or "")[:25] if nm.get("formula") else ""
        L.append(f"| {name} | {formula} | {nm_str} | {nm_v} | {jax_str} | {jax_v} | {delta} |")
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
