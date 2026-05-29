"""Feynman A/B: decomposition pre-pass v2 (power-law + exp-wrapper) OFF vs ON.

Step 3b of the Feynman improvement plan. Extends the v1 decomposition
prepass (power-law only) with an exp-wrapper detector that handles
`f = ±exp(C · ∏ x_i^{a_i})` forms — specifically Gaussian / Boltzmann-
like exponentials that the pure power-law detector rejects.

Hypothesis: I.6.20a `exp(-θ²/2)` and I.6.20 `exp(-(θ/σ)²/2)` were
partials in v1; the exp-wrapper detector recovers them as exact.

Reuses SUBSET from run_feynman_extended.py (30 equations).

Wall-clock: ~12 min (6 min per arm). Uses PYTHONHASHSEED=0 implicitly
via the determinism fix in _OP_SWAP_GROUPS (see test_gp_determinism.py).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_feynman_extended import SUBSET  # noqa: E402


def run_with_prepass(name, sampler, prepass: bool, n_gens: int, pop_size: int):
    """Run one equation with decomposition prepass either enabled or disabled."""
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
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
        use_jax_population_eval=False,
        decompose_prepass_enabled=prepass,
        decompose_prepass_r2_threshold=0.99,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")
    return dict(
        name=name, prepass=prepass, n_vars=len(feature_names),
        runtime=runtime, front_size=len(front),
        best_cx=best.complexity, best_loss=best.train_loss, best_rel=rel,
        best_tree=str(best.tree), var_y=var_y,
    )


def classify_verdict(rel: float) -> str:
    if not np.isfinite(rel):
        return "failed"
    if rel < 0.01:
        return "exact"
    if rel < 0.20:
        return "partial"
    return "failed"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_gens", type=int, default=120)
    p.add_argument("--pop_size", type=int, default=400)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    if args.quick:
        args.n_gens = 30
        args.pop_size = 100

    print(f"=== Feynman decompose v2 A/B (n_gens={args.n_gens}, pop={args.pop_size}) ===")
    print(f"Comparing prepass=OFF vs prepass=ON (power-law + exp-wrapper)")
    print(f"on {len(SUBSET)} equations\n")

    t_start = time.time()
    pairs = []
    for i, (name, formula, sampler, expected) in enumerate(SUBSET):
        print(f"\n[{i+1}/{len(SUBSET)}] {name}: {formula[:50]}")
        try:
            r_off = run_with_prepass(name, sampler, prepass=False,
                                     n_gens=args.n_gens, pop_size=args.pop_size)
            print(f"  OFF: cx={r_off['best_cx']:3d} rel={r_off['best_rel']:.4f} "
                  f"({classify_verdict(r_off['best_rel'])}) in {r_off['runtime']:.1f}s")
        except Exception as e:
            print(f"  OFF: FAILED - {e}")
            r_off = {"name": name, "prepass": False, "best_rel": float("inf"),
                     "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        try:
            r_on = run_with_prepass(name, sampler, prepass=True,
                                    n_gens=args.n_gens, pop_size=args.pop_size)
            print(f"  ON:  cx={r_on['best_cx']:3d} rel={r_on['best_rel']:.4f} "
                  f"({classify_verdict(r_on['best_rel'])}) in {r_on['runtime']:.1f}s")
        except Exception as e:
            print(f"  ON: FAILED - {e}")
            r_on = {"name": name, "prepass": True, "best_rel": float("inf"),
                    "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        pairs.append((r_off, r_on))

    elapsed_total = time.time() - t_start
    print(f"\nTotal wall-clock: {elapsed_total:.1f}s")

    off_verdicts = [classify_verdict(r["best_rel"]) for r, _ in pairs if "error" not in r]
    on_verdicts = [classify_verdict(r["best_rel"]) for _, r in pairs if "error" not in r]

    def tally(vs):
        from collections import Counter
        c = Counter(vs)
        return {"exact": c.get("exact", 0), "partial": c.get("partial", 0),
                "failed": c.get("failed", 0)}
    off_tally = tally(off_verdicts)
    on_tally = tally(on_verdicts)

    print(f"\n=== Summary ({len(pairs)} eqs) ===")
    print(f"prepass=OFF: {off_tally['exact']} exact / {off_tally['partial']} partial / {off_tally['failed']} failed")
    print(f"prepass=ON:  {on_tally['exact']} exact / {on_tally['partial']} partial / {on_tally['failed']} failed")

    moves = {"partial->exact": 0, "failed->partial": 0, "failed->exact": 0,
             "exact->partial": 0, "exact->failed": 0, "partial->failed": 0,
             "same": 0}
    for off, on in pairs:
        if "error" in off or "error" in on:
            continue
        v_off = classify_verdict(off["best_rel"])
        v_on = classify_verdict(on["best_rel"])
        key = f"{v_off}->{v_on}" if v_off != v_on else "same"
        moves[key] = moves.get(key, 0) + 1
    print(f"\nTransitions (OFF -> ON):")
    for k, v in moves.items():
        if v > 0:
            print(f"  {k}: {v}")

    out_path = HERE / "results" / "feynman_decompose_v2_ab.md"
    write_report(pairs, args, elapsed_total, off_tally, on_tally, moves, out_path)
    return 0


def write_report(pairs, args, runtime, off_tally, on_tally, moves, out_path):
    L = []
    L.append("# Feynman A/B: decomposition v2 (power-law + exp-wrapper) OFF vs ON")
    L.append("")
    L.append("**Per-equation A/B of multiplicative-separability + exp-wrapper")
    L.append("pre-pass on the 30-equation Feynman subset.**")
    L.append("")
    L.append(f"**GP config**: pop_size={args.pop_size}, n_gens={args.n_gens}, ")
    L.append(f"init_max_depth=5, optimize_constants_every=3, optimize_constants_maxiter=30, ")
    L.append(f"Nelder-Mead, pointwise_only=True, seed=2026")
    L.append("")
    L.append(f"**Pre-pass config**: R² threshold = 0.99. Orchestrator tries")
    L.append(f"power-law `C · ∏ x_i^{{a_i}}` first; falls back to exp-wrapper")
    L.append(f"`±exp(C · ∏ x_i^{{a_i}})` if power-law rejects.")
    L.append("")
    L.append(f"**Total wall-clock**: {runtime:.1f}s")
    L.append("")

    L.append("## Headline tally")
    L.append("")
    L.append("| Prepass | Exact (rel<0.01) | Partial (rel<0.20) | Failed |")
    L.append("|---|---|---|---|")
    L.append(f"| OFF (baseline) | {off_tally['exact']} | {off_tally['partial']} | {off_tally['failed']} |")
    L.append(f"| ON             | {on_tally['exact']} | {on_tally['partial']} | {on_tally['failed']} |")
    L.append("")

    delta_exact = on_tally['exact'] - off_tally['exact']
    if delta_exact > 0:
        L.append(f"**Prepass=ON wins on exact-count by +{delta_exact}**.")
    elif delta_exact < 0:
        L.append(f"**Prepass=ON loses on exact-count by {delta_exact}**. Investigate.")
    else:
        L.append("**Same exact count.** Look at the transitions table for finer detail.")
    L.append("")

    L.append("## Transitions (OFF -> ON)")
    L.append("")
    L.append("| Transition | Count |")
    L.append("|---|---|")
    for k in ["partial->exact", "failed->partial", "failed->exact",
              "exact->partial", "exact->failed", "partial->failed", "same"]:
        if k in moves:
            L.append(f"| {k} | {moves[k]} |")
    L.append("")

    L.append("## Per-equation results")
    L.append("")
    L.append("| Eq | OFF cx | OFF rel | OFF verdict | ON cx | ON rel | ON verdict | Δ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for off, on in pairs:
        if "error" in off:
            off_str, off_v, off_cx = "ERROR", "—", "—"
        else:
            off_str = f"{off['best_rel']:.4f}"
            off_v = classify_verdict(off["best_rel"])
            off_cx = str(off["best_cx"])
        if "error" in on:
            on_str, on_v, on_cx = "ERROR", "—", "—"
        else:
            on_str = f"{on['best_rel']:.4f}"
            on_v = classify_verdict(on["best_rel"])
            on_cx = str(on["best_cx"])
        delta = ""
        if "error" not in off and "error" not in on:
            if on["best_rel"] < off["best_rel"] * 0.5:
                delta = "**ON much better**"
            elif on["best_rel"] < off["best_rel"] * 0.95:
                delta = "ON better"
            elif on["best_rel"] > off["best_rel"] * 2.0:
                delta = "OFF much better"
            elif on["best_rel"] > off["best_rel"] * 1.05:
                delta = "OFF better"
            else:
                delta = "tie"
        name = off.get("name", "?")
        L.append(f"| {name} | {off_cx} | {off_str} | {off_v} | "
                 f"{on_cx} | {on_str} | {on_v} | {delta} |")
    L.append("")

    L.append("## Discovered expressions (prepass=ON arm)")
    L.append("")
    for _off, on in pairs:
        if "error" in on:
            continue
        L.append(f"### {on['name']}")
        tree = on.get("best_tree", "?")
        ascii_tree = tree.encode("ascii", "replace").decode("ascii")
        L.append(f"- cx={on['best_cx']}, rel={on['best_rel']:.4f}")
        L.append("  ```")
        L.append(f"  {ascii_tree[:250]}{'...' if len(ascii_tree) > 250 else ''}")
        L.append("  ```")
        L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_feynman_decompose_v2_ab.py --n_gens {args.n_gens} --pop_size {args.pop_size}")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
