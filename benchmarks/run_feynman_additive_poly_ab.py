"""Feynman A/B: decompose v2 vs decompose v2 + C8 additive polynomial.

Tests Conjecture C8: does an additive-polynomial structure detector
(fit y via polynomial OLS up to total degree D, seed with top-N
monomials) ADD exact transitions beyond what decompose v2 already
produces?

Architectural note
------------------
Both arms enable production decompose v2 (`decompose_prepass_enabled=True`).
ON arm computes a C8 seed externally and injects via
`precomputed_seed_trees`. The injection is ADDITIVE — C8's seed is
inserted alongside decompose's seed (if both fire), not instead.
This isolates C8's empirical contribution.

Per the experimental discipline, production code does NOT import
from `tessera.experimental`. The C8 invocation happens in this
runner.

Pre-analysis prediction
-----------------------
Smoke test on Feynman targets showed C8 reaches R² > 0.99 on 25/30
equations at D=4 — including equations decompose v2 missed:
  - I.11.19 (dot product, genuine polynomial)
  - I.6.20a, I.6.20 (Gaussian — polynomial approximation)
  - I.8.14 (distance — polynomial approximation of squared form)
  - I.12.11 (additive sum with trig — polynomial approximation)
  - I.15.3t, I.16.6 (Lorentz forms — polynomial approximation)
  - I.18.4 (center of mass — local polynomial approximation)

The interesting question: do polynomial-approximation seeds help the
GP converge to the TRUE (non-polynomial) form?

  - Genuine polynomial cases (I.11.19): seed IS the answer → exact
  - Approximation cases: seed gives reasonable starting point; GP
    must simplify and discover the true wrapper (exp, sqrt, trig).
    May or may not help.

Honest expectation: +2-4 exact for genuine polynomial cases; mixed
results on approximation cases. Net: positive on Feynman.

Reuses SUBSET from run_feynman_extended.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig
from tessera.expression.mutation import validate_tree

# Experimental import — runner-side only (per discipline).
from tessera.experimental.additive_polynomial import (
    additive_polynomial_seed,
)

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_feynman_extended import SUBSET  # noqa: E402
from feynman_common import classify_feynman_verdict as classify_verdict  # noqa: E402


def run_with_c8(name, sampler, c8_on: bool, n_gens: int, pop_size: int,
                c8_max_degree: int = 3):
    """Run one equation with both production decompose v2 enabled +
    optionally a C8 seed injected via precomputed_seed_trees."""
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    seed_trees: tuple = ()
    if c8_on:
        try:
            seed_tree, fit = additive_polynomial_seed(
                env, y,
                max_degree=c8_max_degree,
                r2_threshold=0.99,
                top_n=8,
            )
            if seed_tree is not None:
                if validate_tree(seed_tree, set(feature_names)) is None:
                    seed_trees = (seed_tree,)
        except Exception:
            seed_trees = ()

    cfg = GPConfig(
        pop_size=pop_size, n_gens=n_gens, init_max_depth=5,
        parsimony=max(var_y * 0.005, 1e-4),
        early_stop_patience=25, seed=2026,
        pointwise_only=True, verbose=False,
        optimize_constants_every=3,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
        use_jax_population_eval=False,
        # decompose v2 enabled on BOTH arms — C8 is incremental
        decompose_prepass_enabled=True,
        decompose_prepass_r2_threshold=0.99,
        precomputed_seed_trees=seed_trees,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")
    return dict(
        name=name, c8=c8_on, runtime=runtime, front_size=len(front),
        best_cx=best.complexity, best_loss=best.train_loss, best_rel=rel,
        best_tree=str(best.tree),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_gens", type=int, default=120)
    p.add_argument("--pop_size", type=int, default=400)
    p.add_argument("--c8_max_degree", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    if args.quick:
        args.n_gens = 30
        args.pop_size = 100

    print(f"=== Feynman C8 (additive polynomial) A/B ===")
    print(f"n_gens={args.n_gens}, pop={args.pop_size}, c8_max_degree={args.c8_max_degree}")
    print(f"BOTH arms: decompose_prepass_enabled=True (C8 is incremental)")
    print(f"ON arm: C8 seed injected via precomputed_seed_trees\n")

    t_start = time.time()
    pairs = []
    for i, (name, formula, sampler, _) in enumerate(SUBSET):
        print(f"\n[{i+1}/{len(SUBSET)}] {name}: {formula[:50]}")
        try:
            r_off = run_with_c8(name, sampler, c8_on=False,
                                n_gens=args.n_gens, pop_size=args.pop_size,
                                c8_max_degree=args.c8_max_degree)
            print(f"  OFF: cx={r_off['best_cx']:3d} rel={r_off['best_rel']:.4f} "
                  f"({classify_verdict(r_off['best_rel'])}) in {r_off['runtime']:.1f}s")
        except Exception as e:
            print(f"  OFF FAILED: {e}")
            r_off = {"name": name, "best_rel": float("inf"),
                     "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        try:
            r_on = run_with_c8(name, sampler, c8_on=True,
                               n_gens=args.n_gens, pop_size=args.pop_size,
                               c8_max_degree=args.c8_max_degree)
            print(f"  ON:  cx={r_on['best_cx']:3d} rel={r_on['best_rel']:.4f} "
                  f"({classify_verdict(r_on['best_rel'])}) in {r_on['runtime']:.1f}s")
        except Exception as e:
            print(f"  ON FAILED: {e}")
            r_on = {"name": name, "best_rel": float("inf"),
                    "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        pairs.append((r_off, r_on))

    elapsed = time.time() - t_start
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

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
    print(f"OFF (decompose only): {off_tally['exact']} exact / "
          f"{off_tally['partial']} partial / {off_tally['failed']} failed")
    print(f"ON  (decompose + C8): {on_tally['exact']} exact / "
          f"{on_tally['partial']} partial / {on_tally['failed']} failed")

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

    out_path = HERE / "results" / "feynman_additive_poly_ab.md"
    write_report(pairs, args, elapsed, off_tally, on_tally, moves, out_path)
    return 0


def write_report(pairs, args, runtime, off_tally, on_tally, moves, out_path):
    L = [
        "# Feynman A/B: decompose v2 vs decompose v2 + C8 additive polynomial",
        "",
        "**Tests Conjecture C8 (additive polynomial structure detector). Both arms",
        "enable production decompose v2; ON arm adds a C8 polynomial seed via",
        "`precomputed_seed_trees`. The delta isolates C8's incremental contribution.**",
        "",
        f"**GP config**: pop_size={args.pop_size}, n_gens={args.n_gens}, seed=2026, ",
        f"Nelder-Mead, decompose_prepass_enabled=True (BOTH arms), ",
        f"C8 max_degree={args.c8_max_degree}, top_n=8.",
        "",
        f"**Total wall-clock**: {runtime:.1f}s",
        "",
        "## Headline tally",
        "",
        "| Arm | Exact | Partial | Failed |",
        "|---|---|---|---|",
        f"| OFF (decompose v2 only) | {off_tally['exact']} | "
        f"{off_tally['partial']} | {off_tally['failed']} |",
        f"| ON (decompose v2 + C8)  | {on_tally['exact']} | "
        f"{on_tally['partial']} | {on_tally['failed']} |",
        "",
    ]
    delta = on_tally["exact"] - off_tally["exact"]
    if delta > 0:
        L.append(f"**C8 ON wins by +{delta} exact** beyond decompose v2.")
    elif delta < 0:
        L.append(f"**C8 ON loses by {delta} exact.** Investigate regressions.")
    else:
        L.append("**Same exact count.** Look at transitions for finer detail.")
    L.append("")

    L.append("## Transitions")
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
        delta_label = ""
        if "error" not in off and "error" not in on:
            if on["best_rel"] < off["best_rel"] * 0.5:
                delta_label = "**ON much better**"
            elif on["best_rel"] < off["best_rel"] * 0.95:
                delta_label = "ON better"
            elif on["best_rel"] > off["best_rel"] * 2.0:
                delta_label = "OFF much better"
            elif on["best_rel"] > off["best_rel"] * 1.05:
                delta_label = "OFF better"
            else:
                delta_label = "tie"
        name = off.get("name", "?")
        L.append(f"| {name} | {off_cx} | {off_str} | {off_v} | "
                 f"{on_cx} | {on_str} | {on_v} | {delta_label} |")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"PYTHONHASHSEED=0 python benchmarks/run_feynman_additive_poly_ab.py "
             f"--n_gens {args.n_gens} --pop_size {args.pop_size} "
             f"--c8_max_degree {args.c8_max_degree}")
    L.append("```")
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
