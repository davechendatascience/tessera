"""Feynman A/B: coordinate-discovery (C7) OFF vs ON.

Tests Conjecture C7: does a target-space coordinate-discovery prepass
(library {identity, log_abs, sqrt_abs, square, inverse}) produce more
exact recoveries than the production power-law + exp-wrapper detectors
alone?

Architectural note
------------------
Per the experimental discipline (`tessera.experimental.__init__`),
production code does NOT import from `tessera.experimental`. The
prepass for this A/B is computed in the runner using
`detect_coord_discovery_seed`, and the resulting seed tree is passed
to GP via the public `precomputed_seed_trees` config option (added
2026-05-29 to gp.py).

Pre-analysis prediction
-----------------------
Neutral or modest improvement on Feynman because:
- sqrt_abs, square, inverse are all linearly equivalent to identity
  in log-log space (since detect_power_law fits log|y_t| as linear in
  log|x_i|, applying sqrt/square/inverse to y just rescales the
  regression target's slope). So these add no new detections.
- log_abs IS genuinely different — and that's already the exp-wrapper
  detector's domain.

The honest expectation: C7 reproduces decompose v2's +10 exact count,
not improves on it. If it does, that confirms the architecture is
correct and we know C7 is the right *generalization* but adds no
empirical value on Feynman specifically. C7 would need to be tested
on benchmarks where the natural target coordinate isn't already
covered by power-law/exp-wrapper (e.g., real-data benchmarks with
additive offsets or non-monomial transforms).

Reuses SUBSET from run_feynman_extended.py (30 equations).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig
from tessera.expression.tree import Var, Const, BinOp, UnOp
from tessera.expression.mutation import validate_tree

# Experimental import — explicit per discipline (the prepass code is
# experimental; the A/B runner is the place where experimental imports
# are allowed).
from tessera.experimental.coordinate_discovery import (
    detect_coord_discovery_seed,
)

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_feynman_extended import SUBSET  # noqa: E402
from feynman_common import classify_feynman_verdict as classify_verdict  # noqa: E402


def run_with_c7(name, sampler, c7_on: bool, n_gens: int, pop_size: int):
    """Run one equation; if c7_on, run the C7 detector and inject any
    resulting seed via precomputed_seed_trees. The GP itself never
    imports from experimental."""
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    seed_trees: tuple = ()
    if c7_on:
        result = detect_coord_discovery_seed(
            env, y, r2_threshold=0.99, margin_over_identity=0.0,
        )
        if result is not None:
            # Validate before passing through
            if validate_tree(result.seed_tree, set(feature_names)) is None:
                seed_trees = (result.seed_tree,)

    cfg = GPConfig(
        pop_size=pop_size, n_gens=n_gens, init_max_depth=5,
        parsimony=max(var_y * 0.005, 1e-4),
        early_stop_patience=25, seed=2026,
        pointwise_only=True, verbose=False,
        optimize_constants_every=3,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
        use_jax_population_eval=False,
        # IMPORTANT: NO decompose_prepass enabled — this A/B isolates
        # C7's effect from the production prepass. C7 includes identity
        # transform so it should subsume the production power-law detector.
        decompose_prepass_enabled=False,
        precomputed_seed_trees=seed_trees,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")
    return dict(
        name=name, c7=c7_on, runtime=runtime, front_size=len(front),
        best_cx=best.complexity, best_loss=best.train_loss, best_rel=rel,
        best_tree=str(best.tree),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_gens", type=int, default=120)
    p.add_argument("--pop_size", type=int, default=400)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    if args.quick:
        args.n_gens = 30
        args.pop_size = 100

    print(f"=== Feynman C7 (coord-discovery) A/B ===")
    print(f"n_gens={args.n_gens}, pop={args.pop_size}")
    print(f"production prepass DISABLED on both arms — isolates C7 alone.\n")

    t_start = time.time()
    pairs = []
    for i, (name, formula, sampler, _) in enumerate(SUBSET):
        print(f"\n[{i+1}/{len(SUBSET)}] {name}: {formula[:50]}")
        try:
            r_off = run_with_c7(name, sampler, c7_on=False,
                                n_gens=args.n_gens, pop_size=args.pop_size)
            print(f"  OFF: cx={r_off['best_cx']:3d} rel={r_off['best_rel']:.4f} "
                  f"({classify_verdict(r_off['best_rel'])}) in {r_off['runtime']:.1f}s")
        except Exception as e:
            print(f"  OFF FAILED: {e}")
            r_off = {"name": name, "best_rel": float("inf"),
                     "best_cx": -1, "runtime": 0, "error": str(e)[:80]}
        try:
            r_on = run_with_c7(name, sampler, c7_on=True,
                               n_gens=args.n_gens, pop_size=args.pop_size)
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
    print(f"C7=OFF: {off_tally['exact']} exact / {off_tally['partial']} partial / "
          f"{off_tally['failed']} failed")
    print(f"C7=ON:  {on_tally['exact']} exact / {on_tally['partial']} partial / "
          f"{on_tally['failed']} failed")

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

    out_path = HERE / "results" / "feynman_coord_discovery_ab.md"
    write_report(pairs, args, elapsed, off_tally, on_tally, moves, out_path)
    return 0


def write_report(pairs, args, runtime, off_tally, on_tally, moves, out_path):
    L = [
        "# Feynman A/B: coordinate-discovery (C7) OFF vs ON",
        "",
        "**Per-equation A/B of the C7 coordinate-discovery prepass on the",
        "30-equation Feynman subset. C7 tries five target-space transforms",
        "{identity, log_abs, sqrt_abs, square, inverse} and seeds the GP",
        "with the inverse-transformed power-law of the highest-R² fit.**",
        "",
        f"**GP config**: pop_size={args.pop_size}, n_gens={args.n_gens}, ",
        f"seed=2026, Nelder-Mead, decompose_prepass_enabled=False (BOTH arms).",
        "",
        "**Why decompose disabled on both arms**: this A/B isolates C7's",
        "effect. Since C7's transform library includes identity (which",
        "subsumes the production power-law detector) and log_abs (which",
        "subsumes the exp-wrapper detector), C7 alone should equal-or-",
        "beat decompose v2.",
        "",
        f"**Total wall-clock**: {runtime:.1f}s",
        "",
        "## Headline tally",
        "",
        "| C7 | Exact (rel<0.01) | Partial (rel<0.20) | Failed |",
        "|---|---|---|---|",
        f"| OFF | {off_tally['exact']} | {off_tally['partial']} | {off_tally['failed']} |",
        f"| ON  | {on_tally['exact']}  | {on_tally['partial']}  | {on_tally['failed']}  |",
        "",
    ]

    delta = on_tally['exact'] - off_tally['exact']
    if delta > 0:
        L.append(f"**C7=ON wins by +{delta} exact**.")
    elif delta < 0:
        L.append(f"**C7=ON loses by {delta} exact.** Investigate (C7 should subsume decompose).")
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
    L.append(f"PYTHONHASHSEED=0 python benchmarks/run_feynman_coord_discovery_ab.py "
             f"--n_gens {args.n_gens} --pop_size {args.pop_size}")
    L.append("```")
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
