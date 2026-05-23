"""Cross-algorithm comparison: GP vs SA vs RandomSearch on a known
synthetic target.

Target: y = 2.371 * x + 0.183 * x*x + small_noise
        (a + b * x + c * x*x with non-trivial constants)

What we measure
---------------
- Wall-clock to reach the Pareto front
- Best (lowest) train MSE on the front
- Pareto-front size
- Composition: do the algorithms find similar structures, or diverge?

Why this target
---------------
Linear-plus-quadratic in `x` is the simplest non-trivial smooth target
where:
  (a) the tree structure (x + x*x) is easy to find
  (b) the numerical constants matter (const-opt makes a difference)
  (c) MSE is a smooth loss with a clean optimum

Usage:
    python benchmarks/run_search_compare.py
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np

from tessera.search import (
    GP, GPConfig, SimulatedAnnealing, SAConfig,
    RandomSearch, RSConfig, pareto_front,
)


OUT_REPORT = Path(__file__).parent / "results" / "search_compare.md"


def main():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    a_true, b_true = 2.371, 0.183
    y = a_true * x + b_true * x * x + 0.01 * rng.standard_normal(n)
    env = {"x": x}
    var_y = float(y.var())

    print(f"=== Cross-algorithm comparison: y = {a_true}*x + {b_true}*x*x + noise ===")
    print(f"n={n}, var(y)={var_y:.4f}, noise floor ~ 1e-4\n")

    # ---- GP ----
    gp_cfg = GPConfig(pop_size=80, n_gens=40, parsimony=1e-4,
                      seed=42, verbose=False,
                      optimize_constants_every=3, optimize_constants_maxiter=30)
    t0 = time.time()
    gp_front = GP(gp_cfg).run(env, y, ["x"])
    gp_time = time.time() - t0
    gp_best = min(gp_front, key=lambda c: c.train_loss)
    print(f"[GP]     runtime={gp_time:.2f}s  |F|={len(gp_front)}  "
          f"best cx={gp_best.complexity} loss={gp_best.train_loss:.4g}")
    print(f"         tree: {gp_best.tree}\n")

    # ---- SA ----
    sa_cfg = SAConfig(n_steps=4000, T_initial=1.0, T_final=1e-4,
                      n_restarts=3, parsimony=1e-4,
                      seed=42, verbose=False,
                      optimize_constants_every=50, optimize_constants_maxiter=30)
    t0 = time.time()
    sa_front = SimulatedAnnealing(sa_cfg).run(env, y, ["x"])
    sa_time = time.time() - t0
    sa_best = min(sa_front, key=lambda c: c.train_loss)
    print(f"[SA]     runtime={sa_time:.2f}s  |F|={len(sa_front)}  "
          f"best cx={sa_best.complexity} loss={sa_best.train_loss:.4g}")
    print(f"         tree: {sa_best.tree}\n")

    # ---- Random search ----
    # Match the candidate budget: GP=80*40=3200; SA=4000*3=12000; RS=8000.
    rs_cfg = RSConfig(n_trees=8000, parsimony=1e-4, seed=42, verbose=False)
    t0 = time.time()
    rs_front = RandomSearch(rs_cfg).run(env, y, ["x"])
    rs_time = time.time() - t0
    rs_best = min(rs_front, key=lambda c: c.train_loss)
    print(f"[Random] runtime={rs_time:.2f}s  |F|={len(rs_front)}  "
          f"best cx={rs_best.complexity} loss={rs_best.train_loss:.4g}")
    print(f"         tree: {rs_best.tree}\n")

    # ---- Merged Pareto front ----
    merged = pareto_front(gp_front + sa_front + rs_front)
    print(f"Merged Pareto front: |F|={len(merged)}")
    for c in merged:
        ascii_tree = str(c.tree).encode("ascii", "replace").decode("ascii")
        print(f"  cx={c.complexity:2d} loss={c.train_loss:.4g}  "
              f"({100*c.train_loss/var_y:.2f}% of var)  | {ascii_tree[:60]}")

    # ---- Report ----
    L = [
        "# tessera.search — GP vs SA vs RandomSearch comparison",
        "",
        f"**Target:** `y = {a_true}*x + {b_true}*x*x + 0.01*noise`",
        f"**Samples:** n={n}, var(y) = {var_y:.4f}",
        f"**Noise floor:** ~ 1e-4 (sigma=0.01)",
        "",
        "## Headline",
        "",
        "| Searcher | Candidate budget | Runtime | Best cx | Best loss | % of var |",
        "|---|---|---|---|---|---|",
        f"| GP (pop=80 × gens=40 + opt-const) | ~3,200 | {gp_time:.2f}s | "
        f"{gp_best.complexity} | {gp_best.train_loss:.4g} | "
        f"{100*gp_best.train_loss/var_y:.2f}% |",
        f"| SA (steps=4000 × restarts=3 + opt-const) | ~12,000 | {sa_time:.2f}s | "
        f"{sa_best.complexity} | {sa_best.train_loss:.4g} | "
        f"{100*sa_best.train_loss/var_y:.2f}% |",
        f"| RandomSearch | 8,000 | {rs_time:.2f}s | "
        f"{rs_best.complexity} | {rs_best.train_loss:.4g} | "
        f"{100*rs_best.train_loss/var_y:.2f}% |",
        "",
        "## Best expression per searcher",
        "",
        f"- **GP:**     `{gp_best.tree}`",
        f"- **SA:**     `{sa_best.tree}`",
        f"- **Random:** `{rs_best.tree}`",
        "",
        "## Merged Pareto front",
        "",
        "Combining all three searchers' Pareto-front members and re-running",
        "`pareto_front()`:",
        "",
        "| Cx | Loss | % of var | Tree |",
        "|---|---|---|---|",
    ]
    for c in merged:
        tree = str(c.tree)
        if len(tree) > 100:
            tree = tree[:97] + "..."
        L.append(f"| {c.complexity} | {c.train_loss:.4g} | "
                 f"{100*c.train_loss/var_y:.2f}% | `{tree}` |")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- Candidate budget mismatch is intentional: each searcher uses")
    L.append("  the canonical config you'd reach for in practice. SA scales")
    L.append("  cheaper per-candidate but needs more total steps to explore.")
    L.append("- RandomSearch as baseline: any 'directed' searcher should")
    L.append("  beat it on this kind of problem. If RS wins, the directed")
    L.append("  searcher is over-tuned to a different regime.")
    L.append("- Smooth MSE + small problem favours const-opt: GP and SA both")
    L.append("  get a significant boost from the polish step on this target.")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
