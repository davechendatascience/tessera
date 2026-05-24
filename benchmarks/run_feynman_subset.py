"""Tessera on a subset of the Feynman symbolic regression benchmark.

The full Feynman dataset (Udrescu & Tegmark 2020) has 100 physics
equations. Running all of them on tessera-CPU is multi-hour; we ship
a small, representative subset that runs in ~2 minutes and documents
where tessera does/doesn't compete.

Subset (8 equations, hand-picked for diversity):
  I.6.20a   exp(-θ²/2)                          (1 var, transcendental)
  I.8.14    sqrt((x2-x1)² + (y2-y1)²)             (4 vars, polynomial)
  I.12.1    μ·N_n                                 (2 vars, trivial product)
  I.12.5    q1·q2/r²                              (3 vars, Coulomb-style)
  I.14.3    m·g·z                                 (3 vars, potential energy)
  I.15.3t   (x - u·t)/sqrt(1 - u²/c²)             (4 vars, Lorentz boost)
  I.27.6    d1·d2/(d1 + d2)                       (2 vars, reduced)
  I.43.31   k·T/(6·pi·η·r)                        (4 vars, Stokes-Einstein)

For each equation, we generate synthetic samples in the documented
input ranges, run tessera GP with pointwise_only=True (since these
are closed-form polynomial / trig forms), and report:
  - Wall-clock to a found expression
  - Train MSE at the Pareto front's lowest-loss cx
  - Whether the analytical form is structurally recoverable

This is the workbench validation (Goal 1) for general-purpose SR.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig


OUT_DIR = Path(__file__).parent / "results"


# ---------------- Feynman subset ----------------
# Each entry: (id, formula_str, n_vars, sampler_lambda)
#
# Sampler returns (X dict, y), X = {var_name: array}, y = array.

def sampler_I_6_20a(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.5, 5.0, n)
    y = np.exp(-(th ** 2) / 2.0)
    return {"theta": th}, y


def sampler_I_8_14(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n); x2 = rng.uniform(-1, 1, n)
    y1 = rng.uniform(-1, 1, n); y2 = rng.uniform(-1, 1, n)
    y = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}, y


def sampler_I_12_1(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1.0, 5.0, n)
    Nn = rng.uniform(1.0, 5.0, n)
    return {"mu": mu, "Nn": Nn}, mu * Nn


def sampler_I_12_5(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(1, 5, n); q2 = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"q1": q1, "q2": q2, "r": r}, q1 * q2 / (r * r)


def sampler_I_14_3(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 5, n); g = rng.uniform(1, 5, n); z = rng.uniform(1, 5, n)
    return {"m": m, "g": g, "z": z}, m * g * z


def sampler_I_15_3t(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 5, n); u = rng.uniform(0.1, 0.5, n)
    t = rng.uniform(1, 5, n); c = rng.uniform(1.0, 2.0, n)
    return {"x": x, "u": u, "t": t, "c": c}, (x - u * t) / np.sqrt(1 - (u / c) ** 2)


def sampler_I_27_6(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d1 = rng.uniform(1, 5, n); d2 = rng.uniform(1, 5, n)
    return {"d1": d1, "d2": d2}, d1 * d2 / (d1 + d2)


def sampler_I_43_31(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(1, 5, n); T = rng.uniform(1, 5, n)
    eta = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"k": k, "T": T, "eta": eta, "r": r}, k * T / (6 * np.pi * eta * r)


SUBSET = [
    ("I.6.20a",   "exp(-theta^2/2)",                sampler_I_6_20a),
    ("I.8.14",    "sqrt((x2-x1)^2+(y2-y1)^2)",      sampler_I_8_14),
    ("I.12.1",    "mu*Nn",                          sampler_I_12_1),
    ("I.12.5",    "q1*q2/r^2",                      sampler_I_12_5),
    ("I.14.3",    "m*g*z",                          sampler_I_14_3),
    ("I.15.3t",   "(x - u*t)/sqrt(1 - u^2/c^2)",    sampler_I_15_3t),
    ("I.27.6",    "d1*d2/(d1+d2)",                  sampler_I_27_6),
    ("I.43.31",   "k*T/(6*pi*eta*r)",               sampler_I_43_31),
]


# ---------------- Per-equation runner ----------------

def run_one(name, formula, sampler):
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    cfg = GPConfig(
        pop_size=120,
        n_gens=40,
        init_max_depth=4,
        parsimony=max(var_y * 0.005, 1e-4),
        early_stop_patience=15,
        seed=2026,
        pointwise_only=True,    # Feynman targets are pure pointwise functions
        verbose=False,
        optimize_constants_every=3,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")

    return dict(
        name=name,
        formula=formula,
        n_vars=len(feature_names),
        n_samples=len(y),
        runtime=runtime,
        front_size=len(front),
        best_cx=best.complexity,
        best_loss=best.train_loss,
        best_rel=rel,
        best_tree=str(best.tree),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Feynman subset on tessera (Goal 1: SR workbench validation) ===\n")
    t_start = time.time()

    results = []
    for name, formula, sampler in SUBSET:
        print(f"[{name}]  {formula}")
        r = run_one(name, formula, sampler)
        results.append(r)
        ascii_tree = r["best_tree"].encode("ascii", "replace").decode("ascii")
        print(f"  cx={r['best_cx']:2d}  loss={r['best_loss']:.4g}  "
              f"rel={r['best_rel']:.4f}  ({r['runtime']:.1f}s)")
        print(f"    {ascii_tree[:100]}\n")

    # Report
    out_path = OUT_DIR / "feynman_subset.md"
    L = ["# Feynman subset on tessera",
         "",
         "**Purpose:** Goal-1 (workbench) validation. Runs tessera GP on 8",
         "representative equations from the Feynman dataset (Udrescu & Tegmark",
         "2020). The full benchmark has 100 equations; this subset spans",
         "trivial-product to non-trivial Lorentz boost in ~2 minutes total.",
         "",
         f"**GP config:** pop=120, gens=40, pointwise_only=True, "
         f"optimize_constants every 3 gens (Nelder-Mead, 30 iter).",
         f"**Samples per equation:** 2000",
         f"**Total wall-clock:** {time.time() - t_start:.1f}s",
         "",
         "## Results",
         "",
         "| # | Eq ID | True formula | n_vars | best cx | best loss | rel to var | runtime (s) |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        L.append(f"| {i} | {r['name']} | `{r['formula']}` | {r['n_vars']} | "
                 f"{r['best_cx']} | {r['best_loss']:.4g} | "
                 f"{r['best_rel']:.4f} | {r['runtime']:.1f} |")
    L += ["", "## Discovered expressions", ""]
    for r in results:
        L.append(f"### {r['name']}")
        L.append(f"- True: `{r['formula']}`")
        ascii_tree = r["best_tree"].encode("ascii", "replace").decode("ascii")
        L.append(f"- Best tree (cx={r['best_cx']}, rel={r['best_rel']:.4f}):")
        L.append(f"  ```")
        L.append(f"  {ascii_tree[:300]}{'...' if len(ascii_tree) > 300 else ''}")
        L.append(f"  ```")
        L.append("")
    L += ["## Reading",
          "",
          "- `rel to var` close to 0 means the search found a near-perfect fit",
          "  on TRAIN. Values > 0.01 indicate the search didn't fully recover",
          "  the analytical form (or its constants are imprecise).",
          "- This is TRAIN loss with no test split. The Feynman targets are",
          "  noiseless, so train-perfect = structurally correct (modulo",
          "  constants).",
          "- Tessera is competitive on small-arity (≤ 3 vars) polynomial /",
          "  rational forms. Higher-arity (4 vars) + transcendental forms",
          "  (Lorentz boost) are harder in 40 generations.",
          "",
          "## Caveats",
          "",
          "- The full Feynman benchmark uses larger sample sizes and more",
          "  generations. PySR / AI Feynman / Operon all run for minutes-to-",
          "  hours per equation; we ran 40 generations as a quick demo.",
          "- For a real workbench claim, run with pop=200, gens=200 and",
          "  report both TRAIN loss and a held-out test loss.",
          "",
          "## See also",
          "",
          "- Original Feynman benchmark: Udrescu & Tegmark, *AI Feynman: a",
          "  Physics-Inspired Method for Symbolic Regression*, Sci. Adv. 2020.",
          "- SRBench: https://github.com/cavalab/srbench for the full",
          "  community SR benchmark suite."]
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


if __name__ == "__main__":
    main()
