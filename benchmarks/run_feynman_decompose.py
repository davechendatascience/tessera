"""Decomposition-directed csp_sr on the extended Feynman benchmark.

Head-to-head: single-layer `discover` (baseline) vs `discover_decompose`
(Strategy A: outer-op peel + polynomial-STLSQ leaf + separability), on the
SAME 30 equations, scored on a HELD-OUT split (fit on train, rel on test).

The question: does top-down decomposition recover MORE equations exactly —
specifically the DEEP-structure forms (sqrt(1 - v^2/c^2)-class) that a
single bounded-size enumeration cannot reach?

Scoring (as in run_feynman_csp.py): exact = rel < 1e-8 (genuine symbolic,
machine precision), approx = rel < 1e-2 (good fit, Class-B not the true
form), else fail.

Usage:
    python benchmarks/run_feynman_decompose.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_feynman_extended import SUBSET                       # noqa: E402

from tessera.experimental.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.experimental.csp_decompose import discover_decompose
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "feynman_decompose.md"

CFG = CSPSRConfig(
    unary=["neg", "sqrt", "exp", "sin", "cos", "log"],
    binary=["add", "sub", "mul", "div"],
    max_size=3, max_terms=4, beam_width=12, max_features=25000,
)

EXACT_REL = 1e-8
APPROX_REL = 1e-2


def classify(rel):
    if not np.isfinite(rel):
        return "fail"
    if rel < EXACT_REL:
        return "exact"
    if rel < APPROX_REL:
        return "approx"
    return "fail"


def _rel(expr, env, y):
    try:
        pred = np.asarray(evaluate(expr, {k: np.asarray(v) for k, v in env.items()}),
                          dtype=np.float64)
    except Exception:
        return float("nan")
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30))


def _split(env, y):
    n = len(y)
    ntr = max(50, (n * 7) // 10)
    if ntr >= n:                      # tiny sample: no split
        return env, y, env, y
    etr = {k: np.asarray(v)[:ntr] for k, v in env.items()}
    ete = {k: np.asarray(v)[ntr:] for k, v in env.items()}
    return etr, y[:ntr], ete, y[ntr:]


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jax", action="store_true", help="GPU eval path (use_jax)")
    args = p.parse_args(argv)
    cfg = replace(CFG, use_jax=args.jax)
    print("=== Feynman: single-layer discover vs discover_decompose "
          "(held-out) ===\n")
    rows = []
    t0 = time.time()
    for i, (name, formula, sampler, expected) in enumerate(SUBSET):
        env, y = sampler()
        y = np.asarray(y, dtype=np.float64)
        etr, ytr, ete, yte = _split(env, y)
        # baseline
        try:
            rb = discover(etr, ytr, cfg)
            relb = _rel(rb.expr, ete, yte)
        except Exception:
            relb = float("nan")
        # decomposition
        try:
            rd = discover_decompose(etr, ytr, cfg, max_depth=2)
            reld = _rel(rd.expr, ete, yte)
            method = rd.method
        except Exception as e:
            reld, method = float("nan"), f"ERR {e}"
        vb, vd = classify(relb), classify(reld)
        rows.append((name, formula, vb, relb, vd, reld, method))
        flag = "  <== gain" if (vd == "exact" and vb != "exact") else ""
        print(f"[{i+1:2d}/30] {name:10s} base={vb:5s}({relb:.1g})  "
              f"decomp={vd:5s}({reld:.1g})  {method[:24]}{flag}")
    elapsed = time.time() - t0

    nb = sum(1 for r in rows if r[2] == "exact")
    nd = sum(1 for r in rows if r[4] == "exact")
    gains = [r[0] for r in rows if r[4] == "exact" and r[2] != "exact"]
    losses = [r[0] for r in rows if r[2] == "exact" and r[4] != "exact"]
    print(f"\nEXACT: baseline {nb}/30  ->  decompose {nd}/30   "
          f"(wall-clock {elapsed:.0f}s)")
    print(f"gains: {gains}\nlosses: {losses}")

    L = ["# Feynman: `discover` vs `discover_decompose` (held-out)", "",
         "Single-layer csp_sr vs Strategy A (outer-op peel + polynomial-STLSQ",
         "leaf + separability). Fit on a 70% train split, `rel = mse/var(y)`",
         "on the 30% held-out test. exact `rel<1e-8`, approx `rel<1e-2`.", "",
         f"Config max_size={CFG.max_size}, vocab `{CFG.unary + CFG.binary}`, "
         f"decompose max_depth=2.", "",
         f"**EXACT: baseline {nb}/30 -> decompose {nd}/30.** "
         f"gains={gains}, losses={losses}. Wall-clock {elapsed:.0f}s.", "",
         "| eq | formula | base | decompose | method |",
         "|---|---|---|---|---|"]
    for name, formula, vb, relb, vd, reld, method in rows:
        L.append(f"| {name} | `{formula[:30]}` | {vb} ({relb:.1g}) | "
                 f"{vd} ({reld:.1g}) | `{method[:34]}` |")
    L += ["", "## Reading", "",
          "- **gains** are equations the single-layer enumeration could not",
          "  reach but decomposition does — the deep-structure forms broken by",
          "  outer-op peel (e.g. sqrt(1-v^2/c^2) -> peel sqrt -> 1 - v^2/c^2,",
          "  a 2-term linear fit) and polynomial-after-peel (STLSQ).",
          "- **losses** would be regressions (decomposition's high threshold",
          "  or a wrong peel beating the base fit) — expected ~none, since",
          "  decompose tries base first and only overrides on a verified",
          "  higher-precision result.",
          "- Decomposition does NOT manufacture Class-B approximations: it",
          "  short-circuits only at machine precision (rel<1e-9), so a smooth",
          "  target is recovered in its true form (via peel), not as a",
          "  high-degree polynomial fit.", "",
          "## Reproducing", "", "```",
          "python benchmarks/run_feynman_decompose.py", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
