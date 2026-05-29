"""csp_sr on the extended Feynman benchmark (30 equations).

General (free-form) mode: enumerate const-free tessera Expr trees over a
rich vocabulary, fit a sparse linear combination by beam search. This is
the real test that csp_sr is GENERAL (not polynomial-only) — and it
stress-tests where enumeration hits its size limit.

Honest expectation:
  - RECOVER: products / ratios / polynomial / multi-term-sum forms whose
    const-free feature is small (≤ size 3) and whose constants are linear
    coefficients (q1*q2/r^2, m*g*z, x1*y1+x2*y2+x3*y3, n*k*T/V, ...).
  - FAIL: (a) constants buried inside a nonlinearity (the `1` in
    sqrt(1 - v^2/c^2), the `/2` in exp(-x^2/2)); (b) high-variable
    equations whose true feature is deeper than the enumeration cap
    (I.9.18 has 9 variables). These mark the boundary of the approach.

Usage:
    python benchmarks/run_feynman_csp.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_feynman_extended import SUBSET                       # noqa: E402

from tessera.experimental.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.expression.tree import evaluate

# Stricter than the AI-Feynman rel<0.01: a flexible multi-term linear
# basis APPROXIMATES smooth functions well enough to pass 0.01 without
# being the true symbolic form (Class-B natural overfit). Genuine
# symbolic recovery hits machine-precision rel (~1e-30); approximations
# sit at 1e-5..1e-3. We separate them.
EXACT_REL = 1e-8        # genuine symbolic match (machine precision)
APPROX_REL = 0.01       # AI-Feynman threshold (good numerical fit)
PARTIAL_REL = 0.20


def classify(rel):
    if not np.isfinite(rel):
        return "failed"
    if rel < EXACT_REL:
        return "exact"        # genuine symbolic recovery
    if rel < APPROX_REL:
        return "approx"       # passes AI-Feynman, but not the true form
    if rel < PARTIAL_REL:
        return "partial"
    return "failed"

OUT = Path(__file__).parent / "results" / "feynman_csp.md"

CFG = CSPSRConfig(
    unary=["neg", "sqrt", "exp", "sin", "cos", "log"],
    binary=["add", "sub", "mul", "div"],
    max_size=3, max_terms=4, beam_width=12, max_features=25000,
)


def run_one(sampler):
    env, y = sampler()
    y = np.asarray(y, dtype=np.float64)
    t = time.time()
    res = discover(env, y, CFG)
    pred = np.asarray(evaluate(res.expr, {k: np.asarray(v) for k, v in env.items()}),
                      dtype=np.float64)
    rel = float(np.sum((pred - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30))
    return res, rel, time.time() - t


def main():
    print("=== csp_sr on extended Feynman (30 eqs, general mode) ===\n")
    rows = []
    t0 = time.time()
    for i, (name, formula, sampler, expected) in enumerate(SUBSET):
        try:
            res, rel, dt = run_one(sampler)
            verdict = classify(rel)
            expr = expr_to_str(res.expr)
            nfeat, nterms = res.n_features, res.n_terms
        except Exception as e:
            verdict, rel, dt, expr, nfeat, nterms = "failed", float("nan"), 0.0, f"ERR {e}", 0, 0
        rows.append((name, formula, verdict, rel, nterms, nfeat, expr, dt))
        print(f"[{i+1:2d}/30] {name:10s} {verdict:7s} rel={rel:.3g} "
              f"terms={nterms} feats={nfeat} ({dt:.1f}s)  {formula[:32]}")
        if verdict == "exact":
            print(f"         = {expr[:90]}")
    elapsed = time.time() - t0

    n_exact = sum(1 for r in rows if r[2] == "exact")
    n_approx = sum(1 for r in rows if r[2] == "approx")
    n_part = sum(1 for r in rows if r[2] == "partial")
    print(f"\nGENUINE exact {n_exact}/30 (rel<1e-8, symbolic match); "
          f"approx {n_approx}/30 (pass AI-Feynman rel<0.01 but not the true form); "
          f"partial {n_part}/30. Wall-clock {elapsed:.0f}s")
    print(f"AI-Feynman-threshold count (exact+approx) = {n_exact + n_approx}/30")

    L = ["# csp_sr on the extended Feynman benchmark (general mode)", "",
         "CSP-enumerated const-free tessera trees (rich vocab) + sparse",
         "linear (beam) fit, gradient-free. `rel = mse/var(y)`, verified via",
         "tessera.evaluate.", "",
         f"Config: max_size={CFG.max_size}, beam_width={CFG.beam_width}, "
         f"max_features={CFG.max_features}, vocab={CFG.unary + CFG.binary}.", "",
         "**Honest scoring.** A flexible multi-term linear basis APPROXIMATES",
         "smooth functions well enough to pass the AI-Feynman `rel<0.01`",
         "threshold WITHOUT being the true symbolic form (Class-B natural",
         "overfit). We separate:",
         "- **exact** (`rel < 1e-8`): genuine symbolic recovery — the fit is",
         "  machine-precision because the true form IS in the dictionary;",
         "- **approx** (`1e-8 ≤ rel < 0.01`): passes AI-Feynman but is a",
         "  multi-term numerical approximation, not the true form;",
         "- **partial** (`0.01 ≤ rel < 0.2`), **failed** otherwise.", "",
         f"**Genuine exact {n_exact}/30** (parsimonious symbolic match); "
         f"approx {n_approx}/30; partial {n_part}/30.",
         f"AI-Feynman-threshold count (exact+approx) = {n_exact + n_approx}/30. "
         f"Wall-clock {elapsed:.0f}s.", "",
         "| eq | formula | verdict | rel | terms | feats | found |",
         "|---|---|---|---|---|---|---|"]
    for name, formula, verdict, rel, nterms, nfeat, expr, dt in rows:
        shown = expr[:60] if verdict in ("exact", "approx", "partial") else "—"
        L.append(f"| {name} | `{formula[:34]}` | {verdict} | "
                 f"{rel:.3g} | {nterms} | {nfeat} | `{shown}` |")
    L += ["", "## Reading", "",
          "- **Genuine recoveries** are 1-term, machine-precision fits of",
          "  products / ratios / sqrt-of-product forms — the linear-in-one-",
          "  feature class (mu·Nn, q1q2/r², m·g·z, q/C, d1d2/(d1+d2), ω/c,",
          "  sqrt(γ·pr/ρ), …). These are real symbolic recoveries.",
          "- **`approx`** are the honest false-positive warning: 4-term fits",
          "  that pass `rel<0.01` but are NOT the true form (e.g. sqrt(1−v²/c²)",
          "  approximated by exp/cos terms on the benign sampling range). A",
          "  loose threshold + flexible basis manufactures these; machine-",
          "  precision separation exposes them.",
          "- **Boundary (expected):** constants buried inside nonlinearities",
          "  (the `1` in sqrt(1−v²/c²), `/2` in exp(-x²/2)) aren't a linear",
          "  combo of const-free features → only approximated; high-variable",
          "  / deep equations exceed the enumeration cap (I.9.18, 9 vars).",
          "- **Honest verdict:** csp_sr cleanly recovers linear-in-parameter",
          "  symbolic forms with small features; it APPROXIMATES (does not",
          "  recover) embedded-constant nonlinear forms. The latter needs the",
          "  per-feature nonlinear-constant refine (1-step Gauss-Newton) or",
          "  deeper structured enumeration — the clear next extension.", "",
          "## Reproducing", "", "```", "python benchmarks/run_feynman_csp.py", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
