"""csp_sr on digit classification (per-class symbolic scorers).

Architecture for image classification with symbolic regression:
  - Reduce each image to a feature vector (here: the raw pixels — 8×8=64
    for sklearn's load_digits, which needs no download; full 28×28 MNIST
    would add a pooling feature-layer, same downstream code).
  - For each class c, csp_sr fits a symbolic scorer s_c(features) to the
    one-vs-rest target (+1 if digit==c else −1).
  - Classify by argmax_c s_c. Report test accuracy + scorer sparsity.

This is the honest feasibility test: a SYMBOLIC classifier (sparse,
interpretable per class), not a SOTA CNN. poly_degree=1 = sparse linear
symbolic classifier; degree 2 adds pixel-product (nonlinear) features.

Usage:
    python benchmarks/run_mnist_csp.py
    python benchmarks/run_mnist_csp.py --degree 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.experimental.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "mnist_csp.md"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.03)
    args = p.parse_args(argv)

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X = d.data / 16.0                      # normalize 0..16 -> 0..1
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    n_feat = X.shape[1]
    names = [f"p{i}" for i in range(n_feat)]
    envtr = {names[i]: Xtr[:, i] for i in range(n_feat)}
    envte = {names[i]: Xte[:, i] for i in range(n_feat)}

    cfg = CSPSRConfig(poly_degree=args.degree, stlsq_threshold=args.threshold,
                      max_terms=24)
    print(f"=== csp_sr digit classifier (8x8, degree={args.degree}) ===")
    print(f"train={len(ytr)} test={len(yte)} features={n_feat}\n")

    t0 = time.time()
    scorers, term_counts = [], []
    for c in range(10):
        target = 2.0 * (ytr == c) - 1.0
        res = discover(envtr, target, cfg)
        scorers.append(res)
        term_counts.append(res.n_terms)
        print(f"  class {c}: {res.n_terms} terms, train R2={res.r2:.3f}")
    fit_t = time.time() - t0

    def scores(env):
        return np.stack([np.asarray(evaluate(s.expr, env), dtype=np.float64)
                         for s in scorers], axis=1)
    tr_acc = float((np.argmax(scores(envtr), axis=1) == ytr).mean())
    te_acc = float((np.argmax(scores(envte), axis=1) == yte).mean())
    avg_terms = float(np.mean(term_counts))

    print(f"\nTRAIN acc={tr_acc:.4f}  TEST acc={te_acc:.4f}  "
          f"avg terms/class={avg_terms:.1f}  fit={fit_t:.1f}s")

    L = ["# csp_sr digit classifier (sklearn 8x8 digits)", "",
         "Per-class symbolic scorers fit by csp_sr (one-vs-rest, +1/−1),",
         "classify by argmax. A SYMBOLIC, sparse, interpretable classifier —",
         "feasibility test, not a CNN. Full 28×28 MNIST runs the same code",
         "with a pooling feature-layer (Colab GPU).", "",
         f"degree={args.degree}, stlsq_threshold={args.threshold}, "
         f"train={len(ytr)}, test={len(yte)}, features={n_feat}.", "",
         f"**TRAIN acc {tr_acc:.4f}, TEST acc {te_acc:.4f}**, "
         f"avg {avg_terms:.1f} terms/class, fit {fit_t:.1f}s.", "",
         "| class | terms | train R² |", "|---|---|---|"]
    for c in range(10):
        L.append(f"| {c} | {term_counts[c]} | {scorers[c].r2:.3f} |")
    L += ["", "## Example scorer (class 0)", "```",
          expr_to_str(scorers[0].expr)[:400], "```", "",
          "## Reading", "",
          "- A sparse symbolic linear (degree 1) or low-order (degree 2)",
          "  classifier over pixels, gradient-free (STLSQ per class).",
          "- Each class scorer is an explicit, inspectable expression —",
          "  the interpretability SR buys, unlike a dense net.",
          "- Accuracy is the honest feasibility number for symbolic scorers",
          "  over raw pixels; richer features (pooling / gradients) or",
          "  degree 2 trade interpretability for accuracy.", "",
          "## Reproducing", "", "```",
          f"python benchmarks/run_mnist_csp.py --degree {args.degree}", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
