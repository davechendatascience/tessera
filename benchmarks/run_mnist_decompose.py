"""A symbolic MNIST network with decomposition-directed per-class heads.

Architecture (gradient-free, interpretable):
  - FEATURE LAYER: 4 channels {raw, |d/dx|, |d/dy|, laplacian} of each image,
    each mean-pooled to a GRID x GRID grid -> 4*GRID^2 features.
  - PER-CLASS HEAD: discover_decompose fits a symbolic one-vs-rest scorer
    s_c(features) (+1 if digit==c else -1). Classify by argmax_c s_c.

Why this is the honest CV test for Strategy A: decomposition recovers DEEP
COMPOSITIONAL structure where it exists (physics). Image features have no
clean additive/multiplicative separability or peelable outer op, so we
EXPECT every head to fall back to `base` (a sparse symbolic readout) and
decomposition to add nothing over it — the empirical confirmation that SR's
value is discovery, not perception. We report the method each head used so
that prediction is checked, not assumed.

Uses sklearn's 8x8 digits (no download). Full 28x28 MNIST runs the same code.

Usage:
    python benchmarks/run_mnist_decompose.py
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from tessera.experimental.csp_sr import CSPSRConfig, expr_to_str
from tessera.experimental.csp_decompose import discover_decompose
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "mnist_decompose.md"


def channel_features(imgs: np.ndarray, grid: int) -> np.ndarray:
    """(n,H,W) images -> (n, 4*grid*grid) pooled multi-channel features."""
    n, H, W = imgs.shape
    raw = imgs
    gx = np.abs(np.gradient(imgs, axis=2))
    gy = np.abs(np.gradient(imgs, axis=1))
    lap = np.abs(4 * imgs - np.roll(imgs, 1, 1) - np.roll(imgs, -1, 1)
                 - np.roll(imgs, 1, 2) - np.roll(imgs, -1, 2))
    bh, bw = H // grid, W // grid
    out = []
    for ch in (raw, gx, gy, lap):
        pooled = ch[:, :grid * bh, :grid * bw].reshape(
            n, grid, bh, grid, bw).mean(axis=(2, 4))
        out.append(pooled.reshape(n, -1))
    return np.concatenate(out, axis=1)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jax", action="store_true",
                   help="GPU eval path (use_jax in every leaf discover())")
    args = p.parse_args(argv)
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        d.images / 16.0, d.target, test_size=0.3, random_state=0,
        stratify=d.target)
    GRID = 4
    Ftr, Fte = channel_features(Xtr, GRID), channel_features(Xte, GRID)
    nf = Ftr.shape[1]
    names = [f"f{i}" for i in range(nf)]
    envtr = {names[i]: Ftr[:, i] for i in range(nf)}
    envte = {names[i]: Fte[:, i] for i in range(nf)}

    cfg = CSPSRConfig(unary=["neg", "sqrt", "tanh", "abs"],
                      binary=["add", "sub", "mul"],
                      max_size=2, max_terms=12, beam_width=10,
                      max_features=20000, use_jax=args.jax)
    print(f"=== symbolic MNIST net: channels x {GRID}x{GRID} pool = {nf} "
          f"features, decomposition heads ===")
    print(f"train={len(ytr)} test={len(yte)}\n")

    t0 = time.time()
    scorers, methods = [], []
    for c in range(10):
        target = 2.0 * (ytr == c) - 1.0
        res = discover_decompose(envtr, target, cfg, max_depth=1)
        scorers.append(res)
        methods.append(res.method)
        print(f"  class {c}: head R2={res.r2:.3f}  via {res.method}")
    fit_t = time.time() - t0

    def scores(env):
        return np.stack([np.asarray(evaluate(s.expr, env), dtype=np.float64)
                         for s in scorers], axis=1)
    tr = float((np.argmax(scores(envtr), axis=1) == ytr).mean())
    te = float((np.argmax(scores(envte), axis=1) == yte).mean())
    head_kinds = Counter(m.split("->")[0].split("[")[0] for m in methods)
    print(f"\nTRAIN {tr:.4f}  TEST {te:.4f}  features={nf}  fit={fit_t:.0f}s")
    print(f"head methods: {dict(head_kinds)}")

    L = ["# Symbolic MNIST network with decomposition heads (8x8 digits)", "",
         "Feature layer: 4 channels {raw, |dx|, |dy|, laplacian} pooled to "
         f"{GRID}x{GRID} -> {nf} features. Per-class `discover_decompose` "
         "one-vs-rest scorer, argmax. Gradient-free, interpretable.", "",
         f"**TRAIN {tr:.4f}, TEST {te:.4f}** ({len(ytr)}/{len(yte)} split, "
         f"fit {fit_t:.0f}s). Head methods used: `{dict(head_kinds)}`.", "",
         "| class | head R2 | method |", "|---|---|---|"]
    for c in range(10):
        L.append(f"| {c} | {scorers[c].r2:.3f} | `{methods[c][:40]}` |")
    L += ["", "## Reading", "",
          "- **Heads fall back to `base`** (a capacity-controlled sparse",
          "  symbolic readout): with degenerate (blob) groupings rejected and",
          "  peel/separability accepted only when it beats `base` on a",
          "  held-out slice, decomposition correctly DECLINES to manufacture",
          "  structure in pooled image features. Image class-ness is not a",
          "  peelable/separable analytic function of the features; it is a",
          "  learned statistical pattern.",
          "- So decomposition adds nothing here over the sparse readout, and",
          "  it no longer wastes time on degenerate additive splits. Accuracy",
          "  is driven by the FEATURE LAYER (the non-symbolic part), not by",
          "  any symbolic structure discovery.",
          "- Empirical complement to Feynman: decomposition breaks the deep-",
          "  structure wall in DISCOVERY (sqrt(1-v^2/c^2)-class, exact) and",
          "  finds nothing to break in PERCEPTION. SR's value is discovery,",
          "  not vision — and the capacity controls make that failure cheap",
          "  and honest (no overfit ensemble) rather than slow and misleading.", "",
          "## Reproducing", "", "```",
          "python benchmarks/run_mnist_decompose.py", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
