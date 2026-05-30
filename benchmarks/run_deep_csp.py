"""Deep symbolic network (csp_sr `discover_deep`) — depth-vs-shallow study.

`discover_deep` stacks csp_sr layers: layer 0 is the inputs; each later
layer has `width` nodes, every node a free-form csp expression over a pool
of prior nodes, fit to the running residual (boosting gives each hidden
node a target — no backprop). The combined model is one self-contained
tessera Expr.

This benchmark maps the three honest regimes, on noise-free analytic
targets, scored on a HELD-OUT split (deep stacking adds capacity, so we
guard against memorization):

  1. shallow target            -> a single shallow layer already suffices;
                                  depth is unnecessary.
  2. deep but single-tractable -> a single DEEPER enumeration recovers it
                                  exactly and BEATS the stack (when the
                                  true form fits under the enumeration cap,
                                  enumerate; don't stack).
  3. deep + single-intractable -> the const-free enumeration explodes
                                  before reaching the depth, but the target
                                  is COMPOSITIONAL, so the stack reaches it
                                  (approximately) where a single layer can't
                                  -- depth breaks the wall via composition.

It also contrasts connectivity:
  - bounded (default): each layer reads the previous layer + the inputs
    (skip). Pool stays O(n_inputs + width); per-layer dictionary is
    constant-size; cost is depth-linear. A deep node still depends on every
    earlier node, transitively, through composition.
  - dense: each layer reads ALL prior nodes; the pool grows by `width` each
    layer, so the dictionary grows polynomially. More expressive per layer,
    but the search space re-explodes -- which is exactly the wall.

GPU note: `--jax` routes Phi-build through the jit'd opcode-tape
interpreter (CSPSRConfig.use_jax). GPU accelerates EVAL (the F x N feature
matrix + BLAS fit) at scale -- it does NOT shrink the search space, and the
depth dimension is sequential (each layer needs the previous residual). The
combinatorial explosion is algorithmic (pruning / bounded pool), not a
hardware problem.

Usage:
    python benchmarks/run_deep_csp.py
    python benchmarks/run_deep_csp.py --jax      # GPU eval path
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.experimental.csp_sr import discover, discover_deep, CSPSRConfig
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "deep_csp.md"

UB = ["neg", "sqrt", "tanh"]                 # unary vocab (no div: stable)
BB = ["add", "sub", "mul"]                   # binary vocab
N = 1500                                      # samples (2/3 train, 1/3 test)
DEPTH, WIDTH = 6, 3                            # deep-net shape (18 nodes)


def make_suite(rng):
    """(name, formula, X(N,d), y(N,), regime)."""
    def U(lo, hi, d):
        return rng.uniform(lo, hi, (N, d))
    S = []
    X = U(-1.5, 1.5, 4)
    S.append(("prod_sum", "x0*x1 + x2*x3", X,
              X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3], "shallow"))
    X = U(-1.5, 1.5, 4)
    S.append(("sq_prodsum", "(x0*x1 + x2*x3)^2", X,
              (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) ** 2, "deep/tractable"))
    X = U(-1.2, 1.2, 8)
    S.append(("norm_prodsum8", "sqrt((x0x1+x2x3)^2 + (x4x5+x6x7)^2)", X,
              np.sqrt((X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) ** 2
                      + (X[:, 4] * X[:, 5] + X[:, 6] * X[:, 7]) ** 2),
              "deep/intractable"))
    X = U(-1.5, 1.5, 6)
    S.append(("norm_triple6", "sqrt((x0x1x2)^2 + (x3x4x5)^2)", X,
              np.sqrt((X[:, 0] * X[:, 1] * X[:, 2]) ** 2
                      + (X[:, 3] * X[:, 4] * X[:, 5]) ** 2),
              "deep/intractable"))
    return S


def _r2(expr, X, y, d):
    env = {f"x{i}": X[:, i] for i in range(d)}
    try:
        pred = np.asarray(evaluate(expr, env), dtype=np.float64)
    except Exception:
        return float("nan")
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return float("nan")
    ss = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    return float(1.0 - np.sum((y - pred) ** 2) / ss)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--jax", action="store_true", help="GPU eval path (use_jax)")
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    suite = make_suite(rng)
    ntr = (N * 2) // 3

    def cfg(max_size, beam=12, feats=40000):
        return CSPSRConfig(unary=UB, binary=BB, max_size=max_size,
                           beam_width=beam, max_features=feats, max_terms=8,
                           use_jax=args.jax)

    print(f"=== deep symbolic csp study (test R2, depth={DEPTH} width={WIDTH}, "
          f"{'JAX' if args.jax else 'numpy'}) ===\n")
    rows = []
    for name, formula, X, y, regime in suite:
        d = X.shape[1]
        Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
        envtr = {f"x{i}": Xtr[:, i] for i in range(d)}

        def fit_single(max_size, feats=40000):
            t = time.time()
            r = discover(envtr, ytr, cfg(max_size, feats=feats))
            return _r2(r.expr, Xte, yte, d), r.n_features, time.time() - t

        def fit_deep(lr):
            t = time.time()
            r = discover_deep(envtr, ytr, depth=DEPTH, width=WIDTH,
                              cfg=cfg(2, beam=14), lr=lr, dense=False)
            return _r2(r.expr, Xte, yte, d), time.time() - t

        rs, fs, tsm = fit_single(2)
        rb, fb, tbg = fit_single(5, feats=60000)
        rd1, td1 = fit_deep(1.0)               # naive boosting (no shrinkage)
        rd3, td3 = fit_deep(0.3)               # shrinkage (the fair deep)
        # winner uses the FAIR deep (shrinkage), not the naive one
        cands = {"single-small": rs, "single-big": rb, "deep(lr.3)": rd3}
        win = max(cands, key=lambda k: cands[k] if np.isfinite(cands[k]) else -9)
        print(f"[{name:14s}] {regime:16s} small={rs:.3f} "
              f"big={rb:.3f}(F={fb}) deep_lr1={rd1:.3g} deep_lr.3={rd3:.3f}"
              f"  -> {win}")
        rows.append((name, formula, regime, rs, rb, fb, rd1, rd3, win))

    L = ["# Deep symbolic csp (`discover_deep`) — depth vs shallow "
         "(a negative result)", "",
         "Stacked csp_sr: `depth` layers of `width` nodes; each node is a",
         "free-form csp expression over prior nodes, fit to the running",
         "residual (boosting -> each hidden node has a target, no backprop).",
         "Combined model is one self-contained tessera Expr. **Scored on a",
         f"held-out 1/3 split** (train {ntr}, test {N - ntr}); noise-free",
         "analytic targets. Vocab `" + str(UB + BB) + "`, "
         f"deep shape depth={DEPTH} width={WIDTH} ({DEPTH * WIDTH} nodes).", "",
         "**Headline: a single deeper enumeration (single-big) wins or ties",
         "on every target; stacking does not robustly extend it.** The",
         "train-R2 gains from depth are largely MEMORIZATION — the held-out",
         "split exposes catastrophic overfitting that train R2 hides.", "",
         "- **single-small**: one layer, `max_size=2` (the shallow wall).",
         "- **single-big**: one layer, `max_size=5` (a deeper enumeration; "
         "`F` = dictionary size, cap 60000 = intractable signal).",
         "- **deep (lr=1.0)**: stack, naive boosting (no shrinkage).",
         "- **deep (lr=0.3)**: stack with shrinkage — the FAIR deep (the "
         "winner column uses this one).", "",
         "| target | formula | regime | single-small | single-big (F) | "
         "deep lr=1.0 | deep lr=0.3 | winner |",
         "|---|---|---|---|---|---|---|---|"]
    for (name, formula, regime, rs, rb, fb, rd1, rd3, win) in rows:
        L.append(f"| {name} | `{formula}` | {regime} | {rs:.3f} | "
                 f"{rb:.3f} ({fb}) | {rd1:.3g} | {rd3:.3f} | **{win}** |")
    L += ["", "## Reading", "",
          "- **Train R2 lies.** With `lr=1.0`, depth drives train R2 up while",
          "  held-out R2 goes hugely NEGATIVE (sq_prodsum -1.5e6, "
          "norm_prodsum8 -383):",
          "  the augmented-feature compositions (squares/products of fitted",
          "  intermediates) are high-variance, and the linear fit assigns",
          "  large cancelling coefficients that explode on unseen points — in",
          "  distribution, on noise-free data. Always score deep stacks on a",
          "  held-out split (cf. the Feynman exact-vs-approx separation).",
          "- **Shrinkage helps but does NOT reliably fix it.** `lr=0.3` pulls",
          "  most targets back from the cliff, but it is unstable and",
          "  data-dependent: on `sq_prodsum` it is still negative here (~ -1.0)",
          "  while a different random draw reached +0.82, and it is",
          "  non-monotonic in lr. It needs per-problem tuning a single",
          "  enumeration never does.",
          "- **Deep does not robustly beat single-big.** It wins on at most",
          "  one target (`norm_triple6`, 0.936 vs 0.925 — within the",
          "  instability noise), loses decisively on the tractable target",
          "  (single-big exact, 1.000), and loses on `norm_prodsum8` (0.291",
          "  vs 0.429).",
          "- **Shallow target** (`prod_sum`): a single shallow layer already",
          "  recovers it; depth is pure overhead.", "",
          "## Honest verdict", "",
          "- **`discover_deep` does not provide a robust generalization",
          "  advantage over single-layer csp enumeration.** When the true",
          "  form fits under the enumeration cap, enumerate (exact, stable,",
          "  no tuning). When it does not, the stack only sometimes matches a",
          "  single layer and adds a fragile shrinkage hyperparameter.",
          "- It remains a real capability (gradient-free deep composition,",
          "  one self-contained Expr); bounded connectivity beat dense at",
          "  equal cost in separate train-fit runs — but neither generalizes",
          "  without shrinkage, and shrinkage does not lift it past single-big.",
          "- The bottleneck is the SEARCH SPACE (exponential in tree size),",
          "  not eval. GPU (`--jax`) accelerates the F x N Phi-build + fit at",
          "  scale but cannot shrink the dictionary, and depth is sequential.",
          "  The combinatorial explosion is beaten algorithmically (bounded",
          "  pool, symmetry-breaking, caps), not with hardware.", "",
          "## Reproducing", "", "```",
          "python benchmarks/run_deep_csp.py          # numpy (CPU)",
          "python benchmarks/run_deep_csp.py --jax    # GPU eval path", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {OUT}")


if __name__ == "__main__":
    main()
