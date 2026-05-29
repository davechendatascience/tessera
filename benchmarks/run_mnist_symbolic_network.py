"""MNIST 0 vs 1 symbolic network — Milestone A CLI runner.

Companion to `notebooks/tessera_symbolic_network_mnist.ipynb`.
Same configuration but headless; writes a results report.

Per `docs/research/hybrid_symbolic_networks.md` Milestone A:
- K=4 layer-1 symbolic feature trees + 1 layer-2 classifier tree
- Network-aware GP (slot mutation, network crossover)
- Binary classification, digit 0 vs digit 1
- Success: TEST accuracy > 0.95 (vs 0.80 single-tree baseline)

Usage
-----
    python benchmarks/run_mnist_symbolic_network.py
    python benchmarks/run_mnist_symbolic_network.py --pop 60 --gens 60 --K 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.experimental.symbolic_network import (
    NetworkGPConfig, run_network_gp, network_accuracy,
)


OUT_DIR = Path(__file__).parent / "results"


def downsample_2x(img: np.ndarray) -> np.ndarray:
    """28×28 → 14×14 via 2×2 block-mean."""
    return img.reshape(14, 2, 14, 2).mean(axis=(1, 3))


def load_mnist_pair(target_a: int = 0, target_b: int = 1,
                    n_per_train: int = 400, n_per_test: int = 200,
                    seed: int = 2026, downsample: bool = True):
    """Load + preprocess MNIST pair classification subset."""
    from sklearn.datasets import fetch_openml
    print(f"[data] loading MNIST (digits {target_a} vs {target_b}) via sklearn...")
    t0 = time.time()
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
    print(f"[data] loaded in {time.time() - t0:.1f}s")
    X = mnist.data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    rng = np.random.default_rng(seed)
    idx_a = rng.permutation(np.where(y == target_a)[0])
    idx_b = rng.permutation(np.where(y == target_b)[0])
    tr_a, tr_b = idx_a[:n_per_train], idx_b[:n_per_train]
    te_a, te_b = (idx_a[n_per_train:n_per_train + n_per_test],
                  idx_b[n_per_train:n_per_train + n_per_test])
    tr_idx = np.concatenate([tr_a, tr_b])
    te_idx = np.concatenate([te_a, te_b])
    rng.shuffle(tr_idx); rng.shuffle(te_idx)

    imgs_tr = X[tr_idx]
    labels_tr = (y[tr_idx] == target_b).astype(int)
    imgs_te = X[te_idx]
    labels_te = (y[te_idx] == target_b).astype(int)
    if downsample:
        imgs_tr = np.stack([downsample_2x(im) for im in imgs_tr], axis=0)
        imgs_te = np.stack([downsample_2x(im) for im in imgs_te], axis=0)
    return imgs_tr, labels_tr, imgs_te, labels_te


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target_a", type=int, default=0)
    p.add_argument("--target_b", type=int, default=1)
    p.add_argument("--n_per_train", type=int, default=400)
    p.add_argument("--n_per_test", type=int, default=200)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--parsimony", type=float, default=0.002)
    p.add_argument("--no_2d", action="store_true",
                   help="Disable FunctionalOp2D in layer-1 random trees.")
    p.add_argument("--jax", action="store_true",
                   help="Enable JAX-batched evaluation (vmap over images, "
                        "JIT'd per tree). Required for Colab GPU.")
    args = p.parse_args(argv)

    imgs_tr, labels_tr, imgs_te, labels_te = load_mnist_pair(
        target_a=args.target_a, target_b=args.target_b,
        n_per_train=args.n_per_train, n_per_test=args.n_per_test,
        seed=args.seed,
    )
    print(f"\n[data] TRAIN: {imgs_tr.shape}, labels = {np.bincount(labels_tr)}")
    print(f"[data] TEST:  {imgs_te.shape}, labels = {np.bincount(labels_te)}")

    cfg = NetworkGPConfig(
        pop_size=args.pop, n_gens=args.gens, K=args.K,
        layer_1_max_depth=3, layer_2_max_depth=3,
        enable_2d=not args.no_2d,
        parsimony=args.parsimony,
        tournament_size=3, crossover_rate=0.3,
        seed=args.seed, early_stop_patience=12,
        verbose=True,
        use_jax_eval=args.jax,
    )
    print(f"\n[gp] pop={cfg.pop_size}, gens={cfg.n_gens}, K={cfg.K}, "
          f"enable_2d={cfg.enable_2d}")
    t0 = time.time()
    best, history = run_network_gp(imgs_tr, labels_tr, cfg, imgs_te, labels_te)
    runtime = time.time() - t0

    tr_acc = best.accuracy
    te_acc = network_accuracy(best.network, imgs_te, labels_te)

    print()
    print("=" * 70)
    print(f"Milestone A — MNIST {args.target_a} vs {args.target_b}, "
          f"K={args.K}, pop={args.pop}, gens={args.gens}")
    print("=" * 70)
    print(f"  Runtime: {runtime:.1f}s")
    print(f"  TRAIN accuracy: {tr_acc:.3f}  "
          f"({int(tr_acc * len(labels_tr))} / {len(labels_tr)})")
    print(f"  TEST accuracy:  {te_acc:.3f}  "
          f"({int(te_acc * len(labels_te))} / {len(labels_te)})")
    print(f"  Network cx:     {best.network.complexity}")
    print()
    print(f"  Single-tree baseline (existing): TEST 0.80 (0-vs-rest)")
    print(f"  Milestone A success bar:         TEST > 0.95")
    if te_acc > 0.95:
        print("  >>> SUCCESS")
    elif te_acc > 0.80:
        print("  >>> PROGRESS (above single-tree baseline; below 0.95 target)")
    else:
        print("  >>> NOT YET (≤ single-tree baseline)")

    # Write report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L = [
        f"# MNIST {args.target_a}-vs-{args.target_b} symbolic network (Milestone A)",
        "",
        f"**GP**: pop_size={args.pop}, n_gens={args.gens}, K={args.K}, ",
        f"enable_2d={cfg.enable_2d}, parsimony={args.parsimony}, seed={args.seed}",
        f"**Data**: {args.n_per_train}/class TRAIN, {args.n_per_test}/class TEST, "
        f"14×14 downsampled",
        f"**Runtime**: {runtime:.1f}s",
        "",
        "## Result",
        "",
        f"- TRAIN accuracy: {tr_acc:.3f}",
        f"- TEST accuracy: {te_acc:.3f}",
        f"- Network complexity: {best.network.complexity}",
        "",
        "## Discovered network",
        "",
        "```",
        str(best.network).encode("ascii", "replace").decode("ascii"),
        "```",
        "",
        "## Verdict vs Milestone A success criterion",
        "",
    ]
    if te_acc > 0.95:
        L.append("**SUCCESS** — TEST accuracy exceeds the 0.95 target.")
    elif te_acc > 0.80:
        L.append("**PROGRESS** — beats single-tree baseline (0.80) but below 0.95 target. "
                 "Try larger pop/gens or K.")
    else:
        L.append("**NOT YET** — TEST accuracy ≤ single-tree baseline. "
                 "Architecture may need infrastructure refinement.")
    L.append("")
    L.append("## Training history")
    L.append("")
    L.append("| gen | best loss | TRAIN acc | TEST acc | cx |")
    L.append("|---|---|---|---|---|")
    for h in history:
        L.append(f"| {h['gen']} | {h['best_loss']:.4f} | "
                 f"{h['best_train_acc']:.3f} | "
                 f"{h['best_test_acc']:.3f} | {h['best_cx']} |")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_mnist_symbolic_network.py "
             f"--target_a {args.target_a} --target_b {args.target_b} "
             f"--K {args.K} --pop {args.pop} --gens {args.gens}")
    L.append("```")
    out_path = OUT_DIR / f"mnist_symbolic_network_{args.target_a}vs{args.target_b}.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
