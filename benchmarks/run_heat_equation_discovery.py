"""PDE-discovery benchmark: rediscover the heat equation from data.

We simulate a 1-D heat equation on a grid:
    U(t+1, x) = U(t, x) + α · (U(t, x-1) − 2·U(t, x) + U(t, x+1))    [forward Euler]
and ask the tessera GP to find a symbolic relationship of the form
    ∂T/∂t  ≈  f( U, ∇²U, ∂U/∂x, … )

Target: `dt_U` = U(t+1) − U(t)  (a 2-D field, same shape as U with the
                                 last row dropped)
Features available to the GP:
    U      — the field itself, shape (T, X)
The GP is run with enable_2d=True so it can compose FunctionalOp2D nodes
(Laplacian, ∂/∂x, ∂/∂t, custom atomic stencils) with pointwise ops.

Acceptance: the Pareto front should contain a low-cx expression of
form  α · ∇²(U)  where α ≈ 0.05 (the simulated diffusivity).

Usage:
    python benchmarks/run_heat_equation_discovery.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression import (
    GP, GPConfig,
    Var, Const, BinOp, UnOp, FunctionalOp2D,
    measure_2d_laplacian_5pt, measure_2d_diff_t,
    iter_subtrees, complexity,
)


# ---------------- Simulation ----------------

def simulate_heat_1d(
    T: int = 200,
    X: int = 64,
    alpha: float = 0.05,
    noise_std: float = 0.002,
    seed: int = 0,
    amplitude: float = 10.0,
) -> np.ndarray:
    """Forward-Euler simulation of 1-D heat equation on [0, X) with
    Dirichlet BCs (zeros at boundaries). Returns array of shape (T, X).

    `amplitude` scales the initial condition. Larger amplitude → larger
    dt_U variance, which makes the predict-zero trivial answer less
    attractive to the GP."""
    rng = np.random.default_rng(seed)
    U = np.zeros((T, X), dtype=np.float64)
    xs = np.arange(X) - X / 2
    U[0] = amplitude * np.exp(-(xs ** 2) / (2 * 5.0 ** 2))
    U[0] += 0.5 * amplitude * np.exp(-((xs - 10) ** 2) / (2 * 3.0 ** 2))

    for t in range(1, T):
        prev = U[t - 1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        U[t] = prev + alpha * lap + noise_std * rng.standard_normal(X)
    return U


# ---------------- Inspect Pareto front ----------------

def measure_2d_summary(node) -> str:
    """Extract a compact summary of which Measure2Ds appear in a tree."""
    parts = []
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            parts.append(str(sub.measure_2d))
    return " | ".join(parts) if parts else "(no Measure2D)"


# ---------------- Report writer ----------------

def write_report(front, cfg, runtime, target_var, oracle_loss, diff_t_loss,
                 out_path: Path) -> None:
    lines = ["# Heat-equation discovery benchmark", ""]
    lines.append(f"**Target:** `dt_U = U[t+1] − U[t]`  on a heat-equation trajectory")
    lines.append(f"**Simulated α:** 0.05 (true diffusivity)")
    lines.append(f"**Grid:** T=200, X=64; Dirichlet BCs (zero); small Gaussian noise")
    lines.append(f"**Target variance:** {target_var:.4g}")
    lines.append(f"**GP runtime:** {runtime:.1f} s  ({cfg.pop_size} candidates × {cfg.n_gens} gens)")
    lines.append(f"**enable_2d:** {cfg.enable_2d}")
    lines.append("")
    lines.append("## Oracle baselines (for honest comparison)")
    lines.append("")
    lines.append("| Hand-coded expression | MSE | % of target var |")
    lines.append("|---|---|---|")
    lines.append(f"| `α · Laplacian(U)` (the true heat equation) | {oracle_loss:.4g} | {100*oracle_loss/target_var:.1f}% |")
    lines.append(f"| `diff_t(U, lag=1)` (off-by-one-shift equivalent) | {diff_t_loss:.4g} | {100*diff_t_loss/target_var:.1f}% |")
    lines.append("")
    lines.append("Both expressions give similar loss because the heat-equation "
                 "trajectory is smooth in time: `U[t] − U[t−1] ≈ α · Laplacian(U[t−1])`. ")
    lines.append("SR's parsimony bias will prefer the simpler cx=2 form (`diff_t(U)`) "
                 "over the cx=4 form (`α · Laplacian(U)`) whenever they tie on accuracy.")
    lines.append("")
    lines.append("## Pareto front")
    lines.append("")
    lines.append("| Cx | TRAIN loss | rel. to var | Tree |")
    lines.append("|---|---|---|---|")
    for c in front:
        rel = c.train_loss / target_var if target_var > 0 else float("nan")
        tree_str = str(c.tree)
        if len(tree_str) > 120:
            tree_str = tree_str[:117] + "..."
        lines.append(f"| {c.complexity} | {c.train_loss:.4g} | {rel:.2%} | `{tree_str}` |")
    lines.append("")
    lines.append("## Measures appearing in the best (lowest-loss) tree")
    if front:
        best = min(front, key=lambda c: c.train_loss)
        lines.append("")
        lines.append(f"`{best.tree}`")
        lines.append("")
        lines.append(f"**Measure2Ds used:** {measure_2d_summary(best.tree)}")
        lines.append("")
        lines.append(f"**Acceptance**: best train_loss = {best.train_loss:.4g} "
                     f"({100*best.train_loss/target_var:.1f}% of target variance). "
                     f"Recovery of the true α·∇² form is best inspected by reading "
                     f"the tree: look for a `Laplacian_5pt(U)` subexpression multiplied "
                     f"by a Const ≈ 0.05.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ---------------- Main ----------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--pop", type=int, default=150)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-amplitude", type=float, default=10.0,
                   help="Amplitude of the initial Gaussian bump; larger → "
                        "larger dt_U variance, easier for GP to escape the "
                        "predict-zero trap.")
    args = p.parse_args(argv)

    # ---- Simulate ----
    U = simulate_heat_1d(T=args.T, X=args.X, alpha=args.alpha, seed=0,
                         amplitude=args.init_amplitude)
    # Build the target: forward difference along time
    dt_U = np.zeros_like(U)
    dt_U[:-1] = U[1:] - U[:-1]
    # Use only the "interior" (t < T-1) for training; we mask the last row
    # by setting it to 0 (any prediction at t=T-1 won't be sensible anyway).
    target_var = float(np.var(dt_U[1:-1, 1:-1]))
    print(f"[sim] U shape={U.shape}, dt_U variance (interior)={target_var:.4g}")

    # ---- Compute oracle baselines ----
    from tessera.expression.measure_2d import (
        measure_2d_laplacian_5pt, measure_2d_diff_t,
    )
    lap_U = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    diff_U = measure_2d_diff_t(lag_t=1).apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    oracle_loss = float(np.mean((args.alpha * lap_U[interior] - dt_U[interior]) ** 2))
    diff_t_loss = float(np.mean((diff_U[interior] - dt_U[interior]) ** 2))
    print(f"[oracle] α·Laplacian loss = {oracle_loss:.4g}  "
          f"({100*oracle_loss/target_var:.1f}% of target var)")
    print(f"[oracle] diff_t(U) loss   = {diff_t_loss:.4g}  "
          f"({100*diff_t_loss/target_var:.1f}% of target var)")

    # ---- Run GP ----
    # Parsimony must be ≪ target variance else the GP optimises only cx.
    # We set it to ~1% of the target variance.
    parsimony = max(target_var * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=args.pop,
        n_gens=args.gens,
        init_max_depth=4,
        parsimony=parsimony,
        early_stop_patience=20,
        seed=args.seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=True,
    )
    print(f"[gp] parsimony={parsimony:.4g} (scaled to ~1% of target_var)")
    gp = GP(cfg)
    env = {"U": U}
    t0 = time.time()
    front = gp.run(env, dt_U, feature_names=["U"])
    runtime = time.time() - t0

    print()
    print("=" * 70)
    print("Pareto front (sorted by complexity)")
    print("=" * 70)
    for c in front:
        rel = c.train_loss / target_var if target_var > 0 else float("nan")
        print(f"  cx={c.complexity:2d}  loss={c.train_loss:8.4g}  "
              f"({100*rel:.1f}% of var)  | {str(c.tree)[:80]}")

    # ---- Report ----
    out = Path(__file__).resolve().parent / "results" / "heat_equation_discovery.md"
    write_report(front, cfg, runtime, target_var, oracle_loss, diff_t_loss, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
