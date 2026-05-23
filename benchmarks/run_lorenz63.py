"""Lorenz-63 dynamical system: SR rediscover the analytical equations.

The Lorenz system:
    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz
with the canonical chaotic parameters σ=10, ρ=28, β=8/3.

For each component, we ask tessera's GP to find a symbolic formula for
the time derivative given (x, y, z). Acceptance: a low-cx expression in
the Pareto front matches the analytical form to within the numerical
finite-difference floor.

Why this is a good SR benchmark
-------------------------------
- Each true equation is a small polynomial in 2-3 variables.
- The discrete derivative dx/dt has the same magnitude as the variables
  themselves, so parsimony scaling is easy.
- It's a CHAOTIC system, so the GP can't just memorise sequences —
  it has to find the underlying STRUCTURE.

Usage:
    python benchmarks/run_lorenz63.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression import GP, GPConfig


# ---------------- Simulate ----------------

def simulate_lorenz(
    T_final: float = 30.0,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    x0: float = 1.0, y0: float = 1.0, z0: float = 1.0,
    transient_skip: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RK4 integration of Lorenz 63. Returns (t, x, y, z) post-transient."""
    N = int(T_final / dt)
    x = np.zeros(N + 1); y = np.zeros(N + 1); z = np.zeros(N + 1)
    x[0], y[0], z[0] = x0, y0, z0

    def rhs(xi, yi, zi):
        return (sigma * (yi - xi),
                xi * (rho - zi) - yi,
                xi * yi - beta * zi)

    for k in range(N):
        dx1, dy1, dz1 = rhs(x[k], y[k], z[k])
        dx2, dy2, dz2 = rhs(x[k] + 0.5*dt*dx1, y[k] + 0.5*dt*dy1, z[k] + 0.5*dt*dz1)
        dx3, dy3, dz3 = rhs(x[k] + 0.5*dt*dx2, y[k] + 0.5*dt*dy2, z[k] + 0.5*dt*dz2)
        dx4, dy4, dz4 = rhs(x[k] + dt*dx3, y[k] + dt*dy3, z[k] + dt*dz3)
        x[k+1] = x[k] + dt * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6.0
        y[k+1] = y[k] + dt * (dy1 + 2*dy2 + 2*dy3 + dy4) / 6.0
        z[k+1] = z[k] + dt * (dz1 + 2*dz2 + 2*dz3 + dz4) / 6.0

    t = np.arange(N + 1) * dt
    sl = slice(transient_skip, None)
    return t[sl], x[sl], y[sl], z[sl]


def finite_difference(series: np.ndarray, dt: float) -> np.ndarray:
    """Centred finite difference: dot{u}[i] ≈ (u[i+1] - u[i-1]) / (2 dt).
    Pad edges with one-sided differences."""
    n = len(series)
    out = np.empty(n, dtype=np.float64)
    out[1:-1] = (series[2:] - series[:-2]) / (2.0 * dt)
    out[0] = (series[1] - series[0]) / dt
    out[-1] = (series[-1] - series[-2]) / dt
    return out


# ---------------- Per-component GP search ----------------

def search_component(
    component_name: str,
    env: dict[str, np.ndarray],
    target: np.ndarray,
    cfg: GPConfig,
) -> dict:
    print()
    print("=" * 70)
    print(f"Search for d{component_name}/dt")
    print("=" * 70)
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, target, feature_names=list(env.keys()))
    runtime = time.time() - t0
    best = min(front, key=lambda c: c.train_loss) if front else None
    return dict(
        component=component_name,
        front=front,
        best=best,
        runtime=runtime,
        target_var=float(np.var(target)),
    )


# ---------------- Report ----------------

def write_report(results: list[dict], cfg: GPConfig, out_path: Path) -> None:
    lines = ["# Lorenz-63 symbolic rediscovery benchmark", ""]
    lines.append(f"**System:** σ=10, ρ=28, β=8/3 (canonical chaotic regime)")
    lines.append(f"**Trajectory:** T=30, dt=0.01 (~3000 RK4 samples post-transient)")
    lines.append(f"**Observed:** all of x, y, z; finite-difference derivatives.")
    lines.append(f"**GP:** pop={cfg.pop_size}, gens={cfg.n_gens}, parsimony={cfg.parsimony}")
    lines.append("")
    lines.append("## Analytical truth")
    lines.append("```")
    lines.append("dx/dt = σ (y − x)              = 10 y - 10 x")
    lines.append("dy/dt = x (ρ − z) − y          = 28 x - x z - y")
    lines.append("dz/dt = x y − β z              = x y - (8/3) z")
    lines.append("```")
    lines.append("")
    lines.append("## Per-component results")
    for r in results:
        lines.append(f"\n### d{r['component']}/dt   (target var = {r['target_var']:.4g}, runtime = {r['runtime']:.1f}s)\n")
        lines.append("| Cx | TRAIN loss | rel. to var | Expression |")
        lines.append("|---|---|---|---|")
        for c in r["front"]:
            rel = c.train_loss / r["target_var"] if r["target_var"] > 0 else float("nan")
            tree_str = str(c.tree)
            if len(tree_str) > 120:
                tree_str = tree_str[:117] + "..."
            lines.append(f"| {c.complexity} | {c.train_loss:.4g} | {rel:.2%} | `{tree_str}` |")
        b = r["best"]
        lines.append("")
        lines.append(f"**Best (by loss)**: cx={b.complexity}, "
                     f"loss = {b.train_loss:.4g} "
                     f"({100*b.train_loss/r['target_var']:.2f}% of target var)")
        lines.append(f"   `{b.tree}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ---------------- Main ----------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T-final", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--pop", type=int, default=120)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)

    # ---- Simulate ----
    t, x, y, z = simulate_lorenz(T_final=args.T_final, dt=args.dt)
    dxdt = finite_difference(x, args.dt)
    dydt = finite_difference(y, args.dt)
    dzdt = finite_difference(z, args.dt)
    print(f"[sim] trajectory: T={len(x)}, dt={args.dt}")
    print(f"[sim] var(x)={x.var():.2f}, var(y)={y.var():.2f}, var(z)={z.var():.2f}")
    print(f"[sim] var(dx/dt)={dxdt.var():.2f}, var(dy/dt)={dydt.var():.2f}, var(dz/dt)={dzdt.var():.2f}")

    env = {"x": x, "y": y, "z": z}
    cfg = GPConfig(
        pop_size=args.pop,
        n_gens=args.gens,
        init_max_depth=4,
        parsimony=0.5,           # target variances are ~50-500, parsimony ~1% of that
        early_stop_patience=15,
        seed=args.seed,
        enable_2d=False,
        fill_warmup=0.0,
        verbose=False,
    )

    # Per-component search
    results = []
    for name, target in [("x", dxdt), ("y", dydt), ("z", dzdt)]:
        # Scale parsimony per-component (different target variances)
        per_cfg = GPConfig(
            pop_size=cfg.pop_size, n_gens=cfg.n_gens,
            init_max_depth=cfg.init_max_depth,
            parsimony=max(target.var() * 0.005, 1e-3),
            early_stop_patience=cfg.early_stop_patience,
            seed=cfg.seed,
            enable_2d=cfg.enable_2d,
            pointwise_only=True,    # Lorenz is a pure ODE; no functionals
            fill_warmup=cfg.fill_warmup,
            verbose=False,
        )
        r = search_component(name, env, target, per_cfg)
        results.append(r)
        # Quick console summary
        print(f"d{name}/dt: best cx={r['best'].complexity}, "
              f"loss={r['best'].train_loss:.4g} "
              f"({100*r['best'].train_loss/r['target_var']:.2f}% of var)  "
              f"-- {str(r['best'].tree)[:80]}")

    out = Path(__file__).resolve().parent / "results" / "lorenz63.md"
    write_report(results, cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
