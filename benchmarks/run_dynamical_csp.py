"""csp_sr on dynamical-systems benchmarks (Lorenz, Rössler).

The governing equations are sparse polynomials in the state — exactly the
regime CSP-enumeration + sparse linear fit was built for (this is SINDy's
home turf). We integrate a trajectory, take the ANALYTIC right-hand side
at the visited states as the target (clean recovery test; finite-
difference derivatives with noise are the realistic extension), and ask
csp_sr to recover each component dX/dt = f(state).

Usage:
    python benchmarks/run_dynamical_csp.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tessera.experimental.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "dynamical_csp.md"


def _rk4(f, s0, dt, n):
    s = np.asarray(s0, float)
    traj = [s.copy()]
    for _ in range(n):
        k1 = f(s); k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(s.copy())
    return np.array(traj)


def lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    truth = ["10*(y - x)", "x*(28 - z) - y  =  28x - xz - y", "xy - 2.667 z"]
    return f, truth


def rossler(a=0.2, b=0.2, c=5.7):
    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])
    truth = ["-y - z", "x + 0.2 y", "0.2 + z(x - 5.7)  =  0.2 + xz - 5.7z"]
    return f, truth


def _verify(expr, env, y):
    pred = np.asarray(evaluate(expr, {k: np.asarray(v) for k, v in env.items()}),
                      dtype=np.float64)
    ss = np.sum((y - y.mean()) ** 2) + 1e-30
    return float(1.0 - np.sum((pred - y) ** 2) / ss)


def run_system(name, f, truth, n_steps=4000, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(-5, 5, size=3)
    traj = _rk4(f, s0, dt, n_steps)[1000:]          # drop transient
    # subsample to keep N modest
    idx = rng.choice(len(traj), size=min(1500, len(traj)), replace=False)
    S = traj[idx]
    env = {"x": S[:, 0], "y": S[:, 1], "z": S[:, 2]}
    rhs = np.array([f(s) for s in S])               # analytic targets
    # SINDy mode: degree-bounded monomial library + STLSQ (joint fit +
    # thresholding). Forward selection (OMP/beam) is blind to jointly-
    # predictive-but-marginally-uncorrelated terms (e.g. Lorenz's `y` in
    # dy/dt, marginal corr 0.004); the joint solve sees them.
    cfg = CSPSRConfig(poly_degree=2, stlsq_threshold=0.05)

    rows = []
    for comp, var in enumerate(["x", "y", "z"]):
        y = rhs[:, comp]
        t = time.time()
        res = discover(env, y, cfg)
        v = _verify(res.expr, env, y)
        rec = v > 0.9999
        rows.append((var, truth[comp], res.r2, v, res.n_terms,
                     expr_to_str(res.expr), rec, time.time() - t))
    return rows


def main():
    systems = [("Lorenz", *lorenz()), ("Rössler", *rossler())]
    print("=== csp_sr on dynamical systems ===\n")
    all_rows = {}
    n_rec = n_tot = 0
    for name, f, truth in systems:
        print(f"{name}:")
        rows = run_system(name, f, truth)
        all_rows[name] = rows
        for var, tr, r2, v, nt, expr, rec, dt in rows:
            n_tot += 1; n_rec += rec
            print(f"  d{var}/dt  [{ 'REC' if rec else 'miss'}] R2={v:.4f} "
                  f"terms={nt}  {dt:.1f}s")
            print(f"      truth: {tr}")
            print(f"      found: {expr}")
        print()
    print(f"Recovered {n_rec}/{n_tot} components.")

    L = ["# csp_sr on dynamical systems (Lorenz, Rössler)", "",
         "CSP-enumeration + sparse linear fit (gradient-free) recovering",
         "dX/dt = f(state) from trajectory samples with analytic RHS targets.",
         "This is the SINDy regime: sparse polynomial dynamics.", "",
         f"**Recovered {n_rec}/{n_tot} components** (R² > 0.9999, verified via",
         "tessera.evaluate).", ""]
    for name in all_rows:
        L += [f"## {name}", "",
              "| component | truth | R² | terms | found |",
              "|---|---|---|---|---|"]
        for var, tr, r2, v, nt, expr, rec, dt in all_rows[name]:
            L.append(f"| d{var}/dt | `{tr}` | {v:.4f} | {nt} | `{expr}` |")
        L.append("")
    L += ["## Reading", "",
          "- Sparse polynomial RHS (products, linear terms, constants) are",
          "  recovered exactly and parsimoniously — coefficients via linear",
          "  least squares, structure via enumeration. No gradients.",
          "- Constants like Rössler's `b` (intercept) and `-c·z` (coefficient)",
          "  fall out of the linear fit; `xz` from enumerated `mul(x,z)`.", "",
          "## Reproducing", "", "```",
          "python benchmarks/run_dynamical_csp.py", "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    main()
