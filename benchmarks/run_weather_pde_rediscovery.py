"""Weather PDE rediscovery on real gridded atmospheric data.

Loads daily 2-m air temperature from NCEP/NCAR Reanalysis 2 over a CONUS
cutout (13 x 29 grid, 365 days) and asks tessera's 2-D GP to find a
symbolic relationship of form

    dT/dt(t, y, x) ≈ f( T, Laplacian(T), ∂T/∂x, ∂T/∂y, persistence, ... )

This is the natural successor to the heat-equation benchmark: same 2-D
machinery, but real data (no analytical ground truth — only oracle
baselines that we hand-code and have to beat).

The point isn't to "discover" the Navier-Stokes / radiative-transfer
equations from one variable. It's to verify that tessera's
FunctionalOp2D + GP loop discovers SOMETHING USEFUL on real geophysical
data — i.e., that the best Pareto-front expression beats trivial
baselines.

Oracle baselines (computed and reported alongside GP)
------------------------------------------------------
1. **Predict zero** (no change):                    MSE = var(dT/dt)
2. **Newtonian relaxation**:  dT/dt ≈ -α(T - T₀)
3. **Diffusion**:             dT/dt ≈ α · Laplacian(T)
4. **AR(1) persistence**:     dT/dt ≈ β · (T[t-1] - T[t-2])

Data source
-----------
NCEP/NCAR Reanalysis 2 (R-2) daily T2m, downloaded once and cached.
For higher resolution (ERA5 0.25°), supply a CDS API key — see
benchmarks/data/load_ncep_reanalysis.py docstring.

Usage:
    python benchmarks/run_weather_pde_rediscovery.py
    python benchmarks/run_weather_pde_rediscovery.py --year 2023 --pop 150 --gens 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script or via `python -m`
sys.path.insert(0, str(Path(__file__).parent))

from tessera.expression import (
    GP, GPConfig,
    FunctionalOp2D, iter_subtrees,
    measure_2d_laplacian_5pt, measure_2d_diff_t,
)
from data.load_ncep_reanalysis import load_ncep_t2m


# ---------------- Oracle baselines ----------------

def laplacian_5pt(field: np.ndarray) -> np.ndarray:
    """5-point Laplacian, zero-padded at edges. Shape preserved."""
    lap = np.zeros_like(field)
    lap[:, 1:-1, 1:-1] = (field[:, 1:-1, :-2] + field[:, 1:-1, 2:]
                          + field[:, :-2, 1:-1] + field[:, 2:, 1:-1]
                          - 4.0 * field[:, 1:-1, 1:-1])
    return lap


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, mse). Both x and y are flattened."""
    A = np.stack([x.ravel(), np.ones_like(x.ravel())], axis=1)
    coef, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    pred = A @ coef
    return float(coef[0]), float(coef[1]), float(np.mean((pred - y.ravel())**2))


def compute_oracle_baselines(T: np.ndarray) -> dict:
    """Compute MSE for hand-coded baselines.

    T shape: (T_days, Y, X)
    Target: dT/dt = T[t+1] - T[t]
    All baselines evaluated on the same "interior" mask (drop boundaries
    + first/last days) to match the GP's training domain.
    """
    dt_T = np.zeros_like(T)
    dt_T[:-1] = T[1:] - T[:-1]
    var_dT = float(np.var(dt_T[1:-1, 1:-1, 1:-1]))

    # 1. Predict zero
    mse_zero = var_dT  # since target is zero-mean by construction it's ~var

    # 2. Newtonian relaxation: dT/dt = -α (T - T̄)
    T_bar = float(T.mean())
    anom = T[1:-1, 1:-1, 1:-1] - T_bar
    tgt = dt_T[1:-1, 1:-1, 1:-1]
    alpha, intercept, mse_newton = fit_linear(anom, tgt)

    # 3. Diffusion: dT/dt = α · Laplacian(T)
    lap = laplacian_5pt(T)
    a, b, mse_diff = fit_linear(lap[1:-1, 1:-1, 1:-1], tgt)

    # 4. AR(1) persistence: dT/dt[t] = β · (T[t] - T[t-1])
    delta_prev = np.zeros_like(T)
    delta_prev[1:] = T[1:] - T[:-1]   # delta_prev[t] = T[t] - T[t-1]; 0 at t=0
    a4, b4, mse_ar1 = fit_linear(
        delta_prev[1:-1, 1:-1, 1:-1],
        tgt,
    )

    return dict(
        var_dT=var_dT,
        mse_zero=mse_zero,
        mse_newton=mse_newton, alpha_newton=alpha, intercept_newton=intercept,
        mse_diff=mse_diff, alpha_diff=a,
        mse_ar1=mse_ar1, beta_ar1=a4,
    )


def compute_oracle_baselines_on_slice(T_slice: np.ndarray) -> dict:
    """Same 4 oracles, but fit on the (T_days, X) mid-latitude row.

    This is the apples-to-apples comparison for the GP: the GP also sees
    only this slice (since FunctionalOp2D expects a (time, space) array,
    not a (time, lat, lon) cube). Diffusion drops to a 1D horizontal
    Laplacian since there's no y-direction in a single row.

    Note: tessera's mse_loss uses a "fill_warmup" convention that scores
    on indices `[fill_warmup * T:]`. With fill_warmup=0.0 it scores on
    the entire array — INCLUDING the last row where dt_T = 0 (target
    is undefined). The oracle MSE here matches that convention so the
    numbers compare directly to GP loss.
    """
    dt_T = np.zeros_like(T_slice)
    dt_T[:-1] = T_slice[1:] - T_slice[:-1]
    # GP scores over the FULL (T, X) including dt_T[-1] = 0. So variance
    # should also be computed over the full array.
    var_dT = float(np.var(dt_T))
    tgt = dt_T   # full array, matches GP's training domain

    # 1. Predict zero
    mse_zero = float(np.mean(tgt ** 2))   # = var_dT + mean^2 ≈ var_dT

    # 2. Newtonian relaxation on the slice
    T_bar = float(T_slice.mean())
    anom = T_slice - T_bar
    alpha, intercept, mse_newton = fit_linear(anom, tgt)

    # 3. 1D horizontal Laplacian on the slice
    lap_x = np.zeros_like(T_slice)
    lap_x[:, 1:-1] = T_slice[:, :-2] - 2.0 * T_slice[:, 1:-1] + T_slice[:, 2:]
    a, b, mse_diff = fit_linear(lap_x, tgt)

    # 4. AR(1) persistence on the slice: dT/dt[t] = β · (T[t] - T[t-1])
    delta_prev = np.zeros_like(T_slice)
    delta_prev[1:] = T_slice[1:] - T_slice[:-1]
    a4, b4, mse_ar1 = fit_linear(delta_prev, tgt)

    return dict(
        var_dT=var_dT,
        mse_zero=mse_zero,
        mse_newton=mse_newton, alpha_newton=alpha, intercept_newton=intercept,
        mse_diff=mse_diff, alpha_diff=a,
        mse_ar1=mse_ar1, beta_ar1=a4,
    )


# ---------------- GP search ----------------

def run_gp_2d(T: np.ndarray, dt_T: np.ndarray, cfg: GPConfig,
              var_dT: float) -> tuple:
    """Run tessera GP in 2-D mode. T has shape (T_days, Y, X) — tessera's
    2-D evaluator treats axis 0 as time, axis 1 as space.

    Returns (front, runtime, T_slice).
    """
    # Pick a horizontal "slice" — use the middle latitude row as a
    # representative 1-D space-time field. This is the natural way to
    # use FunctionalOp2D which expects (time, space) shape.
    # For full 2-D we'd need either a 3-D FunctionalOp or to flatten
    # (Y, X) into a 1-D spatial index; for v1 we use a single mid-lat row.
    Y = T.shape[1]
    mid = Y // 2
    T_slice = T[:, mid, :]            # (T_days, X)
    dt_T_slice = dt_T[:, mid, :]      # (T_days, X)
    print(f"[gp] taking mid-latitude row idx={mid}; slice shape {T_slice.shape}")
    print(f"[gp] slice var(T)={T_slice.var():.2f}, var(dT/dt)={dt_T_slice.var():.4f}")

    env = {"T": T_slice}
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, dt_T_slice, feature_names=["T"])
    runtime = time.time() - t0
    print(f"[gp] runtime {runtime:.1f}s, Pareto front size {len(front)}")
    return front, runtime, T_slice


def measure_2d_summary(node) -> str:
    parts = []
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            parts.append(str(sub.measure_2d))
    return " | ".join(parts) if parts else "(no Measure2D)"


# ---------------- Report ----------------

def write_report(front, cfg, runtime, oracle, oracle_slice, T_shape, year,
                 lats, lons, out_path: Path) -> None:
    var_dT = oracle["var_dT"]
    var_dT_slice = oracle_slice["var_dT"]
    L = ["# Weather PDE rediscovery benchmark (NCEP/NCAR Reanalysis 2)", ""]
    L.append(f"**Data:** NCEP-DOE AMIP-II Reanalysis daily 2m temperature, year {year}")
    L.append(f"**Grid:** {T_shape[0]} days × {T_shape[1]} lats × {T_shape[2]} lons "
             f"(lat {lats[0]:.1f}..{lats[-1]:.1f}, lon {lons[0]:.1f}..{lons[-1]:.1f})")
    L.append(f"**Target:** `dT/dt = T[t+1] − T[t]`, var = {var_dT:.4g} K²/day² "
             f"(full grid); {var_dT_slice:.4g} K²/day² (mid-lat row slice)")
    L.append(f"**GP:** pop={cfg.pop_size}, gens={cfg.n_gens}, "
             f"parsimony={cfg.parsimony:.3g}, enable_2d={cfg.enable_2d}, "
             f"runtime={runtime:.1f}s")
    L.append("")
    L.append("## Oracle baselines — full grid (context)")
    L.append("")
    L.append("Fit on all (T, Y, X) interior points. Provided for context;"
             " the GP only sees the slice, so use the next table for the"
             " apples-to-apples comparison.")
    L.append("")
    L.append("| # | Expression | MSE | % of full var |")
    L.append("|---|---|---|---|")
    L.append(f"| 0 | predict 0 (no change) | {oracle['mse_zero']:.4g} | 100.0% |")
    L.append(f"| 1 | `{oracle['alpha_newton']:+.4g}·(T − T̄)`  (Newtonian relaxation) "
             f"| {oracle['mse_newton']:.4g} "
             f"| {100*oracle['mse_newton']/var_dT:.1f}% |")
    L.append(f"| 2 | `{oracle['alpha_diff']:+.4g}·∇²T`  (diffusion, 2D) "
             f"| {oracle['mse_diff']:.4g} "
             f"| {100*oracle['mse_diff']/var_dT:.1f}% |")
    L.append(f"| 3 | `{oracle['beta_ar1']:+.4g}·(T[t]−T[t-1])`  (AR(1) persistence) "
             f"| {oracle['mse_ar1']:.4g} "
             f"| {100*oracle['mse_ar1']/var_dT:.1f}% |")
    L.append("")
    L.append("## Oracle baselines — mid-latitude row slice (apples-to-apples)")
    L.append("")
    L.append(f"Same baselines refit on the (T, X) slice the GP sees "
             f"(var = {var_dT_slice:.4g} K²/day²). Diffusion drops to a "
             f"1D horizontal Laplacian (no y-direction in a single row).")
    L.append("")
    L.append("| # | Expression | MSE | % of slice var |")
    L.append("|---|---|---|---|")
    L.append(f"| 0 | predict 0 (no change) | {oracle_slice['mse_zero']:.4g} | "
             f"{100*oracle_slice['mse_zero']/var_dT_slice:.1f}% |")
    L.append(f"| 1 | `{oracle_slice['alpha_newton']:+.4g}·(T − T̄)`  (Newtonian relaxation) "
             f"| {oracle_slice['mse_newton']:.4g} "
             f"| {100*oracle_slice['mse_newton']/var_dT_slice:.1f}% |")
    L.append(f"| 2 | `{oracle_slice['alpha_diff']:+.4g}·∂²T/∂x²`  (1D horizontal diffusion) "
             f"| {oracle_slice['mse_diff']:.4g} "
             f"| {100*oracle_slice['mse_diff']/var_dT_slice:.1f}% |")
    L.append(f"| 3 | `{oracle_slice['beta_ar1']:+.4g}·(T[t]−T[t-1])`  (AR(1) persistence) "
             f"| {oracle_slice['mse_ar1']:.4g} "
             f"| {100*oracle_slice['mse_ar1']/var_dT_slice:.1f}% |")
    L.append("")
    L.append("These are the bars the GP needs to beat.")
    L.append("")
    L.append("## Pareto front (mid-latitude row slice)")
    L.append("")
    L.append("| Cx | TRAIN loss | rel. to slice var | Tree |")
    L.append("|---|---|---|---|")
    for c in front:
        rel = c.train_loss / var_dT_slice if var_dT_slice > 0 else float("nan")
        tree_str = str(c.tree)
        if len(tree_str) > 140:
            tree_str = tree_str[:137] + "..."
        L.append(f"| {c.complexity} | {c.train_loss:.4g} | {rel:.2%} | `{tree_str}` |")
    if front:
        best = min(front, key=lambda c: c.train_loss)
        L.append("")
        L.append("## Best expression")
        L.append("")
        L.append(f"`{best.tree}`")
        L.append("")
        L.append(f"- complexity = {best.complexity}")
        L.append(f"- loss = {best.train_loss:.4g} "
                 f"({100*best.train_loss/var_dT_slice:.1f}% of slice var)")
        L.append(f"- Measure2Ds used: {measure_2d_summary(best.tree)}")

        # Compare best to slice oracle ceiling (apples-to-apples)
        oracle_best = min(oracle_slice['mse_newton'],
                          oracle_slice['mse_diff'],
                          oracle_slice['mse_ar1'])
        oracle_best_name = min(
            [('Newton', oracle_slice['mse_newton']),
             ('diff_1D', oracle_slice['mse_diff']),
             ('AR(1)', oracle_slice['mse_ar1'])],
            key=lambda kv: kv[1],
        )[0]
        if best.train_loss < oracle_best:
            verdict = (f"BEATS best slice oracle ({oracle_best_name}, "
                       f"MSE={oracle_best:.4g}) by "
                       f"{100*(1 - best.train_loss/oracle_best):.1f}%")
        else:
            verdict = (f"does NOT beat best slice oracle ({oracle_best_name}, "
                       f"MSE={oracle_best:.4g})")
        L.append(f"- vs slice oracles: {verdict}")

    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- This benchmark validates the FunctionalOp2D path on REAL data, "
             "not the canonical heat-equation simulator.")
    L.append("- The slice-based formulation (mid-lat row) is the cheapest way to "
             "exercise 2D measures; full grid would need either a 3D FunctionalOp "
             "or flattened (Y,X) → 1D spatial.")
    L.append("- Slice oracles are computed on the SAME (T, X) array the GP sees, "
             "so the verdict above is apples-to-apples.")
    L.append("- For ERA5 (0.25° resolution, 3-hourly), supply a CDS API key and "
             "swap `load_ncep_t2m` for an ERA5 retrieve; the rest is unchanged.")
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ---------------- Main ----------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--pop", type=int, default=120)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lat-min", type=float, default=25.0)
    p.add_argument("--lat-max", type=float, default=50.0)
    p.add_argument("--lon-min", type=float, default=235.0)
    p.add_argument("--lon-max", type=float, default=290.0)
    args = p.parse_args(argv)

    # ---- Load ----
    T, lats, lons, days = load_ncep_t2m(
        year=args.year,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
    )

    # ---- Build target ----
    dt_T = np.zeros_like(T)
    dt_T[:-1] = T[1:] - T[:-1]
    print(f"[setup] T shape {T.shape}, var(T)={T.var():.2f} K^2, "
          f"var(dT/dt)={dt_T[1:-1, 1:-1, 1:-1].var():.4f} K^2/day^2")

    # ---- Oracle baselines ----
    oracle = compute_oracle_baselines(T)
    print(f"[oracle] zero:    MSE={oracle['mse_zero']:.4g}  (100.0%)")
    print(f"[oracle] Newton:  MSE={oracle['mse_newton']:.4g}  "
          f"({100*oracle['mse_newton']/oracle['var_dT']:.1f}% of var, "
          f"α={oracle['alpha_newton']:+.4g})")
    print(f"[oracle] diff:    MSE={oracle['mse_diff']:.4g}  "
          f"({100*oracle['mse_diff']/oracle['var_dT']:.1f}% of var, "
          f"α={oracle['alpha_diff']:+.4g})")
    print(f"[oracle] AR(1):   MSE={oracle['mse_ar1']:.4g}  "
          f"({100*oracle['mse_ar1']/oracle['var_dT']:.1f}% of var, "
          f"β={oracle['beta_ar1']:+.4g})")

    # ---- GP ----
    cfg = GPConfig(
        pop_size=args.pop, n_gens=args.gens,
        init_max_depth=4,
        parsimony=max(oracle["var_dT"] * 0.005, 1e-4),
        early_stop_patience=15,
        seed=args.seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=True,
    )
    front, runtime, T_slice = run_gp_2d(T, dt_T, cfg, oracle["var_dT"])

    # ---- Slice-level oracles (apples-to-apples with GP) ----
    oracle_slice = compute_oracle_baselines_on_slice(T_slice)
    print(f"[oracle/slice] zero:   MSE={oracle_slice['mse_zero']:.4g}  "
          f"({100*oracle_slice['mse_zero']/oracle_slice['var_dT']:.1f}% of slice var)")
    print(f"[oracle/slice] Newton: MSE={oracle_slice['mse_newton']:.4g}  "
          f"({100*oracle_slice['mse_newton']/oracle_slice['var_dT']:.1f}%, "
          f"a={oracle_slice['alpha_newton']:+.4g})")
    print(f"[oracle/slice] diff1D: MSE={oracle_slice['mse_diff']:.4g}  "
          f"({100*oracle_slice['mse_diff']/oracle_slice['var_dT']:.1f}%, "
          f"a={oracle_slice['alpha_diff']:+.4g})")
    print(f"[oracle/slice] AR(1):  MSE={oracle_slice['mse_ar1']:.4g}  "
          f"({100*oracle_slice['mse_ar1']/oracle_slice['var_dT']:.1f}%, "
          f"b={oracle_slice['beta_ar1']:+.4g})")

    print()
    print("=" * 70)
    print("Pareto front")
    print("=" * 70)

    def _ascii_safe(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    for c in front:
        rel = c.train_loss / oracle["var_dT"]
        print(f"  cx={c.complexity:2d}  loss={c.train_loss:8.4g}  "
              f"({100*rel:5.1f}%)  | {_ascii_safe(str(c.tree))[:80]}")

    # ---- Report ----
    out = Path(__file__).parent / "results" / "weather_pde_rediscovery.md"
    write_report(front, cfg, runtime, oracle, oracle_slice, T.shape,
                 args.year, lats, lons, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
