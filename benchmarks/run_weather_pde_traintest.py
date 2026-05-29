"""Weather PDE rediscovery with TRAIN/TEST split — Class A/B/C taxonomy.

Extends `run_weather_pde_rediscovery.py` with a held-out test split,
so the benchmark inherits the §6 methodological discipline that
`from_data_to_mechanism.md` established for the heat-equation work:
distinguish Class A (generic), Class B (natural-overfit / TRAIN-only
fit), and Class C (mechanism that generalizes).

What's new vs the v1 weather runner
-----------------------------------
- **Chronological 75/25 split** on the mid-latitude row slice. GP
  trains on the first 75% of days; TEST is the last 25%.
- **Oracle baselines refit on TRAIN only**, then evaluated on both
  TRAIN and TEST. Sets the bar for both halves separately.
- **Per-candidate TEST evaluation**: evaluate each Pareto-front tree
  on the TEST slice, compute MSE, report alongside TRAIN.
- **Class verdict column** per the A/B/C taxonomy in
  `docs/research/from_data_to_mechanism.md` §5:
    - **trivial**  — TRAIN and TEST both ≥ 95% of var (predict-zero)
    - **natural-overfit (B)** — TEST/TRAIN > 2.0  (fits TRAIN-specific
      pattern that doesn't transfer)
    - **mild-overfit** — 1.3 < TEST/TRAIN ≤ 2.0
    - **mechanism-ish (C)** — TRAIN < 96% of var AND TEST/TRAIN ≤ 1.3
    - **generic (A)** — none of the above

The class verdict is heuristic — it categorizes the failure mode but
doesn't claim ground-truth mechanism. Weather doesn't have a single
canonical PDE the way heat-equation does.

Why temporal split (and what it tests)
--------------------------------------
TRAIN = Jan-Sep, TEST = Oct-Dec (in a normal year). Climatology
differs between the two halves, so a candidate that fit TRAIN by
memorizing seasonal patterns will degrade on TEST. A true PDE
operator (∇²T, ∂T/∂x, etc.) should degrade only modestly because
the underlying physics is the same.

This is the cleanest test of "did the GP find Class B or Class C
on real data."

Usage
-----
    python benchmarks/run_weather_pde_traintest.py
    python benchmarks/run_weather_pde_traintest.py --year 2023 --pop 150 --gens 60
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
    evaluate as eval_tree,
)
from tessera.search.losses import mse_loss

from data.load_ncep_reanalysis import load_ncep_t2m


# ---------------- Oracle baselines (TRAIN-fit, TEST-evaluated) -----

def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) from OLS."""
    A = np.stack([x.ravel(), np.ones_like(x.ravel())], axis=1)
    coef, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    return float(coef[0]), float(coef[1])


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_oracle_baselines_traintest(
    T_slice: np.ndarray, train_end: int,
) -> dict:
    """Fit each oracle on TRAIN only, evaluate on both TRAIN and TEST.

    Returns a dict with per-oracle (mse_train, mse_test, var_dT_train,
    var_dT_test) + the fitted coefficient.

    Same four oracles as v1: predict-zero, Newtonian relaxation,
    1D horizontal Laplacian (diffusion), AR(1) persistence.
    """
    dt_T = np.zeros_like(T_slice)
    dt_T[:-1] = T_slice[1:] - T_slice[:-1]

    # Split
    T_tr = T_slice[:train_end]
    T_te = T_slice[train_end:]
    dt_tr = dt_T[:train_end]
    dt_te = dt_T[train_end:]
    var_dT_train = float(np.var(dt_tr))
    var_dT_test = float(np.var(dt_te))

    out: dict = dict(
        var_dT_train=var_dT_train,
        var_dT_test=var_dT_test,
    )

    # 1. Predict zero
    out["zero"] = dict(
        coef=0.0,
        mse_train=float(np.mean(dt_tr ** 2)),
        mse_test=float(np.mean(dt_te ** 2)),
    )

    # 2. Newtonian relaxation: dT/dt = α (T - T̄). Fit T̄ on TRAIN.
    T_bar = float(T_tr.mean())
    anom_tr = T_tr - T_bar
    anom_te = T_te - T_bar
    alpha, intercept = _fit_linear(anom_tr, dt_tr)
    out["newton"] = dict(
        coef=alpha, T_bar=T_bar, intercept=intercept,
        mse_train=_mse(alpha * anom_tr + intercept, dt_tr),
        mse_test=_mse(alpha * anom_te + intercept, dt_te),
    )

    # 3. 1D horizontal Laplacian on the slice
    def lap_x(T):
        out = np.zeros_like(T)
        out[:, 1:-1] = T[:, :-2] - 2.0 * T[:, 1:-1] + T[:, 2:]
        return out
    lap_tr = lap_x(T_tr)
    lap_te = lap_x(T_te)
    a, b = _fit_linear(lap_tr, dt_tr)
    out["diff_1D"] = dict(
        coef=a, intercept=b,
        mse_train=_mse(a * lap_tr + b, dt_tr),
        mse_test=_mse(a * lap_te + b, dt_te),
    )

    # 4. AR(1) persistence
    def delta_prev(T):
        out = np.zeros_like(T)
        out[1:] = T[1:] - T[:-1]
        return out
    dp_tr = delta_prev(T_tr)
    dp_te = delta_prev(T_te)
    a4, b4 = _fit_linear(dp_tr, dt_tr)
    out["ar1"] = dict(
        coef=a4, intercept=b4,
        mse_train=_mse(a4 * dp_tr + b4, dt_tr),
        mse_test=_mse(a4 * dp_te + b4, dt_te),
    )
    return out


# ---------------- Per-candidate TEST evaluation -----

def evaluate_pareto_traintest(
    front, T_slice: np.ndarray, train_end: int,
    fill_warmup: float = 0.0,
) -> list[dict]:
    """For each Pareto candidate, evaluate on TRAIN and TEST slices
    separately. Returns list of dicts with (cx, tree, train_loss,
    test_loss, ratio, verdict).
    """
    dt_T = np.zeros_like(T_slice)
    dt_T[:-1] = T_slice[1:] - T_slice[:-1]
    T_tr = T_slice[:train_end]
    T_te = T_slice[train_end:]
    dt_tr = dt_T[:train_end]
    dt_te = dt_T[train_end:]
    var_dT_train = float(np.var(dt_tr))
    var_dT_test = float(np.var(dt_te))

    rows: list[dict] = []
    for c in front:
        # Compute TRAIN and TEST losses by direct evaluation. We pass
        # fill_warmup=0.0 to score the entire slice (matches the GP's
        # convention from `run_weather_pde_rediscovery.py`).
        try:
            pred_tr = eval_tree(c.tree, {"T": T_tr}, fill_warmup=fill_warmup)
            pred_tr = np.asarray(pred_tr, dtype=np.float64)
            if pred_tr.ndim == 0:
                pred_tr = np.full_like(dt_tr, float(pred_tr))
            elif pred_tr.shape != dt_tr.shape:
                pred_tr = np.broadcast_to(pred_tr, dt_tr.shape)
            train_loss = mse_loss(pred_tr, dt_tr)
        except Exception:
            train_loss = float("nan")

        try:
            pred_te = eval_tree(c.tree, {"T": T_te}, fill_warmup=fill_warmup)
            pred_te = np.asarray(pred_te, dtype=np.float64)
            if pred_te.ndim == 0:
                pred_te = np.full_like(dt_te, float(pred_te))
            elif pred_te.shape != dt_te.shape:
                pred_te = np.broadcast_to(pred_te, dt_te.shape)
            test_loss = mse_loss(pred_te, dt_te)
        except Exception:
            test_loss = float("nan")

        if np.isfinite(train_loss) and train_loss > 1e-12:
            ratio = test_loss / train_loss
        else:
            ratio = float("nan")

        train_rel = train_loss / var_dT_train if var_dT_train > 0 else float("nan")
        test_rel = test_loss / var_dT_test if var_dT_test > 0 else float("nan")

        verdict = classify_verdict(train_rel, test_rel, ratio)

        rows.append(dict(
            cx=c.complexity,
            tree=str(c.tree),
            train_loss=train_loss, test_loss=test_loss,
            train_rel=train_rel, test_rel=test_rel,
            ratio=ratio, verdict=verdict,
        ))
    return rows


def classify_verdict(
    train_rel: float, test_rel: float, ratio: float,
) -> str:
    """Heuristic Class A/B/C verdict from TRAIN/TEST behaviour.

    Thresholds chosen to match the spirit of the heat-equation
    taxonomy (`docs/research/from_data_to_mechanism.md` §5) — small
    enough to flag clear cases, loose enough to avoid over-claiming.
    """
    if not (np.isfinite(train_rel) and np.isfinite(test_rel)):
        return "invalid"
    # Trivial: barely better than predict-zero
    if train_rel >= 0.98 and test_rel >= 0.98:
        return "trivial"
    if not np.isfinite(ratio):
        return "invalid"
    # Natural-overfit: big TEST blowup vs TRAIN
    if ratio > 2.0:
        return "B-natural-overfit"
    if ratio > 1.3:
        return "B-mild-overfit"
    # Below oracle ceiling on TRAIN and decent TEST: mechanism-ish
    if train_rel < 0.96 and ratio <= 1.3:
        return "C-mechanism-ish"
    return "A-generic"


# ---------------- Reporting -----

def write_report(
    rows: list[dict], oracles: dict, cfg: GPConfig, runtime: float,
    T_shape: tuple, year: int, lats, lons,
    train_end: int, train_days_label: str, test_days_label: str,
    out_path: Path,
) -> None:
    var_dT_tr = oracles["var_dT_train"]
    var_dT_te = oracles["var_dT_test"]

    L = ["# Weather PDE rediscovery: TRAIN/TEST split + Class A/B/C taxonomy", ""]
    L.append(f"**Data:** NCEP-DOE AMIP-II Reanalysis daily 2m temperature, year {year}")
    L.append(f"**Grid:** {T_shape[0]} days × {T_shape[1]} lats × {T_shape[2]} lons "
             f"(lat {lats[0]:.1f}..{lats[-1]:.1f}, lon {lons[0]:.1f}..{lons[-1]:.1f})")
    L.append(f"**Split:** chronological 75/25 — TRAIN = first {train_end} days "
             f"({train_days_label}), TEST = remaining ({test_days_label})")
    L.append(f"**Target:** `dT/dt = T[t+1] − T[t]`; "
             f"var(TRAIN) = {var_dT_tr:.4g} K²/day², "
             f"var(TEST) = {var_dT_te:.4g} K²/day²")
    L.append(f"**GP:** pop={cfg.pop_size}, gens={cfg.n_gens}, "
             f"parsimony={cfg.parsimony:.3g}, enable_2d={cfg.enable_2d}, "
             f"runtime={runtime:.1f}s")
    L.append("")
    L.append("## Oracle baselines (TRAIN-fit, TRAIN+TEST evaluated)")
    L.append("")
    L.append("Each baseline's coefficient is fit on TRAIN only, then evaluated on both halves.")
    L.append("If the baseline form is genuine physics, TRAIN ≈ TEST. If it's TRAIN-overfit, TEST blows up.")
    L.append("")
    L.append("| Oracle | TRAIN MSE | TRAIN %var | TEST MSE | TEST %var | TEST/TRAIN |")
    L.append("|---|---|---|---|---|---|")
    for name, label in [("zero", "predict 0"),
                        ("newton", "Newton relax (α·(T−T̄))"),
                        ("diff_1D", "1D diffusion (α·∂²T/∂x²)"),
                        ("ar1", "AR(1) (β·(T[t]−T[t-1]))")]:
        o = oracles[name]
        ratio = o["mse_test"] / o["mse_train"] if o["mse_train"] > 1e-12 else float("nan")
        L.append(f"| {label} | {o['mse_train']:.4g} | "
                 f"{100*o['mse_train']/var_dT_tr:.1f}% | "
                 f"{o['mse_test']:.4g} | "
                 f"{100*o['mse_test']/var_dT_te:.1f}% | "
                 f"{ratio:.2f} |")
    L.append("")
    L.append("## Pareto front — TRAIN/TEST + class verdict")
    L.append("")
    L.append("Each candidate's TRAIN loss is what the GP optimized; TEST loss is from")
    L.append("evaluating the same tree on the held-out slice. Class verdict per the")
    L.append("taxonomy in `docs/research/from_data_to_mechanism.md` §5.")
    L.append("")
    L.append("| Cx | TRAIN | TRAIN %var | TEST | TEST %var | TEST/TRAIN | Verdict | Tree |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tree_str = r["tree"]
        if len(tree_str) > 140:
            tree_str = tree_str[:137] + "..."
        L.append(f"| {r['cx']} | {r['train_loss']:.4g} | "
                 f"{100*r['train_rel']:.1f}% | "
                 f"{r['test_loss']:.4g} | "
                 f"{100*r['test_rel']:.1f}% | "
                 f"{r['ratio']:.2f} | "
                 f"{r['verdict']} | `{tree_str}` |")
    L.append("")
    L.append("## Class tally")
    L.append("")
    from collections import Counter
    counts = Counter(r["verdict"] for r in rows)
    for cls in ["C-mechanism-ish", "A-generic", "B-mild-overfit",
                "B-natural-overfit", "trivial", "invalid"]:
        n = counts.get(cls, 0)
        if n:
            L.append(f"- **{cls}**: {n}")
    L.append("")

    # Pick the most useful "best" candidate. Prefer Class C; fall back
    # to Class A; report B candidates as warnings.
    rows_c = [r for r in rows if r["verdict"] == "C-mechanism-ish"]
    rows_a = [r for r in rows if r["verdict"] == "A-generic"]
    if rows_c:
        best = min(rows_c, key=lambda r: r["test_loss"])
        L.append(f"## Best Class-C candidate (mechanism-ish)")
    elif rows_a:
        best = min(rows_a, key=lambda r: r["test_loss"])
        L.append(f"## Best Class-A candidate (generic, no mechanism found)")
    else:
        best = min(rows, key=lambda r: r["test_loss"])
        L.append(f"## Best by TEST loss (no Class C/A; verdict = {best['verdict']})")
    L.append("")
    L.append(f"```\n{best['tree']}\n```")
    L.append("")
    L.append(f"- cx = {best['cx']}")
    L.append(f"- TRAIN MSE = {best['train_loss']:.4g}  "
             f"({100*best['train_rel']:.1f}% of TRAIN var)")
    L.append(f"- TEST MSE = {best['test_loss']:.4g}  "
             f"({100*best['test_rel']:.1f}% of TEST var)")
    L.append(f"- TEST/TRAIN ratio = {best['ratio']:.2f}")
    L.append(f"- verdict: **{best['verdict']}**")
    L.append("")

    # Compare to best oracle
    best_oracle_train = min(
        ("Newton", oracles["newton"]["mse_train"]),
        ("diff_1D", oracles["diff_1D"]["mse_train"]),
        ("AR(1)", oracles["ar1"]["mse_train"]),
        key=lambda kv: kv[1],
    )
    best_oracle_test = min(
        ("Newton", oracles["newton"]["mse_test"]),
        ("diff_1D", oracles["diff_1D"]["mse_test"]),
        ("AR(1)", oracles["ar1"]["mse_test"]),
        key=lambda kv: kv[1],
    )
    L.append("## vs best oracle")
    L.append("")
    L.append(f"- best TRAIN oracle: **{best_oracle_train[0]}** at MSE={best_oracle_train[1]:.4g}")
    L.append(f"- best TEST oracle: **{best_oracle_test[0]}** at MSE={best_oracle_test[1]:.4g}")
    if best["train_loss"] < best_oracle_train[1]:
        L.append(f"- GP TRAIN beats best oracle by "
                 f"{100*(1 - best['train_loss']/best_oracle_train[1]):.1f}%")
    else:
        L.append(f"- GP TRAIN does NOT beat best oracle")
    if best["test_loss"] < best_oracle_test[1]:
        L.append(f"- GP TEST beats best oracle by "
                 f"{100*(1 - best['test_loss']/best_oracle_test[1]):.1f}%")
    else:
        L.append(f"- GP TEST does NOT beat best oracle "
                 f"({100*best['test_loss']/best_oracle_test[1]:.1f}% of oracle)")
    L.append("")

    L.append("## Reading the result")
    L.append("")
    L.append("- A Class-C result is the goal: TRAIN beats the trivial baselines AND")
    L.append("  TEST follows along (TEST/TRAIN ≤ 1.3). That's the closest equivalent")
    L.append("  on real data of the heat-equation 'canonical mechanism' verdict.")
    L.append("- A Class-B result means the GP found a TRAIN-specific pattern that")
    L.append("  doesn't transfer. On weather this often shows up as a candidate that")
    L.append("  uses scalar reductions or seasonally-biased windowing — same failure")
    L.append("  mode the `reduce_*` downweight (CHANGELOG 2026-05-25) was designed for.")
    L.append("- A Class-A result is honest underfit: nothing meaningful found, but")
    L.append("  also no false positive. Better than B; worse than C.")
    L.append("- 'trivial' = predict-zero level. Means the GP search didn't find any")
    L.append("  signal at all, even on TRAIN.")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_weather_pde_traintest.py --year {year} "
             f"--pop {cfg.pop_size} --gens {cfg.n_gens}")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ---------------- Main -----

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
    p.add_argument("--train-frac", type=float, default=0.75)
    args = p.parse_args(argv)

    # ---- Load ----
    T, lats, lons, days = load_ncep_t2m(
        year=args.year,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
    )
    print(f"[setup] T shape {T.shape}, var(T)={T.var():.2f} K^2")

    # ---- Slice to mid-latitude row ----
    Y = T.shape[1]
    mid = Y // 2
    T_slice = T[:, mid, :]  # (T_days, X)
    print(f"[slice] mid-lat row idx={mid}; slice shape {T_slice.shape}")

    # ---- TRAIN/TEST split ----
    n_days = T_slice.shape[0]
    train_end = int(n_days * args.train_frac)
    train_days_label = f"days 1-{train_end}"
    test_days_label = f"days {train_end+1}-{n_days}"
    print(f"[split] TRAIN: {train_days_label}, TEST: {test_days_label}")

    T_train_slice = T_slice[:train_end]
    dt_T = np.zeros_like(T_slice)
    dt_T[:-1] = T_slice[1:] - T_slice[:-1]
    dt_train_slice = dt_T[:train_end]

    # ---- Oracle baselines (TRAIN-fit, both-evaluated) ----
    oracles = compute_oracle_baselines_traintest(T_slice, train_end)
    print(f"\n[oracle] var(dT/dt) TRAIN={oracles['var_dT_train']:.4g}, "
          f"TEST={oracles['var_dT_test']:.4g}")
    for name in ("zero", "newton", "diff_1D", "ar1"):
        o = oracles[name]
        ratio = o["mse_test"] / o["mse_train"] if o["mse_train"] > 1e-12 else float("nan")
        print(f"[oracle] {name:8s}: TRAIN={o['mse_train']:.4g}  "
              f"TEST={o['mse_test']:.4g}  ratio={ratio:.2f}")

    # ---- GP on TRAIN slice only ----
    cfg = GPConfig(
        pop_size=args.pop, n_gens=args.gens,
        init_max_depth=4,
        parsimony=max(oracles["var_dT_train"] * 0.005, 1e-4),
        early_stop_patience=15,
        seed=args.seed,
        enable_2d=True,
        fill_warmup=0.0,
        verbose=True,
    )
    env_train = {"T": T_train_slice}
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env_train, dt_train_slice, feature_names=["T"])
    runtime = time.time() - t0
    print(f"\n[gp] runtime {runtime:.1f}s, Pareto front size {len(front)}")

    # ---- TEST evaluation for each Pareto candidate ----
    rows = evaluate_pareto_traintest(front, T_slice, train_end, cfg.fill_warmup)

    print()
    print("=" * 80)
    print("Pareto front — TRAIN + TEST + class verdict")
    print("=" * 80)
    for r in rows:
        ascii_tree = r["tree"].encode("ascii", "replace").decode("ascii")
        print(f"  cx={r['cx']:2d}  tr={r['train_loss']:7.4g}  "
              f"te={r['test_loss']:7.4g}  ratio={r['ratio']:5.2f}  "
              f"{r['verdict']:20s}  | {ascii_tree[:60]}")

    # ---- Report ----
    out = Path(__file__).parent / "results" / "weather_pde_traintest.md"
    write_report(rows, oracles, cfg, runtime, T.shape, args.year, lats, lons,
                 train_end, train_days_label, test_days_label, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
