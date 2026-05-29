"""CAMELS-style streamflow rediscovery for one basin.

Same methodological scaffolding as the weather benchmark
(TRAIN/TEST + Class A/B/C taxonomy) but for a real-data hydrology
problem with no known ground-truth equation.

The hypothesis tessera is testing
---------------------------------
Catchment hydrology has decades of empirical work pointing toward a
convolution form:

    Q[t] ≈ ∫ κ(s) · P_eff[t-s] ds   +   baseflow

where κ is the basin's instantaneous unit hydrograph (IUH) — peaked,
right-tailed, ~days-to-weeks timescale — and P_eff is "effective"
precipitation (rain that doesn't evaporate). The form has been a
conceptual workhorse since Sherman (1932); modern conceptual models
(SAC-SMA, HBV) implement it with hand-picked κ shapes. NO closed-form
equation derives κ from physics — it's catchment-specific and
parameterized.

This is EXACTLY what tessera's `LinearFunctional[Measure[κ]](P)`
primitive was built for. The GP has the structure to discover the
convolution form; the question is whether it does, and what κ it
finds.

Comparison baselines
--------------------
1. Predict yesterday's Q (persistence / AR(1))         — trivial bar
2. Linear regression of Q[t+1] on (Q[t], P[t], P-lag) — engineering bar
3. Reported LSTM benchmark (Kratzert et al. 2019)      — citation only

Kratzert's EA-LSTM on the same 531-basin CAMELS subset achieves
~0.74 median NSE on a TEST period 1989-1999 (TRAIN 1999-2008). That's
the bar in literature; we cite, we don't re-run.

Class A/B/C verdict
-------------------
- **C-mechanism-ish**: convolution form discovered, TRAIN ≤ engineering
  baseline AND TEST/TRAIN ratio matches baseline's ratio
- **B-natural-overfit**: candidate uses TRAIN-specific scalars
  (reduce_*) and TEST blows up — same failure mode as the heat-eq
  paired diagnostic
- **A-generic**: beats trivial but not engineering baseline
- **trivial**: tied with persistence baseline

Usage
-----
    python benchmarks/run_camels_streamflow.py
    python benchmarks/run_camels_streamflow.py --gauge 06614800 --pop 300 --gens 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from tessera.expression import (
    GP, GPConfig, FunctionalOp, iter_subtrees, evaluate as eval_tree,
)
from tessera.search.losses import mse_loss

from data.load_camels_basin import load_camels_basin, CAMELS_REFERENCE_BASINS


# ---------------- Engineering baselines ----------------

def fit_baselines_traintest(
    P: np.ndarray, T: np.ndarray, Q: np.ndarray, train_end: int,
) -> dict:
    """Fit baselines on TRAIN; evaluate on TRAIN + TEST.

    Target convention: predict Q[t+1] from features available at time t.
    """
    # Shift target by 1: y[t] = Q[t+1]
    y = np.roll(Q, -1)
    y[-1] = np.nan  # last sample has no Q[t+1]

    # Make a clean training mask: drop NaN + first 30 days (warmup for
    # lagged features) + last sample (no target).
    n = len(Q)
    warmup = 30
    valid = np.zeros(n, dtype=bool)
    valid[warmup:-1] = True
    valid &= np.isfinite(y) & np.isfinite(P) & np.isfinite(T) & np.isfinite(Q)

    tr_mask = valid.copy()
    tr_mask[train_end:] = False
    te_mask = valid.copy()
    te_mask[:train_end] = False

    var_y_train = float(np.var(y[tr_mask]))
    var_y_test = float(np.var(y[te_mask]))

    out: dict = dict(var_y_train=var_y_train, var_y_test=var_y_test)

    # 0. Predict zero (i.e. log_q = log_q_mean ≈ const)
    mean_y_train = float(np.mean(y[tr_mask]))
    pred_zero_tr = np.full(tr_mask.sum(), mean_y_train)
    pred_zero_te = np.full(te_mask.sum(), mean_y_train)
    out["mean"] = dict(
        mse_train=float(np.mean((pred_zero_tr - y[tr_mask]) ** 2)),
        mse_test=float(np.mean((pred_zero_te - y[te_mask]) ** 2)),
    )

    # 1. Persistence: predict Q[t+1] = Q[t]
    persistence_tr = Q[tr_mask]
    persistence_te = Q[te_mask]
    out["persistence"] = dict(
        mse_train=float(np.mean((persistence_tr - y[tr_mask]) ** 2)),
        mse_test=float(np.mean((persistence_te - y[te_mask]) ** 2)),
    )

    # 2. AR(1) with intercept: y[t] = a*Q[t] + b
    A_tr = np.column_stack([Q[tr_mask], np.ones(tr_mask.sum())])
    A_te = np.column_stack([Q[te_mask], np.ones(te_mask.sum())])
    coef_ar1, *_ = np.linalg.lstsq(A_tr, y[tr_mask], rcond=None)
    pred_tr = A_tr @ coef_ar1
    pred_te = A_te @ coef_ar1
    out["ar1"] = dict(
        a=float(coef_ar1[0]), b=float(coef_ar1[1]),
        mse_train=float(np.mean((pred_tr - y[tr_mask]) ** 2)),
        mse_test=float(np.mean((pred_te - y[te_mask]) ** 2)),
    )

    # 3. Multi-lag linear: y[t] = Σ_k a_k * P[t-k] + b*Q[t] + c (lags 0..7)
    lags = list(range(0, 8))
    cols = []
    for k in lags:
        cols.append(np.roll(P, k))
    cols.append(Q)
    cols.append(np.ones(n))
    X = np.column_stack(cols)
    coef_ml, *_ = np.linalg.lstsq(X[tr_mask], y[tr_mask], rcond=None)
    pred_tr = X[tr_mask] @ coef_ml
    pred_te = X[te_mask] @ coef_ml
    out["multilag_linear"] = dict(
        coef=coef_ml.tolist(), lags=lags,
        mse_train=float(np.mean((pred_tr - y[tr_mask]) ** 2)),
        mse_test=float(np.mean((pred_te - y[te_mask]) ** 2)),
    )

    out["train_mask"] = tr_mask
    out["test_mask"] = te_mask
    out["y"] = y
    return out


# ---------------- TEST evaluation for Pareto candidates ----------------

def evaluate_pareto_traintest(
    front, env_full: dict, y: np.ndarray,
    train_mask: np.ndarray, test_mask: np.ndarray,
    var_y_train: float, var_y_test: float,
    persistence_train: float, persistence_test: float,
    engineering_train: float, engineering_test: float,
    fill_warmup: float = 0.0,
) -> list[dict]:
    """For each Pareto candidate, evaluate on TRAIN and TEST.

    Returns list of dicts with (cx, tree, train_loss, test_loss,
    ratio, train_rel, test_rel, verdict).
    """
    rows: list[dict] = []
    for c in front:
        try:
            pred = eval_tree(c.tree, env_full, fill_warmup=fill_warmup)
            pred = np.asarray(pred, dtype=np.float64)
            if pred.ndim == 0:
                pred = np.full_like(y, float(pred))
            else:
                pred = np.broadcast_to(pred, y.shape).astype(np.float64).copy()
        except Exception:
            rows.append(dict(cx=c.complexity, tree=str(c.tree),
                             train_loss=float("nan"), test_loss=float("nan"),
                             ratio=float("nan"),
                             train_rel=float("nan"), test_rel=float("nan"),
                             verdict="invalid"))
            continue

        # MSE on the same masks the baselines used
        train_resid = pred[train_mask] - y[train_mask]
        test_resid = pred[test_mask] - y[test_mask]
        train_loss = float(np.mean(train_resid ** 2))
        test_loss = float(np.mean(test_resid ** 2))
        ratio = test_loss / train_loss if train_loss > 1e-12 else float("nan")
        train_rel = train_loss / var_y_train if var_y_train > 0 else float("nan")
        test_rel = test_loss / var_y_test if var_y_test > 0 else float("nan")
        verdict = classify_verdict(
            train_loss, test_loss, ratio,
            train_rel, test_rel,
            persistence_train, persistence_test,
            engineering_train, engineering_test,
        )
        rows.append(dict(
            cx=c.complexity, tree=str(c.tree),
            train_loss=train_loss, test_loss=test_loss, ratio=ratio,
            train_rel=train_rel, test_rel=test_rel,
            verdict=verdict,
        ))
    return rows


def classify_verdict(
    train_loss: float, test_loss: float, ratio: float,
    train_rel: float, test_rel: float,
    persistence_train: float, persistence_test: float,
    engineering_train: float, engineering_test: float,
) -> str:
    """Two-anchor verdict: persistence sets the trivial bar, multilag
    linear is the engineering bar that Class C must match or beat.

    - **trivial**:  TRAIN ≥ persistence * 0.98  (didn't escape persistence)
    - **B-natural-overfit**:  TEST > persistence_test * 1.3 (worse than
        persistence on TEST despite some TRAIN improvement)
    - **B-mild-overfit**:  ratio > engineering_ratio * 1.3  (worse
        TEST/TRAIN profile than engineering baseline)
    - **C-mechanism-ish**:  TRAIN ≤ engineering * 1.05  AND
        TEST ≤ engineering * 1.10  (matches or beats engineering on both
        halves; mechanism-shaped)
    - **A-generic**:  beats persistence on TRAIN but not at engineering
        level (modest signal, not the convolution kernel)
    """
    if not (np.isfinite(train_loss) and np.isfinite(test_loss)
            and np.isfinite(ratio)):
        return "invalid"

    # Trivial: never escaped persistence on TRAIN
    if train_loss >= persistence_train * 0.98:
        return "trivial"

    # TEST blew up vs persistence baseline → suspicious natural-overfit
    if test_loss > persistence_test * 1.3:
        return "B-natural-overfit"

    eng_ratio = (engineering_test / engineering_train
                 if engineering_train > 1e-12 else float("inf"))
    if np.isfinite(eng_ratio) and ratio > eng_ratio * 1.3:
        return "B-mild-overfit"

    # Class C: matches or beats engineering baseline on both halves
    if (train_loss <= engineering_train * 1.05
            and test_loss <= engineering_test * 1.10):
        return "C-mechanism-ish"

    # Class A: meaningfully beat persistence, didn't match engineering
    return "A-generic"


# ---------------- Reporting ----------------

def write_report(
    basin_id: str, basin_meta: dict, df: pd.DataFrame,
    baselines: dict, rows: list[dict], cfg: GPConfig, runtime: float,
    train_end: int, out_path: Path,
) -> None:
    var_y_tr = baselines["var_y_train"]
    var_y_te = baselines["var_y_test"]
    L = [f"# CAMELS streamflow rediscovery: {basin_meta['name']}", ""]
    L.append(f"**Basin:** USGS {basin_id} — {basin_meta['name']}")
    L.append(f"**Centroid:** lat={basin_meta['lat']}, lon={basin_meta['lon']}")
    L.append(f"**Area:** {basin_meta['area_km2']} km²")
    L.append(f"**Data:** {len(df):,} aligned daily samples "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    L.append(f"**Target:** `log Q[t+1]` (next-day log streamflow)")
    L.append(f"**Forcing:** DAYMET single-pixel (precip, tmax, tmin)")
    L.append(f"**Streamflow:** USGS NWIS daily mean discharge")
    L.append(f"**Split:** chronological — TRAIN = first {train_end} samples, "
             f"TEST = remaining ({len(df) - train_end})")
    L.append(f"**GP:** pop={cfg.pop_size}, gens={cfg.n_gens}, "
             f"pointwise_only={cfg.pointwise_only}, "
             f"runtime={runtime:.1f}s")
    L.append("")
    L.append("**TRAIN var(target) = "
             f"{var_y_tr:.4g}, TEST var(target) = {var_y_te:.4g}**")
    L.append("")

    # Baselines table
    L.append("## Baselines (TRAIN-fit, TRAIN+TEST evaluated)")
    L.append("")
    L.append("| Baseline | TRAIN MSE | TRAIN %var | TEST MSE | TEST %var | TEST/TRAIN |")
    L.append("|---|---|---|---|---|---|")
    for name, label in [
        ("mean", "predict mean"),
        ("persistence", "persistence (Q[t+1]=Q[t])"),
        ("ar1", "AR(1) with intercept"),
        ("multilag_linear", "multi-lag linear (P lags 0..7 + Q)"),
    ]:
        b = baselines[name]
        ratio = b["mse_test"] / b["mse_train"] if b["mse_train"] > 1e-12 else float("nan")
        L.append(f"| {label} | {b['mse_train']:.4g} | "
                 f"{100*b['mse_train']/var_y_tr:.1f}% | "
                 f"{b['mse_test']:.4g} | "
                 f"{100*b['mse_test']/var_y_te:.1f}% | "
                 f"{ratio:.2f} |")
    L.append("")
    L.append("Multi-lag linear is the *engineering* baseline — what a")
    L.append("hydrologist would build in 10 lines of `numpy.linalg.lstsq`.")
    L.append("The GP needs to beat this AND generalize to TEST.")
    L.append("")
    L.append("**Reported LSTM benchmark (citation, not run): Kratzert et al.")
    L.append("(2019) EA-LSTM on 531-basin CAMELS subset, median NSE ≈ 0.74**")
    L.append("on TEST period 1989-1999. NSE = 1 - MSE/var ≈ a TEST %var of ~26%.")
    L.append("Tessera's discrete-tree SR is not expected to match an LSTM's")
    L.append("expressive ceiling, but the *form* it discovers is interpretable")
    L.append("where the LSTM's parameters aren't.")
    L.append("")

    # Pareto table
    L.append("## Pareto front — TRAIN/TEST + class verdict")
    L.append("")
    L.append("| Cx | TRAIN | TRAIN %var | TEST | TEST %var | TEST/TRAIN | Verdict | Tree |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tree_str = r["tree"]
        if len(tree_str) > 160:
            tree_str = tree_str[:157] + "..."
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
    for cls in ["C-mechanism-ish", "A-generic", "A-baseline-ish",
                "B-mild-overfit", "B-natural-overfit", "trivial", "invalid"]:
        n = counts.get(cls, 0)
        if n:
            L.append(f"- **{cls}**: {n}")
    L.append("")

    # Pick best candidate
    rows_c = [r for r in rows if r["verdict"] == "C-mechanism-ish"]
    rows_other = [r for r in rows if r["verdict"] in ("A-generic", "A-baseline-ish")]
    if rows_c:
        best = min(rows_c, key=lambda r: r["test_loss"])
        header = "## Best Class-C candidate (mechanism-ish)"
    elif rows_other:
        best = min(rows_other, key=lambda r: r["test_loss"])
        header = "## Best non-trivial candidate"
    else:
        best = min(rows, key=lambda r: r["test_loss"])
        header = f"## Best by TEST loss (verdict={best['verdict']})"
    L.append(header)
    L.append("")
    L.append(f"```\n{best['tree']}\n```")
    L.append("")
    L.append(f"- cx = {best['cx']}")
    L.append(f"- TRAIN MSE = {best['train_loss']:.4g}  "
             f"({100*best['train_rel']:.1f}% of TRAIN var)")
    L.append(f"- TEST MSE = {best['test_loss']:.4g}  "
             f"({100*best['test_rel']:.1f}% of TEST var)")
    L.append(f"- TEST/TRAIN = {best['ratio']:.2f}")
    L.append(f"- verdict: **{best['verdict']}**")
    L.append("")

    L.append("## Reading the result")
    L.append("")
    L.append("- The engineering baseline (multi-lag linear) is what a")
    L.append("  hydrologist would write — already captures the convolution")
    L.append("  form at a fixed lag basis. GP needs to do BETTER than this")
    L.append("  to claim mechanism discovery; doing better than persistence")
    L.append("  alone is the trivial bar.")
    L.append("- A Class-C result means: discovered a convolution-shaped")
    L.append("  expression AND it generalizes to TEST years. The shape of")
    L.append("  the discovered Measure (EMA half-life, lag) is then")
    L.append("  interpretable as the basin's effective IUH timescale.")
    L.append("- A Class-B result would use trajectory-specific scalars")
    L.append("  (reduce_max(Q_train) etc.) that don't transfer to TEST.")
    L.append("  Same failure mode the `reduce_*` downweight (CHANGELOG")
    L.append("  2026-05-25) was designed to suppress.")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_camels_streamflow.py --gauge {basin_id} "
             f"--pop {cfg.pop_size} --gens {cfg.n_gens}")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")


# ---------------- Main ----------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gauge", default="01013500",
                   choices=list(CAMELS_REFERENCE_BASINS.keys()))
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--pop", type=int, default=300)
    p.add_argument("--gens", type=int, default=120)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--polish-every", type=int, default=5,
                   help="Sufficient-stats polish every K gens. 0 = disable. "
                        "Injects closed-form-optimal linear additive term "
                        "over lagged P features. See "
                        "tessera/search/sufficient_stats.py.")
    p.add_argument("--lags", type=str, default="1,3,7,14",
                   help="Comma-separated precipitation lags (days) to expose "
                        "to the sufficient-stats polish basis. Default 1,3,7,14.")
    args = p.parse_args(argv)

    basin = CAMELS_REFERENCE_BASINS[args.gauge]
    print(f"=== CAMELS streamflow: {args.gauge} — {basin['name']} ===\n")

    # ---- Load ----
    df = load_camels_basin(
        gauge_id=args.gauge,
        lat=basin["lat"], lon=basin["lon"],
        start_year=args.start_year, end_year=args.end_year,
        refresh=args.refresh,
    )

    # Features and target
    P = df["prcp"].values.astype(np.float64)
    Tm = df["tmean"].values.astype(np.float64)
    Q = df["log_q"].values.astype(np.float64)  # log-transformed
    Q_raw = df["q_obs"].values.astype(np.float64)

    n = len(df)
    train_end = int(n * args.train_frac)
    print(f"[split] n={n}, train_end={train_end} ({df.index[train_end].date()}), "
          f"TRAIN={train_end} / TEST={n-train_end}")

    # ---- Baselines ----
    baselines = fit_baselines_traintest(P, Tm, Q, train_end)
    print()
    print("[baselines]")
    for name in ("mean", "persistence", "ar1", "multilag_linear"):
        b = baselines[name]
        print(f"  {name:18s}  TRAIN={b['mse_train']:.4g}  TEST={b['mse_test']:.4g}  "
              f"ratio={b['mse_test']/b['mse_train']:.2f}")

    # ---- GP env: raw + lagged precipitation features ----
    # Target: y[t] = Q[t+1]
    y = baselines["y"]
    tr_mask = baselines["train_mask"]

    # Build lagged precipitation features. These are what the
    # sufficient-stats polish will use to inject the closed-form
    # optimal linear additive term — the "instantaneous unit
    # hydrograph" projection of the residual `Q[t+1] - Q[t]` onto
    # lagged precipitation, computed without GP search.
    lag_days = [int(x) for x in args.lags.split(",") if x.strip()]
    print(f"\n[features] precipitation lags: {lag_days}")

    def lagged(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.empty_like(arr)
        if k <= 0:
            return arr.copy()
        out[:k] = arr[0]  # pad with first value; sufficient stats handles warmup
        out[k:] = arr[:-k]
        return out

    P_lags = {f"P_lag{k}": lagged(P, k) for k in lag_days}

    # For the GP run, we slice to TRAIN-only env to avoid info leakage.
    env_train = {
        "P":  P[:train_end].copy(),
        "T":  Tm[:train_end].copy(),
        "Q":  Q[:train_end].copy(),
    }
    for name, series in P_lags.items():
        env_train[name] = series[:train_end].copy()
    y_train = y[:train_end].copy()
    # Mask out the warmup + last-element positions during scoring by
    # passing fill_warmup; tessera's mse_loss handles NaNs in y_pred but
    # not in y_true, so we replace y NaN with 0 + count the valid mask.
    y_train_clean = np.where(np.isfinite(y_train), y_train, 0.0)

    var_y_train = baselines["var_y_train"]

    # Two anchors for the verdict:
    #   persistence = "trivial" bar (Q[t+1] = Q[t])
    #   multilag    = "engineering" bar — 10 LOC of lstsq, the bar tessera
    #                 needs to match for a Class C verdict
    persistence_train = baselines["persistence"]["mse_train"]
    persistence_test = baselines["persistence"]["mse_test"]
    engineering_train = baselines["multilag_linear"]["mse_train"]
    engineering_test = baselines["multilag_linear"]["mse_test"]

    # ---- GP run ----
    parsimony = max(var_y_train * 0.001, 1e-5)
    # Sufficient-stats polish features: P + the lagged P variants.
    # We deliberately exclude Q from the polish basis because the GP's
    # best candidate ALREADY contains Q (persistence is too strong to
    # ignore); polishing on (residual after Q) projected onto lagged P
    # is the operation we want — exactly the instantaneous unit
    # hydrograph regression.
    polish_features = ("P",) + tuple(f"P_lag{k}" for k in lag_days)
    cfg = GPConfig(
        pop_size=args.pop, n_gens=args.gens,
        init_max_depth=4,
        parsimony=parsimony,
        early_stop_patience=20,
        seed=args.seed,
        enable_2d=False,
        pointwise_only=False,  # LinearFunctional needed for convolution
        fill_warmup=0.0,
        verbose=True,
        # Use BFGS const-opt (smooth log-MSE landscape).
        optimize_constants_every=3,
        optimize_constants_method="BFGS",
        optimize_constants_maxiter=30,
        # Sufficient-stats polish: inject closed-form linear additive
        # term on lagged precipitation features.
        sufficient_stats_polish_every=args.polish_every,
        sufficient_stats_feature_names=polish_features,
        sufficient_stats_max_degree=1,
        sufficient_stats_top_n_terms=5,
        sufficient_stats_coef_threshold=1e-6,
        sufficient_stats_include_constant=False,
    )
    if args.polish_every > 0:
        print(f"[polish] sufficient_stats every {args.polish_every} gens, "
              f"features={polish_features}, degree=1, top_n=5")
    print(f"\n[gp] cfg: pop={cfg.pop_size}, gens={cfg.n_gens}, "
          f"pointwise_only={cfg.pointwise_only}")
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env_train, y_train_clean, feature_names=["P", "T", "Q"])
    runtime = time.time() - t0
    print(f"\n[gp] runtime {runtime:.1f}s, Pareto front size {len(front)}")

    # ---- TEST evaluation: evaluate each tree on the FULL env and use masks ----
    env_full = {
        "P":  P.copy(),
        "T":  Tm.copy(),
        "Q":  Q.copy(),
    }
    for name, series in P_lags.items():
        env_full[name] = series.copy()
    rows = evaluate_pareto_traintest(
        front, env_full, y,
        baselines["train_mask"], baselines["test_mask"],
        var_y_train=baselines["var_y_train"],
        var_y_test=baselines["var_y_test"],
        persistence_train=persistence_train,
        persistence_test=persistence_test,
        engineering_train=engineering_train,
        engineering_test=engineering_test,
        fill_warmup=cfg.fill_warmup,
    )

    print()
    print("=" * 90)
    print("Pareto front — TRAIN/TEST + verdict")
    print("=" * 90)
    for r in rows:
        ascii_tree = r["tree"].encode("ascii", "replace").decode("ascii")
        print(f"  cx={r['cx']:2d}  tr={r['train_loss']:7.4g}  "
              f"te={r['test_loss']:7.4g}  ratio={r['ratio']:5.2f}  "
              f"{r['verdict']:20s}  | {ascii_tree[:60]}")

    # ---- Report ----
    out = Path(__file__).parent / "results" / f"camels_{args.gauge}.md"
    write_report(args.gauge, basin, df, baselines, rows, cfg, runtime,
                 train_end, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
