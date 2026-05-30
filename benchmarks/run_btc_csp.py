"""csp_sr on a real-data, no-closed-form-GT benchmark: BTC hourly.

The "unknown-GT real-world" benchmark class (project memory): real data,
possibly-solvable dynamics, no ground-truth equation. Honest, time-respecting,
out-of-sample evaluation.

Two targets from BTC/USDT hourly OHLCV:
  - next-hour LOG-RETURN  (the hard one: near-efficient, ~unpredictable)
  - next-hour |LOG-RETURN| = volatility (the discoverable one: vol clusters)

Protocol: chronological 70/30 split (NO shuffle), features standardised on
train stats only (no leakage), score R^2 and directional accuracy on the
held-out FUTURE. Compare naive, linear OLS, and csp_sr on identical features.
The honest question: does csp find a GENERALISING symbolic relationship out of
sample — none for returns (expected), real structure for volatility (expected)?

Usage: python benchmarks/run_btc_csp.py
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np

from tessera.experimental.csp_sr import discover, CSPSRConfig, expr_to_str
from tessera.expression.tree import evaluate

OUT = Path(__file__).parent / "results" / "btc_csp.md"


def fetch_btc(n_hours=4000):
    import requests
    out = []
    end = None
    while len(out) < n_hours:
        p = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 1000}
        if end:
            p['endTime'] = end
        r = requests.get('https://api.binance.com/api/v3/klines', params=p, timeout=20)
        k = r.json()
        if not isinstance(k, list) or not k:
            break
        out = k + out
        end = k[0][0] - 1
        time.sleep(0.25)
    a = np.array([[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                  for c in out[-n_hours:]], dtype=np.float64)
    return a                                       # (N,5) open,high,low,close,vol


def build(arr):
    o, h, lo, c, v = arr.T
    r = np.diff(np.log(c))                          # log-returns (N-1,)
    a = np.abs(r)
    lv = np.log(v[1:] + 1.0)
    rng = (h[1:] - lo[1:]) / c[1:]                  # intraday range
    n = len(r)
    feats, names, rows = [], [], []
    L = 24                                          # history needed
    for t in range(L, n - 1):
        f = {}
        for k in (1, 2, 3, 5):
            f[f'r{k}'] = r[t-k+1]
            f[f'a{k}'] = a[t-k+1]
        f['vol6'] = r[t-5:t+1].std()
        f['vol12'] = r[t-11:t+1].std()
        f['vol24'] = r[t-23:t+1].std()
        f['mom6'] = r[t-5:t+1].sum()
        f['lv'] = lv[t]
        f['dlv'] = lv[t] - lv[t-1]
        f['rng'] = rng[t]
        if not names:
            names = list(f.keys())
        rows.append([f[k] for k in names])
        feats.append((r[t+1], a[t+1]))             # targets: next ret, next |ret|
    X = np.array(rows); Y = np.array(feats)
    return X, Y[:, 0], Y[:, 1], names               # X, y_ret, y_vol, names


def _r2(pred, y):
    pred = np.asarray(pred, np.float64)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return float('nan')
    return float(1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30))


def evaluate_target(Xtr, Xte, ytr, yte, names, label, directional, log):
    # standardise features on TRAIN stats
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    envtr = {names[i]: Ztr[:, i] for i in range(len(names))}
    envte = {names[i]: Zte[:, i] for i in range(len(names))}

    # naive: train mean
    naive = np.full_like(yte, ytr.mean())
    r2_naive = _r2(naive, yte)

    # linear OLS (closed form, ridge for stability)
    A = np.c_[Ztr, np.ones(len(Ztr))]
    w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ ytr)
    lin = np.c_[Zte, np.ones(len(Zte))] @ w
    r2_lin = _r2(lin, yte)

    # csp_sr (free-form), fit on train, score test
    cfg = CSPSRConfig(unary=['neg', 'sqrt', 'tanh', 'abs'],
                      binary=['add', 'sub', 'mul'],
                      max_size=2, max_terms=6, beam_width=12, max_features=20000)
    res = discover(envtr, ytr, cfg)
    csp = np.asarray(evaluate(res.expr, envte), np.float64)
    r2_csp = _r2(csp, yte)

    def diracc(pred):
        return float((np.sign(pred) == np.sign(yte)).mean()) if directional else float('nan')

    log(f"\nTARGET: {label}   (test n={len(yte)})")
    log(f"  naive(mean)   R2_oos={r2_naive:+.4f}" +
        (f"  dir={diracc(naive):.3f}" if directional else ""))
    log(f"  linear OLS    R2_oos={r2_lin:+.4f}" +
        (f"  dir={diracc(lin):.3f}" if directional else ""))
    log(f"  csp_sr        R2_oos={r2_csp:+.4f}" +
        (f"  dir={diracc(csp):.3f}" if directional else "") +
        f"   ({res.n_terms} terms)")
    log(f"    csp expr: {expr_to_str(res.expr)[:120]}")
    return dict(label=label, naive=r2_naive, lin=r2_lin, csp=r2_csp,
                dir_csp=diracc(csp), expr=expr_to_str(res.expr)[:150],
                terms=res.n_terms)


def main():
    out_lines = []
    def log(s):
        print(s); out_lines.append(s)
    log("=== csp_sr on BTC hourly (real data, no closed-form GT) ===")
    arr = fetch_btc(4000)
    log(f"fetched {len(arr)} hourly candles (BTCUSDT), last close={arr[-1,3]:.0f}")
    X, y_ret, y_vol, names = build(arr)
    ntr = int(len(X) * 0.7)
    log(f"features={len(names)}, samples={len(X)}, chronological split "
        f"train={ntr} test={len(X)-ntr} (NO shuffle)")

    rows = []
    rows.append(evaluate_target(X[:ntr], X[ntr:], y_ret[:ntr], y_ret[ntr:],
                                names, "next log-return", True, log))
    rows.append(evaluate_target(X[:ntr], X[ntr:], y_vol[:ntr], y_vol[ntr:],
                                names, "next |log-return| (volatility)", False, log))

    log("\n## Reading")
    log("- Returns: any out-of-sample R2 > 0 / dir > 0.50 is a (weak) real edge;")
    log("  expected ~0 (markets are near-efficient at 1h). Honest if it's nil.")
    log("- Volatility: vol clusters, so a generalising R2_oos >> 0 is the genuine")
    log("  discoverable structure (GARCH-like). csp beating naive+linear here =")
    log("  it found a real symbolic relationship in real data, no GT needed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# csp_sr on BTC hourly (out-of-sample)\n\n```\n"
                   + "\n".join(out_lines) + "\n```\n", encoding="utf-8")
    print(f"\n[report] wrote {OUT}")


if __name__ == "__main__":
    main()
