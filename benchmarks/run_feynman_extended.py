"""Tessera on an extended Feynman SR benchmark (~30 equations).

The full Feynman dataset (Udrescu & Tegmark 2020) has 100 physics
equations. This extended runner covers ~30 representative equations
from Table II of the paper, hand-picked to span the main complexity
classes:

  - Trivial products / sums (3 eqs)
  - Rational forms / quotients (5 eqs)
  - Polynomial / squared terms (4 eqs)
  - Trigonometric (2 eqs) — uses sin/cos which tessera lacks; will fail
  - Exponential / Gaussian forms (3 eqs)
  - Square-root forms (4 eqs)
  - Logarithmic forms (2 eqs)
  - Multi-variable compound (7 eqs)

Larger search budget than the 8-equation subset:
  pop=400, gens=120, init_max_depth=5, optimize_constants every 3 gens.

Expected wall-clock: ~30-60 minutes on 8-core CPU.

For each equation, we report:
  - Wall-clock to a found expression
  - Train MSE at Pareto front's lowest-loss complexity
  - rel = train_loss / var(y)  (0 = perfect; near 1 = no fit)
  - Whether the analytical form is structurally recoverable

This is the Goal-1 (workbench) headline benchmark.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig


OUT_DIR = Path(__file__).parent / "results"


# ============ Equation samplers ============
# Each: returns (env_dict, y_array). Sampling ranges follow the
# original paper's Table II where given; otherwise sensible bounded
# uniform.

def s_I_6_20a(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.5, 5.0, n)
    return {"theta": th}, np.exp(-(th ** 2) / 2.0)


def s_I_6_20(n=2000, seed=0):
    """exp(-(theta/sigma)^2/2)"""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.5, 5.0, n); sig = rng.uniform(0.5, 2.0, n)
    return {"theta": th, "sigma": sig}, np.exp(-((th / sig) ** 2) / 2.0)


def s_I_8_14(n=2000, seed=0):
    """Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)"""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n); x2 = rng.uniform(-1, 1, n)
    y1 = rng.uniform(-1, 1, n); y2 = rng.uniform(-1, 1, n)
    return ({"x1": x1, "x2": x2, "y1": y1, "y2": y2},
            np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def s_I_9_18(n=2000, seed=0):
    """G*m1*m2 / ((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)"""
    rng = np.random.default_rng(seed)
    G = rng.uniform(1, 2, n)
    m1 = rng.uniform(1, 2, n); m2 = rng.uniform(1, 2, n)
    x1 = rng.uniform(1, 2, n); x2 = rng.uniform(3, 4, n)
    y1 = rng.uniform(1, 2, n); y2 = rng.uniform(3, 4, n)
    z1 = rng.uniform(1, 2, n); z2 = rng.uniform(3, 4, n)
    r2 = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    return ({"G": G, "m1": m1, "m2": m2,
             "x1": x1, "x2": x2, "y1": y1, "y2": y2, "z1": z1, "z2": z2},
            G * m1 * m2 / r2)


def s_I_10_7(n=2000, seed=0):
    """m_0 / sqrt(1 - v^2/c^2)   (relativistic mass)"""
    rng = np.random.default_rng(seed)
    m0 = rng.uniform(1, 5, n); v = rng.uniform(0.1, 0.5, n)
    c = rng.uniform(1.0, 2.0, n)
    return {"m0": m0, "v": v, "c": c}, m0 / np.sqrt(1 - (v / c) ** 2)


def s_I_11_19(n=2000, seed=0):
    """x1*y1 + x2*y2 + x3*y3 (dot product 3D)"""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(1, 5, n); x2 = rng.uniform(1, 5, n); x3 = rng.uniform(1, 5, n)
    y1 = rng.uniform(1, 5, n); y2 = rng.uniform(1, 5, n); y3 = rng.uniform(1, 5, n)
    return ({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2, "y3": y3},
            x1*y1 + x2*y2 + x3*y3)


def s_I_12_1(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(1, 5, n); Nn = rng.uniform(1, 5, n)
    return {"mu": mu, "Nn": Nn}, mu * Nn


def s_I_12_2(n=2000, seed=0):
    """q1*q2 / (4*pi*eps*r^2)"""
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(1, 5, n); q2 = rng.uniform(1, 5, n)
    eps = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"q1": q1, "q2": q2, "eps": eps, "r": r}, q1 * q2 / (4 * np.pi * eps * r * r)


def s_I_12_4(n=2000, seed=0):
    """q1*q2 / (4*pi*eps*r^2)  (same as 12.2 but different naming/range)"""
    return s_I_12_2(n, seed + 1)


def s_I_12_5(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    q1 = rng.uniform(1, 5, n); q2 = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"q1": q1, "q2": q2, "r": r}, q1 * q2 / (r * r)


def s_I_12_11(n=2000, seed=0):
    """q*(Ef + B*v*sin(theta))  — uses sin; expected to fail"""
    rng = np.random.default_rng(seed)
    q = rng.uniform(1, 5, n); Ef = rng.uniform(1, 5, n)
    B = rng.uniform(1, 5, n); v = rng.uniform(1, 5, n)
    th = rng.uniform(0, np.pi, n)
    return ({"q": q, "Ef": Ef, "B": B, "v": v, "theta": th},
            q * (Ef + B * v * np.sin(th)))


def s_I_14_3(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 5, n); g = rng.uniform(1, 5, n); z = rng.uniform(1, 5, n)
    return {"m": m, "g": g, "z": z}, m * g * z


def s_I_14_4(n=2000, seed=0):
    """0.5 * k * x^2 (spring potential)"""
    rng = np.random.default_rng(seed)
    k = rng.uniform(1, 5, n); x = rng.uniform(1, 5, n)
    return {"k": k, "x": x}, 0.5 * k * x * x


def s_I_15_3t(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 5, n); u = rng.uniform(0.1, 0.5, n)
    t = rng.uniform(1, 5, n); c = rng.uniform(1.0, 2.0, n)
    return ({"x": x, "u": u, "t": t, "c": c},
            (x - u * t) / np.sqrt(1 - (u / c) ** 2))


def s_I_15_3x(n=2000, seed=0):
    """(x - u*t) / sqrt(1 - u^2/c^2)  — Lorentz x-coordinate"""
    return s_I_15_3t(n, seed)


def s_I_16_6(n=2000, seed=0):
    """(u + v) / (1 + u*v/c^2)  (relativistic velocity addition)"""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.1, 0.5, n); v = rng.uniform(0.1, 0.5, n)
    c = rng.uniform(1.0, 2.0, n)
    return {"u": u, "v": v, "c": c}, (u + v) / (1 + u * v / (c * c))


def s_I_18_4(n=2000, seed=0):
    """(m1*r1 + m2*r2) / (m1 + m2)  (center of mass)"""
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(1, 5, n); m2 = rng.uniform(1, 5, n)
    r1 = rng.uniform(1, 5, n); r2 = rng.uniform(1, 5, n)
    return ({"m1": m1, "m2": m2, "r1": r1, "r2": r2},
            (m1 * r1 + m2 * r2) / (m1 + m2))


def s_I_24_6(n=2000, seed=0):
    """0.5 * m * (omega^2 + omega_0^2) * x^2"""
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 5, n); w = rng.uniform(1, 5, n)
    w0 = rng.uniform(1, 5, n); x = rng.uniform(1, 5, n)
    return ({"m": m, "omega": w, "omega0": w0, "x": x},
            0.5 * m * (w * w + w0 * w0) * x * x)


def s_I_25_13(n=2000, seed=0):
    """q / C  (capacitor voltage; trivial)"""
    rng = np.random.default_rng(seed)
    q = rng.uniform(1, 5, n); C = rng.uniform(1, 5, n)
    return {"q": q, "C": C}, q / C


def s_I_26_2(n=2000, seed=0):
    """arcsin(n*sin(theta2))  — uses arcsin AND sin, expected to fail"""
    rng = np.random.default_rng(seed)
    nn = rng.uniform(0.7, 1.3, n); th2 = rng.uniform(0.1, 1.0, n)
    return {"n": nn, "theta2": th2}, np.arcsin(nn * np.sin(th2))


def s_I_27_6(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d1 = rng.uniform(1, 5, n); d2 = rng.uniform(1, 5, n)
    return {"d1": d1, "d2": d2}, d1 * d2 / (d1 + d2)


def s_I_29_4(n=2000, seed=0):
    """omega / c  (trivial ratio)"""
    rng = np.random.default_rng(seed)
    w = rng.uniform(1, 5, n); c = rng.uniform(1, 5, n)
    return {"omega": w, "c": c}, w / c


def s_I_30_3(n=2000, seed=0):
    """Intensity_0 * sin(n*theta/2)^2 / sin(theta/2)^2 — sin; expected fail"""
    rng = np.random.default_rng(seed)
    I0 = rng.uniform(1, 5, n); nn = rng.uniform(2, 5, n)
    th = rng.uniform(0.1, 1.0, n)
    return ({"I0": I0, "n": nn, "theta": th},
            I0 * np.sin(nn * th / 2) ** 2 / np.sin(th / 2) ** 2)


def s_I_32_5(n=2000, seed=0):
    """q^2 * a^2 / (6 * pi * eps * c^3)  (Larmor radiated power)"""
    rng = np.random.default_rng(seed)
    q = rng.uniform(1, 5, n); a = rng.uniform(1, 5, n)
    eps = rng.uniform(1, 5, n); c = rng.uniform(1.5, 3, n)
    return ({"q": q, "a": a, "eps": eps, "c": c},
            q * q * a * a / (6 * np.pi * eps * c ** 3))


def s_I_34_8(n=2000, seed=0):
    """q*v*B / p  — simple ratio"""
    rng = np.random.default_rng(seed)
    q = rng.uniform(1, 5, n); v = rng.uniform(1, 5, n)
    B = rng.uniform(1, 5, n); p = rng.uniform(1, 5, n)
    return {"q": q, "v": v, "B": B, "p": p}, q * v * B / p


def s_I_39_22(n=2000, seed=0):
    """n*k*T / V (ideal gas pressure)"""
    rng = np.random.default_rng(seed)
    n = rng.uniform(1, 5, 2000); k = rng.uniform(1, 5, 2000)
    T = rng.uniform(1, 5, 2000); V = rng.uniform(1, 5, 2000)
    return {"n": n, "k": k, "T": T, "V": V}, n * k * T / V


def s_I_40_1(n=2000, seed=0):
    """n_0 * exp(-m*g*x/(k*T))  (barometric formula)"""
    rng = np.random.default_rng(seed)
    n0 = rng.uniform(1, 5, n); m = rng.uniform(1, 5, n)
    g = rng.uniform(1, 5, n); x = rng.uniform(0.1, 1.0, n)
    k = rng.uniform(1, 5, n); T = rng.uniform(1, 5, n)
    return ({"n0": n0, "m": m, "g": g, "x": x, "k": k, "T": T},
            n0 * np.exp(-m * g * x / (k * T)))


def s_I_43_31(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(1, 5, n); T = rng.uniform(1, 5, n)
    eta = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"k": k, "T": T, "eta": eta, "r": r}, k * T / (6 * np.pi * eta * r)


def s_I_43_43(n=2000, seed=0):
    """kappa * v^2 / (n*sigma)"""
    rng = np.random.default_rng(seed)
    kappa = rng.uniform(1, 5, n); v = rng.uniform(1, 5, n)
    nn = rng.uniform(1, 5, n); sig = rng.uniform(1, 5, n)
    return {"kappa": kappa, "v": v, "n": nn, "sigma": sig}, kappa * v * v / (nn * sig)


def s_I_47_23(n=2000, seed=0):
    """sqrt(gamma * pr / rho) — sound speed"""
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(1, 2, n); pr = rng.uniform(1, 5, n)
    rho = rng.uniform(1, 5, n)
    return {"gamma": gamma, "pr": pr, "rho": rho}, np.sqrt(gamma * pr / rho)


def s_I_48_2(n=2000, seed=0):
    """m * c^2 / sqrt(1 - v^2/c^2) — relativistic energy"""
    rng = np.random.default_rng(seed)
    m = rng.uniform(1, 5, n); c = rng.uniform(1.5, 3, n)
    v = rng.uniform(0.1, 0.5, n)
    return {"m": m, "c": c, "v": v}, m * c * c / np.sqrt(1 - (v / c) ** 2)


# ============ Registry ============
# (id, formula_str, sampler, expected: True if tessera should plausibly find it)

SUBSET = [
    ("I.6.20a",   "exp(-theta^2/2)",                      s_I_6_20a,   True),
    ("I.6.20",    "exp(-(theta/sigma)^2/2)",              s_I_6_20,    True),
    ("I.8.14",    "sqrt((x2-x1)^2+(y2-y1)^2)",            s_I_8_14,    True),
    ("I.9.18",    "G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)", s_I_9_18, False),
    ("I.10.7",    "m0/sqrt(1-v^2/c^2)",                    s_I_10_7,   True),
    ("I.11.19",   "x1*y1+x2*y2+x3*y3",                     s_I_11_19,  True),
    ("I.12.1",    "mu*Nn",                                 s_I_12_1,   True),
    ("I.12.2",    "q1*q2/(4*pi*eps*r^2)",                  s_I_12_2,   False),
    ("I.12.4",    "q1*q2/(4*pi*eps*r^2)",                  s_I_12_4,   False),
    ("I.12.5",    "q1*q2/r^2",                             s_I_12_5,   True),
    ("I.12.11",   "q*(Ef+B*v*sin(theta))",                 s_I_12_11,  False),
    ("I.14.3",    "m*g*z",                                 s_I_14_3,   True),
    ("I.14.4",    "0.5*k*x^2",                             s_I_14_4,   True),
    ("I.15.3t",   "(x-u*t)/sqrt(1-u^2/c^2)",               s_I_15_3t,  True),
    ("I.16.6",    "(u+v)/(1+u*v/c^2)",                     s_I_16_6,   True),
    ("I.18.4",    "(m1*r1+m2*r2)/(m1+m2)",                 s_I_18_4,   True),
    ("I.24.6",    "0.5*m*(omega^2+omega0^2)*x^2",          s_I_24_6,   True),
    ("I.25.13",   "q/C",                                   s_I_25_13,  True),
    ("I.26.2",    "arcsin(n*sin(theta2))",                 s_I_26_2,   False),
    ("I.27.6",    "d1*d2/(d1+d2)",                         s_I_27_6,   True),
    ("I.29.4",    "omega/c",                               s_I_29_4,   True),
    ("I.30.3",    "I0*sin(n*theta/2)^2/sin(theta/2)^2",    s_I_30_3,   False),
    ("I.32.5",    "q^2*a^2/(6*pi*eps*c^3)",                s_I_32_5,   True),
    ("I.34.8",    "q*v*B/p",                               s_I_34_8,   True),
    ("I.39.22",   "n*k*T/V",                               s_I_39_22,  True),
    ("I.40.1",    "n0*exp(-m*g*x/(k*T))",                  s_I_40_1,   True),
    ("I.43.31",   "k*T/(6*pi*eta*r)",                      s_I_43_31,  True),
    ("I.43.43",   "kappa*v^2/(n*sigma)",                   s_I_43_43,  True),
    ("I.47.23",   "sqrt(gamma*pr/rho)",                    s_I_47_23,  True),
    ("I.48.2",    "m*c^2/sqrt(1-v^2/c^2)",                 s_I_48_2,   True),
]


# ============ Runner ============

def run_one(name, sampler):
    env, y = sampler()
    feature_names = list(env.keys())
    var_y = float(np.var(y))

    cfg = GPConfig(
        pop_size=400,
        n_gens=120,
        init_max_depth=5,
        parsimony=max(var_y * 0.005, 1e-4),
        early_stop_patience=25,
        seed=2026,
        pointwise_only=True,
        verbose=False,
        optimize_constants_every=3,
        optimize_constants_method="Nelder-Mead",
        optimize_constants_maxiter=30,
    )
    gp = GP(cfg)
    t0 = time.time()
    front = gp.run(env, y, feature_names=feature_names)
    runtime = time.time() - t0

    best = min(front, key=lambda c: c.train_loss)
    rel = best.train_loss / var_y if var_y > 0 else float("nan")
    return dict(
        name=name, n_vars=len(feature_names), n_samples=len(y),
        runtime=runtime, front_size=len(front),
        best_cx=best.complexity, best_loss=best.train_loss, best_rel=rel,
        best_tree=str(best.tree), var_y=var_y,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Feynman extended (~30 eqs) on tessera ===")
    print(f"GP config: pop=400, gens=120, pointwise_only=True")
    t_start = time.time()

    results = []
    for name, formula, sampler, expected in SUBSET:
        print(f"\n[{name}]  {formula}  (expected={expected})")
        try:
            r = run_one(name, sampler)
            r["formula"] = formula
            r["expected"] = expected
            results.append(r)
            ascii_tree = r["best_tree"].encode("ascii", "replace").decode("ascii")
            print(f"  cx={r['best_cx']:2d}  loss={r['best_loss']:.4g}  "
                  f"rel={r['best_rel']:.4f}  ({r['runtime']:.1f}s)")
            print(f"    {ascii_tree[:120]}")
        except Exception as e:
            print(f"  CRASHED: {e}")
            results.append(dict(name=name, formula=formula, expected=expected,
                                runtime=0.0, best_cx=0, best_loss=float("nan"),
                                best_rel=float("nan"), best_tree=f"CRASH: {e}",
                                n_vars=0, n_samples=0, front_size=0))

    total = time.time() - t_start

    # ------ Report ------
    rel_threshold_exact = 0.01
    rel_threshold_partial = 0.20
    n_exact = sum(1 for r in results if r["best_rel"] < rel_threshold_exact)
    n_partial = sum(1 for r in results
                    if rel_threshold_exact <= r["best_rel"] < rel_threshold_partial)
    n_failed = sum(1 for r in results if r["best_rel"] >= rel_threshold_partial
                   or np.isnan(r["best_rel"]))

    L = ["# Feynman extended on tessera",
         "",
         f"**Equations:** {len(SUBSET)} (representative subset of the canonical 100)",
         f"**GP config:** pop=400, gens=120, pointwise_only=True, "
         f"optimize_constants every 3 gens, Nelder-Mead 30 iter.",
         f"**Samples per equation:** 2000",
         f"**Total wall-clock:** {total:.1f}s ({total/60:.1f} min)",
         f"**Alphabet:** add/sub/mul/div/min/max, gt/lt/ge/le, "
         f"tanh/abs/sign/neg/step, sqrt/exp/log/pow, "
         f"reduce_mean/max/sum/std.",
         "",
         "## Headline",
         "",
         f"- **Exact** (rel < 0.01): {n_exact} / {len(SUBSET)}",
         f"- **Partial** (0.01 ≤ rel < 0.20): {n_partial} / {len(SUBSET)}",
         f"- **Failed** (rel ≥ 0.20): {n_failed} / {len(SUBSET)}",
         "",
         "## Results table",
         "",
         "| # | Eq ID | True formula | n_vars | best cx | best rel | runtime (s) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        if r["best_rel"] < rel_threshold_exact:
            verdict = "**exact**"
        elif r["best_rel"] < rel_threshold_partial:
            verdict = "partial"
        else:
            verdict = "failed"
        L.append(f"| {i} | {r['name']} | `{r['formula']}` | {r['n_vars']} | "
                 f"{r['best_cx']} | {r['best_rel']:.4f} | {r['runtime']:.1f} | "
                 f"{verdict} |")

    L += ["", "## Discovered expressions", ""]
    for r in results:
        L.append(f"### {r['name']}  (`{r['formula']}`)")
        ascii_tree = r["best_tree"].encode("ascii", "replace").decode("ascii")
        L.append(f"- best cx={r['best_cx']}, rel={r['best_rel']:.4f}, "
                 f"runtime={r['runtime']:.1f}s")
        L.append(f"  ```")
        L.append(f"  {ascii_tree[:300]}{'...' if len(ascii_tree) > 300 else ''}")
        L.append(f"  ```")
        L.append("")

    L += ["## Known limitations",
          "",
          "- **No `sin` / `cos`**: equations I.12.11, I.26.2, I.30.3 use",
          "  trigonometric ops that tessera's vocabulary doesn't include.",
          "  These will fail by construction — documenting the gap.",
          "- **Multi-variable inverse products (4+ vars)**: even with",
          "  pop=400, gens=120, the search may not converge cleanly on",
          "  forms like `q1*q2/(4*pi*eps*r^2)` where the constant must",
          "  be fitted alongside structural symbol choice.",
          "",
          "## See also",
          "",
          "- `run_feynman_subset.py` — 8-equation quick run (~10 s)",
          "- Udrescu & Tegmark, *AI Feynman*, Sci. Adv. 2020",
          "- SRBench: https://github.com/cavalab/srbench"]

    out_path = OUT_DIR / "feynman_extended.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] wrote {out_path}")
    print(f"\nHeadline: {n_exact} exact / {n_partial} partial / "
          f"{n_failed} failed out of {len(SUBSET)} equations")


if __name__ == "__main__":
    main()
