"""Head-to-head comparison of simplification chains.

Question: does CAS (sympy) actually catch redundancies our existing
hand-rolled chain misses on REAL GP-discovered trees, or is it
redundant for typical GP output?

Chains compared
---------------
  (a) AC only — `simplify_ac`
  (b) Canonical = AC + rule-based — `simplify_canonical` (current GP default)
  (c) Full = Canonical + polynomial — `simplify_full`
  (d) Full + CAS = simplify_full → cas_simplify

For each chain, we measure on each GP-discovered candidate:
  - cx after that chain
  - latency
  - what the chain produced (for inspection)

This tells us:
  1. Does our hand-rolled chain progressively reduce cx as we add layers?
  2. Does CAS catch what our chain misses?
  3. What's the practical overhead of each layer?

Test data sources
-----------------
We use REAL GP outputs from a few targets:
  - Heat equation (1 baseline run, take Pareto front)
  - Feynman I.6.20a (gaussian; has exp/trig)
  - Feynman I.27.6 (d1·d2/(d1+d2); has division)
  - Hand-crafted redundancy cases (sanity check)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, complexity, evaluate,
)
from tessera.expression.simplify import (
    simplify_ac, simplify_canonical, simplify_full,
    simplify_polynomial, cas_simplify, cas_backend,
)
from tessera.expression.simplify.cas_fallback import clear_cache

from tessera.search import GP, GPConfig


# ----------------------------------------------------------------------
# Chain definitions
# ----------------------------------------------------------------------

CHAINS = [
    ("ac_only",    lambda t: simplify_ac(t)),
    ("canonical",  lambda t: simplify_canonical(t)),
    ("full",       lambda t: simplify_full(t)),
    ("full+cas",   lambda t: cas_simplify(simplify_full(t))),
]


# ----------------------------------------------------------------------
# Apply chain with timing
# ----------------------------------------------------------------------

def apply_chain(name, fn, tree):
    """Apply a simplification chain; return (simplified, cx, time_ms)."""
    try:
        t0 = time.perf_counter()
        result = fn(tree)
        dt = (time.perf_counter() - t0) * 1000
        return result, complexity(result), dt
    except Exception as e:
        return tree, complexity(tree), 0.0


# ----------------------------------------------------------------------
# GP targets we'll run
# ----------------------------------------------------------------------

def gaussian_target(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.5, 5.0, n)
    y = np.exp(-(theta ** 2) / 2.0)
    return {"theta": theta}, y


def reduced_target(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d1 = rng.uniform(1, 5, n)
    d2 = rng.uniform(1, 5, n)
    return {"d1": d1, "d2": d2}, d1 * d2 / (d1 + d2)


def stokes_target(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.uniform(1, 5, n); T = rng.uniform(1, 5, n)
    eta = rng.uniform(1, 5, n); r = rng.uniform(1, 5, n)
    return {"k": k, "T": T, "eta": eta, "r": r}, k * T / (6 * np.pi * eta * r)


TARGETS = [
    ("gaussian_I.6.20a", "exp(-θ²/2)", gaussian_target),
    ("reduced_I.27.6", "d1·d2/(d1+d2)", reduced_target),
    ("stokes_I.43.31", "k·T/(6π·η·r)", stokes_target),
]


# ----------------------------------------------------------------------
# Hand-crafted redundancy cases (control)
# ----------------------------------------------------------------------

def make_handcrafted_cases():
    """Trees with KNOWN redundancies that each chain SHOULD catch."""
    cases = []

    # 1. Pure polynomial collapse: 2x + 3x → 5x
    cases.append(("polynomial 2x+3x", BinOp("add",
                  BinOp("mul", Const(2.0), Var("x")),
                  BinOp("mul", Const(3.0), Var("x")))))

    # 2. Constant folding: 2*3
    cases.append(("constant_fold 2*3", BinOp("mul", Const(2.0), Const(3.0))))

    # 3. AC normalization: (a+b)+c with weird ordering
    cases.append(("ac (c+a)+b", BinOp("add",
                  BinOp("add", Var("c"), Var("a")), Var("b"))))

    # 4. Identity removal: x*1
    cases.append(("x*1", BinOp("mul", Var("x"), Const(1.0))))

    # 5. Polynomial with like terms across layers: (x*x) + 2*(x*x)
    cases.append(("polynomial_x2 + 2x2",
                  BinOp("add",
                        BinOp("mul", Var("x"), Var("x")),
                        BinOp("mul", Const(2.0),
                              BinOp("mul", Var("x"), Var("x"))))))

    # 6. Trig identity: sin²+cos²
    sin_x = UnOp("sin", Var("x"))
    cos_x = UnOp("cos", Var("x"))
    cases.append(("trig sin²+cos²",
                  BinOp("add",
                        BinOp("mul", sin_x, sin_x),
                        BinOp("mul", cos_x, cos_x))))

    # 7. Rational cancellation: (2*x)/x
    cases.append(("rational (2x)/x",
                  BinOp("div", BinOp("mul", Const(2.0), Var("x")), Var("x"))))

    # 8. log(exp(x)) — protected-ops edge case
    cases.append(("log(exp(x))",
                  UnOp("log", UnOp("exp", Var("x")))))

    return cases


# ----------------------------------------------------------------------
# Run GP on a target and return the Pareto front
# ----------------------------------------------------------------------

def get_pareto_front(target_name, sampler, pop=80, gens=30, seed=42):
    env, y = sampler()
    cfg = GPConfig(
        pop_size=pop, n_gens=gens, seed=seed, verbose=False,
        pointwise_only=True, early_stop_patience=gens,
        optimize_constants_every=0,
        parsimony=max(float(np.var(y)) * 0.005, 1e-4),
    )
    gp = GP(cfg)
    front = gp.run(env, y, feature_names=list(env.keys()))
    return [c.tree for c in front]


# ----------------------------------------------------------------------
# Main: run chains on candidates, report
# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--pop", type=int, default=80)
    args = p.parse_args(argv)

    print(f"CAS backend: {cas_backend()}")
    print()

    all_results = []
    t_start = time.time()

    # --- Handcrafted cases ---
    print("=== Handcrafted redundancy cases ===")
    print(f"{'name':<28s} {'orig':>4s}", end="")
    for chain_name, _ in CHAINS:
        print(f" {chain_name:>10s}", end="")
    print()

    for case_name, tree in make_handcrafted_cases():
        orig_cx = complexity(tree)
        row = {"target": "handcrafted", "case": case_name,
               "orig_tree": str(tree), "orig_cx": orig_cx,
               "chain_results": {}}
        print(f"{case_name:<28s} {orig_cx:4d}", end="")
        for chain_name, fn in CHAINS:
            clear_cache()
            result, cx, dt_ms = apply_chain(chain_name, fn, tree)
            row["chain_results"][chain_name] = {
                "cx": cx, "time_ms": dt_ms,
                "result_str": str(result),
            }
            print(f"  {cx:3d}({dt_ms:5.1f}ms)", end="")
        print()
        all_results.append(row)
    print()

    # --- Real GP outputs per target ---
    print("=== Real GP outputs per target ===")
    for target_name, formula, sampler in TARGETS:
        print(f"\n--- {target_name}: {formula} ---")
        front = get_pareto_front(target_name, sampler, pop=args.pop, gens=args.gens)
        print(f"  Pareto front size: {len(front)}")
        print(f"  {'cx':>4s}", end="")
        for chain_name, _ in CHAINS:
            print(f" {chain_name:>10s}", end="")
        print()
        for i, tree in enumerate(front):
            orig_cx = complexity(tree)
            row = {"target": target_name, "case": f"front[{i}]",
                   "orig_tree": str(tree), "orig_cx": orig_cx,
                   "chain_results": {}}
            print(f"  {orig_cx:4d}", end="")
            for chain_name, fn in CHAINS:
                clear_cache()
                result, cx, dt_ms = apply_chain(chain_name, fn, tree)
                row["chain_results"][chain_name] = {
                    "cx": cx, "time_ms": dt_ms,
                    "result_str": str(result),
                }
                print(f"  {cx:3d}({dt_ms:5.1f}ms)", end="")
            print()
            all_results.append(row)

    total_rt = time.time() - t_start
    print(f"\nTotal wall-clock: {total_rt:.1f}s")

    # Aggregate analysis
    print("\n=== Aggregate analysis ===")
    chain_names = [name for name, _ in CHAINS]
    total_orig_cx = 0
    chain_total_cx = {n: 0 for n in chain_names}
    chain_total_time = {n: 0.0 for n in chain_names}
    chain_strict_reductions = {n: 0 for n in chain_names}  # cx strictly less than prev chain
    for row in all_results:
        total_orig_cx += row["orig_cx"]
        prev_cx = row["orig_cx"]
        for cn in chain_names:
            cur_cx = row["chain_results"][cn]["cx"]
            chain_total_cx[cn] += cur_cx
            chain_total_time[cn] += row["chain_results"][cn]["time_ms"]
            if cur_cx < prev_cx:
                chain_strict_reductions[cn] += 1
            prev_cx = cur_cx

    print(f"  Total trees: {len(all_results)}")
    print(f"  Total original cx: {total_orig_cx}")
    for cn in chain_names:
        print(f"  {cn:>12s}: total cx={chain_total_cx[cn]}  "
              f"saved={total_orig_cx - chain_total_cx[cn]}  "
              f"time={chain_total_time[cn]:.1f}ms  "
              f"reductions_beyond_prev_chain={chain_strict_reductions[cn]}")

    out_path = Path(__file__).resolve().parent / "results" / "simplification_chain_comparison.md"
    write_report(all_results, total_rt, args, out_path)
    return 0


def write_report(all_results, total_rt, args, out_path):
    L = ["# Simplification chain head-to-head comparison", ""]
    L.append("**Question:** does CAS (sympy) catch redundancies our existing")
    L.append("hand-rolled chain misses on REAL GP-discovered trees?")
    L.append("")
    L.append("Chains compared:")
    L.append("  (a) `ac_only`: just AC normalization")
    L.append("  (b) `canonical`: AC + rule-based (current GP default)")
    L.append("  (c) `full`: canonical + polynomial like-term collection")
    L.append("  (d) `full+cas`: full → sympy CAS fallback")
    L.append("")
    L.append(f"**CAS backend:** {cas_backend()}")
    L.append(f"**Total wall-clock:** {total_rt:.1f}s")
    L.append("")

    # Aggregate stats
    chain_names = ["ac_only", "canonical", "full", "full+cas"]
    total_orig_cx = 0
    chain_total_cx = {n: 0 for n in chain_names}
    chain_total_time = {n: 0.0 for n in chain_names}
    chain_reductions = {n: 0 for n in chain_names}
    for row in all_results:
        total_orig_cx += row["orig_cx"]
        prev_cx = row["orig_cx"]
        for cn in chain_names:
            cur_cx = row["chain_results"][cn]["cx"]
            chain_total_cx[cn] += cur_cx
            chain_total_time[cn] += row["chain_results"][cn]["time_ms"]
            if cur_cx < prev_cx:
                chain_reductions[cn] += 1
            prev_cx = cur_cx

    L.append("## Aggregate findings")
    L.append("")
    L.append(f"Across {len(all_results)} trees (handcrafted + real GP outputs):")
    L.append("")
    L.append("| Chain | Total cx | Saved vs orig | Time (ms) | Reductions beyond prev |")
    L.append("|---|---|---|---|---|")
    for cn in chain_names:
        L.append(f"| {cn} | {chain_total_cx[cn]} | "
                 f"{total_orig_cx - chain_total_cx[cn]} | "
                 f"{chain_total_time[cn]:.1f} | "
                 f"{chain_reductions[cn]} |")
    L.append("")

    # Headline
    cas_extra = chain_reductions["full+cas"]
    poly_extra = chain_reductions["full"]
    L.append("## Headline")
    L.append("")
    L.append(f"- **`full+cas` reduces cx beyond `full` in {cas_extra}/{len(all_results)} trees**")
    L.append(f"- **`full` reduces cx beyond `canonical` in {poly_extra}/{len(all_results)} trees**")
    cas_marginal_savings = chain_total_cx["full"] - chain_total_cx["full+cas"]
    L.append(f"- **Total cx saved by adding CAS to the chain: {cas_marginal_savings}** "
             f"(across {len(all_results)} trees)")
    L.append("")
    L.append(f"CAS overhead: {chain_total_time['full+cas'] - chain_total_time['full']:.1f} ms total "
             f"({chain_total_time['full+cas'] / max(chain_total_time['full'], 0.01):.1f}× cost of `full`)")
    L.append("")

    # Per-row table
    L.append("## Per-tree comparison")
    L.append("")
    L.append("| Target | Case | orig cx | ac | canonical | full | full+cas |")
    L.append("|---|---|---|---|---|---|---|")
    for row in all_results:
        crs = row["chain_results"]
        markers = []
        for cn in chain_names:
            cx_now = crs[cn]["cx"]
            cx_orig = row["orig_cx"]
            if cx_now < cx_orig:
                markers.append(f"**{cx_now}**")
            else:
                markers.append(f"{cx_now}")
        L.append(f"| {row['target']} | {row['case']} | {row['orig_cx']} | "
                 f"{markers[0]} | {markers[1]} | {markers[2]} | {markers[3]} |")
    L.append("")

    # Cases where CAS made a difference beyond `full`
    L.append("## Cases where CAS reduced cx BEYOND `full`")
    L.append("")
    found_any_cas_wins = False
    for row in all_results:
        full_cx = row["chain_results"]["full"]["cx"]
        cas_cx = row["chain_results"]["full+cas"]["cx"]
        if cas_cx < full_cx:
            found_any_cas_wins = True
            L.append(f"### {row['target']} / {row['case']}")
            L.append("")
            L.append(f"- Original (cx={row['orig_cx']}): `{row['orig_tree'][:80]}`")
            L.append(f"- After `full` (cx={full_cx}): "
                     f"`{row['chain_results']['full']['result_str'][:80]}`")
            L.append(f"- After `full+cas` (cx={cas_cx}): "
                     f"`{row['chain_results']['full+cas']['result_str'][:80]}`")
            L.append("")
    if not found_any_cas_wins:
        L.append("(None on this run. CAS didn't add value beyond `full` for any tree.)")
        L.append("")

    # Verdict
    L.append("## Verdict")
    L.append("")
    if cas_marginal_savings > 0:
        L.append(f"**CAS adds value.** Saved {cas_marginal_savings} cx beyond")
        L.append(f"the existing chain on {cas_extra}/{len(all_results)} trees.")
        L.append("Use case: catches redundancies in trees containing trig/log/")
        L.append("div/etc. that our hand-rolled passes miss.")
    elif cas_extra == 0 and poly_extra == 0:
        L.append("**CAS adds no value over `canonical` on this run.** The hand-")
        L.append("rolled chain already handles all redundancies the GP produces")
        L.append("here. CAS would matter only on trees with unhandled patterns.")
    else:
        L.append(f"**CAS adds marginal value.** {cas_extra} trees benefit; small")
        L.append(f"absolute cx savings ({cas_marginal_savings}). Useful when")
        L.append("redundancy-rich trees appear; negligible otherwise.")
    L.append("")

    L.append("## Implication for tessera's simplification pipeline")
    L.append("")
    L.append("Current GP scoring path uses `simplify_canonical`. The data here")
    L.append("tells us:")
    L.append("")
    if chain_total_cx["full"] < chain_total_cx["canonical"]:
        L.append(f"- **`simplify_full` saves cx vs `simplify_canonical`** "
                 f"({chain_total_cx['canonical'] - chain_total_cx['full']} nodes saved).")
        L.append("  Worth changing the GP default to `simplify_full` (the polynomial pass).")
    else:
        L.append(f"- `simplify_full` doesn't materially improve over `simplify_canonical` "
                 f"on this dataset.")
    L.append("")
    if cas_marginal_savings > 0:
        L.append(f"- CAS catches {cas_extra} trees `full` missed.")
        L.append("  Worth keeping CAS as an opt-in for redundancy-rich workflows")
        L.append("  (PDE discovery, signal processing). Not necessarily a default.")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append(f"python benchmarks/run_simplification_chain_comparison.py "
             f"--pop {args.pop} --gens {args.gens}")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
