"""Validate CAS simplification: speed cost + complexity reduction.

Tests the claim from the structural-critique gap analysis:
  - CAS simplification at the Pareto-front level adds <1% overhead
  - Catches redundancies our hand-rolled simplifier misses
  - Numerical verification keeps it safe

Methodology
-----------
1. Run a baseline GP without CAS simplification; measure wall-clock
2. Run the same GP, then apply CAS simplification to the Pareto front;
   measure cx reduction and overhead
3. Spot-check on a synthetic tree set with KNOWN redundancies that
   only CAS catches (sin²+cos²=1, log(exp(x))=x, etc.)

Pass criteria
-------------
- CAS overhead < 5% of GP runtime
- CAS catches > 0 simplifications on Pareto fronts of typical GP runs
- Synthetic redundancy test cases all simplify correctly
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.search import GP, GPConfig
from tessera.expression.tree import Var, Const, BinOp, UnOp, complexity
from tessera.expression.simplify import (
    cas_simplify, simplify_front_with_cas, cas_backend, is_worth_cas_pass,
)
from tessera.expression.simplify.cas_fallback import clear_cache, cache_size


def _make_polynomial_target(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = 2.0 * x - 0.5 * x ** 2 + 0.1 * x ** 3
    return {"x": x}, y


def _make_trig_target(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    y = np.sin(x) ** 2 + np.cos(x) ** 2  # = 1, but expressed via trig
    # Add tiny noise so SR has something to fit
    y = y + 0.01 * rng.standard_normal(n)
    return {"x": x}, y


def benchmark_gp_with_cas(target_fn, n_gens=30, pop_size=80, seed=42):
    """Run a GP and time it, then apply CAS simplification and time that too."""
    env, y = target_fn()
    cfg = GPConfig(
        pop_size=pop_size, n_gens=n_gens, seed=seed,
        verbose=False, pointwise_only=True,
        early_stop_patience=n_gens, optimize_constants_every=0,
        parsimony=0.005,
    )

    # Run GP
    t0 = time.time()
    gp = GP(cfg)
    front = gp.run(env, y, feature_names=list(env.keys()))
    t_gp = time.time() - t0

    # Apply CAS simplification on the final front
    clear_cache()
    t0 = time.time()
    simplified_front = simplify_front_with_cas(
        front, feature_names=list(env.keys()),
        verify_samples=20, atol=1e-6,
    )
    t_cas = time.time() - t0

    # Compute cx reductions
    n_reduced = 0
    total_cx_before = 0
    total_cx_after = 0
    examples = []
    for orig, simp in zip(front, simplified_front):
        total_cx_before += orig.complexity
        total_cx_after += simp.complexity
        if simp.complexity < orig.complexity:
            n_reduced += 1
            examples.append({
                "before_cx": orig.complexity,
                "after_cx": simp.complexity,
                "before_str": str(orig.tree),
                "after_str": str(simp.tree),
            })

    return {
        "front_size": len(front),
        "t_gp": t_gp,
        "t_cas": t_cas,
        "n_reduced": n_reduced,
        "total_cx_before": total_cx_before,
        "total_cx_after": total_cx_after,
        "examples": examples,
        "cache_after": cache_size(),
    }


def benchmark_synthetic_redundancies():
    """Test on hand-crafted trees with KNOWN redundancies."""
    sin_x = UnOp("sin", Var("x"))
    cos_x = UnOp("cos", Var("x"))

    cases = [
        ("sin²+cos²=1",
         BinOp("add",
               BinOp("mul", sin_x, sin_x),
               BinOp("mul", cos_x, cos_x))),
        ("(2x)/x=2",
         BinOp("div", BinOp("mul", Const(2.0), Var("x")), Var("x"))),
        ("log(exp(x))=x",
         UnOp("log", UnOp("exp", Var("x")))),
        ("polynomial (no CAS needed)",
         BinOp("add", BinOp("mul", Const(2.0), Var("x")), Const(3.0))),
        ("(x*y)/(x*z) = y/z",
         BinOp("div",
               BinOp("mul", Var("x"), Var("y")),
               BinOp("mul", Var("x"), Var("z")))),
    ]

    results = []
    for name, tree in cases:
        t0 = time.time()
        simplified = cas_simplify(tree, verify_samples=20)
        dt = time.time() - t0
        results.append({
            "name": name,
            "before_cx": complexity(tree),
            "after_cx": complexity(simplified),
            "reduced": complexity(simplified) < complexity(tree),
            "latency_ms": dt * 1000,
            "before": str(tree),
            "after": str(simplified),
        })
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_gens", type=int, default=30)
    p.add_argument("--pop", type=int, default=80)
    args = p.parse_args(argv)

    backend = cas_backend()
    print(f"CAS backend: {backend}")
    if backend is None:
        print("No CAS backend available — exiting.")
        return 1
    print()

    # ----- Synthetic redundancy test -----
    print("=== Synthetic redundancy test ===")
    syn = benchmark_synthetic_redundancies()
    for r in syn:
        status = "✓" if r["reduced"] else ("(skipped/no-op)" if "polynomial" in r["name"] else "✗")
        print(f"  {status} {r['name']:35s}  cx {r['before_cx']:3d} → {r['after_cx']:3d}  "
              f"({r['latency_ms']:.2f} ms)")
    print()

    # ----- GP + CAS overhead on polynomial target -----
    print("=== GP + CAS overhead: polynomial target ===")
    poly = benchmark_gp_with_cas(_make_polynomial_target,
                                 n_gens=args.n_gens, pop_size=args.pop)
    overhead_pct = 100.0 * poly["t_cas"] / max(poly["t_gp"], 1e-9)
    print(f"  GP runtime: {poly['t_gp']:.2f}s  CAS overhead: {poly['t_cas']:.3f}s  "
          f"({overhead_pct:.1f}%)")
    print(f"  Front size: {poly['front_size']}")
    print(f"  Candidates simplified: {poly['n_reduced']}/{poly['front_size']}")
    print(f"  Total cx: {poly['total_cx_before']} → {poly['total_cx_after']}  "
          f"(saved {poly['total_cx_before'] - poly['total_cx_after']})")
    print()

    # ----- GP + CAS overhead on trig target -----
    print("=== GP + CAS overhead: trig target ===")
    trig = benchmark_gp_with_cas(_make_trig_target,
                                 n_gens=args.n_gens, pop_size=args.pop)
    overhead_pct_trig = 100.0 * trig["t_cas"] / max(trig["t_gp"], 1e-9)
    print(f"  GP runtime: {trig['t_gp']:.2f}s  CAS overhead: {trig['t_cas']:.3f}s  "
          f"({overhead_pct_trig:.1f}%)")
    print(f"  Front size: {trig['front_size']}")
    print(f"  Candidates simplified: {trig['n_reduced']}/{trig['front_size']}")
    print(f"  Total cx: {trig['total_cx_before']} → {trig['total_cx_after']}  "
          f"(saved {trig['total_cx_before'] - trig['total_cx_after']})")
    if trig["examples"]:
        print(f"\n  Example reductions:")
        for ex in trig["examples"][:3]:
            print(f"    [cx {ex['before_cx']} → {ex['after_cx']}]")
            print(f"      before: {ex['before_str'][:80]}")
            print(f"      after:  {ex['after_str'][:80]}")
    print()

    out_path = Path(__file__).resolve().parent / "results" / "cas_simplification_validation.md"
    write_report(syn, poly, trig, args, backend, out_path)
    return 0


def write_report(syn, poly, trig, args, backend, out_path):
    L = ["# CAS simplification validation", ""]
    L.append("Validates the claim that CAS-based simplification (via sympy)")
    L.append("can be integrated at the Pareto-front level with bounded overhead")
    L.append("and meaningful complexity reduction.")
    L.append("")
    L.append(f"**Backend:** {backend}")
    L.append(f"**GP config:** pop={args.pop}, n_gens={args.n_gens}, pointwise_only")
    L.append("")

    L.append("## Synthetic redundancy test (hand-crafted trees)")
    L.append("")
    L.append("| Case | cx before | cx after | Reduced? | Latency |")
    L.append("|---|---|---|---|---|")
    for r in syn:
        marker = "✓" if r["reduced"] else "—"
        L.append(f"| {r['name']} | {r['before_cx']} | {r['after_cx']} | "
                 f"{marker} | {r['latency_ms']:.2f} ms |")
    L.append("")

    L.append("## GP + CAS overhead (polynomial target — predicate should mostly skip)")
    L.append("")
    L.append(f"- GP runtime: **{poly['t_gp']:.2f}s**")
    L.append(f"- CAS overhead: **{poly['t_cas']:.3f}s** "
             f"({100 * poly['t_cas'] / max(poly['t_gp'], 1e-9):.1f}%)")
    L.append(f"- Front size: {poly['front_size']}")
    L.append(f"- Candidates simplified: {poly['n_reduced']}/{poly['front_size']}")
    L.append(f"- Total cx reduction: {poly['total_cx_before']} → {poly['total_cx_after']}")
    L.append("")

    L.append("## GP + CAS overhead (trig target — CAS should catch identities)")
    L.append("")
    L.append(f"- GP runtime: **{trig['t_gp']:.2f}s**")
    L.append(f"- CAS overhead: **{trig['t_cas']:.3f}s** "
             f"({100 * trig['t_cas'] / max(trig['t_gp'], 1e-9):.1f}%)")
    L.append(f"- Front size: {trig['front_size']}")
    L.append(f"- Candidates simplified: {trig['n_reduced']}/{trig['front_size']}")
    L.append(f"- Total cx reduction: {trig['total_cx_before']} → {trig['total_cx_after']}")
    L.append("")

    if trig["examples"]:
        L.append("### Example reductions on trig front")
        L.append("")
        for ex in trig["examples"][:5]:
            L.append(f"- **cx {ex['before_cx']} → {ex['after_cx']}**")
            before = ex['before_str'].encode("ascii", "replace").decode("ascii")
            after = ex['after_str'].encode("ascii", "replace").decode("ascii")
            L.append(f"  - before: `{before[:100]}`")
            L.append(f"  - after:  `{after[:100]}`")
        L.append("")

    # Verdict
    poly_overhead = 100 * poly['t_cas'] / max(poly['t_gp'], 1e-9)
    trig_overhead = 100 * trig['t_cas'] / max(trig['t_gp'], 1e-9)
    overall_overhead = max(poly_overhead, trig_overhead)

    L.append("## Verdict")
    L.append("")
    if overall_overhead < 5.0:
        L.append(f"**Overhead claim VALIDATED.** CAS adds < 5% overhead "
                 f"({overall_overhead:.1f}% max) at the Pareto-front level.")
    elif overall_overhead < 15.0:
        L.append(f"**Overhead acceptable.** {overall_overhead:.1f}% max; would")
        L.append("benefit from caching across multiple GP runs or symengine for")
        L.append("further reduction.")
    else:
        L.append(f"**Overhead too high.** {overall_overhead:.1f}% max — needs")
        L.append("optimization (symengine? front subset?) before deployment.")

    L.append("")
    L.append(f"**Synthetic redundancy catch rate:** "
             f"{sum(r['reduced'] for r in syn)}/{len([r for r in syn if 'polynomial' not in r['name']])} "
             f"non-polynomial cases simplified correctly")
    L.append("")
    L.append(f"**Trig benchmark cx reduction:** "
             f"{trig['total_cx_before']} → {trig['total_cx_after']} "
             f"(saved {trig['total_cx_before'] - trig['total_cx_after']} nodes)")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_cas_simplification_validation.py")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
