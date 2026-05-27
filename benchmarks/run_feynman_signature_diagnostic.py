"""Diagnostic: run workbench signatures on each of the 30 Feynman equations.

Per the diagnostic-first step (`docs/planned/methodology_workbench_and_library.md`
Stage 5 pipeline, Step 1): before any engine upgrade, run the signature
pipeline on the actual target data and see what it tells us.

For Feynman:
  - All targets are ALGEBRAIC (y = f(x)) with iid input sampling
  - Expectation: Tier A classifies all as algebraic
  - The interesting signal is Tier B — smoothness, effective dim,
    spectral content (skipped for algebraic), modes, symmetry

What we learn:
  - Are any equations misclassified by Tier A? (Sign that the data
    has hidden structure beyond iid algebraic)
  - Do signature values cluster by formula structure
    (polynomial vs trig vs exp)?
  - Which equations have "weird" signatures suggesting they'd benefit
    from specific upgrades (BFGS const-opt, wider vocabulary, etc.)?

Output: benchmarks/results/feynman_signature_diagnostic.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.workbench import Trajectory, ModelClass
from tessera.workbench.signatures import compute_full_signature

# Reuse the SUBSET from the existing Feynman runner — single source of truth
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_feynman_extended import SUBSET  # noqa: E402


def wrap_algebraic(env: dict, y: np.ndarray, name: str,
                   feature_names: list[str]) -> Trajectory:
    """Wrap (env, y) into a Trajectory in the algebraic convention.

    state = y.reshape(-1, 1); meta['inputs'] = stacked columns from env.
    """
    n = len(y)
    inputs = np.column_stack([env[f] for f in feature_names]).astype(np.float64)
    state = y.reshape(-1, 1).astype(np.float64)
    t = np.arange(n, dtype=np.float64)
    return Trajectory(
        t=t, state=state, observable=state, system_id=f"feynman_{name}",
        params={}, ic=np.array([], dtype=np.float64), noise_std=0.0, seed=0,
        meta={"inputs": inputs, "input_names": list(feature_names)},
    )


def diagnose_one(name: str, formula: str, sampler) -> dict:
    """Generate data + run compute_full_signature."""
    env, y = sampler()
    feature_names = list(env.keys())
    traj = wrap_algebraic(env, y, name, feature_names)
    t0 = time.time()
    sig = compute_full_signature(traj)
    elapsed = time.time() - t0
    return {
        "name": name,
        "formula": formula,
        "n_vars": len(feature_names),
        "n_samples": len(y),
        "inferred_class": sig.inferred_model_class,
        "permutation_inv": _v(sig.permutation_invariance),
        "t_acf_max": _v(sig.autocorrelation_structure, key="time_acf_max"),
        "smoothness": _v(sig.smoothness),
        "mode_count": _v(sig.mode_count),
        "effective_dim": _v(sig.effective_dimensionality),
        "symmetry": _v(sig.symmetry),
        "elapsed_sec": elapsed,
    }


def _v(sigval, key: str | None = None):
    """Extract a value from a SignatureValue, with dict-flattening."""
    if sigval is None:
        return None
    v = sigval.value
    if isinstance(v, dict):
        if key is not None:
            return v.get(key)
        # Default flatten: first scalar field
        return next((vv for vv in v.values() if isinstance(vv, (int, float))), None)
    return v


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="Path to write report (default: benchmarks/results/feynman_signature_diagnostic.md)")
    args = p.parse_args(argv)

    out_path = Path(args.out) if args.out else (
        HERE / "results" / "feynman_signature_diagnostic.md"
    )

    print("=== Feynman signature diagnostic ===")
    print(f"Running compute_full_signature on {len(SUBSET)} Feynman equations\n")

    t_start = time.time()
    results = []
    for name, formula, sampler, expected in SUBSET:
        try:
            row = diagnose_one(name, formula, sampler)
            inferred = row["inferred_class"]
            mark = "+" if inferred == "algebraic" else "-"
            print(f"{mark} {name:<10s} {formula[:40]:<40s} → {inferred:<10s} "
                  f"smooth={row['smoothness']!s:>6}  dim={row['effective_dim']!s:>6}")
            results.append(row)
        except Exception as e:
            print(f"! {name}: FAILED — {e}")
            results.append({
                "name": name, "formula": formula, "error": str(e)[:100],
            })

    elapsed_total = time.time() - t_start
    print(f"\nTotal wall-clock: {elapsed_total:.1f}s")

    write_report(results, elapsed_total, out_path)
    return 0


def _fmt(v, w=6):
    if v is None:
        return f"{'—':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.3f}"
    return f"{v!s:>{w}}"


def write_report(results, runtime, out_path):
    L = []
    L.append("# Feynman signature diagnostic")
    L.append("")
    L.append("Workbench signatures applied to each of the 30 Feynman equations.")
    L.append("Goal: identify whether any equation shows hidden structure (non-")
    L.append("algebraic ACF, mode multiplicity, surprising symmetry) that")
    L.append("would route to a different identification path in Stage 5.")
    L.append("")
    L.append(f"**Total wall-clock:** {runtime:.1f}s")
    L.append("")

    n_total = len([r for r in results if "error" not in r])
    n_algebraic = sum(1 for r in results
                      if "error" not in r and r.get("inferred_class") == "algebraic")
    n_other = n_total - n_algebraic

    L.append("## Headline")
    L.append("")
    L.append(f"- **{n_algebraic} of {n_total}** equations classified as ALGEBRAIC by Tier A (expected)")
    if n_other > 0:
        L.append(f"- **{n_other} misclassified** — these warrant investigation:")
        for r in results:
            if "error" not in r and r.get("inferred_class") != "algebraic":
                L.append(f"  - `{r['name']}` ({r['formula']}) → `{r['inferred_class']}`")
    L.append("")

    L.append("## Per-equation signatures")
    L.append("")
    L.append("| Eq | Formula | n_vars | Inferred | perm_inv | t_acf | smooth | modes | eff_dim |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            L.append(f"| {r['name']} | {r['formula'][:35]} | — | ERROR | — | — | — | — | — |")
            continue
        L.append(
            f"| {r['name']} | {r['formula'][:35]} | {r['n_vars']} | "
            f"{r['inferred_class']} | {_fmt(r['permutation_inv'])} | "
            f"{_fmt(r['t_acf_max'])} | {_fmt(r['smoothness'])} | "
            f"{_fmt(r['mode_count'], 4)} | {_fmt(r['effective_dim'])} |"
        )
    L.append("")

    L.append("## Reading")
    L.append("")
    L.append("Feynman targets are PURE algebraic functions of iid inputs. The")
    L.append("Tier A signature should classify all as ALGEBRAIC; any that don't")
    L.append("indicate either (a) the input sampling has hidden correlation")
    L.append("we haven't accounted for, or (b) the formula has structural")
    L.append("regularity that the autocorrelation test picks up as spurious")
    L.append("temporal structure.")
    L.append("")
    L.append("Smoothness ≈ 0 is expected (iid inputs → no temporal smoothness).")
    L.append("Effective dim should track n_vars roughly — high-n_var equations")
    L.append("with low effective dim may have a redundancy SR can exploit.")
    L.append("Mode count > 1 on iid data is a GMM-clustering artifact, not")
    L.append("a topological mode count — already documented limitation.")
    L.append("")

    L.append("## What this diagnostic does NOT measure")
    L.append("")
    L.append("- **Which equations the GP will actually find.** That's the")
    L.append("  Feynman benchmark itself; this is just a data characterization.")
    L.append("- **What const-opt would unlock.** The diagnostic suggests")
    L.append("  whether an equation has structure SR can identify; the")
    L.append("  constant-optimization step is independent.")
    L.append("- **Whether vocabulary is sufficient.** sin/cos/atan2 are in")
    L.append("  the vocabulary; whether they'd be discovered is a GP-search")
    L.append("  question, not a signature question.")
    L.append("")

    L.append("## Implication for Step 2 (const-opt upgrade)")
    L.append("")
    L.append("Equations classified correctly as algebraic with low complexity")
    L.append("signatures (low effective dim, few modes) are the ones most")
    L.append("likely to benefit from BFGS const-opt — they have clean structure")
    L.append("and the GP's bottleneck is parameter precision, not form discovery.")
    L.append("")
    L.append("Equations with surprising signatures (misclassified, high")
    L.append("effective dim, weird mode counts) may need other interventions")
    L.append("(vocabulary expansion, better operator support).")

    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_feynman_signature_diagnostic.py")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
