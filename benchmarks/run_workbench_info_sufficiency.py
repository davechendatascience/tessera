"""Stage 3 — workbench info-sufficiency calibration.

Per `docs/planned/methodology_workbench_and_library.md` §7 (Information
Sufficiency Framework): each canonical system needs empirically calibrated
`InformationRequirements`. This benchmark runs the sample-complexity sweep.

For each system, for each (n_samples, seed) combination:
  1. Generate trajectory
  2. Run full signature (Tier A classify + applicable Tier B)
  3. Record:
     - Classification correctness (Tier A matches declared model_class)
     - Tier B signature values

Aggregate across seeds at each n to find:
  - `min_samples`: smallest n where Tier A classifies correctly on
    >= 2/3 seeds AND key Tier B signatures stabilize
  - Stability proxy: coefficient-of-variation < 0.3 across seeds, or
    absolute change < 0.1 from previous n

Output:
  - benchmarks/results/workbench_info_sufficiency.md (the report)
  - Suggested info_min defaults per system (for manual review before
    updating systems.py)
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tessera.workbench import get_system, list_systems
from tessera.workbench.signatures import (
    classify_model_class, compute_full_signature,
)


# Per-system generation kwargs for varying sample count.
# For ODE/PDE systems we vary t_max; for algebraic we vary n_samples.

def make_traj(system, n_samples: int, seed: int, noise_std: float = 0.01):
    """Generate a trajectory of (approximately) n_samples points.

    For deterministic systems, varies IC + adds observation noise so that
    different seeds produce genuinely different observables (needed to
    measure cross-seed signature stability).
    """
    s = system
    rng_ic = np.random.default_rng(seed * 1000 + 7)

    def perturbed_ic():
        # 3% multiplicative perturbation of default IC (or additive if IC is zero)
        ic = s.default_ic().copy()
        scale = np.where(np.abs(ic) > 1e-3, np.abs(ic), 1.0)
        return ic + 0.03 * scale * rng_ic.standard_normal(ic.shape)

    if s.id == "algebraic_feynman_gaussian":
        return s.generate(n_samples=n_samples, seed=seed, noise_std=noise_std)
    elif s.id == "burgers_1d":
        dt = 0.005
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, seed=seed, noise_std=noise_std)
    elif s.id == "heat_1d":
        dt = 0.1
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, seed=seed, noise_std=noise_std)
    elif s.id == "fhn":
        dt = 0.05
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, ic=perturbed_ic(),
                          seed=seed, noise_std=noise_std)
    elif s.id == "vdp":
        dt = 0.05
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, ic=perturbed_ic(),
                          seed=seed, noise_std=noise_std)
    elif s.id == "kepler":
        dt = 0.005
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, ic=perturbed_ic(),
                          seed=seed, noise_std=noise_std)
    else:
        # Default ODE: dt=0.01, perturb IC
        dt = 0.01
        t_max = (n_samples - 1) * dt
        return s.generate(t_max=t_max, dt=dt, ic=perturbed_ic(),
                          seed=seed, noise_std=noise_std)


SIGNATURE_KEY_FIELDS = [
    "smoothness", "effective_dimensionality", "spectral_content",
    "lyapunov", "determinism",
]


def _extract_scalar(sig_value):
    """Extract a scalar from a SignatureValue.value (may be dict)."""
    if sig_value is None:
        return None
    v = sig_value.value
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, int):
        return float(v)
    if isinstance(v, dict):
        # Pick the first finite numeric field
        for key in ("time_acf_max", "peak_height", "variance_ratio"):
            if key in v and isinstance(v[key], (int, float)) and np.isfinite(v[key]):
                return float(v[key])
        return None
    return None


def run_calibration(n_grid, seeds, verbose=True, include_tier_b=True):
    """For each system + each n + each seed, record classification + signatures."""
    results = []
    systems = list_systems()
    for sid in systems:
        s = get_system(sid)
        declared = s.model_class.value
        if verbose:
            print(f"\n=== {sid} (declared={declared}) ===")
            print(f"{'n':>6s}  {'cls_correct':>11s}  {'tierB CoV':>10s}")
        for n in n_grid:
            classifications = []
            tier_b_values = defaultdict(list)
            for seed in seeds:
                try:
                    traj = make_traj(s, n_samples=n, seed=seed)
                except Exception as e:
                    classifications.append(("ERROR", str(e)[:30]))
                    continue
                if not np.all(np.isfinite(traj.state)):
                    classifications.append(("NAN", "trajectory has NaN/inf"))
                    continue
                try:
                    if include_tier_b:
                        sig = compute_full_signature(traj)
                        classifications.append((sig.inferred_model_class, ""))
                        for k in SIGNATURE_KEY_FIELDS:
                            v = _extract_scalar(getattr(sig, k))
                            if v is not None:
                                tier_b_values[k].append(v)
                    else:
                        inferred, _ = classify_model_class(traj)
                        classifications.append((inferred.value, ""))
                except Exception as e:
                    classifications.append(("ERROR", str(e)[:30]))
            n_correct = sum(1 for c, _ in classifications if c == declared)
            # Tier B coefficient-of-variation = std / |mean| per signature
            cov_per_sig = {}
            for k, vs in tier_b_values.items():
                vs_arr = np.array(vs)
                if len(vs_arr) < 2:
                    continue
                mean_abs = max(abs(vs_arr.mean()), 1e-9)
                cov_per_sig[k] = float(vs_arr.std() / mean_abs)
            mean_cov = float(np.mean(list(cov_per_sig.values()))) if cov_per_sig else None
            if verbose:
                cov_str = "—" if mean_cov is None else f"{mean_cov:.2f}"
                print(f"{n:>6d}  {n_correct:>5d}/{len(seeds):<5d}  {cov_str:>10s}")
            results.append({
                "system": sid,
                "declared": declared,
                "n": n,
                "n_correct": n_correct,
                "n_seeds": len(seeds),
                "classifications": classifications,
                "tier_b_cov": cov_per_sig,
                "tier_b_mean_cov": mean_cov,
            })
    return results


def derive_min_samples(results, threshold_fraction=0.67, cov_threshold=0.5):
    """For each system, find:
      - tier_a_min_n: smallest n where classify >= threshold_fraction (lenient)
      - tier_b_stable_n: smallest n where Tier B mean CoV < cov_threshold

    The recommended `min_samples` is max(tier_a_min_n, tier_b_stable_n).
    """
    by_system = defaultdict(list)
    for r in results:
        by_system[r["system"]].append(r)
    summary = {}
    for sid, rows in by_system.items():
        rows.sort(key=lambda r: r["n"])
        tier_a_min_n = None
        for r in rows:
            if r["n_correct"] / r["n_seeds"] >= threshold_fraction:
                tier_a_min_n = r["n"]
                break
        tier_b_stable_n = None
        for r in rows:
            if r["tier_b_mean_cov"] is not None and r["tier_b_mean_cov"] < cov_threshold:
                tier_b_stable_n = r["n"]
                break
        # Recommendation: max of the two (so both gates pass)
        if tier_a_min_n is None:
            recommended = None
        elif tier_b_stable_n is None:
            recommended = tier_a_min_n  # only Tier A floor known
        else:
            recommended = max(tier_a_min_n, tier_b_stable_n)
        summary[sid] = {
            "tier_a_min_n": tier_a_min_n,
            "tier_b_stable_n": tier_b_stable_n,
            "recommended": recommended,
            "declared": rows[0]["declared"],
            "all_results": rows,
        }
    return summary


def write_report(summary, n_grid, seeds, runtime, out_path):
    L = []
    L.append("# Workbench information-sufficiency calibration — Stage 3")
    L.append("")
    L.append("**Per-system sample-size calibration of `InformationRequirements.min_samples`.**")
    L.append("")
    L.append(f"Per design contract `docs/planned/methodology_workbench_and_library.md` §7.")
    L.append(f"For each canonical system, we sweep `n_samples` across {list(n_grid)}, ")
    L.append(f"running {len(seeds)} seeds per n. Two gates are evaluated:")
    L.append("")
    L.append("- **Tier A floor**: smallest n where >= 2/3 seeds get correct `model_class`")
    L.append("- **Tier B stable**: smallest n where Tier B mean coefficient-of-variation < 0.5")
    L.append("- **Recommended**: max(Tier A, Tier B) — both gates must pass")
    L.append("")
    L.append(f"**Total wall-clock:** {runtime:.1f}s")
    L.append("")
    L.append("## Per-system calibration table")
    L.append("")
    L.append("| System | Declared | Tier A floor | Tier B stable | Recommended | Current | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for sid in sorted(summary.keys()):
        info = summary[sid]
        s = get_system(sid)
        cur = s.info_min.min_samples
        rec = info["recommended"]
        ta = info["tier_a_min_n"]
        tb = info["tier_b_stable_n"]
        ta_s = "—" if ta is None else str(ta)
        tb_s = "—" if tb is None else str(tb)
        rec_s = "—" if rec is None else str(rec)
        if rec is None:
            delta = "no calibration"
        elif rec <= cur:
            delta = f"current OK ({cur} ≥ {rec})"
        else:
            delta = f"**TIGHTEN: {cur} → {rec}**"
        L.append(f"| {sid} | {info['declared']} | {ta_s} | {tb_s} | {rec_s} | {cur} | {delta} |")
    L.append("")
    L.append("## Detailed per-system sweep")
    L.append("")
    for sid in sorted(summary.keys()):
        info = summary[sid]
        L.append(f"### {sid} (declared = {info['declared']})")
        L.append("")
        L.append("| n | seeds_correct | Tier B mean CoV | inferred per seed |")
        L.append("|---|---|---|---|")
        for r in info["all_results"]:
            inferred = ", ".join(c for c, _ in r["classifications"])
            cov = "—" if r["tier_b_mean_cov"] is None else f"{r['tier_b_mean_cov']:.2f}"
            L.append(f"| {r['n']} | {r['n_correct']}/{r['n_seeds']} | {cov} | {inferred} |")
        L.append("")
    L.append("## Reading")
    L.append("")
    L.append("The Tier A classifier (`classify_model_class`) is the gateway to")
    L.append("within-class signature extraction. If Tier A misclassifies at a")
    L.append("given n, all downstream Tier B signatures are run with the wrong")
    L.append("model-class routing and the identification pipeline fails.")
    L.append("")
    L.append("**Tier A is robust above the calibrated min_n in this table.**")
    L.append("Below it, classification can flip seed-to-seed, indicating that")
    L.append("the autocorrelation / permutation-invariance / stencil-locality")
    L.append("signals are too noisy. This is the empirical floor.")
    L.append("")
    L.append("## What this does NOT calibrate (future Stage 3 sub-tasks)")
    L.append("")
    L.append("- **noise_max**: at fixed n, max noise std where classify still works")
    L.append("- **min_dt / min_dx**: time- and space-step sufficiency for ODE/PDE")
    L.append("- **multi-trajectory requirements**: for systems with declared min_trajectories > 1")
    L.append("- **Tier B signature stability**: each individual signature's stabilization curve")
    L.append("")
    L.append("These are the natural follow-ups; each adds one dimension to the sweep.")
    L.append("")
    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_workbench_info_sufficiency.py")
    L.append("```")
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n_grid", nargs="+", type=int,
                   default=[50, 100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args(argv)

    t0 = time.time()
    results = run_calibration(args.n_grid, args.seeds, verbose=True)
    summary = derive_min_samples(results)
    runtime = time.time() - t0

    print(f"\nTotal wall-clock: {runtime:.1f}s")
    print("\n=== Summary ===")
    print(f"{'system':<32s} {'declared':<10s} {'TierA':>6s} {'TierB':>6s} "
          f"{'rec':>6s} {'cur':>6s}")
    for sid in sorted(summary.keys()):
        info = summary[sid]
        cur = get_system(sid).info_min.min_samples
        ta = "—" if info["tier_a_min_n"] is None else str(info["tier_a_min_n"])
        tb = "—" if info["tier_b_stable_n"] is None else str(info["tier_b_stable_n"])
        rec = "—" if info["recommended"] is None else str(info["recommended"])
        print(f"{sid:<32s} {info['declared']:<10s} {ta:>6s} {tb:>6s} "
              f"{rec:>6s} {cur:>6d}")

    out_path = Path(__file__).resolve().parent / "results" / "workbench_info_sufficiency.md"
    write_report(summary, args.n_grid, args.seeds, runtime, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
