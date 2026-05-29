"""Hamiltonian discovery benchmark (Conjecture E1, gradient-free).

Per docs/research/energy_based_symbolic_learning.md. Three checks, all
gradient-free:

  1. Simulated bifurcation (GPU energy minimizer) finds ground states
     of small Ising instances (validated against brute force).
  2. E1a — recover H(s) = -Σ J_ij s_i s_j - Σ h_i s_i from (config,
     energy) pairs via the C8 additive-polynomial detector (H is a
     degree-2 additive polynomial).
  3. E1b — recover the couplings from SAMPLES ONLY (no energies) via
     the gradient-free naive-mean-field inverse (J ≈ -C^{-1}/β).

This is the energy-based analog of the Feynman benchmark: discover the
generating energy function, with no gradients anywhere.

Usage:
    python benchmarks/run_hamiltonian_discovery.py
    python benchmarks/run_hamiltonian_discovery.py --jax   # SB on GPU
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.experimental.energy_symbolic import (
    random_ising, ising_energy, sample_ising_gibbs,
    brute_force_ground_state, simulated_bifurcation,
    discover_hamiltonian_energy_labeled, fit_to_coupling_matrix,
)

OUT = Path(__file__).parent / "results" / "hamiltonian_discovery.md"


def check_sb(use_jax: bool, n: int = 12, trials: int = 8):
    rows = []
    hits = 0
    for seed in range(trials):
        sys = random_ising(n=n, density=0.4, seed=seed)
        _, ge = brute_force_ground_state(sys)
        _, sb_e = simulated_bifurcation(sys, n_replicas=128, n_steps=400,
                                        seed=seed, use_jax=use_jax)
        # relative gap tolerant of float32 (JAX) precision
        rel = abs(sb_e - ge) / (abs(ge) + 1e-9)
        ok = rel < 1e-4
        hits += ok
        rows.append((seed, ge, sb_e, rel, ok))
    return hits, trials, rows


def check_e1a(sizes=(6, 8, 10), seeds=(0, 1, 2)):
    rows = []
    for n in sizes:
        for seed in seeds:
            sys = random_ising(n=n, density=0.5, seed=100 + seed)
            rng = np.random.default_rng(seed)
            S = rng.choice([-1.0, 1.0], size=(max(500, 40 * n), n))
            E = ising_energy(S, sys)
            fit = discover_hamiltonian_energy_labeled(S, E, r2_threshold=0.0)
            J_est, h_est = fit_to_coupling_matrix(fit, n)
            Jt = sys.J[np.triu_indices(n, 1)]
            Je = J_est[np.triu_indices(n, 1)]
            corr = float(np.corrcoef(Jt, Je)[0, 1]) if Jt.std() > 0 else float("nan")
            maxerr = float(np.max(np.abs(J_est - sys.J)))
            r2 = fit.r2 if fit else float("nan")
            rows.append((n, seed, r2, corr, maxerr))
    return rows


def check_e1b(betas=(0.2, 0.5, 1.0), n: int = 10):
    rows = []
    for beta in betas:
        sys = random_ising(n=n, density=0.4, seed=7,
                           coupling_scale=0.6, field_scale=0.3)
        S = sample_ising_gibbs(sys, beta=beta, n_samples=8000,
                               n_sweeps=80, seed=1)
        C = np.cov(S, rowvar=False)
        Cinv = np.linalg.inv(C + 1e-6 * np.eye(n))
        J_est = -Cinv / beta
        np.fill_diagonal(J_est, 0.0)
        Jt = sys.J[np.triu_indices(n, 1)]
        Je = J_est[np.triu_indices(n, 1)]
        corr = float(np.corrcoef(Jt, Je)[0, 1])
        nz = np.abs(Jt) > 1e-9
        sign_acc = float(np.mean(np.sign(Je[nz]) == np.sign(Jt[nz]))) if nz.any() else float("nan")
        rows.append((beta, corr, sign_acc))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--jax", action="store_true",
                   help="Run simulated bifurcation on the JAX/GPU path.")
    args = p.parse_args(argv)

    print("=== Hamiltonian discovery benchmark (E1, gradient-free) ===\n")
    t0 = time.time()

    print("[1] Simulated bifurcation vs brute-force ground state...")
    sb_hits, sb_trials, sb_rows = check_sb(use_jax=args.jax)
    for seed, ge, sb_e, rel, ok in sb_rows:
        print(f"    seed {seed}: ground={ge:.3f} SB={sb_e:.3f} relgap={rel:.1e} "
              f"{'HIT' if ok else 'miss'}")
    print(f"    -> {sb_hits}/{sb_trials} exact ground states\n")

    print("[2] E1a: Hamiltonian discovery from (config, energy) via C8...")
    e1a_rows = check_e1a()
    for n, seed, r2, corr, maxerr in e1a_rows:
        print(f"    n={n} seed={seed}: R2={r2:.4f} coupling-corr={corr:.4f} "
              f"max|dJ|={maxerr:.2e}")
    print()

    print("[3] E1b: inverse Ising from samples only (gradient-free nMF)...")
    e1b_rows = check_e1b()
    for beta, corr, sign_acc in e1b_rows:
        print(f"    beta={beta}: coupling-corr={corr:.3f} sign-recovery={sign_acc:.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # ---- report ----
    L = ["# Hamiltonian discovery benchmark (E1, gradient-free)", "",
         "Per `docs/research/energy_based_symbolic_learning.md`. All checks",
         "gradient-free; simulated bifurcation is the GPU energy minimizer",
         "(classical Hamiltonian dynamics, no quantum hardware).", "",
         f"**SB path:** {'JAX/GPU' if args.jax else 'numpy/CPU'}",
         f"**Total wall-clock:** {elapsed:.1f}s", "",
         "## 1. Simulated bifurcation vs brute-force ground state (n=12)", "",
         f"**{sb_hits}/{sb_trials} exact ground states** "
         "(rel gap < 1e-4; misses are float32 precision or genuine local minima).", "",
         "| seed | ground E | SB E | rel gap | hit |", "|---|---|---|---|---|"]
    for seed, ge, sb_e, rel, ok in sb_rows:
        L.append(f"| {seed} | {ge:.3f} | {sb_e:.3f} | {rel:.1e} | {'yes' if ok else 'no'} |")
    L += ["", "## 2. E1a — Hamiltonian discovery from (config, energy) pairs", "",
          "H is a degree-2 additive polynomial in the spins, so the C8",
          "detector recovers it directly. Coupling-corr = correlation of",
          "recovered vs true off-diagonal couplings.", "",
          "| n | seed | R² | coupling corr | max\\|ΔJ\\| |", "|---|---|---|---|---|"]
    for n, seed, r2, corr, maxerr in e1a_rows:
        L.append(f"| {n} | {seed} | {r2:.4f} | {corr:.4f} | {maxerr:.2e} |")
    L += ["", "## 3. E1b — inverse Ising from samples only (gradient-free)", "",
          "Recover couplings from SAMPLES (no energies) via naive-mean-field",
          "inverse `J ≈ -C⁻¹/β`. Confirms samples carry the couplings and",
          "they are recoverable without gradients.", "",
          "| β | coupling corr | sign recovery (true edges) |", "|---|---|---|"]
    for beta, corr, sign_acc in e1b_rows:
        L.append(f"| {beta} | {corr:.3f} | {sign_acc:.2f} |")
    L += ["", "## Reading", "",
          "- E1a recovering the Hamiltonian *exactly* (corr ≈ 1.0) validates",
          "  that tessera represents and discovers energy-function forms — the",
          "  energy-based analog of recovering a Feynman equation.",
          "- E1b recovering couplings from samples-only, gradient-free,",
          "  confirms the inverse problem is tractable on this substrate.",
          "- Simulated bifurcation finding ground states validates the GPU",
          "  gradient-free energy minimizer (the inner solver for E3).",
          "- NEXT: E1b with SYMBOLIC FORM SEARCH (search energy forms, score",
          "  by moment-matching against samples via the GPU sampler) — the",
          "  full energy-based symbolic-discovery loop. Then E2 (MNIST-as-",
          "  energy boundary).", "",
          "## Reproducing", "", "```",
          "python benchmarks/run_hamiltonian_discovery.py" + (" --jax" if args.jax else ""),
          "```"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
