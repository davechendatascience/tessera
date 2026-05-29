# Hamiltonian discovery benchmark (E1, gradient-free)

Per `docs/research/energy_based_symbolic_learning.md`. All checks
gradient-free; simulated bifurcation is the GPU energy minimizer
(classical Hamiltonian dynamics, no quantum hardware).

**SB path:** JAX/GPU
**Total wall-clock:** 5.1s

## 1. Simulated bifurcation vs brute-force ground state (n=12)

**7/8 exact ground states** (rel gap < 1e-4; misses are float32 precision or genuine local minima).

| seed | ground E | SB E | rel gap | hit |
|---|---|---|---|---|
| 0 | -19.140 | -19.140 | 3.8e-08 | yes |
| 1 | -13.430 | -13.430 | 1.4e-08 | yes |
| 2 | -13.617 | -12.802 | 6.0e-02 | no |
| 3 | -20.187 | -20.187 | 1.2e-08 | yes |
| 4 | -21.902 | -21.902 | 6.2e-08 | yes |
| 5 | -14.156 | -14.156 | 3.0e-08 | yes |
| 6 | -20.945 | -20.945 | 1.5e-07 | yes |
| 7 | -13.329 | -13.329 | 3.9e-08 | yes |

## 2. E1a — Hamiltonian discovery from (config, energy) pairs

H is a degree-2 additive polynomial in the spins, so the C8
detector recovers it directly. Coupling-corr = correlation of
recovered vs true off-diagonal couplings.

| n | seed | R² | coupling corr | max\|ΔJ\| |
|---|---|---|---|---|
| 6 | 0 | 1.0000 | 1.0000 | 1.33e-15 |
| 6 | 1 | 1.0000 | 1.0000 | 1.11e-15 |
| 6 | 2 | 1.0000 | 1.0000 | 2.22e-15 |
| 8 | 0 | 1.0000 | 1.0000 | 3.11e-15 |
| 8 | 1 | 1.0000 | 1.0000 | 2.66e-15 |
| 8 | 2 | 1.0000 | 1.0000 | 6.88e-15 |
| 10 | 0 | 1.0000 | 1.0000 | 3.55e-15 |
| 10 | 1 | 1.0000 | 1.0000 | 3.55e-15 |
| 10 | 2 | 1.0000 | 1.0000 | 2.87e-15 |

## 3. E1b — inverse Ising from samples only (gradient-free)

Recover couplings from SAMPLES (no energies) via naive-mean-field
inverse `J ≈ -C⁻¹/β`. Confirms samples carry the couplings and
they are recoverable without gradients.

| β | coupling corr | sign recovery (true edges) |
|---|---|---|
| 0.2 | 0.975 | 0.88 |
| 0.5 | 0.993 | 1.00 |
| 1.0 | 0.979 | 1.00 |

## Reading

- E1a recovering the Hamiltonian *exactly* (corr ≈ 1.0) validates
  that tessera represents and discovers energy-function forms — the
  energy-based analog of recovering a Feynman equation.
- E1b recovering couplings from samples-only, gradient-free,
  confirms the inverse problem is tractable on this substrate.
- Simulated bifurcation finding ground states validates the GPU
  gradient-free energy minimizer (the inner solver for E3).
- NEXT: E1b with SYMBOLIC FORM SEARCH (search energy forms, score
  by moment-matching against samples via the GPU sampler) — the
  full energy-based symbolic-discovery loop. Then E2 (MNIST-as-
  energy boundary).

## Reproducing

```
python benchmarks/run_hamiltonian_discovery.py --jax
```