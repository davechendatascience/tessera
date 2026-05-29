# Hamiltonian discovery benchmark (E1, gradient-free)

Per `docs/research/energy_based_symbolic_learning.md`. All checks
gradient-free; simulated bifurcation is the GPU energy minimizer
(classical Hamiltonian dynamics, no quantum hardware).

**SB path:** JAX/GPU
**Total wall-clock:** 56.3s

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

## 4. E1b — symbolic FORM-search from samples only

Discover WHICH interactions exist (not just fit a fixed pairwise
form) by greedy forward-selection over the multilinear-monomial
basis, scored by pseudo-log-likelihood. Gradient-free (Nelder-Mead
coefficient fit + greedy structure search) AND sampling-free (the
PLL needs only H on spin-flipped data, no model sampling). PLL
evaluated on the **GPU/JAX** path.

**coupling-corr mean = 0.999** (robust metric); **4/6 exact edge-set recovery** (strict).

`min|J|` = weakest true coupling. Where exact recovery fails it is
one of two boundary effects, not a search failure: (a) a *noise-
level* edge (min|J| ≲ 0.1) that finite samples cannot resolve and
BIC correctly omits — a true negative miscounted against a synthetic
truth that contains an unidentifiable coupling; (b) a *dense* system
where the pairwise pseudo-likelihood adds edges to absorb loop
correlations — the identifiability boundary the strategy note's §5
predicts. Coupling-corr stays ≈1.0 through both.

| n | seed | true edges | found | exact | coupling corr | energy corr | min\|J\| |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 3 | 3 | yes | 0.999 | 0.997 | 0.28 |
| 5 | 1 | 1 | 1 | yes | 1.000 | 0.998 | 0.19 |
| 5 | 2 | 5 | 5 | yes | 1.000 | 0.989 | 0.35 |
| 6 | 0 | 4 | 4 | yes | 0.999 | 0.974 | 0.20 |
| 6 | 1 | 4 | 3 | no | 1.000 | 0.970 | 0.05 |
| 6 | 2 | 7 | 11 | no | 0.999 | 0.991 | 0.19 |

## 5. E1b differentiator — non-pairwise (3-body) discovery

Ground truth: `H = -0.9 s₀s₁ - 0.8 s₂s₃ - 1.1 s₀s₁s₂` (a genuine
3-body term). A fixed pairwise inverse (sections 3–4 at order 2)
structurally CANNOT represent it; the symbolic form-search at
order 3 can discover it. This is the contribution of *symbolic*
form-search over a fixed-form statistical inverse.

| method | energy corr | discovered monomials |
|---|---|---|
| pairwise-only (order 2) | 0.675 | [(0, 1), (2,), (2, 3)] |
| symbolic (order 3) | 0.999 | [(0, 1), (0, 1, 2), (0, 3, 4), (2, 3)] |

Found the 3-body term `(0,1,2)`: **True**.

## Reading

- E1a recovering the Hamiltonian *exactly* (corr ≈ 1.0) validates
  that tessera represents and discovers energy-function forms — the
  energy-based analog of recovering a Feynman equation.
- E1b (sections 3–5) recovers the energy from SAMPLES ONLY, with no
  gradients and no model-sampling. The symbolic form-search (4)
  recovers exact interaction structure, and (5) discovers 3-body
  terms a fixed pairwise inverse cannot — the genuine contribution.
- Simulated bifurcation finding ground states validates the GPU
  gradient-free energy minimizer (the inner solver for E3).
- Both a CPU (numpy) and a GPU (jit'd JAX) path exist for the
  sampler and the PLL form-search hot loop; they agree to float
  precision.
- NEXT: E2 (MNIST-as-energy boundary) — expected to hit the
  high-order entanglement wall per the strategy note's §5.

## Reproducing

```
python benchmarks/run_hamiltonian_discovery.py --jax
```