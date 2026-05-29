# Research note: energy-based symbolic learning (gradient-free, GPU)

**Status:** ? RESEARCH — strategic thesis + experimental plan. Not yet
implemented. This note records the *bet* before we build, per the
discipline that direction decisions should be written down first.

**Provenance:** code-review + scalability discussion 2026-05-29. The
thread: (1) is GP+JAX optimal? → no, per-tree JIT is compile-bound;
(2) is the symbolic network mathematically bounded? → yes, in
*learnability* (discrete search), not expressivity; (3) can we scale
to billions of parameters? → only by making the within-slot
optimization continuous + differentiable (EQL-style); (4) user:
*"the computation model of modern quantum computing, ising,
hamiltonian doesn't rely on differentiability ... can we skip
differentiability because I think this is more valuable than using
the existing deep learning framework."* (5) constraint correction:
**we have a GPU, not a quantum annealer.**

Supersedes the scattered energy threads with a single strategic frame:
- `hamiltonian_ising_for_sr.md` — Ising/QUBO methods catalogue (the four
  directions; SA + Hamiltonian-relaxation losses already shipped).
- `search_as_energy_min.md` — SR-as-optimization energy framing
  (caching, pruning, simplification). DIFFERENT sense of "energy" (the
  search landscape, not the model). Both feed this note.

---

## 1. The bet, in one paragraph

Deep learning owns high-order, entangled function approximation via
**differentiability** (backprop's arbitrary-order credit assignment).
That field is crowded and mature; re-building it (EQL-style
differentiable symbolic nets) makes tessera a worse PyTorch. The
**uncrowded, tessera-native** territory is **gradient-free symbolic
discovery of energy functions** — Hamiltonians, Lagrangians, free
energies, spin models — optimized by **energy minimization on GPU**
(parallel simulated annealing, parallel tempering, Gibbs sampling,
simulated bifurcation), not by gradient descent. The thesis: plant the
flag where the *mechanism is an energy function* and where the
*optimizer is energy minimization*, because that is exactly where
differentiability is unnecessary and where tessera's existing pieces
already point.

## 2. The substrate is the GPU we have — not future quantum hardware

This is the load-bearing correction. We are **not** betting on a
D-Wave / quantum annealer. Energy minimization is mature on the GPU
*today*, and every method below is `jax.jit` + `vmap`-friendly:

| Method | What it is | GPU/JAX fit |
|---|---|---|
| Parallel simulated annealing | many Metropolis chains, cooling schedule | vmap over chains; trivially parallel |
| Parallel tempering / replica exchange | chains at a temperature ladder + swap moves | vmap over replicas; swaps are cheap |
| Gibbs sampling (Ising) | vectorized conditional spin updates | one matmul per sweep on GPU |
| **Simulated bifurcation** (Goto/Toshiba 2019) | Ising solving via *classical Hamiltonian dynamics* (coupled nonlinear oscillators, integrate ODEs) | pure ODE integration — jittable, vmappable over replicas; built for GPU/FPGA |
| Mean-field / TAP / belief propagation | deterministic fixed-point iterations | vectorized |

Simulated bifurcation is the sharpest fit: it solves Ising by
integrating a *Hamiltonian* system on a GPU, no quantum hardware, and
it is competitive with quantum annealers on many benchmarks. So
"energy-based, gradient-free, on the hardware we have" is a present
capability, not a speculative bet.

The choice we are making: use the GPU's parallelism for **energy
minimization** rather than for **backprop**. The GPU is happy to do
either; we choose the gradient-free path because it is native to the
problem class we are targeting.

## 3. The matched-constraint insight (why this can work)

The deepest reason this is coherent — and the thing that makes it more
than contrarianism. Recall the scalability bound: tractable symbolic
*discovery* requires **low epistasis / decomposability** (search cost
is exponential in *entangled* structural complexity). Now look at what
energy-based hardware/algorithms require: the objective must be
expressible as a **low-order (pairwise) energy** `H(s) = Σ Jᵢⱼ sᵢsⱼ +
Σ hᵢ sᵢ` (higher order costs auxiliary variables, exponentially).

**These are the same constraint.** Low-order interaction = low
epistasis = decomposable = pairwise-expressible. Therefore:

> Where the energy is low-order, **both** the discovery (tessera's
> GP/SA) **and** the optimization (GPU energy minimization) are
> tractable — and **neither needs differentiability.**

Differentiability is the tool deep learning invented to handle
*high-order entangled* credit assignment (chain rule through deep
nonlinearity). If you *restrict to low-order energies*, you do not
need that tool: the pairwise coupling matrix `J` **is** the credit
structure, evaluated across all variables in parallel by the energy
solver. Differentiability and low-order-energy-expressibility are dual
constraints; for the low-order regime, the second suffices and the
first is redundant.

And physics is saturated with low-order energy functions. This is not
a niche — it is most of statistical mechanics, condensed matter,
optimization, and constraint satisfaction.

## 4. Three optimization paradigms (where each lives)

| Paradigm | Info per step | Needs | Scales via | Bound |
|---|---|---|---|---|
| Gradient / differentiable (deep learning) | O(P) (backprop) | smoothness + differentiability | parameters (→ billions) | none practical |
| Gradient-free local (GP, hill-climb) | ~1 bit / eval | nothing | — | exponential in entangled complexity |
| **Energy-based global (SA / tempering / Gibbs / bifurcation)** | landscape structure, parallel | **low-order energy expressibility** | variables/spins + replicas (GPU-memory-bound) | hardware variable count × connectivity |

GP is, in this light, a *weak* energy-based search (stochastic
hill-climb, temperature-free). Annealing / tempering / bifurcation are
*principled* energy minimizers with better escape from local minima.
Moving tessera's discrete search from GP toward these is an upgrade
*within* the gradient-free paradigm, not a defection to gradients.

## 5. The trap: do NOT build an energy-based deep classifier

Honest history: energy-based learning models (Boltzmann machines,
EBMs) **lost to backprop** — *not because of differentiability* but
because training them requires **sampling from the model's
distribution** (intractable partition function), and that sampling did
not scale. Quantum annealers were *hoped* to provide native Boltzmann
sampling; they are not there. On a GPU, sampling (Gibbs / tempering) is
better but still the bottleneck for large dense models.

So: building an energy-based **MNIST classifier** walks into the
sampling bottleneck that already killed this approach once — AND MNIST
10-class is high-order/entangled, the worst case for *both* energy
methods and tractable discovery. That is the trap. We test it (E2
below) only to *document the boundary*, not expecting to win.

## 6. The right target: discover the energy function itself

Reframe from *"use energy methods to train a model"* to *"use tessera
to discover the energy function / Hamiltonian of a system."* This is:

- **On-thesis** — tessera discovers generating mechanisms; for vast
  swaths of physics the mechanism *is* an energy function. Discovering
  `H` is the same kind of object as discovering a PDE operator.
- **Native to tessera's algebra** — `Σ hᵢsᵢ` = `LinearFunctional(signed_sum)`,
  `Σ Jᵢⱼsᵢsⱼ` = `Volterra2`/`SeparableBilinear` (per
  `hamiltonian_ising_for_sr.md` §1). The Ising form is already
  representable.
- **Already half-built** — `SimulatedAnnealing` searcher is shipped;
  `pnl_loss_smooth` is a Hamiltonian relaxation; and crucially the
  **C8 additive-polynomial detector IS a degree-2 Hamiltonian-form
  detector** — `H = Σ Jᵢⱼsᵢsⱼ + Σ hᵢsᵢ` is exactly an additive
  polynomial of pairwise products + linear terms. So the
  *energy-labeled* discovery case is already largely solved by shipped
  code.
- **Gradient-free end to end** — discover the *form* with GP/SA, and
  the inner problem (which couplings nonzero, what structure) is a
  selection/QUBO problem a GPU energy minimizer solves.

## 7. Scaling — honestly, on a GPU

You scale not in *parameters* (the gradient axis) but in **number of
variables/spins × replicas**, bounded by **GPU memory** and the
parallelism of the energy algorithm:

- Gibbs/SA/bifurcation in JAX: a dense `J` is O(n²) memory; n ~ 10⁴
  spins is comfortable on a GPU, sparse `J` much more. Replicas
  (chains/temperatures) vmap cheaply.
- This is NOT billions-of-parameters scale, and it is NOT the goal. A
  structured physical Hamiltonian has **sparse** couplings; you do not
  need billions. The win is discovering the right *sparse low-order
  energy* and minimizing it on a substrate where gradient methods are
  not the natural fit.

So the honest answer to "scale to billions like deep nets": **no, and
that is the wrong yardstick.** The yardstick is: discover correct,
interpretable, low-order energy functions and optimize them faster /
more globally than GP, on the GPU, without gradients.

## 8. Experimental plan (gradient-free, GPU, in `tessera.experimental`)

### E1 — Hamiltonian discovery (the benchmark)

Two graded forms:

- **E1a (sanity, energy-labeled)**: generate a random Ising system
  (couplings `J`, fields `h`); sample spin configs `s`; compute
  energies `H(s)`. Give tessera `(s, H(s))` pairs; discover the form.
  This is degree-2 additive-polynomial SR → the **C8 detector** should
  largely solve it. Validates that tessera *represents and recovers*
  Hamiltonian forms. Success: recover the bilinear structure + couplings
  to small error.
- **E1b (the real test, sample-only / inverse Ising)**: give tessera
  only **samples** `s ~ p(s) ∝ exp(-βH(s))` — no energies. Discover `H`
  by a **gradient-free** objective (pseudo-likelihood or moment
  matching), with the inner optimization run by the GPU energy
  minimizer. This is the genuine energy-based-learning test: the model
  defines a distribution, and we recover it without gradients or a
  tractable partition function (pseudo-likelihood sidesteps `Z`).
  Success: recovered `J`/`h` match ground truth (e.g. correlation of
  inferred vs true couplings > 0.9), structure (which `Jᵢⱼ` nonzero)
  recovered.

**Graduation criterion (E1):** tessera recovers a sparse Ising
Hamiltonian's structure from samples (E1b), gradient-free, with
coupling-recovery clearly above a non-energy baseline.

**Removal criterion:** even E1a (energy-labeled, ≈C8) fails, OR the
GPU energy minimizer offers no benefit over the existing GP on the
inner problem.

### E2 — MNIST-as-energy (the stress test / boundary)

Frame classification as an energy model, gradient-free on GPU.
Expected to hit the high-order entanglement wall (the §5 trap). Run it
to *document where the boundary is*, not to win. A clean negative here
is a valuable result: it confirms the matched-constraint thesis (the
approach is for low-order energies, not entangled classifiers).

### E3 (conditional) — energy minimizer as the GP's inner solver

If E1b validates, replace selection/assignment sub-problems in the SR
loop (which terms, which couplings) with the GPU energy minimizer.
This is the `hamiltonian_ising_for_sr.md` direction (c), now with a
concrete GPU substrate.

## 9. Falsification

- E1a fails → tessera cannot even represent/recover Hamiltonian forms;
  the whole thesis is mis-conceived (unlikely — C8 already does degree-2).
- E1b fails but E1a passes → the *inverse* (sample-only) problem is the
  wall; energy-based *discovery* needs more than we have; reduces to
  "tessera fits energies when labeled," a weaker claim.
- GPU energy minimizer ≈ GP on the inner problem → the "energy methods
  beat weak hill-climb" premise is wrong at this scale; keep GP.
- E2 *succeeds* → surprising; the high-order-entanglement bound is
  softer than argued, and energy-based classification deserves a harder
  look.

## 10. What this note explicitly does NOT claim

- NOT that energy-based methods scale to billions of parameters. They
  scale in variables/replicas, GPU-memory-bound; that is a different
  and smaller axis, deliberately.
- NOT that we are using or need quantum hardware. GPU energy
  minimization (esp. simulated bifurcation) is the substrate.
- NOT that this replaces deep learning for general function
  approximation. It targets low-order, energy-function-shaped problems
  — physics, combinatorial structure, constraint systems.
- NOT that differentiability is "bad." It is the right tool for
  high-order entangled credit assignment. We are choosing a problem
  class where it is *unnecessary*, not claiming it is wrong elsewhere.

## 11. Reading list

- Goto, Tatsumura, Dixon (2019) *Combinatorial optimization by
  simulating adiabatic bifurcations in nonlinear Hamiltonian systems*,
  Science Advances. (Simulated bifurcation — GPU/FPGA Ising via
  classical Hamiltonian dynamics.)
- Nguyen, Zecchina, Berg (2017) *Inverse statistical problems: from the
  inverse Ising problem to data science*, Advances in Physics. (Inverse
  Ising / pseudo-likelihood — the E1b machinery.)
- Ackley, Hinton, Sejnowski (1985) *A learning algorithm for Boltzmann
  machines.* (The energy-based-learning origin + the sampling
  bottleneck.)
- Hinton (2002) *Training products of experts by minimizing contrastive
  divergence.* (Why sampling-based energy training is hard.)
- Mohseni, Read, Neven et al. (2017) *Ising machines* survey.
- Tessera internal: `hamiltonian_ising_for_sr.md`, `search_as_energy_min.md`,
  `from_data_to_mechanism.md` (the discover-mechanism thesis),
  `c8_additive_polynomial.md` (= degree-2 Hamiltonian-form detector).

## 12. Reading order / next steps

1. This note (the strategic frame).
2. Build `tessera/experimental/energy_symbolic.py` — synthetic Ising
   generator + JAX GPU energy minimizer (start: parallel SA or
   simulated bifurcation) + inverse objective (pseudo-likelihood).
3. Run E1a (≈C8 sanity), then E1b (the real test).
4. Then E2 (MNIST-as-energy boundary).

## Changelog

- 2026-05-29: initial. Strategic thesis for gradient-free energy-based
  symbolic learning on GPU (not quantum). Matched-constraint insight
  (low-order energy = low epistasis, differentiability unnecessary).
  Discover-the-Hamiltonian reframe; Boltzmann-sampling trap; E1-E3 plan
  with graduation/removal criteria. GPU substrate grounded in simulated
  bifurcation / parallel tempering / Gibbs (JAX-implementable now).
