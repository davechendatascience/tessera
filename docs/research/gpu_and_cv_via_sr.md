# Research note: GPU acceleration + computer vision via SR-evolved architectures

**Status:** open research direction, large scope. Written 2026-05-25
as a honest scoping document for the question "can tessera be moved
to GPU, and can SR organically evolve architectures useful for
computer vision?"

The short answer is **yes in principle**, **multi-month in practice**.
This doc lays out the path and identifies the load-bearing pieces.

## 1. The question

If tessera's SR machinery becomes a strong general-purpose
data-fitter (measure-theoretic operators + canonical forms + L1-norm
bounds + branch-and-bound), three follow-up questions become
interesting:

1. **Can we GPU-accelerate it?** SR is famously CPU-bound. GPU
   acceleration of the inner loop could change what's feasible.
2. **Can we apply it to computer vision?** Tessera's `FunctionalOp2D`
   already operates on (T, X) fields; CV operates on (H, W) images.
   The vocabulary is closer than it looks.
3. **Can SR organically discover novel architectures?** Instead of
   the standard neural-architecture-search (NAS) over human-designed
   primitives (Conv2D, BatchNorm, ReLU, ...), let SR explore the
   measure-theoretic operator algebra and see what falls out.

These three questions stack: (1) makes (2) feasible; (2) provides a
testbed for (3).

## 2. Honest scoping

### What's hard

**SR + GPU is not natively a good fit.** The classical GP inner loop
evaluates one tree at a time on a data array. Vectorisation parallel-
ises *across data samples* but not *across candidates* — different
candidates have different tree structures, so they can't share a
forward pass. Multi-threading at the population level works but
gives modest speedups (memory contention, Python GIL or pickling
overhead).

**The PySR community has gone there and back.** Several papers
(Mundhenk et al., DSRGym, etc.) have tried GPU-accelerated GP-for-SR;
results are modest. The dominant bottleneck is *symbolic divergence* —
the very thing that makes SR interesting fights with GPU parallelism.

**Computer vision is not a natural SR domain.** CV problems are
typically:
- High-dimensional inputs (224×224 RGB ≈ 150k features)
- Translation-invariant solutions desired
- Multi-class classification, not regression
- Standard architectures already very well-optimised

A tessera SR run on CIFAR-10 would be vastly slower than a small CNN
and almost certainly underperform.

### What's tractable

**GPU acceleration of tessera operators specifically** — much more
tractable than generic GP-on-GPU. Reason: tessera's `LinearFunctional`,
`SeparableBilinear`, `Volterra2`, `FunctionalOp2D` are all **batched
convolutions**, which is precisely what GPU frameworks (PyTorch, JAX)
do well. The "different tree structures fight parallelism" problem
becomes "different *measure parameters*" — and parameters can
be batched as a tensor.

**SR-as-NAS for small problems** — there's prior work (e.g., Real et al.
*Regularised Evolution for Image Classifier Architecture Search*,
AAAI 2019; AmoebaNet) showing evolutionary search over architectures
is competitive on CIFAR-10. The combination "evolutionary search over
measure-theoretic operators" is novel and worth exploring.

**Hybrid SR+gradient** — fix the tree structure, train the *measure
parameters* by gradient descent on a CV loss. This makes each
candidate's evaluation O(forward pass) — comparable to a small CNN
training run. NAS-grade scale, not single-tree-eval scale.

## 3. The three-stage path

### Stage 1: GPU backend for tessera operators (1-2 months)

Goals:
- A `tessera.torch` or `tessera.jax` submodule that mirrors the
  numpy-based operators but runs on GPU
- Differentiable measure parameters (kernel weights as `torch.nn.Parameter`
  / `jax.numpy` arrays)
- Batched evaluation across multiple candidates simultaneously

Architecture choice:
- **PyTorch**: easier for the NAS angle (well-known TorchScript /
  fx-based architecture search frameworks)
- **JAX**: better for the gradient-correctness angle (functional
  composition matches measure-theoretic semantics cleanly); slower
  ecosystem for NAS specifically

Recommendation: **JAX**. Tessera's compositional operator design fits
JAX's pure-function philosophy. `jax.vmap` over candidate parameter
sets handles the "batch across candidates" question directly.

Deliverables:
- `tessera.jax.measure` — JAX-array kernel representation
- `tessera.jax.functional` — `LinearFunctional`, `SeparableBilinear`,
  `Volterra2`, `FunctionalOp2D` reimplemented with `jax.numpy`
- Forward-pass equivalence tested vs numpy backend
- Per-operator speed benchmark

### Stage 2: CV-flavored benchmarks (1-2 months)

Goals:
- Validate that tessera's measure-theoretic operators can express
  CV-flavoured architectures
- Compare to small standard CNNs on toy CV problems

Concrete benchmarks (ascending difficulty):
1. **MNIST digit classification** with a *fixed-structure* tessera
   tree (e.g., `Volterra2(measure_2d_x, measure_2d_y)(image)`), and
   gradient-trained measure parameters. Beat random baseline.
2. **CIFAR-10** with a multi-layer tessera tree — multiple `FunctionalOp2D`
   composed via pointwise ops. Beat MLP baseline.
3. **Tiny ImageNet** — full-scale benchmark. Goal: be within 5% of
   ResNet-18.

The interesting research question: **can `Volterra2` operators encode
useful nonlinearity that a standard CNN's ReLU doesn't?** Volterra2 is
literally a quadratic feature interaction — used in some second-order
networks (e.g., bilinear pooling, attention via outer products).
Tessera's expression of it as a first-class operator makes the
architecture more flexible than fixed pooling layers.

### Stage 3: SR-as-NAS over measure-theoretic operators (2-3 months)

Goals:
- Replace human-designed CNN architectures with SR-evolved trees
- Each "candidate" is a tessera tree whose nodes are FunctionalOp2D /
  Volterra2 / SeparableBilinear with learnable measure parameters
- GP outer loop evolves tree structure; gradient inner loop trains
  parameters per candidate
- Compare to AmoebaNet, EfficientNet, hand-designed nets

Architecture sketch:
```
class TesseraNasCandidate:
    tree: Node  # tessera Expr tree with FunctionalOp2D nodes
    
    def fit(self, X, y, n_epochs):
        # Convert tree to a JAX-compatible function with learnable
        # measure parameters; train via gradient descent
        ...
    
    def loss(self):
        # validation loss after fit; used as GP fitness
        ...
```

Risks:
- Each candidate's `fit` takes minutes to hours; population evolution
  becomes prohibitively expensive
- Mitigations: small candidate networks; partial-training proxy
  fitness (train for a few epochs, extrapolate); fitness-predictor
  co-evolution (Schmidt & Lipson 2009)

This is where things get genuinely hard. NAS at this depth is a
PhD-thesis-sized direction. The reward, if it works: novel architectures
that no human would have designed, derived from first-principles
measure theory.

## 4. What tessera already has working for this

Surprisingly, the foundation is mostly in place:

| Component | Status | Notes |
|---|---|---|
| `FunctionalOp2D` for image inputs | shipping | Used in heat-equation benchmark |
| `Measure2D` with separable density | shipping | Direct analog of separable conv kernels |
| `Volterra2` for quadratic features | shipping | Maps to bilinear pooling / second-order nets |
| Tree mutation for structure search | shipping | The GP outer loop |
| L1-norm bounds for B&B pruning | shipping | Per the step (c) work; could prune NAS candidates |
| Hall of Fame | shipping | Per-complexity best-architecture |
| `optimize_constants` polish | shipping | The inner-loop gradient analog at trivial scale |

What's missing for stage 1:
- JAX/torch backend for the operators
- Differentiable measure construction (atoms + density → JAX array)
- Batched candidate evaluation

That's ~1500 LOC of focused work. Not trivial, but not prohibitive.

## 5. The "organically-evolved esoteric architectures" angle

Here's the speculative part the user asked about. If we successfully
deploy SR-as-NAS over measure-theoretic operators, what kinds of
architectures might emerge?

Some speculations:

1. **Density-only convolutions with non-rectangular support** —
   standard CNNs use rectangular (Gaussian, etc.) kernels by
   convention. SR has no such bias; it might evolve power-law,
   harmonic, or even fractional-derivative-like kernels that
   transfer poorly to standard conv but match the data better.

2. **Asymmetric Volterra2 stacking** — `Volterra2(μ_a, μ_b)(x) =
   L_{μ_a}(x) · L_{μ_b}(x)` is a quadratic feature. Stacking
   `Volterra2(Volterra2(x), Volterra2(x))` gives a quartic feature.
   No standard CNN architecture has this directly; tessera could
   discover it organically if quartic features matter for the task.

3. **Measure-algebraic identities discovered late in training** —
   under equality saturation (deferred Exp 4), the search might
   discover that some learned 7-node tree is semantically equal to
   a 3-node canonical form. This is "lottery ticket" hypothesis
   territory: small networks hidden inside large ones, found by
   algebraic equivalence.

4. **Cross-modality architectures** — tessera's `SeparableBilinear`
   over two inputs naturally encodes attention-like mechanisms. An
   SR-evolved net might rediscover transformer-style attention as a
   measure-theoretic primitive, but with different scaling behaviour.

These are speculative — they're what makes the direction interesting,
not what makes it tractable. The honest expectation: stage 1+2 give
real results; stage 3 is a research bet.

## 6. What I'd recommend NOT doing

Three failure modes from "let's apply tessera to CV":

1. **Going straight to ImageNet.** Standard architectures are too
   well-tuned; SR will lose by orders of magnitude in wall-clock and
   meaningfully in accuracy. Start with MNIST/CIFAR-10.

2. **Replacing the entire deep-learning stack.** Tessera's value-add
   is the *operator vocabulary* + *search loop*. The rest (gradient
   descent, batch norm, dropout) should be borrowed from PyTorch/JAX,
   not reinvented.

3. **Insisting on "no human priors."** Even AmoebaNet seeds the
   search with sensible building blocks. Tessera's measure-theoretic
   operators ARE the prior — that's good. Don't insist on starting
   from scratch.

## 7. Connection to tessera's current research

This direction is downstream of essentially everything else:

- The fit-as-perfect-info game framework → SR is a tractable
  optimisation problem
- The measure-theory + perfect-info note → tessera has the right
  vocabulary
- L1-norm bounds (step c) → branch-and-bound for NAS candidate
  pruning
- Hall of Fame → per-complexity best architecture
- The deferred e-graph experiment → equivalence-class collapse for
  candidate dedup

It's a natural application of the workbench tessera is becoming.

## 8. Honest go/no-go criteria

I'd recommend pursuing this IF:

- Tessera has produced at least ONE compelling non-trading success
  (PDE discovery already counts; let's say "discovered an equation
  better than humans by some metric")
- Someone is committing 2-3 months full-time to stage 1 (the JAX
  backend)
- There's a benchmark target (e.g., "beat ResNet-18 on CIFAR-10 with
  10× fewer parameters using tessera-NAS")

I'd recommend NOT pursuing this IF:

- Tessera's pure CPU SR isn't yet competitive with PySR on shared
  benchmarks (the BTC closure suggests we're not quite there)
- The team is also trying to publish on trading
- The compute budget for stage 3 isn't realistic (each NAS run is
  GPU-weeks)

## 9. Tractable next step

If the user wants to start somewhere concrete WITHOUT committing to
the full multi-month direction:

**Implement a JAX backend for `tessera.expression.measure` and
`tessera.expression.functional` only.** ~500 LOC, ~2-4 weeks of focused
work. Validates that the operators ARE JIT-compileable + GPU-friendly
and gives a 10-100× speed-up on the operators themselves. Doesn't
commit to NAS; doesn't commit to CV; just establishes the GPU
backend as a parallel evaluator.

If that succeeds, stages 2 and 3 become reasonable to scope.

If it fails (e.g., the measure-density-with-atoms decomposition
doesn't JIT cleanly), we've learned something important and can
reassess.

## 10. Does the Knuth framework work on GPU?

User's sharper question (2026-05-25). Honest answer below.

### 10.1 The framework's components by GPU-friendliness

The Knuth-grounded fit-as-perfect-info-game framework
(`fit_as_perfect_info_game.md`) has six load-bearing operations.
Each has a different GPU profile:

| Operation | Compute pattern | GPU-friendly? |
|---|---|---|
| Tree mutation | symbolic, branching | **No** (CPU) |
| Per-candidate data evaluation | dense numerical, O(N) | **Yes** (the natural fit) |
| Interval-arithmetic bounds | scalar reductions, small ops | **Mixed** (batchable but small) |
| Pareto threshold lookup | dict / sorted-list lookup | **No** (CPU) |
| Canonical-form simplification | tree rewriting | **No** (CPU) |
| Hall of Fame update | dict update | **No** (CPU) |

**The outer loop (mutation, selection, Pareto, HoF) is fundamentally
CPU. The inner loop (per-candidate data evaluation) is GPU-friendly.**

### 10.2 The structural tension

GPU acceleration favours **batch parallelism**: evaluate N candidates
on the same data simultaneously, get an N-vector of losses back.

Knuth's branch-and-bound favours **sequential pruning**: evaluate one
candidate, update the incumbent, use the tightened incumbent to prune
the next candidate before evaluation.

These are in genuine tension. With pure batch eval, you waste work on
candidates that B&B would have pruned. With pure sequential B&B, you
waste GPU throughput evaluating one candidate at a time.

**The resolution from the literature** (modern compiler optimisation
research, e.g. work on GPU-parallel A*): *batched best-first search*.
Evaluate a frontier of candidates in parallel; after the batch
returns, sequentially update the incumbent and re-check each
batch member for now-applicable pruning. The pruning rate is lower
than pure-sequential, but the wall-clock wins as long as
`(batch_size × parallel_speed) > (sequential_speed × (1 - prune_rate))`.

For tessera:
- Batch size ≈ population size (~100-1000)
- Parallel speed ≈ 100× on a modest GPU vs CPU eval
- Prune rate (from step b/c results) ≈ 20-50% on pointwise grammar
  with L1 bounds
- So: `1000 × 100 = 100k` > `1 × 0.5` ≈ 0.5. Batch wins by ~200,000×
  *effective throughput* — even at 50% pruning rate, batch
  evaluation dominates.

This says: **GPU acceleration is compatible with the Knuth framework
as long as you batch the inner loop.** The framework's B&B character
becomes less aggressive (less pruning, more eval), but the wall-clock
wins overwhelmingly.

### 10.3 What WOULD change on GPU

Concretely, if we GPU-accelerated the framework:

1. **Eval per candidate** drops from ~10ms (numpy/Numba on N=10⁴) to
   ~0.1ms (JAX on GPU with batched candidates). 100× speedup.
2. **Population size** can grow from 100 → 10k cheaply.
3. **Per-generation walltime** stays similar (because population grew).
4. **Total candidates explored per minute** grows by 100×.
5. **Pruning becomes less critical** — the cost of speculative
   evaluation is so low that aggressive pruning doesn't pay back.
6. **Simplification becomes MORE critical** — if you're going to eval
   10k candidates per gen, you want them to be distinct equivalence
   classes (else you're wasting throughput).

So GPU makes the framework's BRANCH-AND-BOUND aspect less central,
but the EQUIVALENCE-CLASS COLLAPSE aspect (canonical forms,
simplification, e-graphs) MORE central. The roadmap shifts.

### 10.4 What WOULDN'T change on GPU

The fundamental claims of the framework still hold:

- (F1) Perfect information — unchanged
- (F2) Decomposable evaluation — unchanged (the per-sample
  decomposability is what makes batch parallelism possible
  in the first place)
- (F3) Algebraic equivalence — unchanged; if anything, more critical
- The conjecture about |E_K| being the effective search-space size —
  unchanged; the framework's complexity analysis is about *unique
  candidates*, not per-candidate evaluation cost

GPU acceleration changes *what's worth doing*, not *what the
theoretical structure is*.

### 10.5 Concrete proposal for testing this

A focused experiment to validate "Knuth framework + GPU" works:

1. Implement `tessera.jax.functional` (the GPU backend, ~500 LOC).
2. Add a batched scorer that evaluates pop_size candidates
   simultaneously on a single forward pass.
3. Re-run a tessera benchmark (e.g., Lorenz-63 or weather PDE) with
   pop_size = 1000 instead of 100.
4. Measure: did the per-generation walltime stay similar? Did the
   final Pareto front improve from richer sampling?

Estimated effort: 2-3 weeks. Worth doing IF the user wants to commit
to the GPU direction. Not worth doing as a one-off experiment.

### 10.6 Honest summary

- **Yes**, the Knuth framework works on GPU, with one structural
  change: batched eval replaces strict-sequential B&B pruning.
- The framework's THEORETICAL structure (F1, F2, F3, the |E_K|
  conjecture) is unchanged.
- The framework's OPERATIONAL focus shifts: less branch-and-bound,
  more equivalence-class collapse (canonical forms, e-graphs).
- The hard part is the JAX/torch backend (~500 LOC of focused work).
  The framework's CPU components (mutation, HoF, Pareto) stay on CPU
  and don't need rewriting.

## Changelog
- 2026-05-25: initial document. Open research direction; honest about
  multi-month scope.
- 2026-05-25: section 10 added — focused answer to "does Knuth's
  framework work on GPU?" Yes with batched eval; equivalence-class
  collapse becomes more central than B&B pruning.
