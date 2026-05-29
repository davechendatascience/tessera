# Research note: hybrid symbolic networks (MNIST scoping)

**Status:** ? RESEARCH (scoping). Direction-setting note, no
implementation yet. Sets up the design space and proposes an MVP
path for the symbolic-network direction.

**Provenance:** discussion 2026-05-29:

> *"We are building one single feature for each digit, but this is
> probably not what we want to do. We want to do symbolic network
> training. And this is a novel idea."*
>
> *"a hybrid architecture probably. And it's quite a new challenge."*

The current MNIST benchmark (`benchmarks/run_mnist_feature_discovery.py`)
discovers ONE symbolic feature per binary task — TEST 0.80 on 0-vs-rest,
nowhere near the 99% a small CNN achieves. The framing is wrong: a
single tree can't encode the compositional structure of digit
recognition. What we need is a **symbolic NETWORK** — multiple
symbolic units composed in layers, trained jointly.

## 1. The conjecture

A symbolic network — multiple symbolic trees composed in a layered
structure, trained as a single unit — can produce interpretable
feature extractors for image classification while remaining
competitive with shallow neural networks. Combined with a neural
teacher in a hybrid architecture, the symbolic network distills
into an interpretable VERSION of the teacher's computation.

The novel claim is **end-to-end symbolic compositional networks**,
not a feature-bag SR system and not a single-equation distillation
(à la Cranmer et al. 2020). The architecture is symbolic at every
layer; the training procedure is GP at every layer.

## 2. Three flavors of hybrid

**Hybrid #1: Symbolic features + neural classifier**
- Layer 1: K symbolic trees → K scalar features
- Layer 2: small MLP (10-class softmax)
- Tessera does interpretable feature discovery; MLP handles the
  messy 10-class boundary.
- Trade-off: interpretability at feature layer only; decision rule
  is black-box.

**Hybrid #2: Neural prepass + symbolic network** *(recommended)*
- Step 1: Train a small CNN teacher (~95% accuracy, baseline-shaped)
- Step 2: Extract teacher's filters and activation patterns
- Step 3: Convert significant filters into symbolic seed trees
- Step 4: Run GP on a symbolic-network architecture with these as
  initial population
- The CNN is the "detector," symbolic is the "seed" — detect-then-
  seed at the network composition layer.
- Strongest novelty claim; generalizes Cranmer et al.'s
  single-equation distillation to whole-network distillation.

**Hybrid #3: Fixed primitives + symbolic composition** (DreamCoder-like)
- Library of hand-coded image features (edge filters, blob detectors,
  moments, holes, symmetry scores)
- GP discovers how to COMPOSE them into per-class scorers
- Strongest interpretability claim
- Weakest novelty (Ellis 2020 did this for program synthesis)

**Recommendation: Hybrid #2.**

The teacher does data-driven discovery (which tessera is not trying
to replace). The symbolic distillation produces an interpretable
network with the teacher's computational structure. This maps
exactly onto the framework's existing detect-then-seed pattern —
just at the network composition layer rather than at the single-tree
layer.

## 3. Architecture sketch

A 2-layer symbolic network:

```
image ─┬→ T_1 ─┬─→ scalar f_1 ─┐
       ├→ T_2 ─┼─→ scalar f_2 ─┤
       ├→ ...                  ├─→ T_class_d(f_1,...,f_K) → score_d (for d=0..9)
       └→ T_K ─┴─→ scalar f_K ─┘
                                         ↓
                                    argmax → predicted digit
```

**Layer 1**: K = 16 symbolic feature trees, each takes the (28×28)
image as input, outputs a scalar via discovered aggregation (mean,
max, sum, ...). Uses Measure2D ops for convolutional structure.

**Layer 2**: 10 symbolic class scorers, each takes the K scalar
features, outputs a class score.

**GP unit**: the full 26-tree network. Mutation operates on one tree
within the network; the other 25 are held fixed. Crossover swaps a
tree-slot between two networks. Fitness is network-level cross-
entropy on the 10-class problem.

## 4. Connection to the existing framework

The framework's themes extend naturally to networks:

- **Detect-then-seed**: the CNN teacher's filters seed the layer-1
  population (Hybrid #2). Decompose v2 / C8 could seed the layer-2
  combiners with sums of products of features.
- **Class A/B/C taxonomy at the network level**: a Class C network
  has features that survive across font/size/noise perturbations;
  Class B has features that overfit TRAIN-specific pixel patterns.
- **Skeleton persistence**: bootstrap-resample the training data; check
  whether the network's discovered tree-topologies are stable. Same
  diagnostic, applied to a 26-tree object instead of a 1-tree object.
- **Counterfactual evaluation (C5, already shipped)**: counterfactual
  perturbations of test images (rotations, noise, occlusions) →
  network-level CF score → rank candidate networks.

## 5. What's missing infrastructure-wise

None of these exist yet:

1. **`Network` dataclass**: a tuple-of-trees with a defined input/
   output schema. Validation against schema. Hashing for caching.
2. **Network-aware GP loop**: mutation operates on slot-within-network
   rather than on a single tree; fitness is network-level.
3. **Image-aware Measure2D vocabulary extensions**: probably need
   max-pooling, stride configurations, multi-scale features.
4. **Aggregator vocabulary**: the existing benchmark hardcoded mean-
   pool because the GP couldn't discover aggregators. The network
   architecture needs an explicit aggregator slot at the layer-1
   output.
5. **Neural prepass** (for Hybrid #2): small CNN training + filter
   extraction + filter-to-symbolic-tree conversion.
6. **Network-level Class A/B/C diagnostic**: bootstrap-persistence on
   a 26-tree network. Likely needs new metric definitions (e.g.,
   how much do the feature trees overlap structurally across
   bootstraps?).
7. **Cross-entropy loss as a tessera loss function**: today's MSE/PnL
   losses don't handle multi-class. Probably want a temperature-
   scaled softmax + cross-entropy that's gradient-friendly for
   const_opt.

## 6. MVP path with milestones

The work is multi-week. Milestones, smallest first:

**Milestone A — Network dataclass + 2-class problem (1 week)**
- Implement `Network` dataclass
- Hardcoded 2-layer architecture: K=4 layer-1 trees + 1 layer-2 tree
- 2-class problem: digit 0 vs digit 1 (binary, easiest contrast)
- Network-aware GP that mutates one tree at a time
- Compare against the existing single-tree 0-vs-rest baseline
- **Success**: TEST accuracy > 0.95 (vs 0.80 for single tree)

**Milestone B — Neural prepass + Hybrid #2 (2-3 weeks)**
- Train small CNN (1-2 conv layers, no dense) on the same 2-class
  problem
- Extract filters as 2D arrays
- Convert significant filters to symbolic seed trees (initial
  attempt: pattern-match against known Measure2D atoms)
- Inject as network initial population
- **Success**: TEST > CNN baseline (or matches with measurably
  smaller cx); interpretability claim documentable

**Milestone C — 10-class MNIST (3-4 weeks)**
- Scale up: K=16 layer-1, 10 layer-2 scorers
- Cross-entropy loss
- Full MNIST TRAIN/TEST
- **Success**: TEST > 0.90 (CNN territory: 0.99). Headline interpretability
  story: each layer-1 feature maps to a human-readable concept
- Or: clear documented falsification — Class B at the network level,
  features overfit TRAIN, doesn't generalize. Still publishable.

**Milestone D — Cross-domain validation (later)**
- Same architecture applied to other small image datasets
  (Fashion-MNIST, CIFAR-10 if scope permits)
- Tests whether the discovered features are domain-specific (Class B)
  or generic visual primitives (Class C)

## 7. Open questions

These don't have obvious answers and need decisions during
implementation:

1. **K (layer-1 width)**: 4 / 16 / 64 / discoverable? Bigger K
   gives more expressiveness but blows up the search space. MVP:
   fixed K=4 for Milestone A; K=16 for Milestone C.
2. **Aggregator**: hardcode mean-pool everywhere (existing benchmark
   choice) or add an aggregator slot per layer-1 tree? Add a slot
   — the existing benchmark explicitly identified this as a
   bottleneck.
3. **Mutation rate per slot**: should layer-1 trees mutate more often
   than layer-2 (more parameters there)? MVP: uniform; refine if
   data shows layer-imbalance.
4. **Crossover semantics**: swap one slot vs swap a whole layer? MVP:
   single-slot swap, the simpler.
5. **Network ensembling**: at the end, do we keep one network or an
   ensemble? MVP: single network; ensembling is a follow-up.
6. **Training data per generation**: full MNIST 60K is expensive per
   eval. MVP: stratified subset (5K-10K), full eval only on test.
7. **Loss landscape**: cross-entropy on a discrete tree population
   has discontinuities at class-boundary crossings. May want a
   temperature-scaled version for smoothness.

## 8. Falsification

What would tell us this direction is wrong:

1. **Milestone A fails**: K=4 symbolic networks can't beat the
   single-tree 0-vs-rest baseline. → The compositional structure
   isn't actually helpful at this scale; either K is too small or
   the GP can't navigate network-space.
2. **Milestone B shows Class B**: neural prepass seeds the network,
   but bootstrap-persistence shows the symbolic features are TRAIN-
   specific (different on each bootstrap). → The distillation isn't
   capturing teacher's mechanism, just its TRAIN-specific
   predictions.
3. **Milestone C plateaus far below CNN**: TEST < 0.85 even with
   K=64 and aggressive search. → The hybrid framing is wrong;
   something more architecturally innovative needed.

## 9. What this note explicitly does NOT claim

- **NOT that symbolic networks will match CNNs on accuracy.** They
  won't. The publishable result is *interpretability + reasonable
  accuracy*, not accuracy.
- **NOT that the infrastructure is small.** This is multi-week work.
  Milestone A alone is ~1 week.
- **NOT that the architecture choices are settled.** Many design
  decisions remain open; the MVP path uses the simplest choice and
  lets empirics drive refinement.
- **NOT a substitute for C8 or other algebraic SR work.** C8 ships
  this session; symbolic networks are a parallel research direction
  that doesn't block ongoing work.

## 10. Reading order + next steps

For someone picking up this thread:

1. Read this note for the design space.
2. Read `c8_additive_polynomial.md` for the current per-tree
   detector work that the network would inherit.
3. Read `from_data_to_mechanism.md` §5 for the Class A/B/C taxonomy
   the network-level diagnostic will extend.
4. Read `process_discovery_sr.md` §6 for the methodological tuning
   discipline that should govern hybrid-network experiments.

Concrete next step when work resumes: start Milestone A. Write
`tessera/experimental/symbolic_network.py` with the `Network`
dataclass + 2-class GP loop + benchmark on digit 0 vs digit 1.

## Changelog

- 2026-05-29: initial scoping note. Three hybrid flavors named.
  Hybrid #2 (neural prepass + symbolic network) recommended.
  Milestone-based MVP path outlined. No implementation yet.
