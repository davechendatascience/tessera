# Research note: invariance, sensor data, and axis-semantic SR

**Status:** open research direction. Written 2026-05-25 in response
to: "we want explainable formula but we run into invariance in the
data recording of sensor data (currently visual sensing). And we
want to be as generalized as possible, and that's one reason why
we need to even make how we categorize variable dimensions a choice."

This is the third of the research notes in this series; it builds on
`fit_as_perfect_info_game.md` (the Knuth-grounded framework) and
`measure_theory_and_perfect_info.md` (the measure-algebra layer)
and sketches what would be the **next architectural layer** for
tessera.

## 1. The tension

Two values are in tension:

- **SR's promise: explainable formulas.** The output is a symbolic
  expression a human can read, audit, falsify. This is the entire
  reason to do SR instead of fitting a black-box neural net.
- **Real sensor data has invariance.** An image of a "7" shifted three
  pixels right is still a "7". An audio recording pitch-shifted by a
  semitone has the same content. A multi-asset basket is unchanged
  under asset reordering. A point cloud is unchanged under rotation.

If SR ignores invariance, two failure modes emerge:

1. **Non-generalising solutions.** The GP finds a formula that works
   on the training data — `image[10, 14] > 0.5` — but fails on shifted
   inputs because the formula references SPECIFIC PIXEL POSITIONS.
2. **Non-interpretable solutions.** The GP brute-forces a workaround
   by combining many specific-position features, producing a tree
   so large no human can audit it. The explainability promise is
   broken.

So invariance isn't a side issue. It's where the SR-for-sensor-data
research direction actually lives.

## 2. The CNN comparison (briefly)

Convolutional neural networks solve translation invariance by:

- **Weight sharing**: the same convolution kernel is applied at every
  spatial position. This makes the *operator* translation-equivariant.
- **Global pooling**: the equivariant feature map is collapsed via
  mean/max to a scalar. This makes the *output* translation-invariant.

CNNs HARDCODE this inductive bias. The architecture forces invariance;
the learned weights are interpretable only in the context of that
architecture.

SR's challenge: get the SAME inductive bias, but in a form a human
can read.

## 3. Tessera's partial solution (what's already there)

Tessera's measure-theoretic operator algebra already addresses HALF
the problem:

- `FunctionalOp` with a 1-D `Measure` is convolution-along-an-axis.
  *The same kernel is applied at every position.* Translation
  equivariance is built into the operator.
- `FunctionalOp2D` with a `Measure2D` is 2-D convolution. Same
  property in 2-D.

What's missing:

- **Aggregation operators.** Tessera has no `reduce_mean`,
  `reduce_max`, `reduce_sum` over a spatial axis. So the GP can't
  convert an equivariant feature map into an invariant scalar
  prediction. The benchmark would have to hardcode the aggregation,
  not discover it.
- **Axis semantics.** Tessera's grammar implicitly treats 1-D inputs
  as "time" and 2-D inputs as "time × space." Both axes get
  translation-equivariant operators by default. But what if the axis
  is *not* translation-invariant? A multi-asset basket's "asset"
  axis is permutation-invariant, not translation-invariant. A
  spectrogram's frequency axis is log-translation-invariant. A
  point cloud's "point index" axis is permutation-invariant.

The right operator for each axis is DIFFERENT.

## 4. The big idea: axis semantics as a first-class search choice

The proposal: every variable in the SR environment carries not just a
*shape* but an *axis semantic* — a declaration of WHAT KIND OF
DIMENSION each axis is. The valid operator set for that variable is
then constrained by the axis semantics.

Example axis types:

| Axis type | Invariance group | Natural operators |
|---|---|---|
| `Translation` | shifts | convolution (1-D measure), conv1d |
| `Translation × Translation` | shifts in both dims | 2-D convolution, conv2d |
| `Permutation` | reorderings | symmetric functions (sum, mean, max, sort) |
| `Cyclic` | rotations on a ring | circular convolution, DFT-basis ops |
| `Log-translation` | scale changes | log-shift convolution (DWT, Mellin) |
| `Rotation (SO(3))` | 3-D rotations | spherical harmonic transforms |
| `Graph` | automorphisms | GNN message passing, spectral ops |

Under this framing:

- A user declares `Var("image", axes=[Translation(H=28), Translation(W=28)])`
- The GP's `random_tree` is *restricted* to compositions of operators
  whose codomain matches the axes' invariance group
- A convolution outputs a translation-equivariant field; a global
  mean over an axis *eliminates* that axis (and its invariance), so
  the output is one dim smaller with the remaining axes' invariance
- A `gt(field, scalar)` returns a field of indicators, preserving the
  axes
- A final `reduce_mean(field, axis=ALL)` produces a scalar:
  translation-invariant

The GP discovers the *composition* — which operators to apply, in
what order — but the *type system* guarantees that whatever it
produces is invariant to the right group.

This is "grammar embedding inductive bias" (per the perfect-info game
framework's §5.4 mention of constrained grammars), pushed all the way
into the axis structure.

## 5. Why this is the right level of abstraction

The proposal isn't just "add more operators to tessera." It's a
declaration that **axis semantics are a search choice**, not a
hardcoded convention.

Why this matters:

1. **Generality across sensor modalities.** Video = `Translation × Translation × Translation`
   (H × W × T). Audio = `Translation`. Multi-channel EEG =
   `Translation × Permutation` (time × electrode-index). The same
   tessera kernel handles all of them, only the axis declarations
   change.

2. **Interpretability through type-level inductive bias.** The user
   reads a discovered tree and SEES the invariance: every operator
   used is provably compatible with the data's symmetries. No black
   box.

3. **Search space compression via the perfect-info framework.** The
   equivalence-class count |E_K| collapses dramatically when we
   restrict to invariance-respecting operators. The framework's
   conjecture (search lives in |E_K|, not |T_K|) gets MUCH tighter
   when the symmetry group is taken into account: by Burnside's
   lemma, |E_K^{invariant}| = |T_K| / |G| asymptotically.

4. **GPU primitives map 1:1 to invariance groups.** Each invariance
   type has a canonical GPU implementation:
   - Translation → `torch.nn.functional.conv*d`
   - Permutation → `torch.sum`, `torch.max` (axis-reduction)
   - Cyclic → `torch.fft.fft`
   - Graph → `torch_geometric.nn.MessagePassing`
   This means the GPU backend (deferred from `gpu_and_cv_via_sr.md`)
   becomes structurally cleaner: dispatch by axis type, not by
   operator name.

## 6. Sketched architecture: `tessera.axes` (or `tessera.invariance`)

A new submodule containing:

```python
# tessera/axes/types.py

@dataclass(frozen=True)
class AxisSemantic:
    """Declares what kind of dimension an axis is."""
    name: str
    size: int

class Translation(AxisSemantic): ...     # 1-D shift symmetry
class Permutation(AxisSemantic): ...     # symmetric group S_n
class Cyclic(AxisSemantic): ...          # Z_n rotations
class LogTranslation(AxisSemantic): ...  # scale group
class Rotation(AxisSemantic): ...        # SO(2), SO(3)
class Graph(AxisSemantic): ...           # carries an edge list

@dataclass(frozen=True)
class TypedVar:
    """A Var with axis semantics."""
    name: str
    axes: tuple[AxisSemantic, ...]
```

```python
# tessera/axes/operators.py

def conv1d(field: TypedField, axis: int, measure: Measure) -> TypedField:
    """Valid iff field.axes[axis] is Translation."""
    ...

def reduce_mean(field: TypedField, axis: int) -> TypedField:
    """Eliminates `axis` from the output; remaining axes' invariance preserved.
    Valid for any axis type (mean is symmetric)."""
    ...

def perm_sum(field: TypedField, axis: int) -> TypedField:
    """Valid iff field.axes[axis] is Permutation. Same as reduce_sum."""
    ...
```

The GP's `random_tree` would consult the axis types when picking
operators. The validator would reject trees that apply
translation-convolution to a permutation axis.

## 7. What this changes about the perfect-info game framework

The companion notes (`fit_as_perfect_info_game.md`,
`measure_theory_and_perfect_info.md`) treat the search space as
generic Expr trees with measure-theoretic operators. Adding axis
semantics:

- **Equivalence-class space shrinks by the invariance group's order.**
  For 1-D translation on a length-N axis, |G| = N. For 2-D, |G| =
  H × W. For permutation on N elements, |G| = N!. The search space
  compression is enormous.

- **F1/F2/F3 properties unchanged** (the framework's structural
  claims hold).

- **Conjecture sharpens:** for axis-semantic tessera at complexity K
  with axis types whose group order is |G|, the search visits at
  most O(|E_K| / |G| · log K) candidates. The 1/|G| factor is
  Burnside-flavoured.

- **Tessera's distinct theoretical contribution becomes:** type-level
  invariance enforcement + measure-theoretic operator algebra +
  branch-and-bound + canonical forms. No other SR engine has all
  four.

## 8. GPU upgrade as natural consequence

The deferred JAX backend (`gpu_and_cv_via_sr.md`) becomes much
cleaner with axis semantics. Instead of "rewrite each tessera
operator in JAX," it becomes "dispatch each axis type to its JAX
primitive."

- Translation → `jax.scipy.signal.convolve`, `jax.lax.conv_general_dilated`
- Permutation → `jnp.sum(axis=...)`, `jnp.mean(axis=...)`
- Cyclic → `jax.numpy.fft.fft`
- Graph → `jax_md` or `pyg.jax_geometric`

Each operator's JAX implementation is short (10-30 LOC) because the
heavy lifting is provided by JAX's built-in primitives for each
invariance group.

Concrete sequencing:

1. Design `tessera.axes` (1 week, no GPU)
2. Add a CPU implementation using numpy (1 week)
3. Add the JAX backend per axis-type (1 week per type; ~4-6 weeks
   total)
4. Benchmark on CV (MNIST, then CIFAR-10)

The total scope is ~6-10 weeks of focused work to reach "tessera on
GPU, doing CV with discovered architectures." Compared to the
"6+ months" estimate in `gpu_and_cv_via_sr.md`, the type-level
approach is 2-3× faster.

## 9. The interpretability claim, sharpened

With axis semantics, a discovered tree like

```
reduce_mean(
    tanh(conv2d(image, measure=ε_horiz_edge_kernel) +
         conv2d(image, measure=ε_vert_edge_kernel))
)
```

is BOTH:
- **Translation-invariant by construction** (provable from the type
  system: conv2d is equivariant; reduce_mean over all spatial axes
  produces an invariant scalar)
- **Interpretable**: the user reads it as "is there a horizontal or
  vertical edge anywhere in the image?" The measure kernels expose
  exactly what shape the model is looking for.

The interpretability + invariance combination is what's unique. CNNs
have invariance without interpretability; vanilla SR has
interpretability without invariance. Tessera + axis semantics has
both.

This is the strongest claim in the entire research-note series:
**tessera's distinct value proposition becomes "the only SR system
that gives invariance-preserving + interpretable formulas for
structured sensor data."**

## 10. Open research questions

1. **What's the right set of axis types?** The list in §4 is a
   starting point; spherical, hyperbolic, Lorentz are all candidates
   for physics applications. How much can be done with just the top
   four (Translation, Permutation, Cyclic, LogTranslation)?

2. **How does AC normalisation interact with axis types?** Currently
   `simplify_canonical` sorts children of commutative ops. With
   axes, sortability might depend on the axes' compatibility — two
   convolutions over the SAME axis can be reordered (commutativity
   of conv along an axis); over DIFFERENT axes, the order matters.

3. **What's the equivalence-class count $|E_K^{(G)}|$ when invariance
   group G is enforced at the type level?** This is the Burnside-
   flavoured refinement of the perfect-info game framework's
   conjecture; empirically measurable via the
   `run_equivalence_class_count.py` experiment extended with axis
   types.

4. **How does the L1-norm bound work on axis-typed operators?** For
   convolution over Translation axis it's tessera's current bound;
   for symmetric functions over Permutation axis, the bound is
   different (sum's L∞ bound is |size| · max(|lo|, |hi|)).

5. **Is there a unified GP grammar for axis-typed operators?** Or
   does each invariance group need its own grammar fragment? The
   former is more elegant; the latter is more flexible.

6. **What's the right Python representation?** Should `TypedVar`
   subclass `Var`, or be a separate type entirely? Backwards
   compatibility with current tessera trees matters.

## 11. The submodule question

The user asked: "we might even need another submodule." Yes, and
the right level of separation is:

- `tessera.expression` (existing) — the untyped measure-theoretic
  Expr tree algebra. Keep as-is.
- `tessera.axes` (new) — axis semantics + typed variables + typed
  operators. Builds on `tessera.expression` for the measure layer.
- `tessera.jax` (new, deferred) — JAX backend that dispatches by
  axis type.

The submodule split mirrors the conceptual layers:
1. Measure-theoretic operator algebra (untyped)
2. Axis-semantic typed operators (with invariance enforcement)
3. GPU backend (per-axis-type implementation)

Each layer can be developed independently. `tessera.axes` doesn't
require GPU; `tessera.jax` doesn't require new operators (just
implements existing ones).

## 12. Concrete first step

Before any of this, run a small experiment that validates the
core claim:

**Does tessera's CURRENT (untyped) machinery, with hardcoded
mean-aggregation, discover useful CV feature kernels on a tractable
dataset?**

If yes, the path to axis-typed tessera is worth the engineering.
If no, the bottleneck is somewhere else and we need to reassess.

That's the MNIST benchmark from the previous discussion, properly
scoped:
- Use `FunctionalOp2D` (translation-equivariant by construction)
- Hardcode a mean-aggregation step at the tree's output
- Train on 500 MNIST 0-vs-rest examples (downsampled to 14×14)
- Report: does the GP discover a kernel that classifies > 90%?
- Plot the discovered kernel — does it look like an interpretable
  feature detector?

If that works, we've validated the underlying claim and can invest
in the axis-types architecture knowing it will pay off.

## Changelog
- 2026-05-25: initial document. Independent companion to
  fit_as_perfect_info_game.md, measure_theory_and_perfect_info.md,
  gpu_and_cv_via_sr.md. Argues for axis semantics as the next
  architectural layer for tessera.
