# Optimization regimes: why csp for small-symbolic, backprop for large-dense

**One line:** The loss-landscape geometry of a model is set by how
*overparameterized* it is. Large dense models have *benign* landscapes
(gradient descent works, local minima don't bite); small *parsimonious*
symbolic models have *rugged, partly-disconnected* landscapes (gradient
descent stalls). So the right optimizer is different per regime — **discrete
search + closed-form (csp) for small-symbolic, backprop for large-dense** —
and the differentiable-symbolic relaxation (diff_eml) sits in the worst spot.
The principled way to *jointly* learn features + symbolic structure is to put
a **differentiable symbolic head (diff_eml) on a CNN** and let the CNN's
overparameterization make the joint problem optimizable.

Date: 2026-05-30. Companion to `deep_symbolic_csp.md`,
`differentiable_eml_jax.md`, and [[project_csp_sr]].

---

## 1. csp does not tune parameters universally

csp_sr's mode is **discrete structure search + closed-form linear fit**. It
solves parameters that enter *linearly* exactly (least squares), and handles a
*few* embedded nonlinear constants via a bounded 1-template refine
(`diff_sr.make_refiner`). It is **not** general continuous parameter
optimization and never does gradient descent over many coupled parameters.
That is a deliberate strength (see below), not a gap to patch.

## 2. Three optimization regimes

| model | landscape | right tool |
|---|---|---|
| **large dense** (CNN, MLP) | **benign** — redundant params, minima near-global, connected | **backprop** (SGD/Adam); local minima don't bite |
| **small symbolic, relaxed** (diff_eml) | **rugged + disconnected** — few narrow basins, discrete structure gaps | *worst spot* — gradient descent stalls; a smarter optimizer only helps at the margins |
| **small symbolic, discrete** (csp_sr) | — (avoided) | **discrete search + closed-form** — enumerate structures, solve the linear part exactly; sidesteps the rugged surface entirely |

## 3. Why overparameterization is benign and parsimony is rugged

The landscape geometry is a consequence of **redundancy**:

- **Overparameterized → benign.** When many weight settings compute (almost)
  the same function, there are many equivalent good basins and lots of room to
  route around bad regions. Established results: spin-glass analysis
  (Choromanska et al. 2015 — almost all local minima sit near the global
  value); deep *linear* nets (Kawaguchi 2016 — every local min is global); the
  NTK / infinite-width regime (Jacot et al. 2018 — wide nets train like a
  *convex* kernel method); mode connectivity (Garipov et al. 2018 — good
  minima are joined by low-loss paths); lottery tickets (Frankle & Carbin
  2019). SGD reliably finds near-global solutions despite non-convexity.

- **Parsimonious → rugged.** A small symbolic model removes exactly that
  redundancy: every parameter matters, basins are few and narrow, and the
  *structure* is partly **discrete** (which operators), so the surface is not
  just rugged but disconnected — gradient descent cannot even move between
  structures. This is the classic "non-convex optimization is genuinely hard"
  regime, and it is *intrinsic* to wanting a small, exact, interpretable model.

**The property that makes symbolic models valuable (small, exact,
interpretable) is the same property that makes their continuous optimization
hard.** You cannot optimize your way out of it with a fancier optimizer; you
change *frame* — to discrete search (csp).

## 4. Implications

- **csp_sr is the correct engine for small-symbolic problems** (discovery:
  Feynman, dynamics) and for the **low-dimensional knobs** of a larger system
  (a scalar activation form, a sparse readout) — it enumerates forms and
  solves the easy part exactly, never touching the rugged continuous surface.
- **backprop is the correct engine for large-dense feature learning**
  (vision): the landscape is benign *because* the model is big, so local
  minima are a non-issue and gradient descent is both available and ideal.
- **diff_eml (the differentiable-symbolic relaxation) is the trap**: it tries
  to gradient-optimize a *small symbolic* model — the one place gradients are
  worst. Standalone, it stalls in local minima and suffers a discretization
  gap; a smarter optimizer (Adam, multi-start, annealing) does not change the
  geometry. This is why the project chose csp over diff_eml for discovery.

## 5. The resolution for JOINT feature + symbolic learning: CNN + diff_eml

The recurring goal — "train the conv features *together with* the symbolic
combination" — was blocked by gradient-freeness: csp is discrete, so no signal
reaches the feature layers (the credit-assignment wall, `deep_symbolic_csp.md`).
**diff_eml is differentiable**, so a diff_eml head on a CNN lets backprop flow
through the symbolic head into the conv layers — features and symbolic
combination train **jointly**.

Crucially, §3 says *why this is optimizable when standalone diff_eml is not*:
**bolted onto a CNN, the joint model is overparameterized**, so the CNN
supplies the redundancy that makes the landscape benign. The CNN effectively
*carries* the optimization, mitigating diff_eml's local-minima problem in the
joint model in a way it never is alone. The overparameterization argument of
§3 predicts CNN+diff_eml succeeds where diff_eml-alone fails.

**Honest costs.** (1) It is **backprop**, not gradient-free — the right trade
for the high-bandwidth feature part, but a genuine identity shift. (2) The
symbolic head is **soft**: diff_eml is a relaxation, so you get sparse,
interpretable *structure*, and extracting a crisp discrete expression needs a
**hardening + verification** step (the discretization gap may cost a little).
(3) It is a real build: a JAX CNN backbone + a diff_eml head (the
differentiable substrate exists in `diff_sr.py`) + an optax/Adam loop + CIFAR
— the project's first backprop training loop, GPU-bound (Colab).

**Build sketch.**
1. JAX CNN backbone (a few conv→nonlinearity→pool blocks) → feature vector.
2. diff_eml head: a fixed-topology differentiable symbolic DAG over the
   feature vector → class logits (reuse `diff_sr.make_core_eval` style eval).
3. End-to-end backprop (optax Adam) on CIFAR cross-entropy; anneal the
   diff_eml soft-selection toward hard.
4. Harden the head to a discrete symbolic expression; verify the hardened
   accuracy vs the soft model (report the discretization gap honestly).
5. Compare to: gradient-free csp-enhanced net (the floor) and a plain
   CNN+linear head (the backprop ceiling). The diff_eml head earns its place
   only if it ~matches the linear head while being interpretable.

## 6. The one-sentence rationale

**Match the optimizer to the landscape: discrete search + closed-form (csp)
for small-parsimonious-symbolic, backprop for large-overparameterized-dense;
and to learn features and symbolic structure *together*, make the symbolic
head differentiable (diff_eml) and let a CNN's overparameterization carry the
joint optimization.**
