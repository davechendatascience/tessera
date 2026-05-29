# Planned: symbolic-network GPU validation + remaining JAX work

**Status:** PLANNED. Follows `docs/shipped/symbolic_network_jax_optimization.md`.
The eval optimizations and the opcode-tape interpreter are shipped and
correct; what remains is **GPU validation** and a few deferred
optimizations whose payoff is GPU- or inference-specific.

**Provenance:** code-review session 2026-05-29.

---

## 1. GPU validation of the interpreter (the key open item)

The opcode-tape interpreter (`tessera.experimental.symbolic_interp`)
compiles once for all trees instead of ~279 times. On CPU it is
*slower* than per-tree JIT (cheap CPU compiles don't justify the
heavier per-eval interpreter). The whole point is GPU, where XLA
compiles are expensive (~0.5–2 s each) and the per-tree path pays
~280 s+ of compilation per run.

**Experiment:** on a Colab GPU runtime, run the binary MNIST 0-vs-1 GP
(pop=30, gens=15, channel bank) twice:
- `use_interpreter=False` (per-tree) — time + wall-clock breakdown
- `use_interpreter=True` (interpreter) — time

**Success criterion:** interpreter total runtime < per-tree total
runtime on GPU by a clear margin (expected multiple ×), with identical
GP results.

**If it wins:** flip the notebook's recommended config to
`use_interpreter=True` on GPU, and the README/Colab story becomes
"compile-once symbolic-network training on GPU."

**If it doesn't:** the GPU compile cost was overestimated, or the
interpreter's dynamic-gather/scatter pattern is GPU-unfriendly. Either
way it's a documented finding; per-tree stays the default.

This needs a GPU; it cannot be validated on the local CPU dev box.

---

## 2. Deferred optimizations (do after GPU validation, if needed)

### 2a. On-device loss + accuracy

`_scores` returns `np.asarray` (device→host of the (N, n_classes) score
matrix), then numpy computes log-softmax / argmax. Compute CE + argmax
accuracy in JAX and transfer 2 scalars instead.

- Gain: small (score matrix is tiny). Worth it only if GPU profiling
  shows the score-matrix transfer is non-trivial.
- Risk: ~1e-7 (float32 reductions); accuracy is exact.

### 2b. Whole-network single-JIT fusion — for INFERENCE only

Fuse the entire forward pass of a SINGLE trained network (all K+N
trees + pooling + softmax) into one jit'd function. Wrong for the GP
loop (per-network-topology destroys cache reuse), but right for:
- the README/demo prediction path,
- the confusion-matrix evaluation,
- any deployment of a discovered network.

Proposed API: `compile_network_inference(network) -> fn(images) -> preds`,
built once for the final network. Never enters the mutating GP loop.

### 2c. Interpreter max_nodes bucketing

The interpreter pads every tape to `interpreter_max_nodes=40`. Trees
are usually much smaller, so padding nodes waste eval. Bucketing
max_nodes to {8, 16, 32} (recompile per bucket — still only ~3
compiles total) would cut padding overhead, helping both CPU and GPU
eval. Only worth it if eval (not compile) becomes the bottleneck.

---

## 3. Training-quality levers (orthogonal to JAX speed)

These improve the *result*, not the speed — listed here so the GPU
budget (once validation passes) is spent well:

- **Fitness cache by canonical tree hash** — identical offspring skip
  re-eval. Common late in GP.
- **Mini-batch fitness** — score on a rotating image subset per
  generation; full eval only for the best candidate. ~5–10× fewer
  per-eval samples.
- **Island populations** — parallel sub-populations with migration
  (PySR's main search-quality lever; also parallelizes across devices).
- **Warm-start at scale** — `warm_start_from_binary` already exists;
  the open question is whether a bigger GPU budget on the warm-started
  direct multi-class network reaches a README-worthy 10-class accuracy.

---

## 4. Reading order

1. `docs/shipped/symbolic_network_jax_optimization.md` — what's done +
   the compile-bound diagnosis.
2. `docs/research/hybrid_symbolic_networks.md` — the architecture +
   milestones (this GPU work unblocks Milestone C, the direct
   10-class network).
3. This note — what's next.

---

## Changelog

- 2026-05-29: initial. GPU validation of the interpreter is the
  headline open item; deferred on-device-metrics / inference-fusion /
  max_nodes-bucketing optimizations and training-quality levers
  catalogued.
