# JAX optimization of the symbolic-network evaluator

**Status:** SHIPPED (correctness validated). One item (the opcode-tape
interpreter) has its *speedup* validated only on CPU as negative; its
GPU speedup is pending — see `docs/planned/symbolic_network_gpu.md`.

**Scope:** the JAX evaluation path of
`tessera.experimental.symbolic_network` (image-classification symbolic
networks). Does NOT touch the tabular SR path
(`tessera.search.GP`) used by Feynman / IK / heat-equation / CAMELS —
those benchmarks never import this module.

**Provenance:** code-review session 2026-05-29, user asked for a
thorough review of the symbolic-network framework, then specifically
"besides JIT compilation, what other JAX improvement can we do?" and
"is GP+JAX actually optimal?"

---

## The diagnosis that frames everything

A symbolic network is evaluated by compiling each tree to a JAX
function. The naive path compiles **one XLA kernel per distinct tree
topology**. Genetic programming produces high topology diversity
(mutation changes structure every step), so a single run generates
hundreds of distinct topologies.

Measured on local CPU (MNIST 14×14, the diagnostic cell in
`notebooks/tessera_symbolic_network_mnist.ipynb`):

| Quantity | Value |
|---|---|
| First eval of a topology (compile + eval) | ~326 ms |
| Second eval (cached) | ~2.7 ms |
| **Compile : eval ratio** | **~123×** |
| Unique topologies per pop=30/gens=15 run | ~279 |

So compilation, not compute, is the dominant cost — the images are
tiny. This is the central fact: **the workload is compile-bound, and
the compile:eval asymmetry is worse on GPU** (GPU XLA compiles are
slower than CPU's, while the tiny eval barely benefits from the GPU).

This is why "the notebook showed no speedup": eval-side optimizations
shave the 2.7 ms, not the 326 ms.

---

## Shipped optimizations (in order, all bit-identical results)

All verified to produce identical GP results (binary MNIST 0-vs-1:
train 0.8925 / test 0.9025) before and after.

### 1. Single forward pass per candidate

`network_loss` and `network_accuracy` each ran a full forward pass, so
every candidate was evaluated **twice**. `run_network_gp._score` now
computes the score matrix once and derives both loss and accuracy from
it (`_loss_from_scores` / `_acc_from_scores`).

Provably identical: both are pure functions of the same score matrix.

### 2. Device-resident channel stack

`evaluate_network_jax_batch` did `jnp.asarray(channels[name])` on every
call → the fixed (N,C,H,W) channel stack was re-uploaded host→device
~1600× per run. `stack_channels_jax` builds the device-resident stack
ONCE; `run_network_gp` reuses it across the whole loop via a `stacked=`
kwarg. The numpy fallback still uses numpy dicts (no host/device mix).

Targets the host→device transfer that is the dominant per-eval JAX
overhead on GPU after compilation.

### 3. Hot-loop fix (the one that mattered most)

The offspring breeding loop — ~94% of all evaluations — still called
the public `network_loss` + `network_accuracy` directly, which (a)
double-evaluated and (b) bypassed the device-stack / interpreter path.
Only the init loop (~6% of evals) used `_score`. So optimizations 1–2
were barely exercised in the hot path until this was fixed. The
offspring loop now uses `_score` too.

This also explains why earlier speedup measurements looked modest.

### CPU result after 1–3

| Path | Time (pop=30, gens=15) | Compiles | Result |
|---|---|---|---|
| numpy | 179.3 s | — | 0.8925 / 0.9025 |
| per-tree JAX (final) | ~24 s | 279 | 0.8925 / 0.9025 |

Per-tree JAX is ~7× faster than numpy on CPU. All results identical.

---

## The opcode-tape interpreter (shipped, GPU-targeted)

`tessera.experimental.symbolic_interp`. The architecture that makes
GP+JAX compile-once instead of compile-per-topology.

- `encode_tree`: post-order encode a pure-pointwise tree into a
  fixed-length opcode tape (ops/arg1/arg2/consts/varidx/root).
- `get_interpreter(max_nodes)`: ONE jit'd interpreter. Unrolled
  fixed-length loop; each node uses `lax.switch(opcode, branches)` so
  it executes exactly ONE op (real branch — the image batch is carried
  inside the element shape, no vmap over samples, so the switch stays a
  real switch and avoids a ~24× compute blowup). Branches reuse
  `BIN_OP_FNS`/`UN_OP_FNS` → bit-identical to the per-tree path.
- The tape is a runtime input, so all trees share one compiled graph.
  Compile count over a whole run: **2** (one per element shape:
  layer-1 fields, layer-2 scalars) instead of ~279.

Enabled via `NetworkGPConfig.use_interpreter=True` (default False).

### Honest performance result

- **Correctness**: max abs error 4.77e-7 vs per-tree JAX over 186
  random trees; full GP runs bit-identical.
- **CPU**: interpreter is SLOWER (41.8 s vs 24.0 s). CPU compiles are
  cheap so per-tree's 279 compiles cost little, while the interpreter's
  per-eval overhead (40-node unroll, dynamic gathers, lax.switch) makes
  each eval ~1.75× heavier.
- **GPU**: expected to win large (GPU compiles ~0.5–2 s each → per-tree
  pays ~280 s+; interpreter pays ~2). **Not yet measured** — this is
  the open validation item.

---

## The answer to "is GP+JAX optimal?"

Not with a single implementation. GP wants high topology diversity;
XLA wants few graphs run on big batches. They are at odds. The
resolution is two backends, selected by device:

- **CPU**: per-tree JIT is optimal (cheap compiles, specialized fast
  eval). `use_interpreter=False`.
- **GPU**: the opcode-tape interpreter is optimal (one compile, eval
  parallelized). `use_interpreter=True`.

This is how serious JAX-GP systems (PySR-style backends) work.

---

## Options considered and NOT shipped (with rationale)

- **On-device loss/accuracy** — compute CE + argmax in JAX, transfer 2
  scalars instead of the (N, n_classes) score matrix. Small gain (the
  matrix is tiny). Deferred — see planned doc.
- **Whole-network single-JIT fusion** — fuse all K+N trees into one
  kernel. Great for INFERENCE on a *trained* network, but net-negative
  for the GP loop (per-network-topology → destroys cache reuse). The
  right home is a `compile_network_inference` for the final network —
  see planned doc.
- **AC-canonicalization of the topology key** — would merge
  commutative-equivalent trees and cut compiles further, but float
  addition isn't associative → ~1e-7 drift → violates "same result."

---

## Changelog

- 2026-05-29: initial. Documents the compile-bound diagnosis, the three
  shipped eval optimizations (single-pass, device-resident channels,
  hot-loop fix), and the opcode-tape interpreter (correct + compile-once;
  CPU-slower / GPU-pending). Companion: `docs/planned/symbolic_network_gpu.md`.
