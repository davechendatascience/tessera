# Multi-trajectory training — heat equation (FINAL EMPIRICAL ANCHOR)

The closing experiment for the heat equation discovery thread. Tests
whether training on K trajectories simultaneously (different ICs,
same α) makes Class B and Class A self-defeat, forcing the GP toward
Class C (clean `Const · Template`).

**Setup:** K=3 TRAIN trajectories of T=200 each, X=32, α=0.05.
Single-trajectory baseline uses T=600 (MATCHED sample count). All
evaluations on shared held-out TEST trajectory (ic_seed=999, never
seen during training).

**Oracles:** MULTI=4.105e-06, SINGLE=3.976e-06, TEST=3.981e-06
**Wall-clock:** 62.6s

## Headline finding

The multi-trajectory training discovered the **canonical heat equation form** at cx=4:

```
(M2D[1·(0,-1) + -2·(0,0) + 1·(0,1)](U) * 0.049883)
                                          ^^^^^^^^
                                          α extracted to 0.2% accuracy
```

This is `Laplacian(U) · α` — the textbook form a physicist would write. TRAIN/oracle = TEST/oracle = 1.00. Mechanism captured EXACTLY on held-out data. **No factory primitives. No grammar machinery. No physics shortcuts. Just multi-IC training with reduce_* downweighted.**

The cleanest Class C result across all our heat-equation experiments. Cx=4 is the minimum possible: one Const, one Var, one BinOp(mul), one FunctionalOp2D — nothing wasted.

## Comparison vs single-trajectory baseline (matched sample count)

| Class | Single-traj (T=600, 1 IC) | Multi-traj (3 × T=200, 3 ICs) |
|---|---|---|
| Class C (clean mechanism) | 1/3 — **cx=17**, train/test=1.37/1.35 | 1/3 — **cx=4**, train/test=1.00/1.00 |
| Class A (diff-style tautology) | 2/3 | **0/3** |
| degenerate (predict-zero) | 0/3 | 2/3 |

Multi-trajectory **eliminates Class A entirely** (the 2-atom diff_t-style trees can't fit consistently across 3 different ICs at the noise floor). What remains: either find the genuine mechanism that works for all 3 trajectories (Class C — the only stable solution), or get stuck at predict-zero (the GP's failure state).

**The trade is:**
- Single-traj: reliable mediocre fit (most seeds find diff-tautology Class A at ~2× oracle); occasional Class C with cruft
- Multi-traj: high-variance — when it works, **canonical mechanism at minimum cx**; when it fails, predict-zero

## Per-seed details — single trajectory (baseline)

| seed | train/oracle | test/oracle | cx | class | tree |
|---|---|---|---|---|---|
| 2026 | 2.04 | 2.12 | 6 | A | `M2D[diff_t](max(0.28, 0.9 · U))` |
| 2027 | 1.37 | 1.35 | 17 | C-partial | `M2D[diff_t](((atan2(..., M2D[diff_t](U)) + ...))` |
| 2028 | 2.03 | 2.11 | 8 | A | `(M2D[diff_t](max(0.25, U/1.12)) - 0.00016)` |

## Per-seed details — multi trajectory (3 ICs)

| seed | train/oracle | test/oracle | cx | class | tree |
|---|---|---|---|---|---|
| 2026 | **1.00** | **1.00** | **4** | **C** | `(M2D[Laplacian](U) * 0.049883)` ★ |
| 2027 | 14.71 | inf | 1 | degenerate | `-0.00142383` (predict zero) |
| 2028 | 14.71 | inf | 1 | degenerate | `-0.00142383` (predict zero) |

★ = the canonical mechanism, found cleanly.

## Why this is the convergence point for this thread

We've now explored the four primary levers for unit-dynamics SR discovery:

| Lever | Effect | Status |
|---|---|---|
| Vocabulary (sin/cos, atan2, Laplacian as factory) | Enables specific physics primitives | Each addition raises ceiling for its target class |
| Scoring (parsimony, MDL, simplification) | Trade loss vs cx | The polynomial simplifier closes ~half the cx gap |
| Search bias (reduce_* downweight, grammars) | Steer mutations toward mechanism-shapes | 5-LOC reduce_* fix gave first Class C |
| **Training data structure (multi-trajectory)** | Punishes trajectory-specific tricks | **Yields canonical Class C at minimum cx** |

Each lever moves the needle. **None alone is sufficient for reliable discovery; combined, they make discovery POSSIBLE but still high-variance.** The multi-trajectory + reduce_*-downweight + sufficient-compute combination produced the cleanest result we've ever achieved (cx=4 with α=0.05) but only in 1/3 seeds. The other 2/3 fail completely.

This is the natural ceiling for tessera at unit-dynamics-recovery:
- The mechanism IS findable with the right bias
- It's not reliably findable without further architectural changes
- More compute helps but with diminishing returns
- The variance comes from initialization sensitivity in non-convex search

For higher reliability (say >90% Class C rate), would need one of:
1. **Better initialization** — warm-start from successful prior runs or template seeds
2. **Mode-2 grammar** — actively construct `Const · Template` from any discovered template
3. **Hybrid neural-symbolic** — neural net biases the symbolic search
4. **Multi-method ensemble** — combine SR with different algorithms; consensus indicates mechanism

These are the next-level research directions, not engineering polish.

## What this opens for the future of tessera

The user identified the natural frontier: **composite dynamics** — discovering systems where multiple unit mechanisms interact (heat + Navier-Stokes, reaction-diffusion, Maxwell's equations as coupled PDEs). This is qualitatively different from unit-dynamics SR:

- **Multi-equation outputs**: tessera currently outputs single expressions; composite systems need coupled outputs
- **Operator algebra**: linear combinations, compositions, commutators — how unit operators combine
- **Conservation constraints**: physics imposes structure across operators (energy, momentum, mass conservation)
- **Cross-mechanism couplings**: T-field affects flow-field via buoyancy; couplings are first-class objects

These aren't extensions of tessera's current architecture. They're a different architecture. The natural framing: **tessera-as-unit-discoverer** is a finished tool; **tessera-as-composite-discoverer** is a new research project.

## Reproducing

```
python benchmarks/run_heat_equation_multitrajectory.py
```

Wall-clock ~63s at default settings.
