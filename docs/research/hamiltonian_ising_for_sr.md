# Research note: Hamiltonian / Ising / QUBO methods in SR

**Status:** ? RESEARCH. Open exploration; not committed to ship.

**Provenance:** extracted from `docs/planned/roadmap.md` section 3 on 2026-05-24 when docs were reorganised into shipped/planned/research buckets. The original `roadmap.md` mixed open-ended research with committed engineering items; this file holds the open-ended part.

---

## 1. They're NOT pointwise — they're quadratic

A QUBO objective is `min_x x^T Q x` over `x ∈ {0, 1}^n`. The Ising Hamiltonian is the same with `s ∈ {-1, +1}`:

$$H(s) = -\sum_{i<j} J_{ij}\, s_i s_j - \sum_i h_i s_i$$

This is a **second-order polynomial** in the variables — a *bilinear form* plus a *linear form*. In tessera's vocabulary, that's exactly:

- The linear term `Σ h_i s_i` ↔ `LinearFunctional(measure_signed_sum)`
- The bilinear term `Σ J_ij s_i s_j` ↔ `SeparableBilinear` / `Volterra2`

So an Ising Hamiltonian is **already expressible in tessera's existing expression algebra**. What it adds is a different *domain* (discrete variables, binary or ±1) and a different *search procedure* (annealing instead of GP).

## 2. Four directions where annealing/Hamiltonian methods could enter tessera

### (a) Replace Nelder-Mead with Simulated Annealing for const-opt

Const-opt currently uses scipy's Nelder-Mead because PnL+flip is non-smooth. SA is also gradient-free, has provable convergence to global optimum (Geman & Geman 1984), and handles discontinuities cleanly. On the BTC PnL benchmark — where Nelder-Mead got stuck — SA might find better constant configurations.

*Effort: ~50 LOC. Just swap the optimiser. Low risk, contained.*

### (b) Use Simulated Annealing as the GP loop's search algorithm

GP and SA are different metaphors for the same problem: search a discrete state space (here, tree structures) by making local moves. Cranmer's PySR is GP-style; you could equally implement an SA-style search where:

- Current state = a single tree (not a population)
- Proposal = a tessera mutation
- Accept with probability `min(1, exp(-(L_new - L_curr) / T))`
- Cool T over time

Pros: simpler than GP; provable convergence theory; some problems are better suited to SA than GP.
Cons: loses the diversity benefit of populations; can't parallelise as naturally; harder to combine with Hall of Fame.

*Effort: ~200 LOC. Multi-day. Worth trying as a comparison to GP.*

**Note (2026-05-24):** `tessera.search.SimulatedAnnealing` is now shipped as one of the parallel searchers alongside GP and RandomSearch. This direction is partially explored; the open question is whether SA-with-cooling-schedule meaningfully outperforms GP on tessera-specific problems.

### (c) QUBO formulation of tree structure search

The most ambitious framing: encode tree-structure choices as binary variables and solve as a QUBO problem. For each "slot" in a fixed-depth tree template, a one-hot binary vector selects which operator goes there. The objective is the loss of the assembled tree. Pairwise interactions encode constraint satisfaction (e.g., "if slot 3 is a BinOp, slots 7 and 8 must be valid subtrees").

If this works, you could solve on classical SA *or* on a real D-Wave quantum annealer. Some recent papers do this for neural-architecture search (NAS) on small problems.

Pros: maps to actual quantum hardware. Provides a principled framing.
Cons: tree-structure search is genuinely combinatorial, and encoding constraints in QUBO is fiddly. Probably small-scale only.

*Effort: research-level. Multi-week. Interesting but uncertain payoff.*

### (d) Hamiltonian-style smooth losses

Most trading losses (PnL+flip-rate) are non-smooth because of `sign(prediction)`. A Hamiltonian-style relaxation would smooth this:

$$H(\text{tree}) = -\langle \text{tanh}(\beta\, \hat y) \cdot y_{\text{fwd}} \rangle + \lambda \langle \text{tanh}(\beta\,\hat y)^2 - 1 \rangle$$

where the second term softly pushes positions toward ±1 (like an Ising double-well potential). As `β → ∞`, this recovers the PnL+flip loss; at finite β it's smooth, so BFGS const-opt works.

Pros: makes const-opt much faster (BFGS vs Nelder-Mead) and might let tessera find the flip-rich regime PySR found.
Cons: changes the loss function — must be careful not to change the optimum.

*Effort: ~30 LOC for the loss function + benchmark comparison. Low risk.*

**Note (2026-05-24):** `tessera.search.losses_trading.pnl_loss_smooth` is now shipped as a Hamiltonian relaxation of `pnl_loss_hard`. Direction (d) is partially explored; the remaining work is empirical — measure whether the smooth relaxation improves discovery on real trading data.

## 3. Recommended research order

If you want to pursue this thread:

1. **Hamiltonian-smooth losses (direction d).** Lowest effort, partially shipped. Measure whether `pnl_loss_smooth` beats `pnl_loss_hard` on benchmark TRAIN/TEST splits with BFGS const-opt.
2. **SA-vs-GP head-to-head.** SA is shipped; design a controlled comparison with same budget on the same problem.
3. **QUBO encoding.** Multi-week research project; worth a paper if it works.

## Changelog
- 2026-05-24: Extracted from `roadmap.md` section 3 during docs reorganisation. Status notes added for (b) and (d) reflecting partial implementations shipped.
