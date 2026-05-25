"""MDL scoring with explicit log-likelihood (Conjecture C3).

Provenance: C3 from `docs/research/process_discovery_sr.md` §6.3.
Theoretical pre-analysis: `docs/research/c3_mdl_analysis.md`.

Status: **FALSIFIED** on heat equation (2026-05-26). Calibration math
        directionally right (α ordering: naive_mdl < adhoc < recal) but
        empirical effect below noise floor at our N/σ. All three modes
        produce essentially equivalent Pareto fronts. Conjecture C3
        ("MDL identifies right amount of model more accurately") is not
        supported. Deeper insight: scoring-function tweaks at parsimony
        scale don't materially affect Class C discovery; search-
        trajectory interventions (reduce_* downweight, multi-traj) do.
        See `benchmarks/results/heat_equation_mdl_mvp_c3.md`.

The conjecture and its dual nature
----------------------------------

C3 has two parts (per the pre-analysis):

  3a (PROVABLE): MDL has stronger theoretical foundation than
       ad-hoc parsimony. PROVEN — MDL is the MAP estimate under a
       universal prior P(m) ∝ 2^(−K(m)).

  3b (EMPIRICAL): MDL produces better Pareto fronts on tessera
       benchmarks. NOT PROVEN — depends on N, σ, and the encoding.

The calibration prediction
--------------------------

Per the pre-analysis, MDL fitness per-sample under Gaussian
likelihood is:

    fitness_MDL = MSE/(2σ²) + K(m)/N

vs ad-hoc:

    fitness_adhoc = MSE + α · cx

For heat equation (N=5940, σ=0.002), the MDL coefficient on
complexity is ~30× SMALLER than ad-hoc's α=0.005. **MDL is predicted
to under-penalize complexity, producing higher cx trees with worse
generalization** (calibration math).

So the experiment is structured to TEST THE PREDICTION:

  - If MDL produces high-cx overfit → math validated; conjecture
    falsified in its original form
  - If MDL doesn't overfit → math is wrong somewhere; debug needed

This is a sanity check on the calibration math, not a discovery
experiment.

Graduation criterion
--------------------
Not designed to graduate as-is. The naive Gaussian-MDL is predicted
to FAIL. The path to graduation:

  1. This module validates the calibration prediction
  2. Future work: structure-function-aware recalibration that
     actually identifies the right model size
  3. Recalibrated version could be tested separately

Removal criterion
-----------------
If MDL passes the test (doesn't overfit), the calibration math is
wrong somewhere. Module preserved until the math is reconciled.

Initial commit: 2026-05-26
Last evaluation: never

Implementation
--------------

Three modes via the `penalty_mode` parameter:

  "adhoc":  fitness = MSE + α_adhoc · cx
            (baseline; same as standard GP — included for clean
             three-way comparison via one code path)

  "naive_mdl":  fitness = MSE/(2σ²) + DL/N
                Per-sample MDL form under Gaussian likelihood with
                fixed or estimated σ. Predicted to overfit at typical
                tessera N/σ regimes.

  "recalibrated_mdl":  fitness = MSE/(2σ²) + DL/√N
                       Heuristic recalibration: the K(m) coefficient
                       grows ~ N^(1/2) rather than N^(-1) (BIC-like).
                       Predicted to balance fit and complexity more
                       like ad-hoc parsimony, but with principled
                       N-scaling.

σ estimation
------------

Two modes:
  - "known" σ: user provides (clean for synthetic data)
  - "estimated" σ: σ̂² = MSE_best running estimate (self-consistent
                  but can be unstable)

Description length encoding
---------------------------

For each node:
  Var:             ⌈log2(num_features)⌉ + 2 bits (type tag + index)
  Const:           ~16 bits (truncated float; aggressive)
  BinOp:           ⌈log2(num_binops)⌉ + 2 bits (type tag + op)
  UnOp:            ⌈log2(num_unops)⌉ + 2 bits
  FunctionalOp:    ~12 bits (type tag + measure approximation)
  FunctionalOp2D:  ~16 bits

These are heuristic but consistent. The relative bit counts between
node types matter more than absolute calibration.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from tessera.search.base import Candidate
from tessera.search.gp import GP, GPConfig
from tessera.expression.tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    iter_subtrees, BIN_OPS, UN_OPS,
)


# ---------------------------------------------------------------------
# Description length per node
# ---------------------------------------------------------------------

def _bits_per_node(node) -> float:
    """Approximate description length in bits for a single node."""
    if isinstance(node, Var):
        return 2.0 + 4.0  # type tag + ~16 features = log2(16) = 4
    if isinstance(node, Const):
        return 2.0 + 16.0  # type tag + truncated float
    if isinstance(node, BinOp):
        return 2.0 + math.log2(max(len(BIN_OPS), 2))
    if isinstance(node, UnOp):
        return 2.0 + math.log2(max(len(UN_OPS), 2))
    if isinstance(node, FunctionalOp):
        return 12.0  # rough; depends on measure complexity
    if isinstance(node, FunctionalOp2D):
        # M2D has atoms + optional density; rough estimate
        # Each atom: ~12 bits (weight + lag_t + lag_x)
        m2d = node.measure_2d
        n_atoms = len(getattr(m2d, "atoms", ()))
        return 4.0 + 12.0 * n_atoms
    return 8.0  # default for unknown


def description_length_bits(tree) -> float:
    """Total approximate description length in bits for a tree.

    Sums per-node bit costs over all subtrees. This is an APPROXIMATION
    of Kolmogorov complexity; the exact value depends on the encoding
    convention, but consistent encoding gives consistent ordering.
    """
    return sum(_bits_per_node(sub) for sub in iter_subtrees(tree))


# ---------------------------------------------------------------------
# MDL scoring
# ---------------------------------------------------------------------

class GPWithMDLScoring(GP):
    """GP that uses MDL-style scoring instead of ad-hoc parsimony.

    Per the C3 pre-analysis, the calibration math predicts naive MDL
    will UNDERPENALIZE complexity at typical tessera N/σ regimes.
    This subclass implements three modes for empirical validation
    of that prediction.

    Parameters
    ----------
    cfg : GPConfig
    sigma : float | None
        Known noise std. If None, sigma is ESTIMATED from current
        best residual (σ̂² = MSE_best, updated each generation).
    penalty_mode : str
        - "adhoc":            MSE + cfg.parsimony · cx (baseline)
        - "naive_mdl":        MSE/(2σ²) + DL/N (predicted to overfit)
        - "recalibrated_mdl": MSE/(2σ²) + DL/√N (BIC-like recalibration)
    loss_fn : callable | None
        Forwarded to GP base.
    """

    def __init__(
        self,
        cfg: GPConfig,
        *,
        sigma: float | None = None,
        penalty_mode: str = "naive_mdl",
        loss_fn: Any = None,
    ) -> None:
        if penalty_mode not in ("adhoc", "naive_mdl", "recalibrated_mdl"):
            raise ValueError(
                f"penalty_mode must be 'adhoc', 'naive_mdl', or "
                f"'recalibrated_mdl', got {penalty_mode!r}"
            )
        super().__init__(cfg, loss_fn=loss_fn)
        self.sigma = sigma  # None means estimate from residuals
        self.penalty_mode = penalty_mode
        # Running estimate of σ² for "estimated" mode; updated as the
        # GP progresses. Initialized large so first-gen scoring isn't
        # dominated by tiny σ.
        self._sigma2_est = max(sigma ** 2 if sigma else 1.0, 1e-6)
        self._n_samples_cached: int | None = None

    def _score(self, tree, env, y_true, born_gen):
        # Standard scoring gives train_loss, complexity, baseline fitness
        cand = super()._score(tree, env, y_true, born_gen)

        if self.penalty_mode == "adhoc":
            # Already uses ad-hoc parsimony via base class; just return.
            return cand

        # MDL modes
        N = self._n_samples_cached
        if N is None:
            N = int(np.asarray(y_true).size)
            self._n_samples_cached = N

        # σ² for the likelihood denominator
        if self.sigma is not None:
            sigma2 = float(self.sigma) ** 2
        else:
            sigma2 = float(self._sigma2_est)
        sigma2 = max(sigma2, 1e-12)

        # Update running estimate if we're using estimated mode
        if self.sigma is None and np.isfinite(cand.train_loss):
            # Exponentially decay toward the latest train_loss
            self._sigma2_est = 0.7 * self._sigma2_est + 0.3 * cand.train_loss

        # Description length (in bits)
        dl = description_length_bits(cand.tree)

        # Log-likelihood per sample (Gaussian; constants dropped)
        log_lik_per_sample = cand.train_loss / (2.0 * sigma2)

        # Complexity penalty per sample
        if self.penalty_mode == "naive_mdl":
            penalty_per_sample = dl / N
        elif self.penalty_mode == "recalibrated_mdl":
            penalty_per_sample = dl / math.sqrt(N)
        else:
            penalty_per_sample = 0.0

        mdl_fitness = log_lik_per_sample + penalty_per_sample

        return Candidate(
            tree=cand.tree,
            train_loss=cand.train_loss,  # Keep MSE for Pareto interpretability
            complexity=cand.complexity,
            fitness=mdl_fitness,
            born_gen=cand.born_gen,
        )


__all__ = [
    "description_length_bits",
    "GPWithMDLScoring",
]
