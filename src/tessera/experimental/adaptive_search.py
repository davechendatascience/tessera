"""Adaptive mutation weights via residual diagnostics (Conjecture C6).

Provenance: C6 from `docs/research/process_discovery_sr.md` §6.6.
Theoretical pre-analysis: `docs/research/c6_residual_diagnostics_analysis.md`.

Status: **VALIDATED-AS-PREDICTED** on heat equation (2026-05-26).
        Pre-analysis predicted adaptive ≈ baseline; experiment confirmed
        EXACTLY: same Class C count (1/5), same Class B count (1/5),
        3/5 seeds produced bit-identical results to baseline. The
        diagnostic→corrective mapping problem played out as predicted.
        Module preserved as documentation of the mapping-problem finding.
        See `benchmarks/results/heat_equation_adaptive_mvp_c6.md`.

The pre-analysis predicted
--------------------------

Generic residual-driven adaptive mutation weights are unlikely to
materially help Class C discovery on heat equation because:

  1. The diagnostic→corrective mapping problem is unsolved without
     domain knowledge
  2. Specific mappings essentially encode the answer (which defeats
     the "adaptive" premise)

Predicted outcome: adaptive ≈ baseline. Class C count similar to or
slightly below baseline.

This module implements the minimal generic version (operator-usage
adaptation) to test that prediction. It is NOT a "double down on
the right answer" hack; it's the cleanest version of adaptive that
doesn't assume domain knowledge.

What this module implements
---------------------------

GPWithAdaptiveSearch:
  Every K generations, examines the current Pareto front to count
  which UN_OPS were used most. Adjusts UN_OP_WEIGHTS (module-level
  in tessera.expression.mutation) to bias future random_tree calls
  toward frequently-used operators.

  The adjustment is multiplicative on the BASELINE weights (which
  include the reduce_* downweight as default). So adaptation tunes
  weights relative to baseline, not from scratch.

  Around the GP run, the module's UN_OP_WEIGHTS is saved + restored
  to prevent leakage between runs.

Graduation criterion
--------------------
If adaptive consistently produces Class C at strictly higher rate
than baseline at the same compute budget (≥2/5 vs 1/5 baseline).

Removal criterion
-----------------
If adaptive produces results indistinguishable from baseline (the
predicted outcome) — module is preserved as documentation of the
mapping-problem finding.

Initial commit: 2026-05-26
Last evaluation: never
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tessera.search.base import Candidate
from tessera.search.gp import GP, GPConfig
from tessera.expression.tree import UnOp, iter_subtrees
from tessera.expression import mutation as _mut


class GPWithAdaptiveSearch(GP):
    """GP with adaptive UN_OP_WEIGHTS based on operator usage in front.

    Parameters
    ----------
    cfg : GPConfig
    adapt_every : int
        Number of generations between adaptation events. Default 10.
    adapt_strength : float
        Fraction of weight that can shift from baseline toward usage
        frequency. 0 = no adaptation; 1 = full replacement with
        usage-weighted distribution. Default 0.5 (moderate).
    floor_weight : float
        Minimum weight any operator can receive. Prevents complete
        elimination. Default 0.01.
    loss_fn : callable | None
        Forwarded to GP base.
    """

    def __init__(
        self,
        cfg: GPConfig,
        *,
        adapt_every: int = 10,
        adapt_strength: float = 0.5,
        floor_weight: float = 0.01,
        loss_fn: Any = None,
    ) -> None:
        super().__init__(cfg, loss_fn=loss_fn)
        self.adapt_every = int(adapt_every)
        self.adapt_strength = float(adapt_strength)
        self.floor_weight = float(floor_weight)
        # Diagnostic state
        self.adapt_history: list[dict] = []
        self._original_un_op_weights: dict[str, float] | None = None

    def run(self, env, y_true, feature_names=None):
        # Save the module-level UN_OP_WEIGHTS so we can restore after
        self._original_un_op_weights = dict(_mut.UN_OP_WEIGHTS)
        try:
            return super().run(env, y_true, feature_names=feature_names)
        finally:
            # Restore baseline (prevent leakage to subsequent GP runs)
            _mut.UN_OP_WEIGHTS.clear()
            _mut.UN_OP_WEIGHTS.update(self._original_un_op_weights)

    def _breed(self, pop, feature_names, env, y_true, gen):
        # Adapt every K generations (skip gen 0)
        if (
            gen > 0
            and gen % self.adapt_every == 0
            and self._current_front
        ):
            self._adapt_weights(self._current_front, gen)
        return super()._breed(pop, feature_names, env, y_true, gen)

    def _adapt_weights(self, front, gen: int) -> None:
        """Adjust UN_OP_WEIGHTS based on operator usage in front."""
        # Count UN_OP usage across all candidates in the front
        un_counts: dict[str, int] = {op: 0 for op in _mut.UN_OPS}
        total_un = 0
        for cand in front:
            for sub in iter_subtrees(cand.tree):
                if isinstance(sub, UnOp) and sub.op in un_counts:
                    un_counts[sub.op] += 1
                    total_un += 1

        if total_un == 0:
            # No UN_OPS in the front; nothing to adapt. Record the
            # snapshot anyway as a diagnostic — knowing the front
            # has no UN_OPS is itself informative.
            self.adapt_history.append({
                "gen": gen,
                "front_size": len(front),
                "total_un_ops_used": 0,
                "weights_snapshot": dict(_mut.UN_OP_WEIGHTS),
                "no_op": True,
            })
            return

        n_ops = len(_mut.UN_OPS)
        uniform_freq = 1.0 / n_ops if n_ops > 0 else 0.0

        # Compute new weights
        new_weights = {}
        for op in _mut.UN_OPS:
            base = self._original_un_op_weights.get(op, 1.0)
            obs_freq = un_counts.get(op, 0) / total_un
            # Adjustment factor: 1.0 means no adaptation; ratio > 1
            # means boost; ratio < 1 means suppress
            if uniform_freq > 0:
                freq_ratio = obs_freq / uniform_freq
            else:
                freq_ratio = 1.0
            # Blend: (1-strength) * 1.0 + strength * freq_ratio
            adjustment = (
                (1.0 - self.adapt_strength)
                + self.adapt_strength * freq_ratio
            )
            new_weights[op] = max(self.floor_weight, base * adjustment)

        # Apply adjustments to module-level dict
        _mut.UN_OP_WEIGHTS.update(new_weights)

        # Record diagnostic snapshot
        self.adapt_history.append({
            "gen": gen,
            "front_size": len(front),
            "total_un_ops_used": total_un,
            "weights_snapshot": dict(new_weights),
        })


__all__ = ["GPWithAdaptiveSearch"]
