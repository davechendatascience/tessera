"""ABC-style scoring with held-out evaluation (Conjecture C1-refined).

Provenance: C1 (refined) from
`docs/research/process_discovery_sr.md` §6.2 + §7.1.

Status: **FALSIFIED at β ∈ {0.1, 1.0}** on heat equation benchmark
        (2026-05-26). See `benchmarks/results/heat_equation_abc_mvp71.md`.
        ABC scoring eliminates Class B (good) but ALSO suppresses
        Class C discovery (bad). Pointwise MSE on held-out data is
        the better tool for this benchmark class. Module preserved
        for potential re-evaluation at β << 0.1 or different summary-
        statistic sets, or on genuinely-stochastic benchmarks (OU,
        Hawkes) where ABC's natural fit might match.

Conjecture C1-refined
---------------------
> "ABC-style summary-statistics scoring evaluated on a held-out
> slice or alternate trajectory suppresses Class B (template/reduce_*
> natural overfit) more than pointwise MSE on the same held-out data."

The discrimination: a model with low TRAIN MSE may have either (a)
captured the mechanism (Class C) or (b) fit TRAIN via trajectory-
specific tweaks (Class B). On held-out data:
- Pointwise MSE penalizes magnitude of error
- ABC-style summary-statistics distance penalizes structural mismatch

ABC may catch a different failure mode than pointwise: a tree that's
average-okay on hold-out but produces wildly wrong distributions of
values gets penalized by ABC but rewarded (mildly) by MSE.

Graduation criterion
--------------------
Mode comparison on heat-equation discovery (4+ budgets, 5+ seeds per
mode) shows: with ABC term active, ratio of (Class C found) /
(Class B found) is strictly higher than with MSE-only hold-out term
at the same compute budget. If the ratio doesn't change, ABC offers
no signal beyond pointwise hold-out CV.

Removal criterion
-----------------
ABC mode produces the SAME or WORSE class distribution than the
pointwise hold-out CV mode across all tested configurations. This
falsifies the additional-information hypothesis.

Initial commit: 2026-05-26
Last evaluation: 2026-05-26 (falsified at β ∈ {0.1, 1.0})

What's in this module
---------------------
    compute_summary_stats(arr) -> dict[str, float]
        Computes 6 summary statistics on a 1-D or 2-D ndarray:
        variance, mean absolute value, temporal/spatial lag-1 ACF,
        and column/row mean variances.

    abc_distance(stats_obs, stats_pred) -> float
        Normalized squared-relative-error distance between two summary-
        statistics dicts. Returns ~0 for matching stats, larger for
        further divergence. Scale-invariant.

    GPWithHoldout(GP)
        GP subclass that augments fitness with hold-out terms. Selection
        (tournament) sees the augmented fitness; Pareto front retains
        (cx, train_loss) semantics so the final output is interpretable.

Design notes
------------
Choosing to augment FITNESS rather than train_loss preserves the
Pareto-front contract (it's still in (cx, train_loss) space) while
biasing tournament selection toward candidates that also fit hold-out.

Over generations, the population fills with double-fit candidates.
The final Pareto front retains candidates that have low TRAIN MSE,
but the selection pressure has shaped WHICH low-TRAIN-MSE candidates
emerged.

The trade-off: this means the Pareto front's train_loss is unchanged
by hold-out scoring. To see the hold-out effect, the experiment must
inspect the TREES on the front, not just the loss numbers — which is
why the benchmark records class A/B/C labels.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tessera.search.base import Candidate
from tessera.search.gp import GP, GPConfig
from tessera.expression.tree import evaluate as eval_tree


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def compute_summary_stats(arr: np.ndarray) -> dict[str, float]:
    """Compute summary statistics on a 1-D or 2-D ndarray.

    For 2-D inputs (shape T × X, typical of dt_U fields), computes:
      - var_total          : variance of all interior values
      - mean_abs           : mean absolute value (scale stat)
      - acf_time_lag1      : Pearson corr(arr[1:], arr[:-1]) over interior
      - acf_space_lag1     : Pearson corr(arr[:, 1:], arr[:, :-1]) over interior
      - spatial_mean_var   : variance of per-column means
      - temporal_mean_var  : variance of per-row means

    For 1-D inputs, computes a subset (no spatial stats):
      - var_total, mean_abs, acf_time_lag1, temporal_mean_var

    NaN-robust: NaNs in input are masked from statistics. Returns
    statistics as float64 dict; -inf if a statistic is undefined
    (e.g., zero variance prevents ACF).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        # Pad to 2-D for unified handling
        return _stats_1d(arr)
    if arr.ndim == 2:
        return _stats_2d(arr)
    raise ValueError(f"compute_summary_stats expects 1-D or 2-D, got {arr.ndim}-D")


def _stats_1d(arr: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {k: 0.0 for k in
                ("var_total", "mean_abs", "acf_time_lag1", "temporal_mean_var")}
    masked = arr[finite]
    var = float(np.var(masked))
    return {
        "var_total": var,
        "mean_abs": float(np.mean(np.abs(masked))),
        "acf_time_lag1": _pearson_safe(arr[:-1], arr[1:]),
        "temporal_mean_var": 0.0,  # 1-D has no temporal-block structure
    }


def _stats_2d(arr: np.ndarray) -> dict[str, float]:
    # Interior: drop first/last row+col (consistent with the heat-eq
    # benchmark's interior convention; boundary rows have garbage from
    # discretization artifacts).
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        # Too small for interior; use the whole array
        interior = arr
    else:
        interior = arr[1:-1, 1:-1]

    finite = np.isfinite(interior)
    if not finite.any():
        return {k: 0.0 for k in (
            "var_total", "mean_abs", "acf_time_lag1", "acf_space_lag1",
            "spatial_mean_var", "temporal_mean_var",
        )}

    masked = interior[finite]
    var_total = float(np.var(masked))
    mean_abs = float(np.mean(np.abs(masked)))

    # Temporal lag-1 ACF: corr of interior rows[t] with rows[t-1]
    if interior.shape[0] >= 2:
        a_t = interior[:-1].reshape(-1)
        b_t = interior[1:].reshape(-1)
        acf_t = _pearson_safe(a_t, b_t)
    else:
        acf_t = 0.0

    # Spatial lag-1 ACF: corr of interior cols[x] with cols[x-1]
    if interior.shape[1] >= 2:
        a_s = interior[:, :-1].reshape(-1)
        b_s = interior[:, 1:].reshape(-1)
        acf_s = _pearson_safe(a_s, b_s)
    else:
        acf_s = 0.0

    # Per-column mean variance (spatial profile)
    col_means = np.nanmean(np.where(finite, interior, np.nan), axis=0)
    col_means = col_means[np.isfinite(col_means)]
    spatial_mean_var = float(np.var(col_means)) if col_means.size >= 2 else 0.0

    # Per-row mean variance (temporal profile)
    row_means = np.nanmean(np.where(finite, interior, np.nan), axis=1)
    row_means = row_means[np.isfinite(row_means)]
    temporal_mean_var = float(np.var(row_means)) if row_means.size >= 2 else 0.0

    return {
        "var_total": var_total,
        "mean_abs": mean_abs,
        "acf_time_lag1": acf_t,
        "acf_space_lag1": acf_s,
        "spatial_mean_var": spatial_mean_var,
        "temporal_mean_var": temporal_mean_var,
    }


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN-robust, returns 0.0 if degenerate."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2)))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denom)


def abc_distance(stats_obs: dict[str, float],
                 stats_pred: dict[str, float]) -> float:
    """Normalized squared-relative-error between two summary-statistics
    dicts.

    For each statistic shared between the two dicts:
        contribution = ((obs - pred) / max(|obs|, ε)) ²

    Returns the MEAN over statistics. Approximately:
      - 0       : perfect match
      - ~1      : ~100% relative error on average
      - >1      : substantial structural divergence

    The normalization makes the distance scale-invariant — variance
    in [1e-6 range] contributes equally to variance in [1e6 range].
    """
    shared = set(stats_obs.keys()) & set(stats_pred.keys())
    if not shared:
        return 0.0
    # Cap individual contributions to prevent overflow when a candidate
    # produces wildly off predictions (relative error can be ~1e200 if
    # the tree's stat is huge). Cap of 1e8 gives "very bad fit" without
    # numerical issues; we're not trying to distinguish 1e8 from 1e10
    # — both mean "rejected."
    MAX_CONTRIB = 1e8
    contributions = []
    for k in shared:
        s_obs = float(stats_obs[k])
        s_pred = float(stats_pred[k])
        if not (np.isfinite(s_obs) and np.isfinite(s_pred)):
            contributions.append(1.0)  # Penalize non-finite predictions
            continue
        scale = max(abs(s_obs), 1e-9)
        # Clip relative error magnitude before squaring to avoid overflow
        diff_rel = (s_obs - s_pred) / scale
        diff_rel = max(-1e4, min(1e4, diff_rel))  # bound at ±1e4 → squared ≤ 1e8
        contributions.append(min(diff_rel ** 2, MAX_CONTRIB))
    return float(np.mean(contributions))


# ---------------------------------------------------------------------
# GP subclass with hold-out scoring
# ---------------------------------------------------------------------

class GPWithHoldout(GP):
    """GP with augmented fitness from hold-out evaluation.

    Augments tournament-selection fitness with terms computed from a
    held-out (env, y_true) pair. Pareto front is unchanged structurally
    (still uses (cx, train_loss)), but SELECTION sees the augmented
    fitness — biasing breeding toward candidates that double-fit.

    Parameters
    ----------
    cfg : GPConfig
        Standard GP config; passed to base class.
    hold_env : dict[str, np.ndarray]
        Hold-out evaluation environment. Keys must overlap with the
        feature_names passed to .run().
    hold_y_true : np.ndarray
        Hold-out target.
    beta_abc : float
        Weight on ABC distance term. 0 disables.
    beta_hold_mse : float
        Weight on hold-out MSE term. 0 disables.
    loss_fn : callable | None
        Forwarded to GP base.

    Mode dispatch via betas
    -----------------------
    beta_abc = 0, beta_hold_mse = 0     →  identical to base GP (baseline)
    beta_abc = 0, beta_hold_mse > 0     →  hold-out CV with pointwise MSE
    beta_abc > 0, beta_hold_mse = 0     →  pure ABC-style scoring on hold
    beta_abc > 0, beta_hold_mse > 0     →  combined
    """

    def __init__(
        self,
        cfg: GPConfig,
        hold_env: dict[str, np.ndarray],
        hold_y_true: np.ndarray,
        *,
        beta_abc: float = 0.0,
        beta_hold_mse: float = 0.0,
        loss_fn: Any = None,
    ) -> None:
        super().__init__(cfg, loss_fn=loss_fn)
        self.hold_env = {k: np.asarray(v, dtype=np.float64) for k, v in hold_env.items()}
        self.hold_y_true = np.asarray(hold_y_true, dtype=np.float64)
        self.beta_abc = float(beta_abc)
        self.beta_hold_mse = float(beta_hold_mse)
        self.hold_stats = compute_summary_stats(self.hold_y_true)

    def _score(self, tree, env, y_true, born_gen):
        # Standard scoring first
        cand = super()._score(tree, env, y_true, born_gen)

        # If both beta weights are zero, nothing to add
        if self.beta_abc == 0.0 and self.beta_hold_mse == 0.0:
            return cand

        # Try evaluating on hold-out
        try:
            hold_pred = eval_tree(
                tree, self.hold_env, fill_warmup=self.cfg.fill_warmup,
            )
            hold_pred = np.asarray(hold_pred, dtype=np.float64)
        except Exception:
            return cand

        # Reject candidates whose hold-out prediction has wrong shape
        # or non-finite values — they fail the experiment automatically
        if hold_pred.shape != self.hold_y_true.shape:
            return cand
        finite_frac = float(np.isfinite(hold_pred).sum()) / hold_pred.size
        if finite_frac < self.cfg.min_valid_frac:
            return cand

        extra_fitness = 0.0

        if self.beta_hold_mse > 0.0:
            # Mask NaNs for the MSE computation
            mask = np.isfinite(hold_pred) & np.isfinite(self.hold_y_true)
            if mask.any():
                hold_mse = float(np.mean(
                    (hold_pred[mask] - self.hold_y_true[mask]) ** 2
                ))
                extra_fitness += self.beta_hold_mse * hold_mse

        if self.beta_abc > 0.0:
            pred_stats = compute_summary_stats(hold_pred)
            abc = abc_distance(self.hold_stats, pred_stats)
            extra_fitness += self.beta_abc * abc

        if extra_fitness == 0.0:
            return cand

        # Return new Candidate with augmented fitness; preserve train_loss
        # and complexity so the Pareto front stays interpretable.
        return Candidate(
            tree=cand.tree,
            train_loss=cand.train_loss,
            complexity=cand.complexity,
            fitness=cand.fitness + extra_fitness,
            born_gen=cand.born_gen,
        )


__all__ = [
    "compute_summary_stats",
    "abc_distance",
    "GPWithHoldout",
]
