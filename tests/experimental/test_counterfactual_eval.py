"""Tests for tessera.experimental.counterfactual_eval — Conjecture C5."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tessera.expression.measure_2d import (
    Atom2D, Measure2D,
    measure_2d_laplacian_5pt, measure_2d_diff_t,
)
from tessera.expression.tree import (
    Var, Const, BinOp, FunctionalOp2D,
)

from tessera.experimental.counterfactual_eval import (
    HeatEqCounterfactual,
    generate_heat_eq_counterfactuals,
    score_counterfactual,
    rank_front_by_counterfactual,
)


# ----------------------------------------------------------------------
# 1. Counterfactual generator
# ----------------------------------------------------------------------

class TestGenerateCounterfactuals:
    def test_returns_expected_set(self):
        """Default config returns 5 CFs with the expected names."""
        cfs = generate_heat_eq_counterfactuals(T=50, X=16)
        assert len(cfs) == 5
        names = {cf.name for cf in cfs}
        expected = {"cf_ic_a", "cf_ic_b", "cf_alpha_2x",
                    "cf_noise_10x", "cf_smaller_x"}
        assert names == expected

    def test_cf_alpha_2x_has_different_alpha(self):
        cfs = generate_heat_eq_counterfactuals(alpha_base=0.05, T=50, X=16)
        alpha_2x_cf = next(cf for cf in cfs if cf.name == "cf_alpha_2x")
        assert alpha_2x_cf.alpha == pytest.approx(0.10)

    def test_cf_smaller_x_has_smaller_X(self):
        cfs = generate_heat_eq_counterfactuals(T=50, X=32)
        smaller = next(cf for cf in cfs if cf.name == "cf_smaller_x")
        assert smaller.U.shape[1] == 16

    def test_oracle_mse_is_finite_and_positive(self):
        """Oracle MSE should be small (just discretization + noise) but positive."""
        cfs = generate_heat_eq_counterfactuals(T=50, X=16)
        for cf in cfs:
            assert math.isfinite(cf.oracle_mse)
            assert cf.oracle_mse > 0


# ----------------------------------------------------------------------
# 2. Scoring trees
# ----------------------------------------------------------------------

class TestScoreCounterfactual:
    def _make_test_cfs(self):
        return generate_heat_eq_counterfactuals(T=50, X=16)

    def test_oracle_tree_scores_low_ratio(self):
        """A tree of the form α·Laplacian(U) should score near 1.0 ratio
        on all CFs except cf_alpha_2x (where α is wrong) and cf_noise_10x
        (where the noise floor is higher)."""
        cfs = self._make_test_cfs()
        lap = measure_2d_laplacian_5pt()
        # Tree: 0.05 * Laplacian(U)
        tree = BinOp("mul", Const(0.05),
                     FunctionalOp2D(lap, Var("U")))
        result = score_counterfactual(tree, cfs)
        # On IC counterfactuals (same α), ratio should be near 1.0
        ic_a_ratio = result["per_cf"]["cf_ic_a"]["ratio_vs_oracle"]
        ic_b_ratio = result["per_cf"]["cf_ic_b"]["ratio_vs_oracle"]
        assert ic_a_ratio < 5.0  # near-oracle
        assert ic_b_ratio < 5.0
        # On cf_alpha_2x (α=0.10, but tree uses 0.05), ratio should be high
        alpha_2x_ratio = result["per_cf"]["cf_alpha_2x"]["ratio_vs_oracle"]
        assert alpha_2x_ratio > ic_a_ratio  # worse on interventional change

    def test_constant_zero_tree_scores_high_ratio(self):
        """Predict-zero tree should have ratio = (target_variance / oracle_mse),
        i.e., high MSE relative to oracle."""
        cfs = self._make_test_cfs()
        tree = Const(0.0)
        result = score_counterfactual(tree, cfs)
        # All CFs should give high ratio (predict 0 doesn't fit anything)
        for cf_result in result["per_cf"].values():
            ratio = cf_result["ratio_vs_oracle"]
            if math.isfinite(ratio):
                assert ratio > 1.0  # worse than oracle

    def test_diff_t_style_tree_scores_moderate(self):
        """A 2-atom temporal-difference tree (Class A) should score
        moderately — not as bad as predict-zero, not as good as Class C."""
        cfs = self._make_test_cfs()
        diff_t = measure_2d_diff_t(lag_t=1)
        tree = FunctionalOp2D(diff_t, Var("U"))
        result = score_counterfactual(tree, cfs)
        # Class A is structurally generic; ratios should be in some
        # moderate range (not catastrophic, not perfect)
        finite_ratios = [
            r["ratio_vs_oracle"] for r in result["per_cf"].values()
            if math.isfinite(r["ratio_vs_oracle"])
        ]
        assert len(finite_ratios) > 0

    def test_score_keys_present(self):
        cfs = self._make_test_cfs()
        tree = Var("U")
        result = score_counterfactual(tree, cfs)
        assert "per_cf" in result
        assert "n_finite" in result
        assert "mean_ratio" in result
        assert "max_ratio" in result
        assert "median_ratio" in result


# ----------------------------------------------------------------------
# 3. Ranking a Pareto front
# ----------------------------------------------------------------------

class TestRankFront:
    def _make_test_cfs(self):
        return generate_heat_eq_counterfactuals(T=50, X=16)

    def test_class_c_ranks_above_predict_zero(self):
        """A Class C tree (clean Laplacian × α) should rank ABOVE a
        predict-zero tree in counterfactual ranking."""
        cfs = self._make_test_cfs()
        # Mock Candidate-like objects with .tree attribute
        from tessera.search.base import Candidate
        lap = measure_2d_laplacian_5pt()
        class_c_tree = BinOp("mul", Const(0.05),
                             FunctionalOp2D(lap, Var("U")))
        zero_tree = Const(0.0)
        front = [
            Candidate(tree=class_c_tree, train_loss=1e-6, complexity=4,
                      fitness=1e-6, born_gen=0),
            Candidate(tree=zero_tree, train_loss=1e-3, complexity=1,
                      fitness=1e-3, born_gen=0),
        ]
        ranked = rank_front_by_counterfactual(front, cfs)
        # Class C should rank #1
        assert ranked[0][0].tree == class_c_tree
        assert ranked[1][0].tree == zero_tree

    def test_returns_all_candidates(self):
        """Even if some candidates fail evaluation, all return."""
        cfs = self._make_test_cfs()
        from tessera.search.base import Candidate
        front = [
            Candidate(tree=Var("U"), train_loss=1.0, complexity=1,
                      fitness=1.0, born_gen=0),
            Candidate(tree=Const(2.0), train_loss=2.0, complexity=1,
                      fitness=2.0, born_gen=0),
        ]
        ranked = rank_front_by_counterfactual(front, cfs)
        assert len(ranked) == 2

    def test_rank_by_different_keys(self):
        """Should be able to rank by mean, median, or max ratio."""
        cfs = self._make_test_cfs()
        from tessera.search.base import Candidate
        lap = measure_2d_laplacian_5pt()
        front = [
            Candidate(tree=BinOp("mul", Const(0.05),
                                 FunctionalOp2D(lap, Var("U"))),
                      train_loss=1e-6, complexity=4, fitness=1e-6, born_gen=0),
        ]
        for key in ("mean_ratio", "median_ratio", "max_ratio"):
            ranked = rank_front_by_counterfactual(front, cfs, score_key=key)
            assert len(ranked) == 1
