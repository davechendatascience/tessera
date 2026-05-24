"""Tests for tessera.combinatorics.dancing_links — canonical DLX.

Covers:
  - Empty / trivial cases.
  - Simple exact-cover matrices with known solution sets.
  - Knuth's paper example (the canonical reference).
  - N-queens enumeration for n in {0, 1, 4, 5, 6, 8} — known counts.
  - Secondary-column behaviour (must accept "uncovered" secondaries).
  - Input validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.combinatorics.dancing_links import (
    ExactCoverMatrix,
    count_exact_covers,
    nqueens_count,
    nqueens_solutions,
    solve_exact_cover,
)


# ----------------------------------------------------------------------
# 1. Trivial cases
# ----------------------------------------------------------------------

class TestTrivial:
    def test_empty_matrix_zero_cols_is_trivially_covered(self):
        """A 0x0 matrix has exactly one cover: the empty set."""
        mat = np.zeros((0, 0), dtype=np.int8)
        solutions = list(solve_exact_cover(mat))
        assert solutions == [[]]

    def test_no_rows_one_primary_col_no_solution(self):
        mat = np.zeros((0, 1), dtype=np.int8)
        assert count_exact_covers(mat) == 0

    def test_single_row_covers_single_col(self):
        mat = np.array([[1]], dtype=np.int8)
        solutions = list(solve_exact_cover(mat))
        assert solutions == [[0]]


# ----------------------------------------------------------------------
# 2. Small known exact-cover problems
# ----------------------------------------------------------------------

class TestKnownProblems:
    def test_unique_solution_identity(self):
        """Identity matrix — exactly one cover, using every row."""
        n = 5
        mat = np.eye(n, dtype=np.int8)
        solutions = list(solve_exact_cover(mat))
        assert len(solutions) == 1
        # Each row is needed; cover is some permutation of 0..n-1.
        assert sorted(solutions[0]) == list(range(n))

    def test_two_alternative_covers(self):
        """Matrix where row 0 alone or rows 1+2 cover all columns."""
        mat = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 1],
        ], dtype=np.int8)
        solutions = sorted(sorted(s) for s in solve_exact_cover(mat))
        assert solutions == [[0], [1, 2]]

    def test_no_solution(self):
        """No subset of rows covers column 2."""
        mat = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.int8)
        assert count_exact_covers(mat) == 0

    def test_knuth_paper_example(self):
        """The canonical example from Knuth (cs/0011047, §3):
            rows 1, 4, 5 form the unique exact cover.
        Encoding the paper's matrix directly:
        """
        mat = np.array([
            #  A  B  C  D  E  F  G
            [0, 0, 1, 0, 1, 1, 0],  # row 0
            [1, 0, 0, 1, 0, 0, 1],  # row 1
            [0, 1, 1, 0, 0, 1, 0],  # row 2
            [1, 0, 0, 1, 0, 0, 0],  # row 3
            [0, 1, 0, 0, 0, 0, 1],  # row 4
            [0, 0, 0, 1, 1, 0, 1],  # row 5
        ], dtype=np.int8)
        solutions = list(solve_exact_cover(mat))
        # Unique cover: rows 0, 3, 4. (Knuth's paper enumerates this.)
        assert len(solutions) == 1
        assert sorted(solutions[0]) == [0, 3, 4]


# ----------------------------------------------------------------------
# 3. Repeated-solve correctness (cover/uncover invariant)
# ----------------------------------------------------------------------

class TestCoverUncoverInvariant:
    def test_matrix_state_restored_after_full_enumeration(self):
        """After enumerating all solutions, every column's `size`
        should equal its original row count — the cover/uncover dance
        must leave the matrix exactly as it was."""
        mat = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        ec = ExactCoverMatrix(mat)
        # Snapshot initial column sizes.
        original_sizes = [c.size for c in ec._columns]
        # Enumerate all (drain the generator).
        _ = list(ec.solve())
        # Sizes should be back to original.
        post_sizes = [c.size for c in ec._columns]
        assert post_sizes == original_sizes

    def test_repeated_solve_yields_same_count(self):
        """A second call to solve() on the same matrix yields the
        same number of solutions (invariant after first enumeration)."""
        mat = np.array([
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
        ], dtype=np.int8)
        ec = ExactCoverMatrix(mat)
        count1 = sum(1 for _ in ec.solve())
        count2 = sum(1 for _ in ec.solve())
        assert count1 == count2


# ----------------------------------------------------------------------
# 4. N-queens — known reference counts
# ----------------------------------------------------------------------

class TestNQueens:
    @pytest.mark.parametrize("n, expected", [
        (0, 1),
        (1, 1),
        (2, 0),
        (3, 0),
        (4, 2),
        (5, 10),
        (6, 4),
        (7, 40),
        (8, 92),
        (9, 352),
    ])
    def test_nqueens_count(self, n, expected):
        """Sequence OEIS A000170 — number of N-queens solutions.
        Our DLX implementation must reproduce it for n ≤ 9 quickly."""
        assert nqueens_count(n) == expected

    def test_nqueens_solutions_are_valid(self):
        """Every reported solution must be a valid queen arrangement:
        no two queens share row, column, or diagonal."""
        for n in [4, 5, 6, 8]:
            solutions = nqueens_solutions(n)
            for sol in solutions:
                self._assert_valid_arrangement(sol, n)
            # Count consistency
            assert len(solutions) == nqueens_count(n)

    def test_nqueens_8_canonical_count(self):
        """The textbook reference: N=8 has exactly 92 solutions."""
        assert nqueens_count(8) == 92

    def test_nqueens_4_explicit_solutions(self):
        """N=4 has exactly the two known solutions."""
        solutions = sorted(nqueens_solutions(4))
        assert solutions == [(1, 3, 0, 2), (2, 0, 3, 1)]

    @staticmethod
    def _assert_valid_arrangement(sol, n):
        assert len(sol) == n
        # Columns must all be distinct.
        assert len(set(sol)) == n
        # No diagonal conflicts.
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                c1, c2 = sol[r1], sol[r2]
                assert abs(c1 - c2) != abs(r1 - r2), (
                    f"diagonal conflict in arrangement {sol} at "
                    f"rows {r1},{r2}"
                )


# ----------------------------------------------------------------------
# 5. Input validation
# ----------------------------------------------------------------------

class TestValidation:
    def test_1d_matrix_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            ExactCoverMatrix(np.array([1, 0, 1]))

    def test_invalid_primary_count_raises(self):
        mat = np.eye(3, dtype=np.int8)
        with pytest.raises(ValueError, match="primary"):
            ExactCoverMatrix(mat, primary=10)
        with pytest.raises(ValueError, match="primary"):
            ExactCoverMatrix(mat, primary=-1)

    def test_negative_n_for_nqueens_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            nqueens_count(-1)


# ----------------------------------------------------------------------
# 6. Secondary-column semantics
# ----------------------------------------------------------------------

class TestSecondaryColumns:
    def test_secondary_can_be_uncovered(self):
        """With one primary column and one secondary, the cover that
        uses a row touching ONLY the primary is valid even though the
        secondary stays uncovered."""
        # Row 0: covers primary only.
        # Row 1: covers both primary and secondary.
        mat = np.array([
            [1, 0],
            [1, 1],
        ], dtype=np.int8)
        # All-primary: both rows can independently cover primary, but
        # only row 0 leaves col 1 uncovered, which is REQUIRED if both
        # primary. So with primary=2, only row 1 is valid (touches both).
        solutions_all_primary = sorted(
            sorted(s) for s in solve_exact_cover(mat, primary=2)
        )
        assert solutions_all_primary == [[1]]
        # With primary=1, col 1 is secondary — row 0 alone (leaving
        # secondary unused) is valid; row 1 also valid (touches secondary
        # once). Two solutions.
        solutions_one_primary = sorted(
            sorted(s) for s in solve_exact_cover(mat, primary=1)
        )
        assert solutions_one_primary == [[0], [1]]

    def test_secondary_cannot_be_double_covered(self):
        """If two rows both touch the same secondary column, they
        can't both appear in a cover."""
        # Both rows cover primary col 0. Both also touch secondary col 1.
        # Each row INDIVIDUALLY covers primary; choosing both would
        # double-cover the primary AND the secondary, invalid on both
        # counts. Choosing one is fine.
        mat = np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8)
        solutions = sorted(
            sorted(s) for s in solve_exact_cover(mat, primary=1)
        )
        assert solutions == [[0], [1]]
