"""Knuth Algorithm X via Dancing Links (DLX).

Canonical reference implementation of Knuth's *Dancing Links* paper
(arXiv:cs/0011047, 2000; TAOCP Vol 4B §7.2.2). Solves the **exact-cover
problem**: given a 0/1 matrix, find subsets of rows that, together, have
exactly one `1` in each (primary) column.

The trick that gives DLX its name: each cell in the toroidal doubly-
linked list is removed by `x.left.right = x.right; x.right.left =
x.left` and restored by the same statements in reverse — `x.left.right
= x; x.right.left = x`. The node "leaves and returns to" its position
without ever being copied. **Backtracking has O(1) undo cost.**

Why this lives in tessera
-------------------------
See `docs/research/dancing_links_for_sr.md` for the SR-side rationale.
Short version: the incremental-state pattern (state mutations with
natural inverses) generalises to the SR per-node-cache + dirty-flag
work planned in `docs/research/analytical_delta_loss.md` §4.3. This
module is a canonical reference; SR adaptation is a separate future
ship.

Public API
----------
    ExactCoverMatrix
        Builds the toroidal linked-list representation from a binary
        matrix (Python lists or 2-D bool / int arrays). Manages cover
        and uncover.

    solve_exact_cover(matrix, *, primary=None)
        Generator yielding each exact cover as a list of row indices.
        `primary` specifies how many of the leftmost columns are
        primary (must-cover); the rest are secondary (at-most-once).
        Default: all columns are primary.

    count_exact_covers(matrix, *, primary=None) -> int
        Convenience: counts exact covers.

    nqueens_solutions(n) -> list[tuple[int, ...]]
        Returns all N-queens solutions as column-per-row tuples.
        Uses DLX with secondary columns for diagonals.

    nqueens_count(n) -> int
        Convenience: number of N-queens solutions.

Complexity
----------
Memory: O(K) where K is the number of 1s in the input matrix.
Time per cover/uncover: O(rows · cols touched), with O(1) per link
manipulation. The bulk of time is the depth-first search itself,
which is problem-specific.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# 1. Node infrastructure
# ----------------------------------------------------------------------

class _Node:
    """Single cell in the toroidal doubly-linked list.

    Each cell carries four directional pointers, a back-pointer to its
    column header, and the row index from the original input matrix
    (so the solver can report cover sets in input coordinates).

    Column headers are also instances of this class — they sit in the
    'header row' and additionally track `size` (number of cells in the
    column) and `name` (the column's identity, used for picking
    minimum-size column during search).
    """

    __slots__ = ("left", "right", "up", "down", "column", "row", "size", "name")

    def __init__(self) -> None:
        # Self-link initially; the matrix builder will splice them in.
        self.left: "_Node" = self
        self.right: "_Node" = self
        self.up: "_Node" = self
        self.down: "_Node" = self
        # Column pointer; column headers point to themselves.
        self.column: "_Node" = self
        # Row index (only meaningful for non-header cells).
        self.row: int = -1
        # Column-header-only fields:
        self.size: int = 0
        self.name: object = None


def _link_left_right(a: _Node, b: _Node) -> None:
    """Splice b to the right of a in horizontal ring."""
    a.right = b
    b.left = a


def _link_up_down(a: _Node, b: _Node) -> None:
    """Splice b below a in vertical ring."""
    a.down = b
    b.up = a


# ----------------------------------------------------------------------
# 2. ExactCoverMatrix
# ----------------------------------------------------------------------

class ExactCoverMatrix:
    """Toroidal doubly-linked-list representation of an exact-cover
    problem.

    Parameters
    ----------
    matrix : 2-D iterable of bool / int
        Rows of the constraint matrix. Each `matrix[i][j]` truthy means
        "row i covers column j."
    primary : int or None
        Number of leftmost columns that are PRIMARY (must be covered
        exactly once). The remaining columns are SECONDARY (may be
        covered at most once). Default None = all columns primary.

        Secondary columns are needed for problems like N-queens where
        diagonals must be coverable AT MOST once but don't all need
        to be hit by exactly one queen. They are NOT included in the
        header row that the search iterates over.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[int]] | np.ndarray,
        *,
        primary: Optional[int] = None,
    ) -> None:
        mat = np.asarray(matrix, dtype=np.int8)
        if mat.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {mat.shape}")
        n_rows, n_cols = mat.shape
        if primary is None:
            primary = n_cols
        if not (0 <= primary <= n_cols):
            raise ValueError(
                f"primary must be in [0, {n_cols}], got {primary}"
            )

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_primary = primary

        # Root sentinel — head of the column-header ring (primary cols only).
        self.root = _Node()
        self.root.name = "__root__"

        # Build column headers. PRIMARY headers are inserted into the
        # root ring; SECONDARY headers exist but stay out of it (they're
        # reachable only via their up/down link from cells).
        columns: list[_Node] = []
        prev = self.root
        for j in range(n_cols):
            col = _Node()
            col.column = col
            col.name = j
            col.up = col
            col.down = col
            if j < primary:
                _link_left_right(prev, col)
                prev = col
            else:
                col.left = col
                col.right = col
            columns.append(col)
        if primary > 0:
            _link_left_right(prev, self.root)
        # else: root remains self-linked; the solver will see an empty header
        # ring and report the trivial empty cover.

        # Fill in cells row by row, splicing into both the column ring and
        # the row ring.
        for i in range(n_rows):
            row_start: Optional[_Node] = None
            for j in range(n_cols):
                if not mat[i, j]:
                    continue
                cell = _Node()
                cell.row = i
                cell.column = columns[j]
                # Append vertically: insert above the column header,
                # which is equivalent to appending at the bottom.
                col = columns[j]
                _link_up_down(col.up, cell)
                _link_up_down(cell, col)
                col.size += 1
                # Append horizontally in row ring.
                if row_start is None:
                    cell.left = cell
                    cell.right = cell
                    row_start = cell
                else:
                    _link_left_right(row_start.left, cell)
                    _link_left_right(cell, row_start)

        self._columns = columns  # kept for debugging / introspection

    # -- the dance ------------------------------------------------------

    @staticmethod
    def _cover(col: _Node) -> None:
        """Remove `col` from the header ring and unlink every row that
        has a cell in `col` from every other column it touches.

        O(rows-in-col · cells-per-row) work. ALL of it via 4 pointer
        writes per cell — no allocations, no copies."""
        col.right.left = col.left
        col.left.right = col.right
        i = col.down
        while i is not col:
            j = i.right
            while j is not i:
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1
                j = j.right
            i = i.down

    @staticmethod
    def _uncover(col: _Node) -> None:
        """Restore in reverse order. Crucially: walk up/left, not
        down/right — undoing in the OPPOSITE order than `cover` did."""
        i = col.up
        while i is not col:
            j = i.left
            while j is not i:
                j.column.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        col.right.left = col
        col.left.right = col

    # -- search ---------------------------------------------------------

    def _choose_column(self) -> Optional[_Node]:
        """Knuth's S heuristic: pick the (primary) column with minimum
        size. Reduces branching factor. Returns None if no primary
        columns remain (= a cover is complete)."""
        best: Optional[_Node] = None
        best_size = float("inf")
        c = self.root.right
        while c is not self.root:
            if c.size < best_size:
                best = c
                best_size = c.size
            c = c.right
        return best

    def solve(self) -> Iterator[list[int]]:
        """Yield each exact cover as a list of row indices from the
        original matrix.

        Implementation is iterative on the stack via recursion; for the
        problem sizes typical of SR experiments and the planned N-queens
        ≤ 12, recursion depth is bounded and safe.
        """
        partial: list[int] = []
        yield from self._search(partial)

    def _search(self, partial: list[int]) -> Iterator[list[int]]:
        col = self._choose_column()
        if col is None:
            # No primary columns left — solution found.
            yield list(partial)
            return
        if col.size == 0:
            # Dead branch — primary column has no row to cover it.
            return
        self._cover(col)
        r = col.down
        while r is not col:
            partial.append(r.row)
            # Cover every other column this row touches.
            j = r.right
            while j is not r:
                self._cover(j.column)
                j = j.right
            yield from self._search(partial)
            # Uncover in reverse.
            j = r.left
            while j is not r:
                self._uncover(j.column)
                j = j.left
            partial.pop()
            r = r.down
        self._uncover(col)


# ----------------------------------------------------------------------
# 3. Public convenience functions
# ----------------------------------------------------------------------

def solve_exact_cover(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    *,
    primary: Optional[int] = None,
) -> Iterator[list[int]]:
    """Yield each exact cover of `matrix` as a list of row indices.

    Convenience wrapper over `ExactCoverMatrix(...).solve()`.
    """
    return ExactCoverMatrix(matrix, primary=primary).solve()


def count_exact_covers(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    *,
    primary: Optional[int] = None,
) -> int:
    """Count exact covers without materialising them. Faster than
    list(solve_exact_cover(...)) when only the count is needed."""
    count = 0
    for _ in ExactCoverMatrix(matrix, primary=primary).solve():
        count += 1
    return count


# ----------------------------------------------------------------------
# 4. N-queens demo
# ----------------------------------------------------------------------

def _nqueens_matrix(n: int) -> tuple[np.ndarray, int]:
    """Encode N-queens as exact-cover with secondary columns.

    Columns (in order):
      [0 .. n-1]            primary: rank (row) — one queen per row
      [n .. 2n-1]           primary: file (col) — one queen per column
      [2n .. 4n-2]          secondary: anti-diagonals r + c
      [4n-1 .. 6n-3]        secondary: main diagonals r - c + (n-1)

    Each potential queen placement (r, c) is a row with exactly four
    1s — one per column it occupies. Diagonals are SECONDARY because
    they can be unused (no queen on them is fine).

    Returns
    -------
    matrix : (n², 6n-2) ndarray of int8
    primary_count : int   — first 2n columns are primary
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    n_diag = 2 * n - 1 if n > 0 else 0
    n_cols = 2 * n + 2 * n_diag
    n_rows = n * n
    mat = np.zeros((n_rows, n_cols), dtype=np.int8)
    for r in range(n):
        for c in range(n):
            row_idx = r * n + c
            mat[row_idx, r] = 1                          # rank
            mat[row_idx, n + c] = 1                      # file
            mat[row_idx, 2 * n + (r + c)] = 1            # anti-diag
            mat[row_idx, 2 * n + n_diag + (r - c + n - 1)] = 1  # main diag
    return mat, 2 * n


def nqueens_solutions(n: int) -> list[tuple[int, ...]]:
    """Return all N-queens solutions.

    Each solution is a length-n tuple where `solution[r] = c` means
    "the queen on rank r is in file c."

    >>> sorted(nqueens_solutions(4))
    [(1, 3, 0, 2), (2, 0, 3, 1)]
    >>> len(nqueens_solutions(8))
    92
    """
    if n == 0:
        return [()]
    mat, primary = _nqueens_matrix(n)
    out: list[tuple[int, ...]] = []
    for cover in ExactCoverMatrix(mat, primary=primary).solve():
        # Each row index in the cover encodes (r, c) = divmod(idx, n).
        # Sort by rank so the output is a column-per-rank tuple.
        cells = sorted(divmod(idx, n) for idx in cover)
        out.append(tuple(c for _, c in cells))
    return out


def nqueens_count(n: int) -> int:
    """Count N-queens solutions without materialising arrangements.

    >>> nqueens_count(8)
    92
    >>> nqueens_count(0)
    1
    """
    if n == 0:
        return 1
    mat, primary = _nqueens_matrix(n)
    return count_exact_covers(mat, primary=primary)
