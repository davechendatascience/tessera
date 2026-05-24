"""tessera.combinatorics — algorithm patterns for combinatorial search.

This subpackage hosts canonical reference implementations of algorithms
from the Knuth / classical-algorithms tradition that may be useful in
SR-side work. See `docs/research/dancing_links_for_sr.md` for the
philosophy and the implementation-budget tracking that gates additions
to this subpackage.

Available modules
-----------------
    dancing_links    — Knuth Algorithm X via Dancing Links (exact-cover
                       solver with O(1) cover/uncover; N-queens wrapper)

Adding to this subpackage
-------------------------
Per the research-note commitment: any new module must be justified by
a concrete SR-side use case, not just "it would be useful." Update the
budget table in `docs/research/dancing_links_for_sr.md` §4 with the
post-addition footprint and a one-paragraph justification.
"""
from __future__ import annotations

from .dancing_links import (
    ExactCoverMatrix,
    solve_exact_cover,
    count_exact_covers,
    nqueens_solutions,
    nqueens_count,
)

__all__ = [
    "ExactCoverMatrix",
    "solve_exact_cover",
    "count_exact_covers",
    "nqueens_solutions",
    "nqueens_count",
]
