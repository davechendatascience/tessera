"""Per-complexity best-ever Candidate store, immune to mutation drift.

A classic GP failure mode: the population discovers a great cx=6
expression at gen 12, then drifts to cx=14 "improvements" that
overfit, losing the cx=6 winner. Without a Hall of Fame, the final
Pareto front only sees the population at the LAST generation; the
intermediate cx=6 gem is gone.

PySR ships this — `HallOfFameMember` in `SymbolicRegression.jl/src/HallOfFame.jl`.
Tessera's version is a small `dict[int, Candidate]` that every searcher
(GP, SA, Random) updates as it evaluates candidates, then returns
`pareto_front(hof.candidates())` at the end of `.run()`.

Design notes
------------
- Keyed by `Candidate.complexity` (the integer cx as computed by
  `tessera.expression.tree.complexity` AFTER simplification).
- Update is monotonic: a candidate replaces the incumbent at its cx
  iff its `train_loss` is strictly lower (with a small ε tolerance to
  avoid floating-point thrash on identical retrains).
- `merge(other)` combines two HoFs — useful when running several
  searchers and unioning their best discoveries.
"""
from __future__ import annotations
from .base import Candidate


_LOSS_IMPROVEMENT_EPSILON = 1e-12


class HallOfFame:
    """Per-complexity best-ever Candidate store.

    Usage
    -----
        hof = HallOfFame()
        hof.update(candidate)              # add one
        hof.update_many(candidates)        # add many
        front = hof.pareto_front()         # extract Pareto front
        merged = hof_a.merge(hof_b)        # combine two HoFs

    Iteration yields candidates ordered by complexity ascending:
        for c in hof: ...
    """

    def __init__(self) -> None:
        self._best_per_cx: dict[int, Candidate] = {}
        # Track how often a slot is "successfully" updated — useful
        # diagnostic for whether the HoF is actually saving discoveries
        # or just rubber-stamping the current population.
        self.n_updates: int = 0

    def update(self, candidate: Candidate) -> bool:
        """Add `candidate` to the HoF.

        Returns True iff this candidate strictly improved the best at
        its complexity (i.e., new train_loss < incumbent train_loss -
        epsilon, or no incumbent yet).
        """
        cx = candidate.complexity
        incumbent = self._best_per_cx.get(cx)
        if incumbent is None or \
           candidate.train_loss < incumbent.train_loss - _LOSS_IMPROVEMENT_EPSILON:
            self._best_per_cx[cx] = candidate
            self.n_updates += 1
            return True
        return False

    def update_many(self, candidates) -> int:
        """Add multiple candidates; return the count of successful
        improvements (each at most once per complexity)."""
        improved = 0
        for c in candidates:
            if self.update(c):
                improved += 1
        return improved

    def candidates(self) -> list[Candidate]:
        """All current Hall-of-Fame entries, ordered by complexity ascending."""
        return [self._best_per_cx[cx] for cx in sorted(self._best_per_cx)]

    def pareto_front(self) -> list[Candidate]:
        """Pareto front over the HoF entries.

        Returns a list sorted by complexity ascending where train_loss
        is monotone non-increasing — same semantics as
        `tessera.search.pareto.pareto_front`.

        Note: the HoF can contain a Pareto-dominated entry (best at cx=8
        may be DOMINATED by best at cx=6 if some cx=6 entry has lower
        loss). The Pareto-front extraction drops these.
        """
        from .pareto import pareto_front
        return pareto_front(self.candidates())

    def merge(self, other: "HallOfFame") -> "HallOfFame":
        """Combine two HoFs. Returns a new HoF whose entry at each cx is
        the better of the two inputs' entries (or the only one)."""
        merged = HallOfFame()
        for c in self.candidates():
            merged.update(c)
        for c in other.candidates():
            merged.update(c)
        return merged

    def best(self) -> Candidate | None:
        """Globally-best entry by `train_loss`. None if empty."""
        if not self._best_per_cx:
            return None
        return min(self._best_per_cx.values(), key=lambda c: c.train_loss)

    def __len__(self) -> int:
        return len(self._best_per_cx)

    def __iter__(self):
        return iter(self.candidates())

    def __contains__(self, cx: int) -> bool:
        return cx in self._best_per_cx

    def __repr__(self) -> str:
        if not self._best_per_cx:
            return "HallOfFame(empty)"
        best = self.best()
        return (
            f"HallOfFame(|F|={len(self._best_per_cx)}, "
            f"cx_range=[{min(self._best_per_cx)}, {max(self._best_per_cx)}], "
            f"best=cx{best.complexity}/loss={best.train_loss:.4g})"
        )
