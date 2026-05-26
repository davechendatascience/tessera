# Research note: Dancing Links (Knuth Algorithm X) as a side track

**Status update (2026-05-26 → `docs/shipped/`):** Moved from `docs/research/`. Canonical Knuth Algorithm X (DLX) **SHIPPED** in `src/tessera/combinatorics/dancing_links.py` with full unit tests. The speculative SR-integration applications (exact-cover formulations of SR sub-problems) remain unimplemented but are not blocking; they could become research notes again if a concrete application emerges.

**Status:** ▷ IN PROGRESS (canonical implementation only). The SR-integration applications below remain ? RESEARCH.

**Provenance:** user (2026-05-24), after Phase 1 of §2.3 shipped: *"I propose also implementing the dancing links algorithm at the side. We'll have to track our implementation as the implementation complexity grows. Afterwards, you can continue on the phase 2 and 3."*

The earlier note [`analytical_delta_loss.md`](analytical_delta_loss.md) §2 named Knuth's Dancing Links (TAOCP Vol 4B §7.2.2) as the canonical analog for Regime A (incremental partial evaluation). This note carves off the **algorithm itself** as a small, self-contained side track — independent of the §2.3 sufficient-statistic mainline — and explicitly tracks its implementation footprint so we know what we're committing to maintain.

---

## 1. What Dancing Links is

Knuth's Algorithm X is a recursive, nondeterministic, depth-first, backtracking algorithm for the **exact-cover problem**: given a 0/1 matrix, find subsets of rows that, together, have exactly one `1` in each column. Many combinatorial problems reduce to this form — N-queens, sudoku, polyomino tiling, set-packing.

The *Dancing Links* implementation (Knuth, *Dancing Links*, arXiv:cs/0011047, 2000) represents the matrix as a **toroidal doubly-linked list of 1-cells**. Each cell knows its up/down/left/right neighbours and the column header. The core trick:

```
To "cover" column c:   unlink c from the header row; for each row that
                       has a 1 in c, unlink that row from every column
                       it touches. O(1) per unlink.

To "uncover":          re-link in reverse order. O(1) per re-link.

The dance: x.left.right = x.right;  x.right.left = x.left
           (later)      x.left.right = x;        x.right.left = x
           — the node "leaves and returns to" the same position.
```

The genius isn't the matrix or the recursion — both are textbook backtracking. The genius is the **O(1) undo**. Backtracking searches that would otherwise spend most of their time *copying state* spend almost none, because state is mutated in-place and restored exactly.

This is the pattern. **It generalises** beyond exact-cover to any incremental search where state mutations have natural inverses.

## 2. Why a side track, not a direct ship into SR

DLX in its canonical form solves exact-cover. SR's GP loop does not have an exact-cover sub-problem at its core. So the canonical algorithm is **not directly applicable** to the current GP — implementing it now wouldn't speed up any benchmark.

The reason to ship it anyway:

1. **Pattern fluency.** We've been citing the Dancing Links pattern in research notes (`analytical_delta_loss.md` §2, `fit_as_perfect_info_game.md` §12) as the analog for incremental state in SR. Having a working canonical reference grounds those citations.

2. **Future application surface.** Three SR applications, in increasing speculation:
   - **(low risk)** Per-node FunctionalCache + dirty-flag invalidation (§4.3 of analytical_delta_loss.md): the dirty-flag propagation IS the Dancing Links uncover pattern applied to a tree, not a toroidal list. A clean canonical reference makes the SR adaptation a small port.
   - **(medium speculation)** Basis-subset enumeration for §2.3 Phase 2: when the GP picks a polynomial template, it implicitly chooses a subset of basis functions. If we wanted to **enumerate** "which K of the N basis functions" exhaustively (rather than GP-randomly), Algorithm X is the canonical method.
   - **(high speculation)** Vocab/constraint gating: tessera has axis-aware types, max-complexity constraints, and the planned ε-lexicase. Composing these into a search is structurally an exact-cover-like problem. Speculative; would need its own research note before commitment.

3. **Tessera as a research workbench (roadmap.md §4).** The library increasingly ports algorithm patterns from the broader algorithms-and-data-structures tradition. Knuth's DLX is one of the canonical such patterns; carrying a clean implementation matches that direction.

The scope of THIS ship is **only** application target (1), with infrastructure that makes (2) and (3) easier later. The implementation is fully decoupled from SR; passes the existing `test_dependency_structure.py` layering check trivially (it's a new top-level depth-0 module).

## 3. What's being implemented (canonical scope)

`tessera.combinatorics.dancing_links`:
- `Node` dataclass — 1-cell with left/right/up/down + column pointer.
- `Column` subclass — header with `size` (count of 1s in column).
- `ExactCoverMatrix` — built from a Python boolean / 0-1 matrix; manages the toroidal links.
- `cover(c)` / `uncover(c)` — the O(1) primitives.
- `solve()` — generator yielding each exact cover (list of row indices).
- `solve_first()`, `count_solutions()` — convenience wrappers.
- `nqueens(n)` — convenience: encode N-queens as exact-cover, return solution count and example arrangements.

Out of scope (deferred):
- Algorithm X+ (Knuth's later refinements with secondary columns for problems like N-queens with non-mandatory columns) — only if the canonical doesn't suffice for tests. **Update (2026-05-24):** secondary-column extension was needed for N-queens; shipped as part of this commit. Out of scope: further refinements (S heuristic variants, parallel solving).
- Any SR integration — separate future ship.
- Specialised solvers (sudoku, polyomino) — easy ports once the canonical exists, but not load-bearing for SR.

## 4. Implementation budget tracking

The user's "track our implementation as the implementation complexity grows" framing. Per the explicit ask, here's where tessera stood before this ship, and where it'll stand after.

| Subsystem | Before §2.3 P1 | After §2.3 P1 | After DLX ship |
|---|---|---|---|
| `tessera.expression.*` | (unchanged) | (unchanged) | (unchanged) |
| `tessera.search.*` | (12 modules) | +1 (sufficient_stats, 243 LOC) | (unchanged) |
| `tessera.koopman.*` | (unchanged) | (unchanged) | (unchanged) |
| `tessera.combinatorics.*` | did not exist | did not exist | **+1 subpackage, 2 files, 436 LOC source** |
| Test files | (existing) | +1 (test_sufficient_stats.py, 311 LOC) | +1 (test_dancing_links.py, 262 LOC) |
| **Test count delta** | — | +21 | +27 |

**Commitment statement:** the `tessera.combinatorics` subpackage is added with a single canonical module. Future additions to this subpackage (sudoku solver, generic constraint-satisfaction wrappers) MUST be justified by a SR-side use case — not just "it would be cool." If 6 months from now we haven't found an SR use for DLX beyond pattern-fluency, the right call is to extract it to its own repo / gist, not let it accumulate without justification.

This tracking lives here. If/when we add to the subpackage, append a new row to the table and a justification paragraph.

## 5. Connection to existing research notes

- [`analytical_delta_loss.md`](analytical_delta_loss.md) §2: cites DLX as the Regime-A canonical analog. With this ship, that citation links to a working reference.
- [`analytical_delta_loss.md`](analytical_delta_loss.md) §4.3: the future per-node-cache + dirty-flag work is the direct SR adaptation of the pattern.
- [`fit_as_perfect_info_game.md`](fit_as_perfect_info_game.md) §12: cites incremental-state as one of the search-time accelerators. Same connection.
- [`network_sr_and_budget_allocation.md`](network_sr_and_budget_allocation.md): the budget-allocation framing is unrelated to DLX directly, but the deterministic-admissible-search philosophy is the same Knuth-tradition stance.

## 6. Falsification

DLX itself is a well-tested algorithm (TAOCP coverage; thousands of implementations). The falsifiable claim of *this note* is narrower:

> **Carrying a canonical DLX implementation in tessera will, within ~6 months, materially accelerate at least one SR-side ship** (per-node cache, basis enumeration, or constraint gating).

If 6 months pass and the answer is "no concrete use found," delete the subpackage and move it to a side repo. The implementation budget tracking in §4 is the mechanism for forcing that audit.

## 7. What this note doesn't claim

This is NOT a research note about backtracking, exact-cover theory, or the long literature of constraint-satisfaction algorithms. It's a small ship carrying a single canonical implementation that we expect to reuse. The reuse case is named (§2 application target 1); the rest is speculative.

The user's framing — "track our implementation as the implementation complexity grows" — is what distinguishes this side track from speculative scope creep. Each addition to `tessera.combinatorics.*` after this initial ship requires a budget-table update and a justification entry. That's the cost-control mechanism.

## Changelog

- 2026-05-24: initial document. Carved off from `analytical_delta_loss.md` §2 reference. Justifies the side track as pattern-fluency + future SR-application infrastructure, with explicit budget tracking per user's framing.
- 2026-05-24: scope-update — secondary-column extension added to canonical scope (needed for N-queens demonstration).
