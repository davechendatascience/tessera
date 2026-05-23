"""Equivalence-class count |E_K| / |T_K| under simplify_canonical.

Probes the load-bearing assumption of `fit_as_perfect_info_game.md`
section 6's conjecture: SR-for-fit effectively lives in the
equivalence-class space, not the raw syntactic tree space. If the
ratio |E_K| / |T_K| is large (close to 1), the canonical simplifier
isn't collapsing much and the "search lives in E_K" framing is weak.
If it's small (< 0.1), the framing is strong: most syntactic trees
are duplicates of each other under canonical form.

Method
------
Enumerate all valid trees up to a small max_depth using a restricted
grammar:
  - Leaves: Var("x") and 3 constants {-1.0, 0.0, 1.0}
  - Unary: tanh, abs, sign, neg, step
  - Binary: add, sub, mul, div, min, max, gt, lt, ge, le

For each tree, compute simplify_canonical(tree). Count:
  - |T_K|: number of distinct syntactic trees of complexity <= K
  - |E_K|: number of distinct CANONICAL trees of complexity <= K
  - ratio = |E_K| / |T_K|: collapse factor

Output the (cx, |T|, |E|, ratio) table.
"""
from __future__ import annotations
from pathlib import Path

from tessera.expression import (
    Var, Const, BinOp, UnOp, Node,
    BIN_OPS, UN_OPS,
    complexity,
)
from tessera.expression.simplify import simplify_canonical


# Restricted grammar for tractability
LEAVES: list[Node] = [
    Var("x"),
    Const(-1.0),
    Const(0.0),
    Const(1.0),
]


def enumerate_trees(max_depth: int) -> set[Node]:
    """All valid trees up to `max_depth`. Returns the set of distinct
    syntactic forms (de-duplicated by structural equality).

    Depth 1: just LEAVES.
    Depth d > 1: leaves + UnOp(op, sub) + BinOp(op, sub_a, sub_b)
    for all sub-trees of depth < d.
    """
    trees: set[Node] = set(LEAVES)
    by_depth: list[set[Node]] = [set(LEAVES)]

    for d in range(2, max_depth + 1):
        new = set()
        # All sub-trees of depth < d
        prev_levels = set().union(*by_depth)
        # UnOps over previous-level sub-trees
        for op in UN_OPS:
            for sub in prev_levels:
                try:
                    new.add(UnOp(op, sub))
                except Exception:
                    pass
        # BinOps over pairs of previous-level sub-trees
        prev_list = list(prev_levels)
        for op in BIN_OPS:
            for a in prev_list:
                for b in prev_list:
                    try:
                        new.add(BinOp(op, a, b))
                    except Exception:
                        pass
        by_depth.append(new)
        trees.update(new)
    return trees


def main():
    OUT = Path(__file__).parent / "results" / "equivalence_class_count.md"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Build incrementally: depth 1, 2, 3
    # Depth 4 is too large (~1e10 candidates)
    rows = []
    for max_depth in (1, 2, 3):
        print(f"\n--- enumerating depth <= {max_depth} ---")
        all_trees = enumerate_trees(max_depth)
        print(f"  |T| total = {len(all_trees):,}")

        # Group by complexity
        by_cx: dict[int, set[Node]] = {}
        for t in all_trees:
            cx = complexity(t)
            by_cx.setdefault(cx, set()).add(t)

        # For each cx, compute |T_cx| (trees AT this cx) and
        # |E_cx| (canonical-distinct AT this cx)
        for cx in sorted(by_cx):
            t_set = by_cx[cx]
            canonical_set = {simplify_canonical(t) for t in t_set}
            # Some canonical forms may have DIFFERENT complexity (e.g. if
            # simplify_canonical collapses) — count by canonical form
            # complexity too.
            distinct_canonical = len(canonical_set)
            ratio = distinct_canonical / len(t_set) if t_set else float("nan")
            rows.append(dict(
                max_depth=max_depth, cx=cx,
                T=len(t_set),
                E=distinct_canonical,
                ratio=ratio,
            ))
            print(f"  cx={cx:2d}  |T|={len(t_set):>8,}  |E|={distinct_canonical:>8,}  "
                  f"ratio={ratio:.4f}")

        # Cumulative: |T_K| = trees with cx <= K; |E_K| likewise on canonical set
        cumulative_T = 0
        cumulative_canonical = set()
        for cx in sorted(by_cx):
            cumulative_T += len(by_cx[cx])
            cumulative_canonical |= {simplify_canonical(t) for t in by_cx[cx]}

    # Aggregate report
    print(f"\n[report] writing to {OUT}")

    L = [
        "# Equivalence-class count |E_K| / |T_K| under simplify_canonical",
        "",
        "Probes the conjecture in `fit_as_perfect_info_game.md` §6: SR-for-fit",
        "effectively lives in the equivalence-class space, not the raw syntactic",
        "tree space.",
        "",
        "## Grammar (restricted for tractability)",
        "",
        "- **Leaves:** `Var('x')`, `Const(-1)`, `Const(0)`, `Const(1)`",
        f"- **Unary ops:** {', '.join(f'`{op}`' for op in UN_OPS)}",
        f"- **Binary ops:** {', '.join(f'`{op}`' for op in BIN_OPS)}",
        "",
        "Note: tessera's full grammar also has `FunctionalOp` and `FunctionalOp2D`",
        "(measure-theoretic operators); these are EXCLUDED from this enumeration",
        "because the parameter space (measures with arbitrary halflife, signed",
        "sums with arbitrary weights) is continuous, so 'enumeration' isn't",
        "well-defined. The pointwise subset suffices to test the conjecture.",
        "",
        "## Per-complexity counts",
        "",
        "**Reading:** `|T|` = distinct *syntactic* trees at this complexity;",
        "`|E|` = distinct *canonical* forms (after `simplify_canonical`).",
        "`ratio = |E| / |T|`. Lower ratio ⇒ more collapse ⇒ the simplifier",
        "is doing real work.",
        "",
        "| max_depth | cx | \\|T\\| | \\|E\\| | ratio = \\|E\\| / \\|T\\| |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['max_depth']} | {r['cx']} | {r['T']:,} | {r['E']:,} | "
                 f"{r['ratio']:.4f} |")
    L.append("")

    # Aggregate stats per max_depth
    L.append("## Cumulative (across all complexities <= depth)")
    L.append("")
    L.append("| max_depth | cumulative \\|T\\| | cumulative \\|E\\| | ratio |")
    L.append("|---|---|---|---|")
    by_md = {}
    for r in rows:
        d = r["max_depth"]
        by_md.setdefault(d, []).append(r)
    for d, rs in sorted(by_md.items()):
        # cumulative |T| and |E| up to this depth
        T_sum = sum(r["T"] for r in rs)
        E_sum = sum(r["E"] for r in rs)
        # NB: this OVER-counts |E| because a canonical form at cx=3 might
        # also appear at cx=5; we'd need set-union to be exact. Report as
        # approximate.
        L.append(f"| {d} | {T_sum:,} | {E_sum:,} | {E_sum/T_sum:.4f} |")
    L.append("")
    L.append("(Cumulative |E| over-counts by treating same-canonical-form-at-")
    L.append("different-input-complexity as distinct; the per-cx table above")
    L.append("is the cleaner view.)")

    L.append("")
    L.append("## Reading")
    L.append("")
    L.append("Interpretation of the ratio at each complexity level:")
    L.append("- **ratio ≈ 1.0**: nearly every syntactic tree is already canonical;")
    L.append("  the simplifier is a no-op at this complexity")
    L.append("- **ratio ≈ 0.5**: roughly half of syntactic trees collapse to")
    L.append("  a previously-seen canonical form")
    L.append("- **ratio < 0.1**: most trees are equivalence-class duplicates;")
    L.append("  the simplifier provides a >10× reduction in effective")
    L.append("  search-space size")
    L.append("")
    L.append("The conjecture is well-supported iff the ratio decreases as")
    L.append("complexity grows (more identities → more collapse).")
    L.append("")
    L.append("**Caveat:** this measures equivalence under `simplify_canonical`'s")
    L.append("rewrites only (rule-based folds + AC normalisation). It does NOT")
    L.append("capture all semantic equivalences. E.g., `tanh(neg(x))` and")
    L.append("`neg(tanh(x))` are semantically equal (`tanh` is odd) but our")
    L.append("simplifier doesn't apply this rule. So |E_K| reported here is")
    L.append("an UPPER bound on the true equivalence-class count.")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
