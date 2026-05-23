# Equivalence-class count |E_K| / |T_K| under simplify_canonical

Probes the conjecture in `fit_as_perfect_info_game.md` §6: SR-for-fit
effectively lives in the equivalence-class space, not the raw syntactic
tree space.

## Grammar (restricted for tractability)

- **Leaves:** `Var('x')`, `Const(-1)`, `Const(0)`, `Const(1)`
- **Unary ops:** `tanh`, `abs`, `sign`, `neg`, `step`
- **Binary ops:** `add`, `sub`, `mul`, `div`, `min`, `max`, `gt`, `lt`, `ge`, `le`

Note: tessera's full grammar also has `FunctionalOp` and `FunctionalOp2D`
(measure-theoretic operators); these are EXCLUDED from this enumeration
because the parameter space (measures with arbitrary halflife, signed
sums with arbitrary weights) is continuous, so 'enumeration' isn't
well-defined. The pointwise subset suffices to test the conjecture.

## Per-complexity counts

**Reading:** `|T|` = distinct *syntactic* trees at this complexity;
`|E|` = distinct *canonical* forms (after `simplify_canonical`).
`ratio = |E| / |T|`. Lower ratio ⇒ more collapse ⇒ the simplifier
is doing real work.

| max_depth | cx | \|T\| | \|E\| | ratio = \|E\| / \|T\| |
|---|---|---|---|---|
| 1 | 1 | 4 | 4 | 1.0000 |
| 2 | 1 | 4 | 4 | 1.0000 |
| 2 | 2 | 20 | 10 | 0.5000 |
| 2 | 3 | 160 | 50 | 0.3125 |
| 3 | 1 | 4 | 4 | 1.0000 |
| 3 | 2 | 20 | 10 | 0.5000 |
| 3 | 3 | 260 | 78 | 0.3000 |
| 3 | 4 | 2,400 | 595 | 0.2479 |
| 3 | 5 | 16,800 | 3,114 | 0.1854 |
| 3 | 6 | 64,000 | 7,322 | 0.1144 |
| 3 | 7 | 256,000 | 19,823 | 0.0774 |

## Cumulative (across all complexities <= depth)

| max_depth | cumulative \|T\| | cumulative \|E\| | ratio |
|---|---|---|---|
| 1 | 4 | 4 | 1.0000 |
| 2 | 184 | 64 | 0.3478 |
| 3 | 339,484 | 30,946 | 0.0912 |

(Cumulative |E| over-counts by treating same-canonical-form-at-
different-input-complexity as distinct; the per-cx table above
is the cleaner view.)

## Reading

Interpretation of the ratio at each complexity level:
- **ratio ≈ 1.0**: nearly every syntactic tree is already canonical;
  the simplifier is a no-op at this complexity
- **ratio ≈ 0.5**: roughly half of syntactic trees collapse to
  a previously-seen canonical form
- **ratio < 0.1**: most trees are equivalence-class duplicates;
  the simplifier provides a >10× reduction in effective
  search-space size

The conjecture is well-supported iff the ratio decreases as
complexity grows (more identities → more collapse).

**Caveat:** this measures equivalence under `simplify_canonical`'s
rewrites only (rule-based folds + AC normalisation). It does NOT
capture all semantic equivalences. E.g., `tanh(neg(x))` and
`neg(tanh(x))` are semantically equal (`tanh` is odd) but our
simplifier doesn't apply this rule. So |E_K| reported here is
an UPPER bound on the true equivalence-class count.