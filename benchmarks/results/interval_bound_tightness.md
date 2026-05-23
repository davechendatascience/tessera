# Interval-bound tightness on real data

Probes `fit_as_perfect_info_game.md` §7 Q2: how informative is the
interval-arithmetic lower bound on average? Branch-and-bound pruning
is genuinely useful if the bound is tight (high ratio) for many trees.

**Method:** sample 2000 random trees (max depth = 5, simplified via `simplify_canonical`) on each workload; compute actual MSE and interval-derived MSE lower bound. The ratio = `bound / actual_loss`.

## Tightness statistics per workload

| Workload | n | median ratio | mean ratio | p10 | p90 | bound=0 | tight>0.1 | tight>0.5 | tight>0.9 |
|---|---|---|---|---|---|---|---|---|---|
| synthetic_xx | 2000 | 0.4743 | 0.4289 | 0.0000 | 0.9691 | 386 | 1475 | 985 | 214 |
| synthetic_sin | 2000 | 0.0013 | 0.2035 | 0.0000 | 0.8550 | 816 | 856 | 238 | 196 |
| synthetic_multi | 2000 | 0.1146 | 0.2375 | 0.0000 | 0.6275 | 548 | 1036 | 326 | 148 |

## Reading

- **bound=0 count**: trees where the interval evaluator returned
  ±∞ (typically: contains `div` with zero-spanning denominator,
  or a `FunctionalOp` that the interval evaluator gives up on).
  These trees CANNOT be pruned by the bound.
- **tight>0.1 count**: trees where the bound is at least 10% of
  the actual loss. These can be pruned when the incumbent's loss
  is ≤ 10× the bound — i.e., very early in search when the GP
  hasn't found anything good yet.
- **tight>0.5 count**: bound ≥ 50% of actual. Pruned when the
  incumbent is within 2× of the bound — mid-to-late search.
- **tight>0.9 count**: bound ≥ 90% of actual. Almost achievable
  — these are the trees the bound REALLY constrains.

**Verdict:** if median ratio > 0.5, branch-and-bound pruning is
valuable. If median ratio < 0.1 and unbounded count is large,
the bound is dead weight on this workload's tree distribution.

## Tightness by complexity (synthetic_xx)

Median ratio binned by tree complexity:

| cx range | n | median ratio | median actual | median bound |
|---|---|---|---|---|
| 1-3 | 390 | 0.1392 | 4.5000 | 0.6263 |
| 4-6 | 273 | 0.5672 | 3.4932 | 1.9806 |
| 7-10 | 535 | 0.5694 | 3.4907 | 1.9806 |
| 11+ | 802 | 0.4811 | 3.3884 | 1.9806 |

## Notes

- The random-tree distribution from `random_tree` is biased
  toward shallow trees. To probe deeper trees, raise MAX_DEPTH.
- FunctionalOp trees get unbounded intervals (per §5 of the
  interval module's comment) and contribute to bound=0. This
  is the FunctionalOp gap identified in step (c) of the
  research note.
- Pre-simplification with `simplify_canonical` tightens bounds
  by removing redundant `x - x` etc. that interval arithmetic
  can't see through (the 'dependency problem').