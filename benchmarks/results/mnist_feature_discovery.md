# MNIST 0-vs-rest feature-discovery validation

**Hypothesis (from invariance_in_sr.md §12):** can tessera's CURRENT
machinery (untyped FunctionalOp2D + hardcoded mean-aggregation +
GP search over Measure2D params) discover a 2D kernel that
classifies MNIST `digit == 0` above chance?

**Train/test:** 200 / 100 samples, downsampled 28×28 → 14×14 via 2× block-mean
**GP:** pop=80, gens=25, init_max_depth=4, parsimony=0.001
**Search loop:** custom (per-image scoring; mean-pool aggregation hardcoded)
**Wall-clock:** 23.3s

## Result

- **Best tree complexity:** 9
- **Best TRAIN loss (MSE + parsimony):** 0.2003
- **TRAIN accuracy:** 0.7500 (chance = 0.5)
- **TEST accuracy:** 0.8000 (chance = 0.5)

### Discovered tree

```
((image + image) + sign(M2D[1??(0,-1) + -2??(0,0) + 1??(0,1)]((image * image))))
```

## Verdict

**TEST accuracy = 0.80.** Above chance but below
the 90% threshold. The framework is doing something useful
but isn't competitive with a single-layer CNN (which gets
~99% on this task). Possible causes:
- GP didn't find the right kernel structure (population too small,
  generations too few)
- Mean-aggregation is too crude (max would distinguish digits better)
- Need to add aggregator operators to the grammar so the GP can
  discover the right pooling rule

## Generation history

| gen | best loss | best cx | elapsed (s) |
|---|---|---|---|
| 0 | 0.2130 | 9 | 0.8 |
| 5 | 0.2078 | 8 | 0.6 |
| 10 | 0.2003 | 9 | 0.4 |
| 15 | 0.2003 | 9 | 0.7 |
| 20 | 0.2003 | 9 | 0.5 |

## Did the GP discover an aggregator?

NO — no reduce_* ops in the discovered tree. The benchmark's
hardcoded mean-pool wrapper is doing the aggregation.

This is an honest empirical finding: ADDING reduce ops to the
grammar isn't sufficient for them to be USED. Three reasons:

1. **random_tree builds bottom-up.** A reduce op is only useful
   if placed at the ROOT (to make the whole tree scalar-valued).
   Random-tree's recursive construction makes reduce ops appear
   at root rarely.
2. **Mean-pool fallback works adequately.** The benchmark wrapper
   mean-pools any array output, so there's no fitness pressure
   to choose a different aggregator — the wrapper gives the GP
   a free aggregation.
3. **No bias mutation.** The mutation dispatcher has no rule like
   'wrap the root in a random reduce op.' Without that bias, the
   GP wanders in the array-output subspace.

Next step to actually discover aggregation: remove the wrapper's
mean-pool fallback (return inf for array outputs) and/or add a
`wrap_in_reduce` mutation. Either forces the GP to discover an
explicit aggregator.