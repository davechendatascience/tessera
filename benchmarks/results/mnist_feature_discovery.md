# MNIST 0-vs-rest feature-discovery validation

**Hypothesis (from invariance_in_sr.md §12):** can tessera's CURRENT
machinery (untyped FunctionalOp2D + hardcoded mean-aggregation +
GP search over Measure2D params) discover a 2D kernel that
classifies MNIST `digit == 0` above chance?

**Train/test:** 200 / 100 samples, downsampled 28×28 → 14×14 via 2× block-mean
**GP:** pop=50, gens=15, init_max_depth=4, parsimony=0.001
**Search loop:** custom (per-image scoring; mean-pool aggregation hardcoded)
**Wall-clock:** 14.2s

## Result

- **Best tree complexity:** 9
- **Best TRAIN loss (MSE + parsimony):** 0.2015
- **TRAIN accuracy:** 0.7200 (chance = 0.5)
- **TEST accuracy:** 0.8200 (chance = 0.5)

### Discovered tree

```
(((image + image) + step(M2D[1??(0,-1) + -2??(0,0) + 1??(0,1)](image))) - 0.044751)
```

## Verdict

**TEST accuracy = 0.82.** Above chance but below
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
| 0 | 0.2126 | 4 | 0.2 |
| 5 | 0.2046 | 7 | 0.6 |
| 10 | 0.2026 | 7 | 0.5 |