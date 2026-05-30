# Deep symbolic csp (`discover_deep`) — depth vs shallow (a negative result)

Stacked csp_sr: `depth` layers of `width` nodes; each node is a
free-form csp expression over prior nodes, fit to the running
residual (boosting -> each hidden node has a target, no backprop).
Combined model is one self-contained tessera Expr. **Scored on a
held-out 1/3 split** (train 1000, test 500); noise-free
analytic targets. Vocab `['neg', 'sqrt', 'tanh', 'add', 'sub', 'mul']`, deep shape depth=6 width=3 (18 nodes).

**Headline: a single deeper enumeration (single-big) wins or ties
on every target; stacking does not robustly extend it.** The
train-R2 gains from depth are largely MEMORIZATION — the held-out
split exposes catastrophic overfitting that train R2 hides.

- **single-small**: one layer, `max_size=2` (the shallow wall).
- **single-big**: one layer, `max_size=5` (a deeper enumeration; `F` = dictionary size, cap 60000 = intractable signal).
- **deep (lr=1.0)**: stack, naive boosting (no shrinkage).
- **deep (lr=0.3)**: stack with shrinkage — the FAIR deep (the winner column uses this one).

| target | formula | regime | single-small | single-big (F) | deep lr=1.0 | deep lr=0.3 | winner |
|---|---|---|---|---|---|---|---|
| prod_sum | `x0*x1 + x2*x3` | shallow | 1.000 | 1.000 (28324) | 1 | 1.000 | **single-small** |
| sq_prodsum | `(x0*x1 + x2*x3)^2` | deep/tractable | 0.394 | 1.000 (28324) | -1.54e+06 | -1.045 | **single-big** |
| norm_prodsum8 | `sqrt((x0x1+x2x3)^2 + (x4x5+x6x7)^2)` | deep/intractable | 0.398 | 0.429 (35596) | -383 | 0.291 | **single-big** |
| norm_triple6 | `sqrt((x0x1x2)^2 + (x3x4x5)^2)` | deep/intractable | 0.725 | 0.925 (24852) | 0.887 | 0.936 | **deep(lr.3)** |

## Reading

- **Train R2 lies.** With `lr=1.0`, depth drives train R2 up while
  held-out R2 goes hugely NEGATIVE (sq_prodsum -1.5e6, norm_prodsum8
  -383): the augmented-feature compositions (squares/products of
  fitted intermediates) are high-variance, and the linear fit assigns
  large cancelling coefficients that explode on unseen points — in
  distribution, on noise-free data. Always score deep stacks on a
  held-out split (cf. the Feynman exact-vs-approx separation).
- **Shrinkage helps but does NOT reliably fix it.** `lr=0.3` pulls
  most targets back from the cliff, but it is unstable and
  data-dependent: on `sq_prodsum` it is still negative here (~ -1.0)
  while a different random draw reached +0.82, and it is
  non-monotonic in lr. It needs per-problem tuning a single
  enumeration never does.
- **Deep does not robustly beat single-big.** It wins on at most
  one target (`norm_triple6`, 0.936 vs 0.925 — within the
  instability noise), loses decisively on the tractable target
  (single-big exact, 1.000), and loses on `norm_prodsum8` (0.291
  vs 0.429).
- **Shallow target** (`prod_sum`): a single shallow layer already
  recovers it; depth is pure overhead.

## Honest verdict

- **`discover_deep` does not provide a robust generalization
  advantage over single-layer csp enumeration.** When the true
  form fits under the enumeration cap, enumerate (exact, stable,
  no tuning). When it does not, the stack only sometimes matches a
  single layer and adds a fragile shrinkage hyperparameter.
- It remains a real capability (gradient-free deep composition,
  one self-contained Expr); bounded connectivity beat dense at
  equal cost in separate train-fit runs — but neither generalizes
  without shrinkage, and shrinkage does not lift it past single-big.
- The bottleneck is the SEARCH SPACE (exponential in tree size),
  not eval. GPU (`--jax`) accelerates the F x N Phi-build + fit at
  scale but cannot shrink the dictionary, and depth is sequential.
  The combinatorial explosion is beaten algorithmically (bounded
  pool, symmetry-breaking, caps), not with hardware.

## Reproducing

```
python benchmarks/run_deep_csp.py          # numpy (CPU)
python benchmarks/run_deep_csp.py --jax    # GPU eval path
```