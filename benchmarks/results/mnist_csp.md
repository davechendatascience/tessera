# csp_sr digit classifier (sklearn 8x8 digits)

Per-class symbolic scorers fit by csp_sr (one-vs-rest, +1/−1),
classify by argmax. A SYMBOLIC, sparse, interpretable classifier —
feasibility test, not a CNN. Full 28×28 MNIST runs the same code
with a pooling feature-layer (Colab GPU).

degree=1, stlsq_threshold=0.03, train=1257, test=540, features=64.

**TRAIN acc 0.9435, TEST acc 0.9407**, avg 53.6 terms/class, fit 0.1s.

| class | terms | train R² |
|---|---|---|
| 0 | 52 | 0.760 |
| 1 | 57 | 0.581 |
| 2 | 53 | 0.722 |
| 3 | 52 | 0.596 |
| 4 | 56 | 0.755 |
| 5 | 56 | 0.727 |
| 6 | 54 | 0.720 |
| 7 | 55 | 0.729 |
| 8 | 52 | 0.497 |
| 9 | 49 | 0.536 |

## Example scorer (class 0)
```
add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(0.08836, p1), mul(0.03964, p2)), mul(0.1987, p3)), mul(-0.1839, p4)), mul(-0.2068, p5)), mul(0.08572, p6)), mul(0.1227, p7)), mul(-0.3251, p8)), mul(-0.1739, p9)), mul(-0.05592, p11)), mul
```

## Reading

- A sparse symbolic linear (degree 1) or low-order (degree 2)
  classifier over pixels, gradient-free (STLSQ per class).
- Each class scorer is an explicit, inspectable expression —
  the interpretability SR buys, unlike a dense net.
- Accuracy is the honest feasibility number for symbolic scorers
  over raw pixels; richer features (pooling / gradients) or
  degree 2 trade interpretability for accuracy.

## Reproducing

```
python benchmarks/run_mnist_csp.py --degree 1
```