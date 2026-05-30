# Symbolic MNIST network with decomposition heads (8x8 digits)

Feature layer: 4 channels {raw, |dx|, |dy|, laplacian} pooled to 4x4 -> 64 features. Per-class `discover_decompose` one-vs-rest scorer, argmax. Gradient-free, interpretable.

**TRAIN 0.9730, TEST 0.9463** (1257/540 split, fit 103s). Head methods used: `{'base': 10}`.

| class | head R2 | method |
|---|---|---|
| 0 | 0.809 | `base` |
| 1 | 0.739 | `base` |
| 2 | 0.816 | `base` |
| 3 | 0.660 | `base` |
| 4 | 0.827 | `base` |
| 5 | 0.747 | `base` |
| 6 | 0.874 | `base` |
| 7 | 0.850 | `base` |
| 8 | 0.465 | `base` |
| 9 | 0.703 | `base` |

## Reading

- **Heads fall back to `base`** (a capacity-controlled sparse
  symbolic readout): with degenerate (blob) groupings rejected and
  peel/separability accepted only when it beats `base` on a
  held-out slice, decomposition correctly DECLINES to manufacture
  structure in pooled image features. Image class-ness is not a
  peelable/separable analytic function of the features; it is a
  learned statistical pattern.
- So decomposition adds nothing here over the sparse readout, and
  it no longer wastes time on degenerate additive splits. Accuracy
  is driven by the FEATURE LAYER (the non-symbolic part), not by
  any symbolic structure discovery.
- Empirical complement to Feynman: decomposition breaks the deep-
  structure wall in DISCOVERY (sqrt(1-v^2/c^2)-class, exact) and
  finds nothing to break in PERCEPTION. SR's value is discovery,
  not vision — and the capacity controls make that failure cheap
  and honest (no overfit ensemble) rather than slow and misleading.

## Reproducing

```
python benchmarks/run_mnist_decompose.py
```