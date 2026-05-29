# MNIST 0-vs-1 symbolic network (Milestone A)

**GP**: pop_size=30, n_gens=20, K=4, 
enable_2d=False, parsimony=0.002, seed=2026
**Data**: 400/class TRAIN, 200/class TEST, 14×14 downsampled
**Runtime**: 14.8s

## Result

- TRAIN accuracy: 0.909
- TEST accuracy: 0.895
- Network complexity: 33

## Discovered network

```
SymbolicNetwork(K=4, n_classes=2, cx=33):
  Layer 1:
    f0 = (max(image, step(image)) + (image + image))
    f1 = sqrt((image + image))
    f2 = (abs(image) - 1.10298)
    f3 = ((sqrt(image) > tanh(image)) + sign(image))
  Layer 2 (one tree per class):
    class_0 = (((f0 + f1) + f2) + f3)
    class_1 = neg(f1)
```

## Verdict vs Milestone A success criterion

**PROGRESS** — beats single-tree baseline (0.80) but below 0.95 target. Try larger pop/gens or K.

## Training history

| gen | best loss | TRAIN acc | TEST acc | cx |
|---|---|---|---|---|
| 0 | 0.6061 | 0.910 | 0.895 | 27 |
| 1 | 0.6061 | 0.910 | 0.895 | 27 |
| 2 | 0.6061 | 0.910 | 0.895 | 27 |
| 3 | 0.5732 | 0.978 | 0.980 | 28 |
| 4 | 0.5732 | 0.978 | 0.980 | 28 |
| 5 | 0.5732 | 0.978 | 0.980 | 28 |
| 6 | 0.5732 | 0.978 | 0.980 | 28 |
| 7 | 0.5732 | 0.978 | 0.980 | 28 |
| 8 | 0.5562 | 0.941 | 0.958 | 29 |
| 9 | 0.5562 | 0.941 | 0.958 | 29 |
| 10 | 0.5562 | 0.941 | 0.958 | 29 |
| 11 | 0.5562 | 0.941 | 0.958 | 29 |
| 12 | 0.5562 | 0.941 | 0.958 | 29 |
| 13 | 0.5562 | 0.941 | 0.958 | 29 |
| 14 | 0.5527 | 0.838 | 0.830 | 33 |
| 15 | 0.5527 | 0.838 | 0.830 | 33 |
| 16 | 0.5527 | 0.838 | 0.830 | 33 |
| 17 | 0.5502 | 0.795 | 0.800 | 32 |
| 18 | 0.5423 | 0.909 | 0.895 | 33 |
| 19 | 0.5423 | 0.909 | 0.895 | 33 |

## Reproducing

```
python benchmarks/run_mnist_symbolic_network.py --target_a 0 --target_b 1 --K 4 --pop 30 --gens 20
```