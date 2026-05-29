# MNIST 0-vs-1 symbolic network (Milestone A)

**GP**: pop_size=30, n_gens=20, K=4, 
enable_2d=False, parsimony=0.002, seed=2026
**Data**: 400/class TRAIN, 200/class TEST, 14×14 downsampled
**Runtime**: 13.8s

## Result

- TRAIN accuracy: 0.975
- TEST accuracy: 0.980
- Network complexity: 28

## Discovered network

```
SymbolicNetwork(K=4, cx=28):
  Layer 1:
    f0 = 0
    f1 = (atan2((image / -0.015401), (image < 2.30902)) - (image + image))
    f2 = neg((image > 0))
    f3 = ((image + image) <= image)
  Layer 2: score = (((f0 + f1) + f2) + f3)
```

## Verdict vs Milestone A success criterion

**SUCCESS** — TEST accuracy exceeds the 0.95 target.

## Training history

| gen | best loss | TRAIN acc | TEST acc | cx |
|---|---|---|---|---|
| 0 | 0.2938 | 0.500 | 0.500 | 25 |
| 1 | 0.2753 | 0.836 | 0.807 | 22 |
| 2 | 0.2753 | 0.836 | 0.807 | 22 |
| 3 | 0.2740 | 0.861 | 0.840 | 22 |
| 4 | 0.2727 | 0.500 | 0.500 | 20 |
| 5 | 0.2727 | 0.500 | 0.500 | 20 |
| 6 | 0.2646 | 0.500 | 0.500 | 19 |
| 7 | 0.2597 | 0.507 | 0.502 | 17 |
| 8 | 0.2529 | 0.535 | 0.532 | 21 |
| 9 | 0.2529 | 0.535 | 0.532 | 21 |
| 10 | 0.2509 | 0.535 | 0.532 | 20 |
| 11 | 0.2414 | 0.676 | 0.660 | 22 |
| 12 | 0.2414 | 0.676 | 0.660 | 22 |
| 13 | 0.2337 | 0.873 | 0.855 | 24 |
| 14 | 0.2337 | 0.873 | 0.855 | 24 |
| 15 | 0.2337 | 0.873 | 0.855 | 24 |
| 16 | 0.2252 | 0.974 | 0.973 | 29 |
| 17 | 0.2213 | 0.975 | 0.980 | 28 |
| 18 | 0.2213 | 0.975 | 0.980 | 28 |
| 19 | 0.2213 | 0.975 | 0.980 | 28 |

## Reproducing

```
python benchmarks/run_mnist_symbolic_network.py --target_a 0 --target_b 1 --K 4 --pop 30 --gens 20
```