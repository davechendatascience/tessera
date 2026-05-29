# MNIST 0-vs-1 symbolic network (Milestone A)

**GP**: pop_size=20, n_gens=15, K=4, 
enable_2d=True, parsimony=0.002, seed=2026
**Data**: 150/class TRAIN, 100/class TEST, 14×14 downsampled
**Runtime**: 15.8s

## Result

- TRAIN accuracy: 0.607
- TEST accuracy: 0.645
- Network complexity: 22

## Discovered network

```
SymbolicNetwork(K=4, cx=22):
  Layer 1:
    f0 = 0.785398
    f1 = (neg((image > -1.50547)) - asin(image))
    f2 = sin(pow(5.8175, image))
    f3 = ((image + image) <= image)
  Layer 2: score = ((f1 + f2) + f3)
```

## Verdict vs Milestone A success criterion

**NOT YET** — TEST accuracy ≤ single-tree baseline. Architecture may need infrastructure refinement.

## Training history

| gen | best loss | TRAIN acc | TEST acc | cx |
|---|---|---|---|---|
| 0 | 0.2940 | 0.500 | 0.500 | 25 |
| 1 | 0.2809 | 0.500 | 0.500 | 24 |
| 2 | 0.2789 | 0.500 | 0.500 | 23 |
| 3 | 0.2769 | 0.500 | 0.500 | 22 |
| 4 | 0.2769 | 0.500 | 0.500 | 22 |
| 5 | 0.2769 | 0.500 | 0.500 | 22 |
| 6 | 0.2749 | 0.500 | 0.500 | 21 |
| 7 | 0.2749 | 0.500 | 0.500 | 21 |
| 8 | 0.2749 | 0.500 | 0.500 | 21 |
| 9 | 0.2729 | 0.500 | 0.500 | 20 |
| 10 | 0.2709 | 0.500 | 0.500 | 19 |
| 11 | 0.2597 | 0.607 | 0.645 | 25 |
| 12 | 0.2597 | 0.607 | 0.645 | 25 |
| 13 | 0.2537 | 0.607 | 0.645 | 22 |
| 14 | 0.2537 | 0.607 | 0.645 | 22 |

## Reproducing

```
python benchmarks/run_mnist_symbolic_network.py --target_a 0 --target_b 1 --K 4 --pop 20 --gens 15
```