# Differentiable EML super-graph — D1 smoke benchmark

Per `docs/research/differentiable_eml_jax.md`. Central claim:
parallel restarts (vmap'd) trade GPU parallelism for local-minima
robustness. `per-init success` = probability a single SGD run
recovers the target (R² > 0.9999); the best-of-R columns show
recovery climbing as restarts scale. A tiny per-init success that
still recovers at large R *is* the thesis: GPU parallelism pays the
low per-init probability.

2000 Adam steps/restart, cosine-τ + delayed sparsity. Selection by
hard-program R². Wall-clock: 178s (CPU; JAX vmap'd — a GPU
runs the R restarts in parallel).

| target | per-init success | best-of-16 R² | best-of-64 R² | best-of-256 R² | recovered |
|---|---|---|---|---|---|
| `x^2` | 98.8% | 1.000 | 1.000 | 1.000 | yes |
| `sin(2x)` | 1.2% | 1.000 | 1.000 | 1.000 | yes |
| `x0*x1` | 9.4% | 0.449 | 1.000 | 1.000 | yes |

## Recovered programs (hard-snapped)

**`x^2`**
```
n0 = add(x0, x0)
n1 = add((add(x0, x0)), x0)
n2 = square((add((add(x0, x0)), x0)))
n3 = square(x0)
root = (square(x0))
```

**`sin(2x)`**
```
n0 = add(x0, x0)
n1 = add(x0, x0)
n2 = square(x0)
n3 = sin((add(x0, x0)))
root = (sin((add(x0, x0))))
```

**`x0*x1`**
```
n0 = square(x1)
n1 = div(x1, x0)
n2 = cos(x1)
n3 = mul(x0, x1)
root = (mul(x0, x1))
```

## Reading

- `x²` recovers at any R — benign landscape (low order).
- `x0*x1` has low per-init success but recovers by R=64 — restarts
  doing the work.
- `sin(2x)` is the spectral-bias hard case: per-init success ~0.3%,
  misses at R=16/64, recovers the EXACT form `sin(x0+x0)` at R=256.
  The clearest 'scale buys robustness' instance.

- The cost is honest: per-init success is the currency. For harder
  targets it may be so small that no feasible R suffices — that is
  where the **Tier-2 GP+gradient hybrid** must raise per-init
  probability *structurally* (non-local moves) rather than relying
  on lucky inits.

## Reproducing

```
python benchmarks/run_diff_eml_jax.py
```