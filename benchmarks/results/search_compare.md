# tessera.search — GP vs SA vs RandomSearch comparison

**Target:** `y = 2.371*x + 0.183*x*x + 0.01*noise`
**Samples:** n=2000, var(y) = 5.5894
**Noise floor:** ~ 1e-4 (sigma=0.01)

## Headline

| Searcher | Candidate budget | Runtime | Best cx | Best loss | % of var |
|---|---|---|---|---|---|
| GP (pop=80 × gens=40 + opt-const) | ~3,200 | 2.90s | 17 | 0.0122 | 0.22% |
| SA (steps=4000 × restarts=3 + opt-const) | ~12,000 | 0.95s | 20 | 0.04807 | 0.86% |
| RandomSearch | 8,000 | 2.15s | 10 | 0.05268 | 0.94% |

## Best expression per searcher

- **GP:**     `((max(x, (x + x)) + (V2[exponential(halflife=720)[0,5041], power_law(scale=168,alpha=1.2)[0,256]](L[exponential(halflife=336)[0,2353]](x)) + (x - 0.218475))) + (x / -7.44308))`
- **SA:**     `(((x + ((x * min(max(x, -0.095303), (x * x))) / x)) - neg(x)) + (x / -3.79004))`
- **Random:** `((x - 0.069053) + (max(x, 0.026565) - neg(x)))`

## Merged Pareto front

Combining all three searchers' Pareto-front members and re-running
`pareto_front()`:

| Cx | Loss | % of var | Tree |
|---|---|---|---|
| 1 | 1.91 | 34.18% | `x` |
| 3 | 0.2201 | 3.94% | `(x + x)` |
| 5 | 0.1903 | 3.40% | `((x + x) - -0.172797)` |
| 6 | 0.174 | 3.11% | `((x / 0.619934) - neg(x))` |
| 7 | 0.07594 | 1.36% | `(max(x, (x + x)) + x)` |
| 9 | 0.03028 | 0.54% | `(max(x, (x + x)) + (x - 0.213685))` |
| 11 | 0.01446 | 0.26% | `(max(x, ((x - 0.313983) + x)) + (x - 0.0780926))` |
| 13 | 0.0122 | 0.22% | `((max(x, (x + x)) + (x - 0.217451)) + (x / -7.4395))` |
| 17 | 0.0122 | 0.22% | `((max(x, (x + x)) + (V2[exponential(halflife=720)[0,5041], power_law(scale=168,alpha=1.2)[0,256]]...` |

## Notes

- Candidate budget mismatch is intentional: each searcher uses
  the canonical config you'd reach for in practice. SA scales
  cheaper per-candidate but needs more total steps to explore.
- RandomSearch as baseline: any 'directed' searcher should
  beat it on this kind of problem. If RS wins, the directed
  searcher is over-tuned to a different regime.
- Smooth MSE + small problem favours const-opt: GP and SA both
  get a significant boost from the polish step on this target.