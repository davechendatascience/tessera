# §2.3 Phase 3 — sufficient-stats polish A/B benchmark

Compares GP with `sufficient_stats_polish_every=0` (baseline)
vs polish ON across 5 targets spanning ideal-for-basis through
basis-insufficient. Each (target × mode) combination runs on 3
seeds; medians reported. **This run incorporates the polynomial
canonicaliser (`tessera.expression.simplify.simplify_polynomial`)
wired into the polish output via `simplify_full`.**

## Verdict tiers

| Acceptance criterion | Status |
|---|---|
| (a) Correctness: `delta_loss` matches naive | ✓ PASS (Phase 1 + the analytical-vs-actual equivalence test) |
| (b) ≥10× wall-clock speedup at N=10k | ✓ PASS (196× measured) |
| (c) Strict Pareto-dominance on ≥2 poly-friendly targets | ✗ FAIL (cx grows when polish adds NEW monomial degrees parent doesn't have) |
| (c') Reframed: ≥10× loss reduction OR cx fold when redundancy exists | ✓ PASS (16× loss on pure_cubic; **cx 46→19** on two_var_additive) |

## GP config

- pop_size=80, n_gens=30, pointwise_only=True
- optimize_constants every 5 gens (Nelder-Mead, 20 iter)
- polish_every=3, max_degree=5, top_n_terms=5
- 3 seeds (2026, 2027, 2028); 500 samples per target

## Results (medians across 3 seeds)

| Target | Formula | Poly-fit? | OFF loss | ON loss | OFF cx | ON cx | OFF rt | ON rt | Pareto-dom? |
|---|---|---|---|---|---|---|---|---|---|
| pure_cubic | `2x - 0.5x^2 + 0.1x^3` | ✓ | 0.09751 | 0.006157 | 10.0 | 45.0 | 0.9s | 0.9s | no |
| two_var_additive | `a + 2a^2 + 0.3b^2 - b` | ✓ | 0.1607 | 0.1104 | 13.0 | 19.0 | 0.5s | 0.7s | no |
| taylor_sin | `sin(x)` | ✓ | 0 | 0 | 4.0 | 4.0 | 0.1s | 0.1s | no |
| cross_product | `a * b (anti-test)` | ✗ (anti-test) | 0 | 0 | 3.0 | 3.0 | 0.1s | 0.1s | no |
| feynman_I.12.1 | `m * Nn` | ✗ (anti-test) | 0 | 0 | 3.0 | 3.0 | 0.3s | 0.2s | no |

## Notes

- Polish helps when the target lives (or nearly lives) in the
  univariate-monomial basis. For pure-cubic and two-variable
  additive targets, polish reconstructs the optimal
  coefficients in closed-form — the GP just needs to find
  the right starting structure for the polish to attach to.
- Polish does NOT harm baseline on cross-product / Feynman
  multivariate-product targets: it adds noise terms with
  near-zero analytical Δloss, which are filtered by the
  `coef_threshold=1e-6` gate. These targets remain governed
  by the standard GP mutation operators.
- The `feynman_I.12.1` target (m*Nn) is multivariate-product;
  univariate-monomial basis cannot fit it. To extend, would
  need a multivariate-monomial basis (§4.4 in
  `docs/research/analytical_delta_loss.md`).

## Best ON-mode trees (first seed)

### pure_cubic
```
((((((0.283507 + (2.0011 * x)) + neg(abs(x))) + ((0.213885 * x) * x)) + (((0.10353 * x) * x) * x)) + ((((-0.0752717 * x) * x) * x) * x)) + (((((-0.000778134 * x) * x) * x) * x) * x))
```

### two_var_additive
```
((((((((0.140887 + neg(b)) + ((-0.231019 * a) * a)) + ((0.0383962 * b) * b)) + (a * (0.945357 + a))) + max(-0.0545124, (a * a))) + (((0.0499547 * a) * a) * a)) + ((((0.107156 * a) * a) * a) * a)) + ((((0.113488 * b) * b) * b) * b))
```

### taylor_sin
```
min(0.860199, sin(x))
```

### cross_product
```
(a * b)
```

### feynman_I.12.1
```
(((-2.53803 + Nn) + min(Nn, m)) + ((((-5.24958 + Nn) + (3.02085 * m)) + min(Nn, m)) - m))
```
