# csp_sr on BTC hourly (out-of-sample)

```
=== csp_sr on BTC hourly (real data, no closed-form GT) ===
fetched 4000 hourly candles (BTCUSDT), last close=73378
features=15, samples=3974, chronological split train=2781 test=1193 (NO shuffle)

TARGET: next log-return   (test n=1193)
  naive(mean)   R2_oos=-0.0006  dir=0.497
  linear OLS    R2_oos=-0.0258  dir=0.500
  csp_sr        R2_oos=-0.0387  dir=0.495   (6 terms)
    csp expr: add(add(add(mul(0.0009735, tanh(mul(mom6, rng))), mul(-0.0002973, sub(a1, mul(dlv, rng)))), add(mul(-0.0001808, sub(a2, 

TARGET: next |log-return| (volatility)   (test n=1193)
  naive(mean)   R2_oos=-0.1635
  linear OLS    R2_oos=-0.0042
  csp_sr        R2_oos=-0.0003   (6 terms)
    csp expr: add(add(add(mul(0.0006908, abs(sub(a1, rng))), mul(-0.0002527, abs(sub(mom6, dlv)))), add(mul(0.0006609, add(add(lv, vol

## Reading
- Returns: any out-of-sample R2 > 0 / dir > 0.50 is a (weak) real edge;
  expected ~0 (markets are near-efficient at 1h). Honest if it's nil.
- Volatility: vol clusters, so a generalising R2_oos >> 0 is the genuine
  discoverable structure (GARCH-like). csp beating naive+linear here =
  it found a real symbolic relationship in real data, no GT needed.
```
