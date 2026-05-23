"""Demonstrate the constant-optimisation polish on a smooth synthetic
problem.

Target: y = a*x + b*x*x + noise   with (a, b) = (2.371, 0.183).

The structure x + x*x is trivial for the GP to find at low cx; the
numerical constants are where polish pays off. With const-opt OFF, the
GP's random `constant_jitter` mutation slowly drifts toward truth.
With const-opt ON, scipy.optimize finds (a, b) within numerical
tolerance in one polish step.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np

from tessera.expression import GP, GPConfig, complexity


def main():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    a_true, b_true = 2.371, 0.183
    y = a_true * x + b_true * x * x + 0.01 * rng.standard_normal(n)
    env = {"x": x}

    print("=== const-opt demo: y = 2.371*x + 0.183*x*x + small_noise ===\n")
    print(f"n={n} samples, target var = {y.var():.4f}\n")

    common = dict(pop_size=80, n_gens=40, init_max_depth=3,
                  parsimony=1e-5, seed=42, verbose=False,
                  early_stop_patience=20)

    # ---- without polish ----
    cfg_off = GPConfig(**common, optimize_constants_every=0)
    t0 = time.time()
    front_off = GP(cfg_off).run(env, y, ["x"])
    t_off = time.time() - t0
    best_off = min(front_off, key=lambda c: c.train_loss)
    print(f"[off]  runtime={t_off:.2f}s, best cx={best_off.complexity}, "
          f"loss={best_off.train_loss:.4g}")
    print(f"       tree: {best_off.tree}\n")

    # ---- with polish every 2 gens ----
    cfg_on = GPConfig(**common, optimize_constants_every=2,
                      optimize_constants_maxiter=100)
    t0 = time.time()
    front_on = GP(cfg_on).run(env, y, ["x"])
    t_on = time.time() - t0
    best_on = min(front_on, key=lambda c: c.train_loss)
    print(f"[on]   runtime={t_on:.2f}s, best cx={best_on.complexity}, "
          f"loss={best_on.train_loss:.4g}")
    print(f"       tree: {best_on.tree}\n")

    if best_off.train_loss > 0:
        improvement = best_off.train_loss / max(best_on.train_loss, 1e-30)
        print(f"const-opt improved best train loss by {improvement:.1f}x")
    print(f"target var = {y.var():.4f}; noise floor ~ 1e-4 (sigma=0.01)")
    print()
    print("Notes:")
    print("- the ON run usually terminates earlier (polish drives to a local")
    print("  min quickly, then mutation alone can't improve and patience ticks).")
    print("  That's the desired behaviour: don't waste budget after the polish")
    print("  has eaten the search horizon's improvement margin.")
    print("- neither GP run found the canonical 'a*x + b*x*x' form because the")
    print("  default operator weights favour growing complexity; the polish")
    print("  refines whatever structure the GP DID land on.")


if __name__ == "__main__":
    main()
