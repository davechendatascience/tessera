"""Differentiable SR — shared substrate + three search paradigms (D1).

Provenance: docs/research/differentiable_eml_jax.md. This module exists
to settle the structure-paradigm question empirically: build all three
searches on ONE shared program substrate so the head-to-head varies only
the SEARCH, not the representation.

Shared representation
---------------------
A fixed-depth program over slots:
  slots = [x0..x_{d-1}]  (data)
        + [c0..c_{m-1}]  (free constants — GRADIENT-tuned in every method)
        + [node_0 .. node_{n-1}]  (computed)
Each node i is a triple (op, left, right) reading earlier slots; the
output is the last node. The discrete part is (op, left, right) per node;
the continuous part is the m constants.

The crucial design choice: free gradient-tuned constants mean `sin(2x)`
is `sin(mul(c, x0))` with the STRUCTURE found by search and `c→2` found
by GRADIENT — the division of labour itself (gradients do O(P) parameter
tuning; search does structure).

Substrate (this file, shared):
  - op table (smooth singular ops so gradients flow)
  - `make_core_eval` : jittable/vmappable hard program evaluator
    (lax.scan + lax.switch opcode interpreter)
  - `make_refiner`   : Adam const-refinement for a FIXED structure
  - target suite, R² scoring, Result

Searches (built on the substrate):
  - Method C  : GP-evolution + gradient const-refinement   [this file]
  - Method A+ : relaxation + Gumbel-STE + L0                [added next]
  - Method B  : DSR-style learned policy                    [added next]

Status: untested. Graduation: the winning paradigm recovers the target
suite (incl. sin(2x)) with a fixed compute budget, WITHOUT relying on
restart luck; then Feynman + vs-GP. Removal: none beats the restart
baseline meaningfully.
Initial commit: 2026-05-29  Last evaluation: never
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------
# Operator table (smooth surrogates so gradients flow everywhere)
# --------------------------------------------------------------------

def make_ops(op_names: List[str]):
    """Return (branch_fns, arities) for the op set. Each branch is
    fn(xl, xr) -> array; unary ops ignore xr."""
    import jax.numpy as jnp

    def _div(a, b):
        return a * b / (b * b + 1e-6)

    def _inv(x):
        return x / (x * x + 1e-6)

    table = {
        "add":    (2, lambda xl, xr: xl + xr),
        "sub":    (2, lambda xl, xr: xl - xr),
        "mul":    (2, lambda xl, xr: xl * xr),
        "div":    (2, lambda xl, xr: _div(xl, xr)),
        "neg":    (1, lambda xl, xr: -xl),
        "square": (1, lambda xl, xr: xl * xl),
        "sqrt":   (1, lambda xl, xr: jnp.sqrt(jnp.abs(xl) + 1e-6)),
        "inv":    (1, lambda xl, xr: _inv(xl)),
        "sin":    (1, lambda xl, xr: jnp.sin(xl)),
        "cos":    (1, lambda xl, xr: jnp.cos(xl)),
        "tanh":   (1, lambda xl, xr: jnp.tanh(xl)),
        "exp":    (1, lambda xl, xr: jnp.exp(jnp.clip(xl, -30.0, 30.0))),
        "log":    (1, lambda xl, xr: jnp.log(jnp.abs(xl) + 1e-6)),
        "id":     (1, lambda xl, xr: xl),
    }
    branches = [table[n][1] for n in op_names]
    arities = [table[n][0] for n in op_names]
    return branches, arities


DEFAULT_OPS = ["add", "sub", "mul", "div", "neg", "square",
               "sqrt", "inv", "sin", "cos", "tanh", "id"]


# --------------------------------------------------------------------
# Shared program evaluator + constant refiner
# --------------------------------------------------------------------

def make_core_eval(d: int, m: int, n_nodes: int, branches):
    """Build a traceable hard-program evaluator.

    core_eval(ops, left, right, consts, X) -> (N,)
      ops/left/right : int (n_nodes,)
      consts         : float (m,)
      X              : float (N, d)
    Jittable, vmappable over programs and/or consts."""
    import jax.numpy as jnp
    from jax import lax
    n_slots = d + m + n_nodes

    def core_eval(ops, left, right, consts, X):
        N = X.shape[0]
        slots = jnp.zeros((n_slots, N))
        slots = slots.at[:d].set(X.T)
        if m > 0:
            slots = slots.at[d:d + m].set(jnp.broadcast_to(consts[:, None], (m, N)))

        def body(slots, i):
            xl = slots[left[i]]
            xr = slots[right[i]]
            res = lax.switch(ops[i], branches, xl, xr)
            res = jnp.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            res = jnp.clip(res, -1e6, 1e6)
            return slots.at[d + m + i].set(res), None

        slots, _ = lax.scan(body, slots, jnp.arange(n_nodes))
        return slots[-1]

    return core_eval


def make_refiner(core_eval, n_steps: int, lr: float = 0.05):
    """Adam refinement of the m constants for a FIXED structure.

    refine(ops, left, right, consts0, X, y) -> (consts, final_mse).
    Vmap over a population to refine all candidates in parallel."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    def refine(ops, left, right, consts0, X, y):
        def loss_c(c):
            pred = core_eval(ops, left, right, c, X)
            return jnp.mean((pred - y) ** 2)

        m0 = jnp.zeros_like(consts0)
        v0 = jnp.zeros_like(consts0)

        def step(carry, t):
            c, m, v = carry
            l, g = jax.value_and_grad(loss_c)(c)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (t + 1))
            vh = v / (1 - 0.999 ** (t + 1))
            c = c - lr * mh / (jnp.sqrt(vh) + 1e-8)
            return (c, m, v), l

        (c, _, _), losses = lax.scan(step, (consts0, m0, v0), jnp.arange(n_steps))
        return c, losses[-1]

    return refine


# --------------------------------------------------------------------
# Rendering + scoring
# --------------------------------------------------------------------

def render(ops, left, right, consts, op_names, d, m,
           var_names: Optional[List[str]] = None) -> str:
    """Render the slot-(d+m+i) chain as a readable expression string,
    inlining only the slots that feed the output (dead nodes dropped)."""
    if var_names is None:
        var_names = [f"x{i}" for i in range(d)]
    labels = list(var_names) + [f"{consts[j]:.4g}" for j in range(m)]
    _, arities = make_ops(op_names)
    exprs = list(labels)
    for i in range(len(ops)):
        op = op_names[int(ops[i])]
        ar = arities[int(ops[i])]
        l = exprs[int(left[i])]
        r = exprs[int(right[i])]
        e = (f"{op}({l})" if ar == 1 else f"{op}({l}, {r})")
        exprs.append(e)
    return exprs[-1]


def r2_of(mse: float, y: np.ndarray) -> float:
    var = float(np.var(y)) + 1e-30
    return float(1.0 - mse / var)


@dataclass
class SRResult:
    method: str
    expr: str
    r2: float
    recovered: bool
    evals: int               # const-refinements spent (compute proxy)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------
# Target suite
# --------------------------------------------------------------------

def target_suite(seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    x = rng.uniform(-2, 2, size=(400, 1))
    out.append(("x^2", x.astype(np.float32), (x[:, 0] ** 2).astype(np.float32)))
    x = rng.uniform(-3, 3, size=(400, 1))
    out.append(("sin(2x)", x.astype(np.float32), np.sin(2 * x[:, 0]).astype(np.float32)))
    x = rng.uniform(-2, 2, size=(400, 2))
    out.append(("x0*x1", x.astype(np.float32), (x[:, 0] * x[:, 1]).astype(np.float32)))
    x = rng.uniform(0.5, 3, size=(400, 2))
    out.append(("x0/x1", x.astype(np.float32), (x[:, 0] / x[:, 1]).astype(np.float32)))
    return out


# --------------------------------------------------------------------
# Method C: GP-evolution + gradient const-refinement
# --------------------------------------------------------------------

@dataclass
class EvoConfig:
    n_nodes: int = 6
    m_consts: int = 2
    op_names: List[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    pop: int = 60
    gens: int = 25
    inner_steps: int = 120
    lr: float = 0.05
    tournament_k: int = 4
    elite_frac: float = 0.1
    p_mut_op: float = 0.25
    p_mut_wire: float = 0.25
    seed: int = 0


def _random_programs(rng, P, d, m, n, K):
    ops = rng.integers(0, K, size=(P, n))
    left = np.zeros((P, n), dtype=np.int64)
    right = np.zeros((P, n), dtype=np.int64)
    for i in range(n):
        hi = d + m + i                      # node i may read slots 0..hi-1
        left[:, i] = rng.integers(0, hi, size=P)
        right[:, i] = rng.integers(0, hi, size=P)
    return ops, left, right


def _crossover(a, b, rng):
    """Per-node uniform crossover (building-block recombination): each
    node's whole (op,left,right) triple comes from parent a or b."""
    n = a[0].shape[0]
    mask = rng.random(n) < 0.5
    ops = np.where(mask, a[0], b[0])
    left = np.where(mask, a[1], b[1])
    right = np.where(mask, a[2], b[2])
    return ops, left, right


def _mutate(prog, rng, d, m, K, cfg):
    ops, left, right = (prog[0].copy(), prog[1].copy(), prog[2].copy())
    n = ops.shape[0]
    for i in range(n):
        if rng.random() < cfg.p_mut_op:
            ops[i] = rng.integers(0, K)
        hi = d + m + i
        if rng.random() < cfg.p_mut_wire:
            left[i] = rng.integers(0, hi)
        if rng.random() < cfg.p_mut_wire:
            right[i] = rng.integers(0, hi)
    return ops, left, right


def evo_search(X, y, cfg: Optional[EvoConfig] = None,
               recover_thresh: float = 0.9999) -> SRResult:
    """GP-evolution over discrete programs with per-candidate gradient
    const-refinement. Selection + crossover supply the building-block
    structure intelligence; Adam supplies O(P) constant tuning."""
    import jax
    import jax.numpy as jnp

    cfg = cfg or EvoConfig()
    d = X.shape[1]; m = cfg.m_consts; n = cfg.n_nodes
    K = len(cfg.op_names)
    branches, _ = make_ops(cfg.op_names)
    core = make_core_eval(d, m, n, branches)
    refine = make_refiner(core, cfg.inner_steps, cfg.lr)
    vrefine = jax.jit(jax.vmap(refine, in_axes=(0, 0, 0, 0, None, None)))

    Xj = jnp.asarray(X, jnp.float32)
    yj = jnp.asarray(y, jnp.float32)
    rng = np.random.default_rng(cfg.seed)

    ops, left, right = _random_programs(rng, cfg.pop, d, m, n, K)
    consts = rng.normal(scale=1.0, size=(cfg.pop, m)).astype(np.float32)

    best = None
    n_elite = max(1, int(cfg.elite_frac * cfg.pop))
    evals = 0
    for gen in range(cfg.gens):
        c_ref, mse = vrefine(jnp.asarray(ops), jnp.asarray(left),
                             jnp.asarray(right), jnp.asarray(consts), Xj, yj)
        c_ref = np.asarray(c_ref); mse = np.asarray(mse)
        evals += cfg.pop
        r2 = np.array([r2_of(float(mse[i]), y) for i in range(cfg.pop)])
        r2 = np.nan_to_num(r2, nan=-1e9, posinf=-1e9, neginf=-1e9)
        consts = c_ref                          # carry refined constants

        gi = int(np.argmax(r2))
        if best is None or r2[gi] > best[0]:
            best = (float(r2[gi]),
                    (ops[gi].copy(), left[gi].copy(), right[gi].copy(),
                     consts[gi].copy()))
        if best[0] > recover_thresh:
            break

        # selection + breeding
        order = np.argsort(-r2)
        elites = order[:n_elite]
        new_ops = [ops[e].copy() for e in elites]
        new_left = [left[e].copy() for e in elites]
        new_right = [right[e].copy() for e in elites]
        new_consts = [consts[e].copy() for e in elites]

        def tourney():
            cand = rng.choice(cfg.pop, size=cfg.tournament_k, replace=False)
            return cand[np.argmax(r2[cand])]

        while len(new_ops) < cfg.pop:
            pa, pb = tourney(), tourney()
            child = _crossover((ops[pa], left[pa], right[pa]),
                               (ops[pb], left[pb], right[pb]), rng)
            child = _mutate(child, rng, d, m, K, cfg)
            new_ops.append(child[0]); new_left.append(child[1]); new_right.append(child[2])
            # inherit fitter parent's constants (+ small noise), then re-refine
            src = pa if r2[pa] >= r2[pb] else pb
            new_consts.append(consts[src] + rng.normal(scale=0.1, size=m).astype(np.float32))

        ops = np.stack(new_ops); left = np.stack(new_left)
        right = np.stack(new_right); consts = np.stack(new_consts).astype(np.float32)

    r2b, (bo, bl, br, bc) = best
    expr = render(bo, bl, br, bc, cfg.op_names, d, m)
    return SRResult(method="C:evo+grad", expr=expr, r2=r2b,
                    recovered=r2b > recover_thresh, evals=evals,
                    info={"n_nodes": n, "m_consts": m})


__all__ = [
    "make_ops", "make_core_eval", "make_refiner", "render", "r2_of",
    "SRResult", "target_suite", "EvoConfig", "evo_search", "DEFAULT_OPS",
]
