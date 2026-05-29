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


# --------------------------------------------------------------------
# Method A+: relaxation + Gumbel-softmax straight-through + entropy L0
# --------------------------------------------------------------------

@dataclass
class RelaxConfig:
    n_nodes: int = 6
    m_consts: int = 1
    op_names: List[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    n_steps: int = 3000
    lr: float = 0.02
    tau0: float = 2.0
    tau1: float = 0.1
    lam_max: float = 1e-2
    sparsity_warmup_frac: float = 0.4
    use_gumbel: bool = False     # deterministic straight-through by default
    n_restarts: int = 16         # A+ is single-run-ish, not a 256-ticket lottery
    seed: int = 0


def _st(logits, tau, key, use_gumbel=False):
    """Straight-through selection: hard one-hot in the forward pass (so
    the program is DISCRETE — closes the discretization gap), soft
    gradient in the backward pass. Gumbel noise optional (off by default;
    it was too disruptive on small programs)."""
    import jax
    import jax.numpy as jnp
    z = logits + jax.random.gumbel(key, logits.shape) if use_gumbel else logits
    y = jax.nn.softmax(z / tau)
    y_hard = jax.nn.one_hot(jnp.argmax(y), logits.shape[0])
    return jax.lax.stop_gradient(y_hard - y) + y


def relax_search(X, y, cfg: Optional[RelaxConfig] = None,
                 recover_thresh: float = 0.9999) -> SRResult:
    """Differentiable relaxation with Gumbel-STE + annealed entropy
    sparsity. The forward pass evaluates SAMPLED DISCRETE programs (STE),
    so the optimizer sees real picks rather than a smeared blend."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    cfg = cfg or RelaxConfig()
    d = X.shape[1]; m = cfg.m_consts; n = cfg.n_nodes
    K = len(cfg.op_names)
    P = d + m + n - 1
    branches, _ = make_ops(cfg.op_names)
    core = make_core_eval(d, m, n, branches)
    Xj = jnp.asarray(X, jnp.float32); yj = jnp.asarray(y, jnp.float32)
    tmap = jax.tree_util.tree_map

    def init(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {
            "alpha": 0.01 * jax.random.normal(k1, (n, K)),
            "bl": 0.01 * jax.random.normal(k2, (n, P)),
            "br": 0.01 * jax.random.normal(k3, (n, P)),
            "consts": jax.random.normal(k4, (m,)),
        }

    def forward(params, tau, key):
        slots = [Xj[:, j] for j in range(d)] + \
                [jnp.full((Xj.shape[0],), params["consts"][j]) for j in range(m)]
        keys = jax.random.split(key, n * 3)
        for i in range(n):
            navail = d + m + i
            prev = jnp.stack(slots, axis=0)
            ug = cfg.use_gumbel
            wl = _st(params["bl"][i, :navail], tau, keys[3 * i], ug)
            wr = _st(params["br"][i, :navail], tau, keys[3 * i + 1], ug)
            xl = jnp.einsum("p,pn->n", wl, prev)
            xr = jnp.einsum("p,pn->n", wr, prev)
            opvals = jnp.stack([f(xl, xr) for f in branches], axis=0)
            wo = _st(params["alpha"][i], tau, keys[3 * i + 2], ug)
            node = jnp.einsum("k,kn->n", wo, opvals)
            node = jnp.clip(jnp.nan_to_num(node, nan=0.0, posinf=1e6, neginf=-1e6),
                            -1e6, 1e6)
            slots.append(node)
        return slots[-1]

    def entropy(params, tau):
        tot = 0.0
        for i in range(n):
            navail = d + m + i
            for lg in (params["alpha"][i], params["bl"][i, :navail],
                       params["br"][i, :navail]):
                p = jax.nn.softmax(lg / tau)
                tot = tot + -jnp.sum(p * jnp.log(p + 1e-12))
        return tot

    def loss_fn(params, tau, lam, key):
        pred = forward(params, tau, key)
        return jnp.mean((pred - yj) ** 2) + lam * entropy(params, tau)

    frac = jnp.arange(cfg.n_steps) / max(cfg.n_steps - 1, 1)
    taus = cfg.tau1 + 0.5 * (cfg.tau0 - cfg.tau1) * (1 + jnp.cos(jnp.pi * frac))
    warm = cfg.sparsity_warmup_frac
    lams = jnp.where(frac < warm, 0.0,
                     cfg.lam_max * (frac - warm) / max(1.0 - warm, 1e-6))

    def train(key):
        params = init(key)
        m0 = tmap(jnp.zeros_like, params); v0 = tmap(jnp.zeros_like, params)
        step_keys = jax.random.split(jax.random.fold_in(key, 99), cfg.n_steps)

        def step(carry, inp):
            params, mm, vv = carry
            t, tau, lam, k = inp
            l, g = jax.value_and_grad(loss_fn)(params, tau, lam, k)
            mm = tmap(lambda a, b: 0.9 * a + 0.1 * b, mm, g)
            vv = tmap(lambda a, b: 0.999 * a + 0.001 * b * b, vv, g)
            bc1 = 1 - 0.9 ** (t + 1); bc2 = 1 - 0.999 ** (t + 1)
            params = tmap(lambda p, a, b: p - cfg.lr * (a / bc1) / (jnp.sqrt(b / bc2) + 1e-8),
                          params, mm, vv)
            return (params, mm, vv), l

        (params, _, _), _ = lax.scan(step, (params, m0, v0),
                                     (jnp.arange(cfg.n_steps), taus, lams, step_keys))
        return params

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.n_restarts)
    trained = jax.jit(jax.vmap(train))(keys)

    # discretize each restart -> hard program -> score
    best = None
    for r in range(cfg.n_restarts):
        pr = {k: np.asarray(v[r]) for k, v in trained.items()}
        ops = np.array([int(np.argmax(pr["alpha"][i])) for i in range(n)])
        left = np.array([int(np.argmax(pr["bl"][i, :d + m + i])) for i in range(n)])
        right = np.array([int(np.argmax(pr["br"][i, :d + m + i])) for i in range(n)])
        consts = pr["consts"]
        pred = np.asarray(core(jnp.asarray(ops), jnp.asarray(left),
                               jnp.asarray(right), jnp.asarray(consts), Xj))
        mse = float(np.mean((pred - y) ** 2))
        r2 = r2_of(mse, y)
        if best is None or r2 > best[0]:
            best = (r2, ops, left, right, consts)

    r2b, bo, bl_, br_, bc = best
    expr = render(bo, bl_, br_, bc, cfg.op_names, d, m)
    return SRResult(method="A+:gumbel-ste", expr=expr, r2=r2b,
                    recovered=r2b > recover_thresh,
                    evals=cfg.n_restarts * cfg.n_steps,
                    info={"n_restarts": cfg.n_restarts})


# --------------------------------------------------------------------
# Method B: learned policy over discrete programs (CEM-style)
# --------------------------------------------------------------------
# A distribution over programs (per-node categoricals) that LEARNS from
# reward: each iteration samples programs, refines their constants, and
# concentrates probability mass on the elite structures. The logits are
# cross-attempt memory — the search gets smarter, unlike independent
# restarts. (Simplified DSR: a learned distribution + risk-seeking elite
# update; no autoregressive controller. The "learned search" property is
# the point of comparison.)

@dataclass
class PolicyConfig:
    n_nodes: int = 6
    m_consts: int = 2
    op_names: List[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    pop: int = 80
    iters: int = 25
    inner_steps: int = 120
    lr: float = 0.05
    elite_frac: float = 0.2
    smooth: float = 0.5         # CEM logit blend toward elite frequencies
    seed: int = 0


def _softmax_np(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _sample_programs(op_logits, left_logits, right_logits, pop, d, m, n, rng):
    K = op_logits.shape[1]
    ops = np.zeros((pop, n), dtype=np.int64)
    left = np.zeros((pop, n), dtype=np.int64)
    right = np.zeros((pop, n), dtype=np.int64)
    for i in range(n):
        navail = d + m + i
        ops[:, i] = rng.choice(K, size=pop, p=_softmax_np(op_logits[i]))
        left[:, i] = rng.choice(navail, size=pop, p=_softmax_np(left_logits[i, :navail]))
        right[:, i] = rng.choice(navail, size=pop, p=_softmax_np(right_logits[i, :navail]))
    return ops, left, right


def policy_search(X, y, cfg: Optional[PolicyConfig] = None,
                  recover_thresh: float = 0.9999) -> SRResult:
    """Learned-distribution (CEM) search over discrete programs with
    per-sample gradient const-refinement."""
    import jax
    import jax.numpy as jnp

    cfg = cfg or PolicyConfig()
    d = X.shape[1]; m = cfg.m_consts; n = cfg.n_nodes
    K = len(cfg.op_names)
    P = d + m + n - 1
    branches, _ = make_ops(cfg.op_names)
    core = make_core_eval(d, m, n, branches)
    refine = make_refiner(core, cfg.inner_steps, cfg.lr)
    vrefine = jax.jit(jax.vmap(refine, in_axes=(0, 0, 0, 0, None, None)))
    Xj = jnp.asarray(X, jnp.float32); yj = jnp.asarray(y, jnp.float32)
    rng = np.random.default_rng(cfg.seed)

    op_logits = np.zeros((n, K)); left_logits = np.zeros((n, P)); right_logits = np.zeros((n, P))
    s = cfg.smooth
    n_elite = max(2, int(cfg.elite_frac * cfg.pop))
    best = None
    evals = 0
    for it in range(cfg.iters):
        ops, left, right = _sample_programs(op_logits, left_logits, right_logits,
                                            cfg.pop, d, m, n, rng)
        consts0 = rng.normal(size=(cfg.pop, m)).astype(np.float32)
        c_ref, mse = vrefine(jnp.asarray(ops), jnp.asarray(left), jnp.asarray(right),
                             jnp.asarray(consts0), Xj, yj)
        c_ref = np.asarray(c_ref); mse = np.asarray(mse)
        evals += cfg.pop
        r2 = np.nan_to_num(np.array([r2_of(float(mse[i]), y) for i in range(cfg.pop)]),
                           nan=-1e9, posinf=-1e9, neginf=-1e9)

        gi = int(np.argmax(r2))
        if best is None or r2[gi] > best[0]:
            best = (float(r2[gi]), ops[gi].copy(), left[gi].copy(),
                    right[gi].copy(), c_ref[gi].copy())
        if best[0] > recover_thresh:
            break

        elite = np.argsort(-r2)[:n_elite]
        for i in range(n):
            navail = d + m + i
            fo = np.bincount(ops[elite, i], minlength=K) / n_elite
            op_logits[i] = (1 - s) * op_logits[i] + s * np.log(fo + 1e-6)
            fl = np.bincount(left[elite, i], minlength=navail) / n_elite
            left_logits[i, :navail] = (1 - s) * left_logits[i, :navail] + s * np.log(fl + 1e-6)
            fr = np.bincount(right[elite, i], minlength=navail) / n_elite
            right_logits[i, :navail] = (1 - s) * right_logits[i, :navail] + s * np.log(fr + 1e-6)

    r2b, bo, bl_, br_, bc = best
    expr = render(bo, bl_, br_, bc, cfg.op_names, d, m)
    return SRResult(method="B:learned-policy", expr=expr, r2=r2b,
                    recovered=r2b > recover_thresh, evals=evals,
                    info={"iters": cfg.iters, "pop": cfg.pop})


# --------------------------------------------------------------------
# Method D: structure search as a CSP — backtracking enumeration
# (Knuth TAOCP Vol 4 Fascicle 7, §7.2.2.3)
# --------------------------------------------------------------------
# The structure search is a CSP: variables = (op, left, right) per node;
# constraints = wiring validity, arity, CONNECTIVITY (root depends on
# inputs), SYMMETRY-BREAKING (commutative ordering, dedup). We enumerate
# canonical expression TREES by increasing size (iterative deepening):
#   - connectivity is automatic (every tree node feeds the root),
#   - symmetry-breaking via commutative child ordering + canonical-key
#     dedup kills the redundant equivalents GP/CEM re-sample,
#   - smallest-first = complete + parsimonious (finds the minimal program
#     that fits, the conditional-guarantee the relaxation can't give).
# Each enumerated skeleton's constants are refined in parallel on the GPU
# (the shared `vrefine`). Deterministic: no restart lottery.

_D_UNARY = ["neg", "square", "sqrt", "inv", "sin", "cos", "tanh"]
_D_BINARY = ["add", "sub", "mul", "div"]
_D_COMM = {"add", "mul"}


@dataclass
class CSPConfig:
    op_names: List[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    max_size: int = 4            # max feature-tree size (operator nodes)
    max_terms: int = 4           # OMP terms (sparsity of the linear fit)
    max_features: int = 40000    # cap on the enumerated dictionary (logged)
    seed: int = 0                # unused (D is deterministic); kept for API


def _tree_key(t):
    if t[0] == "var":
        return f"v{t[1]}"
    if len(t) == 2:
        return f"{t[0]}({_tree_key(t[1])})"
    return f"{t[0]}({_tree_key(t[1])},{_tree_key(t[2])})"


def _gen_trees(size, d, memo):
    """All canonical distinct CONSTANT-FREE expression trees with exactly
    `size` operator nodes (leaves = variables only; symmetry-broken:
    commutative children ordered, dups removed). Constants enter later as
    LINEAR coefficients, not as leaves — so there is no gradient descent."""
    if size in memo:
        return memo[size]
    out, seen = [], set()

    def add(t):
        k = _tree_key(t)
        if k not in seen:
            seen.add(k); out.append(t)

    if size == 0:
        for j in range(d):
            add(("var", j))
        memo[size] = out
        return out
    for op in _D_UNARY:
        for c in _gen_trees(size - 1, d, memo):
            add((op, c))
    for op in _D_BINARY:
        for ls in range(size):
            L = _gen_trees(ls, d, memo)
            R = _gen_trees(size - 1 - ls, d, memo)
            comm = op in _D_COMM
            for cl in L:
                kl = _tree_key(cl)
                for cr in R:
                    if comm and kl > _tree_key(cr):
                        continue           # symmetry-break commutative ops
                    add((op, cl, cr))
    memo[size] = out
    return out


def _render_tree(t):
    if t[0] == "var":
        return f"x{t[1]}"
    if len(t) == 2:
        return f"{t[0]}({_render_tree(t[1])})"
    return f"{t[0]}({_render_tree(t[1])}, {_render_tree(t[2])})"


def _tree_size(t):
    if t[0] == "var":
        return 0
    if len(t) == 2:
        return 1 + _tree_size(t[1])
    return 1 + _tree_size(t[1]) + _tree_size(t[2])


# Numpy feature evaluators (smooth surrogates, matching make_ops). Feature
# evaluation is pure numpy: no JAX compile, no gradient descent — the whole
# point of Method D's revision.
def _np_div(a, b):
    return a * b / (b * b + 1e-6)


_NP_UNARY = {
    "neg": lambda x: -x, "square": lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.abs(x) + 1e-6),
    "inv": lambda x: x / (x * x + 1e-6),
    "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
}
_NP_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": _np_div,
}


def _np_eval(t, X):
    if t[0] == "var":
        return X[:, t[1]]
    if len(t) == 2:
        return _NP_UNARY[t[0]](_np_eval(t[1], X))
    return _NP_BINARY[t[0]](_np_eval(t[1], X), _np_eval(t[2], X))


def _omp(Phi, y, sizes, max_terms, recover_thresh):
    """Orthogonal matching pursuit: greedily add the feature most
    correlated with the residual (with a small parsimony tie-break that
    prefers the SMALLEST feature among near-equal correlations), refit by
    closed-form least squares, repeat. No gradient descent — the residual-
    guided selection IS the fit feeding back into the search."""
    N, F = Phi.shape
    ones = np.ones((N, 1))
    ybar = float(y.mean())
    ss_tot = float(np.sum((y - ybar) ** 2)) + 1e-30
    Phic = Phi - Phi.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Phic, axis=0) + 1e-12
    sizes = np.asarray(sizes, dtype=np.float64)
    selected, coef, intercept = [], np.zeros(0), ybar
    r = y - ybar
    best_r2 = 0.0
    for _ in range(max_terms):
        rn = np.linalg.norm(r) + 1e-12
        rho = np.abs(Phic.T @ r) / (norms * rn)      # normalized corr in [0,1]
        scores = rho - 1e-4 * sizes                  # parsimony tie-break
        if selected:
            scores[selected] = -1e9
        k = int(np.argmax(scores))
        trial = selected + [k]
        A = np.hstack([ones, Phi[:, trial]])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r_new = y - A @ c
        r2 = 1.0 - float(np.sum(r_new ** 2)) / ss_tot
        if r2 <= best_r2 + 1e-9 and selected:
            break                            # no improvement
        selected, coef, intercept, r, best_r2 = trial, c[1:], float(c[0]), r_new, r2
        if best_r2 > recover_thresh:
            break
    return selected, coef, intercept, best_r2


def csp_search(X, y, cfg: Optional[CSPConfig] = None,
               recover_thresh: float = 0.9999, log=lambda s: None) -> SRResult:
    """Method D: CSP-enumerated feature dictionary + sparse LINEAR fit.

    The CSP enumerates constant-free feature trees (the dictionary); the
    coefficients are fit by orthogonal matching pursuit (closed-form least
    squares per step). Gradient-free, no JAX, no iterative refinement —
    the design matrix `Phi` IS the 'weights compressed onto one structure',
    solved once. Returns the sparse linear combination of features that
    fits. Constants buried inside nonlinearities (e.g. sin(c·x), irrational
    c) are the known limit; a 1-step Gauss-Newton refine on the selected
    feature is the cheap extension (not needed for linear-in-params forms).
    """
    cfg = cfg or CSPConfig()
    d = X.shape[1]; K = cfg.max_size
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)

    # ---- enumerate the const-free feature dictionary (smallest first) ----
    memo = {}
    feats, cols, sizes, seen_cols = [], [], [], set()
    for size in range(0, K + 1):
        for t in _gen_trees(size, d, memo):
            col = _np_eval(t, X).astype(np.float64)
            if not np.all(np.isfinite(col)) or float(np.std(col)) < 1e-9:
                continue                       # drop NaN/inf and constants
            key = hash(np.round(col, 6).tobytes())
            if key in seen_cols:
                continue                       # numerical dedup (e.g. neg(neg))
            seen_cols.add(key)
            feats.append(t); cols.append(col); sizes.append(_tree_size(t))
            if len(feats) >= cfg.max_features:
                log(f"feature cap {cfg.max_features} hit at size {size}")
                break
        if len(feats) >= cfg.max_features:
            break
    Phi = np.stack(cols, axis=1)               # (N, F) design matrix
    log(f"dictionary: {Phi.shape[1]} features (sizes 0..{K})")

    # ---- sparse linear fit (OMP), gradient-free ----
    selected, coef, intercept, r2 = _omp(Phi, y, sizes, cfg.max_terms, recover_thresh)

    # ---- render ----
    parts = []
    if abs(intercept) > 1e-3:
        parts.append(f"{intercept:.4g}")
    for c, k in zip(coef, selected):
        if abs(c) < 1e-6:
            continue
        fr = _render_tree(feats[k])
        parts.append(fr if abs(c - 1.0) < 1e-3 else f"{c:.4g}*{fr}")
    expr = " + ".join(parts) if parts else "0"
    return SRResult(method="D:csp+lstsq", expr=expr, r2=r2,
                    recovered=r2 > recover_thresh, evals=Phi.shape[1],
                    info={"max_size": K, "n_terms": len(selected)})


__all__ = [
    "make_ops", "make_core_eval", "make_refiner", "render", "r2_of",
    "SRResult", "target_suite", "EvoConfig", "evo_search",
    "RelaxConfig", "relax_search", "PolicyConfig", "policy_search",
    "CSPConfig", "csp_search", "DEFAULT_OPS",
]
