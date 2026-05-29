"""Differentiable SR via a JAX EML super-graph (Conjecture D1).

Provenance: docs/research/differentiable_eml_jax.md. Ported to JAX from
market-analysis/src/lib/diff_eml (Yarotsky universal-operator DARTS).

Status: untested at module-add time.
Graduation criterion: D1 holds (best-of-R restarts beats single-init on
  x²/sin(2x)/x₀·x₁ and recovers them), then Feynman + vs-GP head-to-head.
Removal criterion: parallel restarts don't beat single-init, or the
  engine can't recover the simple targets.
Initial commit: 2026-05-29
Last evaluation: never

What this is
------------
A fixed-depth DAG. Each internal node holds:
  - alpha: logits over K operators (which operator)
  - beta_left/right: logits over previous node outputs (input wiring)
The forward pass is a SOFT mixture (softmax(·/τ)); τ is annealed soft→
hard, then the program is snapped to discrete (argmax). Differentiable
end-to-end, so one Adam step updates all selection logits at once —
O(P) credit assignment (the property GP lacks).

The whole point — Tier-1 local-minima engineering (see the note):
  - PARALLEL RESTARTS, vmap'd over the GPU. Selection is by hard-program
    MSE, not soft loss. "Scale buys robustness."
  - Delayed-sparsity + cosine-τ annealing (fit before committing).
  - Smooth singular ops (div/inv/sqrt) so gradients flow everywhere.

JAX is imported lazily inside functions (the module imports without jax;
running requires the `jax` extra). Adam is hand-rolled (no optax dep).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------
# Operator dictionary (forward-pass functions)
# --------------------------------------------------------------------
# Each op is fn(xl, xr) -> array. Constants ignore inputs (broadcast);
# unary ops use xl; binary ops use both. Singular ops are SMOOTH
# surrogates (Tier-1) so gradients flow everywhere, unlike hard clamp.

def _build_ops(op_names: List[str]):
    import jax.numpy as jnp

    def _div(a, b):  # smooth: a/b for |b|≫√ε, →0 near b=0 (no blow-up)
        return a * b / (b * b + 1e-6)

    def _inv(x):
        return x / (x * x + 1e-6)

    table = {
        "one":     (0, lambda xl, xr: jnp.ones_like(xl)),
        "two":     (0, lambda xl, xr: 2.0 * jnp.ones_like(xl)),
        "half":    (0, lambda xl, xr: 0.5 * jnp.ones_like(xl)),
        "neg_one": (0, lambda xl, xr: -jnp.ones_like(xl)),
        "add":     (2, lambda xl, xr: xl + xr),
        "sub":     (2, lambda xl, xr: xl - xr),
        "mul":     (2, lambda xl, xr: xl * xr),
        "div":     (2, lambda xl, xr: _div(xl, xr)),
        "neg":     (1, lambda xl, xr: -xl),
        "square":  (1, lambda xl, xr: xl * xl),
        "sqrt":    (1, lambda xl, xr: jnp.sqrt(jnp.abs(xl) + 1e-6)),
        "inv":     (1, lambda xl, xr: _inv(xl)),
        "sin":     (1, lambda xl, xr: jnp.sin(xl)),
        "cos":     (1, lambda xl, xr: jnp.cos(xl)),
        "tanh":    (1, lambda xl, xr: jnp.tanh(xl)),
        "exp":     (1, lambda xl, xr: jnp.exp(jnp.clip(xl, -30.0, 30.0))),
        "log":     (1, lambda xl, xr: jnp.log(jnp.abs(xl) + 1e-6)),
    }
    ops = []
    for name in op_names:
        if name not in table:
            raise ValueError(f"unknown op {name!r}")
        arity, fn = table[name]
        ops.append((name, arity, fn))
    return ops


DEFAULT_OPS = ["one", "two", "neg_one", "add", "sub", "mul", "div",
               "neg", "square", "sqrt", "inv", "sin", "cos", "tanh"]


@dataclass
class EMLConfig:
    n_inputs: int = 1
    n_internal: int = 4
    op_names: List[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    n_steps: int = 1500
    lr: float = 0.05
    n_restarts: int = 64
    tau0: float = 2.0
    tau1: float = 0.05
    lam_max: float = 1e-2
    sparsity_warmup_frac: float = 0.5   # λ=0 until this fraction of steps
    seed: int = 0


# --------------------------------------------------------------------
# Parameters + forward
# --------------------------------------------------------------------

def _init_params(key, cfg: EMLConfig):
    import jax
    import jax.numpy as jnp
    K = len(cfg.op_names)
    P = cfg.n_inputs + cfg.n_internal - 1            # max prev outputs
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "alpha": 0.01 * jax.random.normal(k1, (cfg.n_internal, K)),
        "beta_left": 0.01 * jax.random.normal(k2, (cfg.n_internal, P)),
        "beta_right": 0.01 * jax.random.normal(k3, (cfg.n_internal, P)),
    }


def _forward(params, X, tau, cfg: EMLConfig, ops, hard=False):
    """X: (N, n_inputs) -> (N,). hard=True uses argmax (discrete program)."""
    import jax
    import jax.numpy as jnp

    outs = [X[:, j] for j in range(cfg.n_inputs)]
    for i in range(cfg.n_internal):
        navail = cfg.n_inputs + i
        prev = jnp.stack(outs, axis=0)               # (navail, N)
        al = params["alpha"][i]
        bl = params["beta_left"][i, :navail]
        br = params["beta_right"][i, :navail]
        if hard:
            wo = jax.nn.one_hot(jnp.argmax(al), al.shape[0])
            wl = jax.nn.one_hot(jnp.argmax(bl), navail)
            wr = jax.nn.one_hot(jnp.argmax(br), navail)
        else:
            wo = jax.nn.softmax(al / tau)
            wl = jax.nn.softmax(bl / tau)
            wr = jax.nn.softmax(br / tau)
        xl = jnp.einsum("p,pn->n", wl, prev)
        xr = jnp.einsum("p,pn->n", wr, prev)
        opvals = jnp.stack([fn(xl, xr) for (_, _, fn) in ops], axis=0)  # (K, N)
        node = jnp.einsum("k,kn->n", wo, opvals)
        node = jnp.nan_to_num(node, nan=0.0, posinf=1e6, neginf=-1e6)
        node = jnp.clip(node, -1e6, 1e6)
        outs.append(node)
    return outs[-1]


def _entropy_sparsity(params, tau, cfg: EMLConfig):
    import jax
    import jax.numpy as jnp
    tot = 0.0
    for i in range(cfg.n_internal):
        navail = cfg.n_inputs + i
        for logits in (params["alpha"][i],
                       params["beta_left"][i, :navail],
                       params["beta_right"][i, :navail]):
            p = jax.nn.softmax(logits / tau)
            tot = tot + -jnp.sum(p * jnp.log(p + 1e-12))
    return tot


def _loss(params, X, y, tau, lam, cfg, ops):
    import jax.numpy as jnp
    pred = _forward(params, X, tau, cfg, ops, hard=False)
    mse = jnp.mean((pred - y) ** 2)
    return mse + lam * _entropy_sparsity(params, tau, cfg)


# --------------------------------------------------------------------
# Training (hand-rolled Adam, lax.scan over steps, vmap over restarts)
# --------------------------------------------------------------------

def _schedules(cfg: EMLConfig):
    import jax.numpy as jnp
    t = jnp.arange(cfg.n_steps)
    frac = t / max(cfg.n_steps - 1, 1)
    tau = cfg.tau1 + 0.5 * (cfg.tau0 - cfg.tau1) * (1 + jnp.cos(jnp.pi * frac))
    warm = cfg.sparsity_warmup_frac
    lam = jnp.where(frac < warm, 0.0,
                    cfg.lam_max * (frac - warm) / max(1.0 - warm, 1e-6))
    return t, tau, lam


def _train_one(params0, X, y, cfg, ops):
    import jax
    import jax.numpy as jnp
    from jax import lax
    tmap = jax.tree_util.tree_map

    m0 = tmap(jnp.zeros_like, params0)
    v0 = tmap(jnp.zeros_like, params0)
    t_arr, tau_arr, lam_arr = _schedules(cfg)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, cfg.lr

    def step(carry, inp):
        params, m, v = carry
        t, tau, lam = inp
        loss, grads = jax.value_and_grad(_loss)(params, X, y, tau, lam, cfg, ops)
        m = tmap(lambda mm, g: b1 * mm + (1 - b1) * g, m, grads)
        v = tmap(lambda vv, g: b2 * vv + (1 - b2) * g * g, v, grads)
        bc1 = 1 - b1 ** (t + 1)
        bc2 = 1 - b2 ** (t + 1)
        params = tmap(
            lambda p, mm, vv: p - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps),
            params, m, v)
        return (params, m, v), loss

    (params, _, _), losses = lax.scan(step, (params0, m0, v0),
                                      (t_arr, tau_arr, lam_arr))
    return params, losses


# --------------------------------------------------------------------
# Discrete read-off + reporting
# --------------------------------------------------------------------

def _hard_program(params_np, cfg: EMLConfig):
    """Read the discrete (op, left, right) choice at each node from numpy
    params via argmax."""
    prog = []
    for i in range(cfg.n_internal):
        navail = cfg.n_inputs + i
        op = int(np.argmax(params_np["alpha"][i]))
        li = int(np.argmax(params_np["beta_left"][i, :navail]))
        ri = int(np.argmax(params_np["beta_right"][i, :navail]))
        prog.append((op, li, ri))
    return prog


def render_program(params_np, cfg: EMLConfig,
                   var_names: Optional[List[str]] = None) -> str:
    if var_names is None:
        var_names = [f"x{i}" for i in range(cfg.n_inputs)]
    labels = list(var_names)
    lines = []
    for i, (op, li, ri) in enumerate(_hard_program(params_np, cfg)):
        name, arity, _ = (cfg.op_names[op], None, None)
        # arity lookup without building jax ops
        arity = {"one": 0, "two": 0, "half": 0, "neg_one": 0}.get(name, None)
        if arity is None:
            arity = 1 if name in {"neg", "square", "sqrt", "inv", "sin",
                                  "cos", "tanh", "exp", "log"} else 2
        l, r = labels[li], labels[ri]
        expr = name if arity == 0 else (f"{name}({l})" if arity == 1
                                        else f"{name}({l}, {r})")
        labels.append(f"({expr})")
        lines.append(f"n{i} = {expr}")
    lines.append(f"root = {labels[-1]}")
    return "\n".join(lines)


@dataclass
class DiffEMLResult:
    best_params: dict
    best_r2: float
    hit_rate: float          # fraction of restarts with R² > recover_thresh
    median_r2: float
    all_r2: np.ndarray
    program: str
    cfg: EMLConfig


def discover(X, y, cfg: Optional[EMLConfig] = None,
             recover_thresh: float = 0.9999,
             var_names: Optional[List[str]] = None) -> DiffEMLResult:
    """Run R parallel restarts, select the best by HARD-program R².

    Returns the best discrete program + the restart-hit-rate (the D1
    measurement: how much parallel restarts buy over a single init)."""
    import jax
    import jax.numpy as jnp

    cfg = cfg or EMLConfig(n_inputs=X.shape[1])
    if cfg.n_inputs != X.shape[1]:
        cfg = EMLConfig(**{**cfg.__dict__, "n_inputs": X.shape[1]})
    ops = _build_ops(cfg.op_names)
    Xj = jnp.asarray(np.asarray(X), dtype=jnp.float32)
    yj = jnp.asarray(np.asarray(y), dtype=jnp.float32)

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.n_restarts)
    params0 = jax.vmap(lambda k: _init_params(k, cfg))(keys)
    trained, _ = jax.vmap(lambda p0: _train_one(p0, Xj, yj, cfg, ops))(params0)

    # hard-program predictions for every restart, vmapped
    hard_pred = jax.vmap(lambda p: _forward(p, Xj, cfg.tau1, cfg, ops, hard=True))(trained)
    hard_pred = np.asarray(hard_pred)                    # (R, N)
    y_np = np.asarray(y, dtype=np.float64)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2) + 1e-30
    r2 = 1.0 - np.sum((hard_pred - y_np[None, :]) ** 2, axis=1) / ss_tot
    r2 = np.nan_to_num(r2, nan=-1e9, posinf=-1e9, neginf=-1e9)

    best = int(np.argmax(r2))
    best_params = {k: np.asarray(v[best]) for k, v in trained.items()}
    return DiffEMLResult(
        best_params=best_params,
        best_r2=float(r2[best]),
        hit_rate=float(np.mean(r2 > recover_thresh)),
        median_r2=float(np.median(r2)),
        all_r2=r2,
        program=render_program(best_params, cfg, var_names),
        cfg=cfg,
    )


__all__ = [
    "EMLConfig",
    "DiffEMLResult",
    "discover",
    "render_program",
    "DEFAULT_OPS",
]
