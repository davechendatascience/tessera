"""Opcode-tape interpreter for pure-pointwise trees (JAX, compile-once).

Why this exists
---------------
The per-tree JIT path (`symbolic_network._compile_image_tree_jax`)
compiles ONE XLA kernel per distinct tree topology. GP produces high
topology diversity (mutation changes structure every step), so a run
generates hundreds of distinct topologies → hundreds of compiles. On
GPU, XLA compilation (~0.5-2s each) dwarfs the tiny per-eval compute on
small images: the local CPU diagnostic measured 326 ms compile vs
2.7 ms eval — a 123× ratio. The run is compile-bound, and eval-side
optimizations (device-resident data, single forward pass) can't help.

This module compiles ONCE for all trees. Every tree is encoded as a
fixed-length instruction tape; a single jit'd interpreter executes any
tape. Mutation/crossover change the tape *contents* (runtime inputs),
not the compiled graph → no recompilation.

Design that keeps eval cheap
----------------------------
A naive interpreter computes all ops at every node and selects — a
~24× compute blowup. We avoid that:

  - The image batch is carried as an ARRAY inside the interpreter
    (slots have shape (MAX_NODES, *elem) where elem is the per-sample
    element shape, e.g. (H, W)). We do NOT vmap over samples.
  - Op selection uses `lax.switch(opcode, branches, ...)`. With no vmap
    over the opcode, switch executes exactly ONE branch per node — a
    real branch, not select-all. So each node runs a single op.
  - Different trees are handled by looping in Python over the K trees
    and calling the SAME jit'd interpreter with different tape inputs.
    The compiled graph is shared → 1 compile, K dispatches.

Compile count over a whole GP run: one interpreter per
(max_nodes, element-ndim) combination — i.e. 2 total (layer-1 fields
ndim=2, layer-2 scalars ndim=0) — instead of hundreds.

Scope / fallback
----------------
Supports the pointwise operator vocabulary (BIN_OPS minus none,
UN_OPS minus reduce_*). Trees containing FunctionalOp / FunctionalOp2D
or reduce_* ops, or exceeding `max_nodes`, fail to encode (returns
None); the caller falls back to the per-tree path. Results are
bit-identical to the per-tree path because the interpreter reuses the
SAME BIN_OP_FNS / UN_OP_FNS as `_build_parametric_fn`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from tessera.expression.tree import (
    Node, Var, Const, BinOp, UnOp,
    BIN_OP_FNS, UN_OP_FNS,
)


# ---------------- Opcode vocabulary (fixed order) ----------------

# Binary ops the interpreter supports (all of BIN_OPS).
_BIN_ORDER = ["add", "sub", "mul", "div", "min", "max",
              "gt", "lt", "ge", "le", "pow", "atan2"]
# Unary ops (UN_OPS minus reduce_*, which collapse shape and are
# filtered out of networks anyway).
_UN_ORDER = ["tanh", "abs", "sign", "neg", "step",
             "sqrt", "log", "exp", "sin", "cos", "acos", "asin"]

OP_VAR = 0
OP_CONST = 1
_BIN_CODE = {name: 2 + i for i, name in enumerate(_BIN_ORDER)}
_UN_CODE = {name: 2 + len(_BIN_ORDER) + i for i, name in enumerate(_UN_ORDER)}
N_OPCODES = 2 + len(_BIN_ORDER) + len(_UN_ORDER)

_SUPPORTED_BIN = set(_BIN_ORDER)
_SUPPORTED_UN = set(_UN_ORDER)


# ---------------- Tape encoding ----------------

class Tape:
    """Fixed-length instruction tape for one tree (numpy arrays).

    Post-order: children precede parents, so operand slots are always
    already-computed when a node executes. Padded to max_nodes with
    no-op CONST(0) entries after the root.
    """
    __slots__ = ("ops", "arg1", "arg2", "consts", "varidx", "root", "n_nodes")

    def __init__(self, ops, arg1, arg2, consts, varidx, root, n_nodes):
        self.ops = ops
        self.arg1 = arg1
        self.arg2 = arg2
        self.consts = consts
        self.varidx = varidx
        self.root = root
        self.n_nodes = n_nodes


def encode_tree(
    tree: Node, var_index: dict[str, int], max_nodes: int,
) -> Optional[Tape]:
    """Encode a pure-pointwise tree into a fixed-length Tape.

    Returns None if the tree uses an unsupported op (FunctionalOp,
    reduce_*, unknown var) or has more than `max_nodes` nodes.
    """
    ops: list[int] = []
    arg1: list[int] = []
    arg2: list[int] = []
    consts: list[float] = []
    varidx: list[int] = []

    class _Unsupported(Exception):
        pass

    def visit(node: Node) -> int:
        if isinstance(node, Var):
            if node.name not in var_index:
                raise _Unsupported()
            ops.append(OP_VAR); arg1.append(0); arg2.append(0)
            consts.append(0.0); varidx.append(var_index[node.name])
            return len(ops) - 1
        if isinstance(node, Const):
            ops.append(OP_CONST); arg1.append(0); arg2.append(0)
            consts.append(float(node.value)); varidx.append(0)
            return len(ops) - 1
        if isinstance(node, UnOp):
            if node.op not in _SUPPORTED_UN:
                raise _Unsupported()
            s = visit(node.a)
            ops.append(_UN_CODE[node.op]); arg1.append(s); arg2.append(0)
            consts.append(0.0); varidx.append(0)
            return len(ops) - 1
        if isinstance(node, BinOp):
            if node.op not in _SUPPORTED_BIN:
                raise _Unsupported()
            sa = visit(node.a)
            sb = visit(node.b)
            ops.append(_BIN_CODE[node.op]); arg1.append(sa); arg2.append(sb)
            consts.append(0.0); varidx.append(0)
            return len(ops) - 1
        raise _Unsupported()  # FunctionalOp / FunctionalOp2D / unknown

    try:
        root = visit(tree)
    except _Unsupported:
        return None

    n = len(ops)
    if n > max_nodes:
        return None

    pad = max_nodes - n
    if pad:
        ops += [OP_CONST] * pad
        arg1 += [0] * pad
        arg2 += [0] * pad
        consts += [0.0] * pad
        varidx += [0] * pad

    return Tape(
        ops=np.asarray(ops, dtype=np.int32),
        arg1=np.asarray(arg1, dtype=np.int32),
        arg2=np.asarray(arg2, dtype=np.int32),
        consts=np.asarray(consts, dtype=np.float32),
        varidx=np.asarray(varidx, dtype=np.int32),
        root=int(root),
        n_nodes=n,
    )


# ---------------- The single jit'd interpreter ----------------

_INTERP_CACHE: dict = {}


def _build_branches():
    """Build the lax.switch branch list (index = opcode).

    Each branch has signature (x1, x2, v, c) -> array, all same shape.
    Reuses BIN_OP_FNS / UN_OP_FNS so semantics match the per-tree path
    exactly (same safe_pow / safe_log clipping, etc.)."""
    branches = []
    branches.append(lambda x1, x2, v, c: v)   # OP_VAR
    branches.append(lambda x1, x2, v, c: c)   # OP_CONST
    for name in _BIN_ORDER:
        f = BIN_OP_FNS[name]
        branches.append(lambda x1, x2, v, c, f=f: f(x1, x2))
    for name in _UN_ORDER:
        f = UN_OP_FNS[name]
        branches.append(lambda x1, x2, v, c, f=f: f(x1))
    return branches


def get_interpreter(max_nodes: int):
    """Return a jit'd interpreter for the given max_nodes (cached).

    The returned fn signature:
        interp(ops, arg1, arg2, consts, varidx, root, var_values) -> out
    where:
        ops/arg1/arg2/varidx : int32 (max_nodes,)
        consts               : float32 (max_nodes,)
        root                 : int32 scalar
        var_values           : (n_vars, *elem)  — batch carried in *elem
        out                  : (*elem,)
    Compiled ONCE per max_nodes; the tape is a runtime input so all
    trees share this single compiled graph.
    """
    if max_nodes in _INTERP_CACHE:
        return _INTERP_CACHE[max_nodes]

    import jax
    import jax.numpy as jnp
    from jax import lax

    branches = _build_branches()

    def interp(ops, arg1, arg2, consts, varidx, root, var_values):
        elem = var_values.shape[1:]
        slots0 = jnp.zeros((max_nodes,) + elem, dtype=var_values.dtype)
        steps = jnp.arange(max_nodes)

        def body(slots, t):
            # lax.scan compiles this body ONCE (not unrolled), so the
            # compiled graph is O(1 node) instead of O(max_nodes · n_ops).
            # That keeps XLA compile cheap regardless of max_nodes.
            x1 = slots[arg1[t]]             # dynamic gather (operand < t)
            x2 = slots[arg2[t]]
            v = var_values[varidx[t]]       # dynamic gather over n_vars
            c = jnp.broadcast_to(consts[t].astype(var_values.dtype), elem)
            res = lax.switch(ops[t], branches, x1, x2, v, c)
            slots = slots.at[t].set(res)
            return slots, None

        slots, _ = lax.scan(body, slots0, steps)
        return slots[root]

    jitted = jax.jit(interp)
    _INTERP_CACHE[max_nodes] = jitted
    return jitted


def clear_interpreter_cache() -> None:
    _INTERP_CACHE.clear()


# ---------------- Batch helpers ----------------

def encode_trees(
    trees: list[Node], var_index: dict[str, int], max_nodes: int,
) -> Optional[list[Tape]]:
    """Encode a list of trees; return None if ANY fails to encode."""
    tapes = []
    for t in trees:
        tape = encode_tree(t, var_index, max_nodes)
        if tape is None:
            return None
        tapes.append(tape)
    return tapes


def run_trees(
    tapes: list[Tape], var_values_cnxhxw, max_nodes: int,
):
    """Run a list of tapes on shared var_values via the cached interpreter.

    var_values_cnxhxw: (n_vars, *elem) device array — the batch dim(s)
        live inside *elem (e.g. (n_vars, N, H, W) for image trees, or
        (n_vars, N) for scalar/feature trees).

    Returns a list of per-tape outputs, each shape (*elem).
    Looping in Python over tapes; all calls hit ONE compiled interp.
    """
    import jax.numpy as jnp
    interp = get_interpreter(max_nodes)
    outs = []
    for tape in tapes:
        out = interp(
            jnp.asarray(tape.ops), jnp.asarray(tape.arg1),
            jnp.asarray(tape.arg2), jnp.asarray(tape.consts),
            jnp.asarray(tape.varidx), jnp.asarray(tape.root, dtype=jnp.int32),
            var_values_cnxhxw,
        )
        outs.append(out)
    return outs


__all__ = [
    "Tape",
    "encode_tree",
    "encode_trees",
    "get_interpreter",
    "run_trees",
    "clear_interpreter_cache",
    "N_OPCODES",
]
