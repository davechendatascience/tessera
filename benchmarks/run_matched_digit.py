"""Matched-parameter digit benchmark: plain CNN vs CNN+symbolic at EQUAL params.

The fair test of "does symbolic structure help accuracy?": give the plain CNN
extra channels so its parameter count matches the symbolic version's, then
compare held-out accuracy over several seeds. If symbolic structure is more
expressive per parameter (the Pi-net / KAN / multiplicative-interaction
claim), the symbolic net wins at equal budget; if not, it ties or loses.

The symbolic layer is the additive spatial sym_conv (a dense 3x3 conv PLUS T
symbolic spatial channels: ch = op(in@(dy,dx,a), in@(dy,dx,b))). T=0 is a plain
CNN. Same architecture, same training, only T and channel widths differ.

Usage:
    python benchmarks/run_matched_digit.py                 # 5 seeds
    python benchmarks/run_matched_digit.py --seeds 3 --epochs 15
"""
from __future__ import annotations
import argparse
from functools import partial

import numpy as np
import jax, jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tessera.experimental.diff_sr import make_ops

tmap = jax.tree_util.tree_map
OPS = ['add', 'sub', 'mul', 'div', 'sin', 'tanh', 'square', 'id']
branches, _ = make_ops(OPS); K = len(OPS); ID = OPS.index('id'); PS = 3


def im2col(x, ps=PS):
    pad = ps // 2; x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    B, H, W, C = x.shape; oh, ow = H - ps + 1, W - ps + 1
    return jnp.concatenate([x[:, i:i+oh, j:j+ow, :] for i in range(ps) for j in range(ps)], -1)

def pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

def sym_conv(x, p, tau, T):
    pc = im2col(x); B, H, W, P = pc.shape; h = pc.reshape(-1, P)
    out = jax.nn.relu(h @ p['Wc'] + p['bc'])                    # dense conv part
    if T > 0:
        xl = h @ jax.nn.softmax(p['bl']/tau, 1).T; xr = h @ jax.nn.softmax(p['br']/tau, 1).T
        ov = jnp.clip(jnp.nan_to_num(jnp.stack([f(xl, xr) for f in branches], 0)), -1e4, 1e4)
        sym = jnp.clip(jnp.nan_to_num(jnp.einsum('uk,kbu->bu', jax.nn.softmax(p['alpha']/tau, 1), ov)), -1e4, 1e4)
        out = jnp.concatenate([out, sym], 1)                   # additive symbolic channels
    out = (out - out.mean(0)) / (out.std(0) + 1e-3)
    return out.reshape(B, H, W, -1)

def sc_init(k, Cin, Cout, T):
    kk = jax.random.split(k, 4); P = PS*PS*Cin
    p = {'Wc': jax.random.normal(kk[0], (P, Cout)) * np.sqrt(2.0/P), 'bc': jnp.zeros(Cout)}
    if T > 0:
        bl = 0.01*jax.random.normal(kk[1], (T, P)); bl = bl.at[jnp.arange(T), jnp.arange(T) % P].add(4.0)
        al = 0.01*jax.random.normal(kk[3], (T, K)); al = al.at[:, ID].add(4.0)
        p.update({'bl': bl, 'br': 0.01*jax.random.normal(kk[2], (T, P)), 'alpha': al})
    return p

def _out_ch(C, T): return C + (T if T > 0 else 0)

def model(p, x, tau, T):
    h = pool(sym_conv(x, p['l1'], tau, T)); h = pool(sym_conv(h, p['l2'], tau, T))
    return h.reshape(h.shape[0], -1) @ p['W2'] + p['b2']

def init(key, C1, C2, T):
    k = jax.random.split(key, 3)
    p = {'l1': sc_init(k[0], 1, C1, T), 'l2': sc_init(k[1], _out_ch(C1, T), C2, T)}
    h = pool(sym_conv(jnp.zeros((1, 8, 8, 1)), p['l1'], 1., T)); h = pool(sym_conv(h, p['l2'], 1., T))
    p['W2'] = 0.1*jax.random.normal(k[2], (h.reshape(1, -1).shape[1], 10)); p['b2'] = jnp.zeros(10)
    return p

def nparams(p):
    return int(sum(int(np.prod(np.array(v.shape))) for v in jax.tree_util.tree_leaves(p)))

def loss_fn(p, x, y, tau, T):
    return -jnp.mean(jax.nn.log_softmax(model(p, x, tau, T))[jnp.arange(len(y)), y])

def train_eval(C1, C2, T, seed, data, epochs):
    Xtr, ytr, Xte, yte = data
    p = init(jax.random.PRNGKey(seed), C1, C2, T)
    m = tmap(jnp.zeros_like, p); v = tmap(jnp.zeros_like, p)
    @partial(jax.jit, static_argnums=(6,))
    def step(p, m, v, t, xb, yb, Tt, tau):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb, tau, Tt)
        m = tmap(lambda a, b: 0.9*a+0.1*b, m, g); v = tmap(lambda a, b: 0.999*a+0.001*b*b, v, g)
        p = tmap(lambda pp, a, b: pp - 3e-3*(a/(1-0.9**(t+1)))/(jnp.sqrt(b/(1-0.999**(t+1)))+1e-8), p, m, v)
        return p, m, v, l
    t = 0
    for ep in range(epochs):
        tau = float(1.0*(0.3/1.0)**(ep/max(epochs-1, 1)))
        idx = np.random.default_rng(seed*1000+ep).permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            b = idx[s:s+64]
            p, m, v, l = step(p, m, v, jnp.float32(t), jnp.asarray(Xtr[b]), jnp.asarray(ytr[b]), T, jnp.float32(tau)); t += 1
    te = float((np.asarray(model(p, jnp.asarray(Xte), jnp.float32(0.3), T)).argmax(1) == yte).mean())
    return te, nparams(p)

def match_plain(C1s, C2s, Ts, target):
    """Find plain (T=0) channel widths whose params ~ target (the symbolic budget)."""
    best = None
    for mult in np.arange(1.0, 3.01, 0.05):
        C1p, C2p = int(round(C1s*mult)), int(round(C2s*mult))
        n = nparams(init(jax.random.PRNGKey(0), C1p, C2p, 0))
        if best is None or abs(n-target) < abs(best[2]-target):
            best = (C1p, C2p, n)
        if n >= target:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--C1", type=int, default=16); ap.add_argument("--C2", type=int, default=24)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args(argv)

    d = load_digits(); X = (d.images/16.).astype(np.float32)[..., None]
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.3, random_state=0, stratify=d.target)
    data = (Xtr, ytr, Xte, yte)

    psym = nparams(init(jax.random.PRNGKey(0), args.C1, args.C2, args.T))
    C1p, C2p, pplain = match_plain(args.C1, args.C2, args.T, psym)
    print(f"=== matched-parameter digit benchmark ({args.seeds} seeds, {args.epochs} epochs) ===")
    print(f"symbolic: C1={args.C1} C2={args.C2} T={args.T} -> {psym} params")
    print(f"plain   : C1={C1p} C2={C2p} T=0 -> {pplain} params (matched)\n")

    sym = [train_eval(args.C1, args.C2, args.T, s, data, args.epochs)[0] for s in range(args.seeds)]
    pln = [train_eval(C1p, C2p, 0, s, data, args.epochs)[0] for s in range(args.seeds)]
    sym, pln = np.array(sym), np.array(pln)
    print(f"plain    CNN  test acc = {pln.mean():.4f} +/- {pln.std():.4f}  ({pplain} params)")
    print(f"symbolic CNN  test acc = {sym.mean():.4f} +/- {sym.std():.4f}  ({psym} params)")
    print(f"delta (symbolic - plain) = {sym.mean()-pln.mean():+.4f}  "
          f"({'symbolic WINS' if sym.mean()>pln.mean() else 'plain wins/ties'} at matched params)")


if __name__ == "__main__":
    main()
