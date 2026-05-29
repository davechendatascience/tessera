"""Symbolic network for image classification.

Provenance: `docs/research/hybrid_symbolic_networks.md` Milestone A → C.

Architecture (current)
----------------------
A 2-layer symbolic network for N-class classification:

- Layer 1: K trees, each maps the input channels → a 2-D field, then
  region-pooled (`n_regions` block means) to scalar features.
- Layer 2: N trees (one per class), each maps the K·n_regions pooled
  features → a class score. softmax + argmax → prediction.
- Search: tournament selection + (μ+λ) survival, per-slot mutation,
  per-slot crossover. Cross-entropy loss + parsimony.
- Optional JAX-vmapped JIT evaluation (`use_jax_eval`).

Input channels (multi-channel inputs)
-------------------------------------
Layer-1 trees reference NAMED input channels, not just a single
"image" variable. The default is a single channel `("image",)`, which
reproduces the original behaviour exactly (backwards compatible).

For image tasks where global+pointwise features are too weak (e.g.
10-class MNIST), supply a richer channel bank — e.g.
`("image", "gx", "gy", "lap")` — precomputed once via
`make_image_channels`. Layer-1 trees can then compose edge / gradient /
curvature maps (`abs(gx)+abs(gy)` pooled per quadrant = "edge density
in region"), which pointwise-of-raw-pixels cannot express. This is the
expressivity fix for the MNIST plateau, kept fully JAX-compatible
because the channels are precomputed numpy arrays fed as ordinary Var
inputs.

The mechanism is domain-agnostic: any task expressible as a set of
named 2-D channels works. It is *separate* from the Feynman / tabular
SR path (`tessera.search.GP`), which this module does not touch — the
conceptual parallel (precompute derived inputs) is the same idea the
decompose prepass uses for Feynman, but the code paths are independent.

Status: experimental. Binary MNIST 0-vs-1 with quadrant pooling
reaches TEST 0.968. Direct 10-class is the open problem the channel
bank targets. See `hybrid_symbolic_networks.md` for milestone status.

Key entry points
----------------
    SymbolicNetwork              — frozen dataclass (K L1 + N L2 trees)
    make_image_channels(images)  — precompute (image, gx, gy, lap) bank
    random_network(rng, ...)     — construct a random network
    mutate_network / crossover_networks
    network_loss / network_accuracy
    NetworkGPConfig + run_network_gp(...)

Both `images_*` arguments to run_network_gp / the eval functions accept
either a bare (N,H,W) array (auto-wrapped or auto-expanded to the
configured channel bank) or a precomputed dict {name: (N,H,W)}.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from tessera.expression.tree import (
    Node, Var, Const, BinOp, UnOp, evaluate, complexity as tree_complexity,
)
from tessera.expression.mutation import (
    random_tree, mutate as tree_mutate, validate_tree,
)
from tessera.expression.simplify import simplify
from tessera.expression.jit import is_pure_pointwise


# ---------------------------------------------------------------------
# Input channels — derived 2-D feature maps fed to layer-1 trees
# ---------------------------------------------------------------------

DEFAULT_IMAGE_CHANNELS: tuple[str, ...] = ("image", "gx", "gy", "lap")
"""Recommended channel bank for image classification. `gx`/`gy` are
central-difference spatial gradients (vertical / horizontal edges);
`lap` is the 5-point Laplacian (curvature / blob structure). Together
with the raw `image`, these give layer-1 trees genuine spatial
primitives, which pointwise-of-raw-pixels lacks."""


def make_image_channels(
    images: np.ndarray,
    channels: tuple[str, ...] = DEFAULT_IMAGE_CHANNELS,
) -> dict[str, np.ndarray]:
    """Precompute named spatial channels for a batch (or single) image.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W) for a batch, or (H, W) for a single image.
    channels : tuple[str, ...]
        Which channels to compute. Supported:
          "image" — raw pixels (copy)
          "gx"    — central horizontal difference (vertical-edge map)
          "gy"    — central vertical difference (horizontal-edge map)
          "lap"   — 5-point Laplacian (curvature / blobs)

    Returns
    -------
    dict[name -> array] with each array the same shape as `images`.
    All ops are simple vectorized numpy shifts (no scipy dependency,
    no GP-loop cost — computed once on the dataset).
    """
    arr = np.asarray(images, dtype=np.float64)
    single = (arr.ndim == 2)
    if single:
        arr = arr[None]  # (1, H, W)

    out: dict[str, np.ndarray] = {}
    for name in channels:
        if name == "image":
            out[name] = arr.copy()
        elif name == "gx":
            g = np.zeros_like(arr)
            g[:, :, 1:-1] = arr[:, :, 2:] - arr[:, :, :-2]
            out[name] = g
        elif name == "gy":
            g = np.zeros_like(arr)
            g[:, 1:-1, :] = arr[:, 2:, :] - arr[:, :-2, :]
            out[name] = g
        elif name == "lap":
            g = np.zeros_like(arr)
            g[:, 1:-1, 1:-1] = (
                arr[:, 1:-1, 2:] + arr[:, 1:-1, :-2]
                + arr[:, 2:, 1:-1] + arr[:, :-2, 1:-1]
                - 4.0 * arr[:, 1:-1, 1:-1]
            )
            out[name] = g
        else:
            raise ValueError(f"make_image_channels: unknown channel {name!r}")

    if single:
        out = {k: v[0] for k, v in out.items()}
    return out


def _prepare_channels(data, channel_names: tuple[str, ...]) -> dict:
    """Normalize an eval input to a {name: array} dict over channel_names.

    - dict in → returned as-is (assumed already correct; GP hot loop
      passes the precomputed dict, so this is a no-op passthrough).
    - bare ndarray + single "image" channel → wrap as {"image": arr}.
    - bare ndarray + multi-channel → auto-compute the bank via
      make_image_channels (ergonomic: pass raw images, get channels).
    """
    if isinstance(data, dict):
        return data
    if tuple(channel_names) == ("image",):
        return {"image": np.asarray(data, dtype=np.float64)}
    return make_image_channels(data, channels=tuple(channel_names))


def _single_channels(channels: dict, i: int) -> dict:
    """Slice the i-th sample out of a batched channel dict."""
    return {name: channels[name][i] for name in channels}


def _batch_size(channels: dict) -> int:
    any_name = next(iter(channels))
    return channels[any_name].shape[0]


# ---------------------------------------------------------------------
# JAX batched evaluation (Milestone A.5)
# ---------------------------------------------------------------------
#
# When JAX is installed AND `use_jax_eval=True`, the network is evaluated
# via a vmapped JIT path: each tree is compiled once to a JAX function,
# vmapped over the image batch dimension, and JIT'd. Subsequent calls on
# the same tree topology hit the JIT cache.
#
# Falls back to per-image numpy when:
# - JAX not installed
# - Any layer-1 tree has FunctionalOp / FunctionalOp2D (not supported
#   by the per-tree JIT path)
#
# JIT cache is module-level, keyed by (topology, var_names_tuple,
# kind="image"|"scalar"). Networks share cached compilations when their
# trees have the same topology — common during GP since mutation
# changes one slot at a time.

_JAX_AVAILABLE: Optional[bool] = None


def _jax_available() -> bool:
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax  # noqa: F401
            _JAX_AVAILABLE = True
        except ImportError:
            _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


_JAX_TREE_CACHE: dict = {}


def _compile_image_tree_jax(tree: Node, channel_names: tuple[str, ...],
                            n_regions: int = 1):
    """Compile a layer-1 tree to a JIT'd batched function:
       fn(channel_stack, consts) -> features_batch

    channel_stack: shape (N, C, H, W) — C = len(channel_names), in
        channel_names order.
    consts: 1-D array of Const values in pre-order.
    features_batch: shape (N, n_regions) — region-pooled features per image.

    For n_regions=1, classic global mean-pool → (N, 1). For n_regions=4,
    4 per-quadrant means → (N, 4). Any perfect-square n_regions.

    Returns None if the tree contains FunctionalOp / FunctionalOp2D
    (not supported by the per-tree JIT path).

    Cache key includes channel_names + n_regions so distinct channel
    sets / pooling resolutions get their own JIT.
    """
    if not is_pure_pointwise(tree):
        return None
    from tessera.expression.batched import (
        topology_key, _build_parametric_fn,
    )
    import jax
    import jax.numpy as jnp

    side = int(round(math.sqrt(n_regions)))
    if side * side != n_regions:
        raise ValueError(f"n_regions must be a perfect square; got {n_regions}")

    channel_names = tuple(channel_names)
    C = len(channel_names)
    key = (topology_key(tree), channel_names, f"image_pool_{n_regions}")
    if key in _JAX_TREE_CACHE:
        return _JAX_TREE_CACHE[key]

    var_idx = {name: i for i, name in enumerate(channel_names)}
    counter = [0]
    raw_fn = _build_parametric_fn(tree, var_idx, counter)

    def _per_sample(chan_stack, consts):
        # chan_stack: (C, H, W). Args ordered by channel_names.
        args = tuple(chan_stack[i] for i in range(C))
        out = raw_fn(args, consts)
        out = jnp.asarray(out)
        # Handle scalar output: broadcast to n_regions
        if out.ndim == 0:
            return jnp.full((n_regions,), out)
        # 2-D pooling: split into side × side blocks, mean each
        if n_regions == 1:
            return jnp.array([jnp.mean(out)])
        H, W = out.shape
        h_step = H // side
        w_step = W // side
        cropped = out[: side * h_step, : side * w_step]
        blocked = cropped.reshape(side, h_step, side, w_step)
        return jnp.mean(blocked, axis=(1, 3)).flatten()

    vmapped = jax.vmap(_per_sample, in_axes=(0, None))
    jitted = jax.jit(vmapped)
    _JAX_TREE_CACHE[key] = jitted
    return jitted


def _compile_feature_tree_jax(tree: Node, K: int):
    """Compile a layer-2 tree to a JIT'd batched function:
       fn(features_batch, consts) -> scores_batch

    features_batch: shape (N, K).
    scores_batch: shape (N,).
    """
    if not is_pure_pointwise(tree):
        return None
    from tessera.expression.batched import (
        topology_key, _build_parametric_fn,
    )
    import jax
    import jax.numpy as jnp

    var_names = tuple(f"f{i}" for i in range(K))
    key = (topology_key(tree), var_names, "scalar")
    if key in _JAX_TREE_CACHE:
        return _JAX_TREE_CACHE[key]

    var_idx = {f"f{i}": i for i in range(K)}
    counter = [0]
    raw_fn = _build_parametric_fn(tree, var_idx, counter)

    def _per_sample(features, consts):
        # `features` is shape (K,) — pass each as a scalar arg.
        args = tuple(features[i] for i in range(K))
        out = raw_fn(args, consts)
        return jnp.asarray(out)

    vmapped = jax.vmap(_per_sample, in_axes=(0, None))
    jitted = jax.jit(vmapped)
    _JAX_TREE_CACHE[key] = jitted
    return jitted


def evaluate_network_jax_batch(
    network: "SymbolicNetwork", data,
) -> Optional[np.ndarray]:
    """JAX batched evaluation. Returns scores of shape (N, n_classes)
    as numpy array, or None if any tree in the network has FunctionalOp
    / 2D (caller should fall back to numpy).

    `data` is a bare (N,H,W) array or a {channel: (N,H,W)} dict.
    """
    if not _jax_available():
        return None
    from tessera.expression.batched import extract_constants
    import jax.numpy as jnp

    channel_names = network.input_channels
    channels = _prepare_channels(data, channel_names)
    # Stack channels in input_channels order → (N, C, H, W)
    stacked = jnp.stack(
        [jnp.asarray(channels[name]) for name in channel_names], axis=1
    )

    feature_cols = []
    for tree in network.layer_1_trees:
        fn = _compile_image_tree_jax(tree, channel_names,
                                     n_regions=network.n_regions)
        if fn is None:
            return None
        consts_list = extract_constants(tree)
        consts = (jnp.asarray(consts_list, dtype=stacked.dtype)
                  if consts_list else jnp.zeros(0, dtype=stacked.dtype))
        feats = fn(stacked, consts)  # shape (N, n_regions)
        feature_cols.append(feats)
    # Concat along feature axis: (N, K * n_regions)
    features_matrix = jnp.concatenate(feature_cols, axis=1)

    n_features = network.n_features
    score_cols = []
    for tree in network.layer_2_trees:
        l2_fn = _compile_feature_tree_jax(tree, n_features)
        if l2_fn is None:
            return None
        l2_consts_list = extract_constants(tree)
        l2_consts = (jnp.asarray(l2_consts_list, dtype=stacked.dtype)
                     if l2_consts_list else jnp.zeros(0, dtype=stacked.dtype))
        sc = l2_fn(features_matrix, l2_consts)  # shape (N,)
        score_cols.append(sc)
    scores = jnp.stack(score_cols, axis=1)  # shape (N, n_classes)

    return np.asarray(scores)


def clear_jax_tree_cache() -> None:
    """Drop the JAX JIT cache for network trees. Useful when benchmarking
    cold-path compilation or managing memory between unrelated runs."""
    _JAX_TREE_CACHE.clear()


# ---------------------------------------------------------------------
# Network dataclass
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolicNetwork:
    """K-feature 2-layer symbolic network for N-class classification with
    spatial region pooling.

    Layer 1: K trees, each takes the image (named "image") and produces
    a 2-D output. The output is pooled to `n_regions` scalar features by
    splitting into a √n_regions × √n_regions block grid and taking the
    mean of each block. So each layer-1 tree contributes n_regions
    features to layer 2.

    Pooling choices:
      - n_regions=1: classic global mean-pool (loses all spatial info).
      - n_regions=4: 2×2 quadrant means (top-left/top-right/bottom-left/
        bottom-right). Distinguishes location-sensitive patterns like
        "7 has ink at the top" vs "1 has ink in the middle".
      - n_regions=9: 3×3 block grid (finer spatial resolution).
      - Any perfect-square value supported.

    Layer 2: N trees (one per class). Each takes K · n_regions named
    scalar features "f0", "f1", ..., "f{K·n_regions − 1}" and produces
    a scalar score. Feature indexing: `f_{k * n_regions + q}` is the
    q-th regional pool of tree k.

    Prediction: softmax(scores) for class probabilities; argmax(scores)
    for the predicted class. Binary is the N=2 special case.
    """
    layer_1_trees: tuple[Node, ...]
    layer_2_trees: tuple[Node, ...]   # one per class; len = n_classes
    n_regions: int = 1                 # pooling resolution; backwards-compat default
    input_channels: tuple[str, ...] = ("image",)  # named layer-1 inputs

    @property
    def K(self) -> int:
        return len(self.layer_1_trees)

    @property
    def n_classes(self) -> int:
        return len(self.layer_2_trees)

    @property
    def n_features(self) -> int:
        """Total layer-2 input features = K · n_regions."""
        return len(self.layer_1_trees) * self.n_regions

    @property
    def complexity(self) -> int:
        """Total complexity = Σ tree complexities (all K+N trees)."""
        return (sum(tree_complexity(t) for t in self.layer_1_trees)
                + sum(tree_complexity(t) for t in self.layer_2_trees))

    def __str__(self) -> str:
        l1 = "\n    ".join(f"T_{i} = {t}" for i, t in enumerate(self.layer_1_trees))
        l2 = "\n    ".join(f"class_{c} = {t}" for c, t in enumerate(self.layer_2_trees))
        return (f"SymbolicNetwork(K={self.K}, n_regions={self.n_regions}, "
                f"n_classes={self.n_classes}, "
                f"n_features={self.n_features}, "
                f"channels={self.input_channels}, cx={self.complexity}):\n"
                f"  Layer 1 (inputs {self.input_channels} → {self.n_regions} "
                f"pooled features each):\n    {l1}\n"
                f"  Layer 2 (one tree per class, takes {self.n_features} features):\n    {l2}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def _pool_regions_np(arr: np.ndarray, n_regions: int) -> np.ndarray:
    """Pool a 2-D array into a flat array of n_regions regional means.

    n_regions must be a perfect square (1, 4, 9, ...). The image is
    split into a √n × √n block grid; each block's mean (ignoring NaN)
    is one output value. Result is row-major flattened.

    Edge handling: if H or W isn't divisible by √n, the image is
    cropped to the largest divisible sub-rectangle. (Tessera's
    benchmark images are 14×14 → cleanly divisible by 1, 2, 7, 14.)

    Scalar input is broadcast to n_regions identical values.
    """
    if n_regions == 1:
        if arr.ndim == 0:
            v = float(arr)
        else:
            v = float(np.nanmean(arr))
        return np.array([v if math.isfinite(v) else 0.0])
    side = int(round(math.sqrt(n_regions)))
    if side * side != n_regions:
        raise ValueError(f"n_regions must be a perfect square; got {n_regions}")
    if arr.ndim == 0:
        v = float(arr)
        return np.full(n_regions, v if math.isfinite(v) else 0.0)
    if arr.ndim != 2:
        v = float(np.nanmean(arr))
        return np.full(n_regions, v if math.isfinite(v) else 0.0)
    H, W = arr.shape
    h_step = H // side
    w_step = W // side
    if h_step == 0 or w_step == 0:
        v = float(np.nanmean(arr))
        return np.full(n_regions, v if math.isfinite(v) else 0.0)
    cropped = arr[: side * h_step, : side * w_step]
    # NaN-safe mean per block
    blocked = cropped.reshape(side, h_step, side, w_step)
    pooled = np.nanmean(blocked, axis=(1, 3)).flatten()
    pooled = np.where(np.isfinite(pooled), pooled, 0.0)
    return pooled.astype(np.float64)


def evaluate_network(network: SymbolicNetwork, data) -> np.ndarray:
    """Score one image; returns array of shape (n_classes,).

    `data` is a single-image channel dict {name: (H,W)} or a bare (H,W)
    array (auto-prepared to the network's channels). Layer-1 trees are
    evaluated on the channels, region-pooled, then layer-2 trees produce
    per-class scores.
    """
    channels = _prepare_channels(data, network.input_channels)
    env = {name: np.asarray(channels[name], dtype=np.float64)
           for name in network.input_channels}

    features: list[float] = []
    n_regions = network.n_regions
    for tree in network.layer_1_trees:
        try:
            out = evaluate(tree, env, fill_warmup=0.0)
            out = np.asarray(out, dtype=np.float64)
            pooled = _pool_regions_np(out, n_regions)
        except Exception:
            pooled = np.zeros(n_regions, dtype=np.float64)
        features.extend(float(v) for v in pooled)

    feature_env = {f"f{i}": np.array([f], dtype=np.float64)
                   for i, f in enumerate(features)}
    scores = np.zeros(network.n_classes, dtype=np.float64)
    for c, tree in enumerate(network.layer_2_trees):
        try:
            s = evaluate(tree, feature_env, fill_warmup=0.0)
            s = np.asarray(s, dtype=np.float64).ravel()[0]
            if math.isfinite(s):
                scores[c] = float(s)
        except Exception:
            scores[c] = 0.0
    return scores


def evaluate_network_batch(
    network: SymbolicNetwork, data,
) -> np.ndarray:
    """Evaluate on N images. Returns shape (N, n_classes).

    `data` is a batched channel dict {name: (N,H,W)} or a bare (N,H,W)
    array (auto-prepared to the network's channels).
    """
    channels = _prepare_channels(data, network.input_channels)
    n = _batch_size(channels)
    out = np.zeros((n, network.n_classes), dtype=np.float64)
    for i in range(n):
        out[i] = evaluate_network(network, _single_channels(channels, i))
    return out


# ---------------------------------------------------------------------
# Random initialization
# ---------------------------------------------------------------------

def _tree_uses_reduce(node: Node) -> bool:
    """Check if a tree contains any reduce_* operator (which collapses
    spatial structure to a scalar and kills layer-1 image dependence)."""
    from tessera.expression.tree import iter_subtrees, UnOp
    for sub in iter_subtrees(node):
        if isinstance(sub, UnOp) and sub.op.startswith("reduce_"):
            return True
    return False


def random_network(
    rng: random.Random,
    K: int = 4,
    n_classes: int = 2,
    n_regions: int = 1,
    input_channels: tuple[str, ...] = ("image",),
    layer_1_max_depth: int = 3,
    layer_2_max_depth: int = 3,
    enable_2d: bool = True,
    max_attempts_per_slot: int = 30,
    seed_layer_2_sum: bool = True,
) -> SymbolicNetwork:
    """Construct a random K-feature network over the given input channels.

    Layer-1 trees reference the named `input_channels` (default just
    "image") and use a restricted alphabet for non-degenerate features:
    - pointwise + Measure2D (if enable_2d) for spatial structure
    - 1D FunctionalOp (L[], V2[], B[]) DISABLED — they expect 1D
      time-series, not 2D fields; degenerate on images
    - reduce_* operators FILTERED OUT — collapse the field to a scalar

    With a multi-channel bank (e.g. image/gx/gy/lap), a layer-1 tree
    like `abs(gx) + abs(gy)` becomes an edge-density feature — the
    expressivity the single-"image" channel lacks.

    Layer-2 trees use a pointwise-only alphabet over the K·n_regions
    pooled features f0..f{K·n_regions − 1}. Seeding: see inline comments.

    Failures to generate a valid layer-1 tree fall back to the first
    channel as a bare Var.
    """
    # Layer 1
    l1_features = list(input_channels)
    l1_feature_set = set(l1_features)
    fallback_l1 = Var(l1_features[0])
    l1_trees: list[Node] = []
    for _ in range(K):
        chosen: Optional[Node] = None
        for _ in range(max_attempts_per_slot):
            try:
                t = random_tree(rng, l1_features, max_depth=layer_1_max_depth,
                                enable_2d=enable_2d,
                                pointwise_only=True)
            except Exception:
                continue
            if validate_tree(t, l1_feature_set) is not None:
                continue
            if _tree_uses_reduce(t):
                # Disallow reduce_* — collapses field to scalar.
                continue
            chosen = t
            break
        if chosen is None:
            chosen = fallback_l1
        try:
            chosen = simplify(chosen)
        except Exception:
            pass
        l1_trees.append(chosen)

    # Layer 2 — one tree per class
    n_features = K * n_regions
    l2_features = [f"f{i}" for i in range(n_features)]
    l2_feature_set = set(l2_features)
    l2_trees: list[Node] = []

    # Build sum-of-features tree (used as the binary positive-class seed).
    sum_features: Node = Var("f0")
    for i in range(1, n_features):
        sum_features = BinOp("add", sum_features, Var(f"f{i}"))

    for c in range(n_classes):
        l2_tree: Optional[Node] = None
        # Default seed picks based on n_classes:
        # - Binary (n_classes=2): class 0 = sum of features, class 1 = 0.
        #   argmax(sum, 0) = 0 if sum > 0, else 1 — equivalent to the
        #   classic threshold-at-0 of a single-tree binary network. This
        #   preserves the strong binary performance.
        # - Multi-class (n_classes >= 3): each class anchored on a single
        #   feature f_{c % K}. Distinct trees that don't collapse under
        #   simplify_ac. The GP refines these into class-discriminating
        #   functions.
        if seed_layer_2_sum:
            if n_classes == 2:
                l2_tree = sum_features if c == 0 else Const(0.0)
            else:
                anchor = c % n_features
                l2_tree = Var(f"f{anchor}")
        else:
            for _ in range(max_attempts_per_slot):
                try:
                    t = random_tree(rng, l2_features, max_depth=layer_2_max_depth,
                                    enable_2d=False, pointwise_only=True)
                except Exception:
                    continue
                if validate_tree(t, l2_feature_set) is None:
                    l2_tree = t
                    break
            if l2_tree is None:
                l2_tree = Var("f0")
                for i in range(1, n_features):
                    l2_tree = BinOp("add", l2_tree, Var(f"f{i}"))
        try:
            l2_tree = simplify(l2_tree)
        except Exception:
            pass
        l2_trees.append(l2_tree)

    return SymbolicNetwork(
        layer_1_trees=tuple(l1_trees),
        layer_2_trees=tuple(l2_trees),
        n_regions=n_regions,
        input_channels=tuple(input_channels),
    )


# ---------------------------------------------------------------------
# Warm-start: combine one-vs-rest binary nets into one direct N-class net
# ---------------------------------------------------------------------

def _remap_feature_indices(node: Node, offset: int) -> Node:
    """Return a copy of a pointwise layer-2 tree with every feature Var
    `f{i}` renamed to `f{i+offset}`. Used to relocate a binary net's
    classifier into its slot within a combined feature vector."""
    if isinstance(node, Var):
        if node.name.startswith("f") and node.name[1:].isdigit():
            return Var(f"f{int(node.name[1:]) + offset}")
        return node
    if isinstance(node, Const):
        return node
    if isinstance(node, UnOp):
        return UnOp(node.op, _remap_feature_indices(node.a, offset))
    if isinstance(node, BinOp):
        return BinOp(node.op,
                     _remap_feature_indices(node.a, offset),
                     _remap_feature_indices(node.b, offset))
    return node  # FunctionalOp/2D shouldn't appear in layer-2 trees


def warm_start_from_binary(binary_networks: list) -> SymbolicNetwork:
    """Build one direct N-class network from N one-vs-rest binary nets.

    Each binary net `d` discovered K layer-1 features tuned to detect
    digit d, plus a 2-tree layer-2 head (class 0 = "not d", class 1 =
    "is d"). This combines them:

      - Layer 1 = concatenation of ALL nets' layer-1 trees
        (N · K trees → N · K · n_regions features).
      - Class-d head = (net_d.class1 − net_d.class0), with its feature
        indices remapped to net d's slot in the combined feature vector.

    At gen 0 the combined network's argmax over the N heads exactly
    reproduces the one-vs-rest argmax(p1 − p0) prediction — so GP
    refinement starts from the one-vs-rest accuracy and (under μ+λ)
    can only improve from there.

    All input nets must share (K, n_regions, input_channels).
    """
    nets = list(binary_networks)
    if not nets:
        raise ValueError("warm_start_from_binary: empty network list")
    K = nets[0].K
    R = nets[0].n_regions
    ch = nets[0].input_channels
    for net in nets:
        if net.K != K or net.n_regions != R or net.input_channels != ch:
            raise ValueError("warm_start_from_binary: nets must share "
                             "(K, n_regions, input_channels)")
        if net.n_classes != 2:
            raise ValueError("warm_start_from_binary: expects binary nets")

    layer_1: list[Node] = []
    heads: list[Node] = []
    for d, net in enumerate(nets):
        offset = d * K * R
        layer_1.extend(net.layer_1_trees)
        c1 = _remap_feature_indices(net.layer_2_trees[1], offset)
        c0 = _remap_feature_indices(net.layer_2_trees[0], offset)
        heads.append(BinOp("sub", c1, c0))

    return SymbolicNetwork(
        layer_1_trees=tuple(layer_1),
        layer_2_trees=tuple(heads),
        n_regions=R,
        input_channels=ch,
    )


# ---------------------------------------------------------------------
# Mutation + crossover (at the network level)
# ---------------------------------------------------------------------

def mutate_network(
    network: SymbolicNetwork,
    rng: random.Random,
    enable_2d: bool = True,
) -> SymbolicNetwork:
    """Pick one slot (K + n_classes choices); mutate that tree.

    Returns a new SymbolicNetwork. Original unchanged.
    """
    n_slots = network.K + network.n_classes
    slot = rng.randrange(n_slots)

    if slot < network.K:
        parent = network.layer_1_trees[slot]
        channel_list = list(network.input_channels)
        for _ in range(5):
            try:
                new_tree = tree_mutate(
                    [parent], rng, channel_list,
                    pointwise_only=True,
                    enable_2d=enable_2d,
                )
            except Exception:
                return network
            if new_tree is None:
                return network
            if not _tree_uses_reduce(new_tree):
                break
        try:
            new_tree = simplify(new_tree)
        except Exception:
            pass
        new_l1 = list(network.layer_1_trees)
        new_l1[slot] = new_tree
        return SymbolicNetwork(
            layer_1_trees=tuple(new_l1),
            layer_2_trees=network.layer_2_trees,
            n_regions=network.n_regions,
            input_channels=network.input_channels,
        )

    # Layer-2 slot: 0..n_classes-1
    class_idx = slot - network.K
    parent = network.layer_2_trees[class_idx]
    feature_names = [f"f{i}" for i in range(network.n_features)]
    try:
        new_tree = tree_mutate(
            [parent], rng, feature_names,
            pointwise_only=True, enable_2d=False,
        )
    except Exception:
        return network
    if new_tree is None:
        return network
    try:
        new_tree = simplify(new_tree)
    except Exception:
        pass
    new_l2 = list(network.layer_2_trees)
    new_l2[class_idx] = new_tree
    return SymbolicNetwork(
        layer_1_trees=network.layer_1_trees,
        layer_2_trees=tuple(new_l2),
        n_regions=network.n_regions,
        input_channels=network.input_channels,
    )


def crossover_networks(
    a: SymbolicNetwork, b: SymbolicNetwork, rng: random.Random,
) -> SymbolicNetwork:
    """Swap one slot from b into a. Networks must agree on shape."""
    if a.K != b.K or a.n_classes != b.n_classes or a.n_regions != b.n_regions:
        raise ValueError(
            f"crossover: shape mismatch "
            f"(a: K={a.K}/n_classes={a.n_classes}/n_regions={a.n_regions} vs "
            f"b: K={b.K}/n_classes={b.n_classes}/n_regions={b.n_regions})"
        )
    n_slots = a.K + a.n_classes
    slot = rng.randrange(n_slots)
    if slot < a.K:
        new_l1 = list(a.layer_1_trees)
        new_l1[slot] = b.layer_1_trees[slot]
        return SymbolicNetwork(
            layer_1_trees=tuple(new_l1),
            layer_2_trees=a.layer_2_trees,
            n_regions=a.n_regions,
            input_channels=a.input_channels,
        )
    class_idx = slot - a.K
    new_l2 = list(a.layer_2_trees)
    new_l2[class_idx] = b.layer_2_trees[class_idx]
    return SymbolicNetwork(
        layer_1_trees=a.layer_1_trees,
        layer_2_trees=tuple(new_l2),
        n_regions=a.n_regions,
        input_channels=a.input_channels,
    )


# ---------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------

def _log_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over axis=-1."""
    max_s = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - max_s
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


def _scores(network: SymbolicNetwork, images: np.ndarray,
            *, use_jax: bool = False) -> np.ndarray:
    """Compute per-class scores; shape (N, n_classes).
    Tries JAX if enabled and tree-compatible; falls back to numpy."""
    if use_jax:
        scores = evaluate_network_jax_batch(network, images)
        if scores is not None:
            return scores
    return evaluate_network_batch(network, images)


def _loss_from_scores(
    scores: np.ndarray, labels: np.ndarray, complexity: int,
    parsimony: float, complexity_divisor: float,
) -> float:
    """Cross-entropy + parsimony penalty from a precomputed score matrix."""
    if not np.isfinite(scores).all():
        return float("inf")
    log_probs = _log_softmax(scores)
    n = scores.shape[0]
    true_log_probs = log_probs[np.arange(n), labels.astype(int)]
    ce = float(-np.mean(true_log_probs))
    penalty = parsimony * (complexity / max(complexity_divisor, 1e-9))
    return ce + penalty


def _acc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """argmax accuracy from a precomputed score matrix."""
    return float(np.mean(scores.argmax(axis=-1) == labels.astype(int)))


def network_loss(
    network: SymbolicNetwork,
    images, labels: np.ndarray,
    *, parsimony: float = 0.001, use_jax: bool = False,
    complexity_divisor: float = 1.0,
) -> float:
    """Cross-entropy loss (mean over samples) + parsimony penalty.

    For each sample i with true class y_i:
        loss_i = -log_softmax(scores[i])[y_i]

    Penalty = parsimony · (total_complexity / complexity_divisor).
    complexity_divisor=1.0 (default) preserves the original behaviour.
    Set it to n_classes (via NetworkGPConfig.normalize_parsimony_by_classes)
    so the per-class complexity penalty doesn't grow with the number of
    layer-2 trees — important for multi-class where total cx scales with
    n_classes and would otherwise over-penalize relative to binary.

    Returns +inf on non-finite scores (selection drops broken candidates).
    """
    scores = _scores(network, images, use_jax=use_jax)
    return _loss_from_scores(scores, labels, network.complexity,
                             parsimony, complexity_divisor)


def network_accuracy(
    network: SymbolicNetwork,
    images, labels: np.ndarray,
    *, use_jax: bool = False,
) -> float:
    """argmax(scores) vs labels."""
    scores = _scores(network, images, use_jax=use_jax)
    return _acc_from_scores(scores, labels)


# ---------------------------------------------------------------------
# Network-aware GP
# ---------------------------------------------------------------------

@dataclass
class NetworkCandidate:
    network: SymbolicNetwork
    loss: float
    accuracy: float


@dataclass
class NetworkGPConfig:
    pop_size: int = 30
    n_gens: int = 30
    K: int = 4
    n_classes: int = 2    # binary by default; set to N for N-class
    n_regions: int = 4    # spatial pooling resolution (1=global mean, 4=quadrants, 9=3x3)
    input_channels: tuple[str, ...] = ("image",)  # layer-1 named inputs;
        # use ("image","gx","gy","lap") for spatial edge/curvature features
    layer_1_max_depth: int = 3
    layer_2_max_depth: int = 3
    enable_2d: bool = True
    parsimony: float = 0.001
    normalize_parsimony_by_classes: bool = False  # divide cx penalty by n_classes
    tournament_size: int = 3
    crossover_rate: float = 0.3
    mutation_rate: float = 0.7   # ignored; kept for symmetry
    seed: int = 2026
    early_stop_patience: int = 12
    verbose: bool = True
    # JAX batched evaluation (see Milestone A.5 section at module top).
    use_jax_eval: bool = False


def run_network_gp(
    images_train: np.ndarray, labels_train: np.ndarray,
    cfg: NetworkGPConfig,
    images_test: Optional[np.ndarray] = None,
    labels_test: Optional[np.ndarray] = None,
    seed_networks: Optional[list] = None,
) -> tuple[NetworkCandidate, list[dict]]:
    """Run the network-aware GP. Returns (best_candidate_ever, history).

    History is a list of per-generation dicts:
      {gen, best_loss, best_train_acc, best_test_acc, best_cx, mean_loss}

    `seed_networks`: optional list of SymbolicNetwork objects to inject
    into the initial population (e.g. a warm-start network from
    `warm_start_from_binary`). They are scored and placed at the front;
    random networks fill the remainder. Seeds must match cfg's
    (n_regions, input_channels); their K may differ from cfg.K (the GP
    handles heterogeneous K across individuals via per-slot ops).
    """
    rng = random.Random(cfg.seed)

    # Decide JAX path once per run.
    use_jax = cfg.use_jax_eval and _jax_available()
    if cfg.use_jax_eval and not _jax_available() and cfg.verbose:
        print("[gp] use_jax_eval=True requested but JAX not installed; "
              "falling back to numpy.")

    # Prepare channels ONCE (auto-computes the spatial bank if the config
    # asks for multi-channel inputs but raw images are passed). The hot
    # loop then re-uses these dicts (no per-eval recompute).
    train_ch = _prepare_channels(images_train, cfg.input_channels)
    test_ch = (_prepare_channels(images_test, cfg.input_channels)
               if images_test is not None else None)

    cdiv = float(cfg.n_classes) if cfg.normalize_parsimony_by_classes else 1.0

    def _score(net: SymbolicNetwork) -> NetworkCandidate:
        # Single forward pass; derive BOTH loss and accuracy from it.
        # (Previously network_loss + network_accuracy each ran a full
        # forward pass — 2× the work. Provably identical: both are pure
        # functions of the same score matrix.)
        scores = _scores(net, train_ch, use_jax=use_jax)
        loss = _loss_from_scores(scores, labels_train, net.complexity,
                                 cfg.parsimony, cdiv)
        acc = _acc_from_scores(scores, labels_train)
        return NetworkCandidate(network=net, loss=loss, accuracy=acc)

    pop: list[NetworkCandidate] = []

    # Seed networks (e.g. warm-start) go in first. The population must be
    # shape-homogeneous for crossover to work, so when seeds are present
    # the random fill matches the seed's K / n_classes (which can differ
    # from cfg.K — a warm-start net has K = N·K_binary).
    fill_K = cfg.K
    fill_n_classes = cfg.n_classes
    if seed_networks:
        for snet in seed_networks:
            pop.append(_score(snet))
            if cfg.verbose:
                print(f"[gp] seeded network: K={snet.K}, "
                      f"n_classes={snet.n_classes}, cx={snet.complexity}, "
                      f"acc={pop[-1].accuracy:.3f}")
        fill_K = seed_networks[0].K
        fill_n_classes = seed_networks[0].n_classes

    while len(pop) < cfg.pop_size:
        net = random_network(
            rng, K=fill_K, n_classes=fill_n_classes, n_regions=cfg.n_regions,
            input_channels=cfg.input_channels,
            layer_1_max_depth=cfg.layer_1_max_depth,
            layer_2_max_depth=cfg.layer_2_max_depth,
            enable_2d=cfg.enable_2d,
        )
        pop.append(_score(net))

    best_ever = min(pop, key=lambda c: c.loss)
    history: list[dict] = []
    gens_no_improve = 0

    for gen in range(cfg.n_gens):
        offspring: list[NetworkCandidate] = []
        for _ in range(cfg.pop_size):
            parents = rng.sample(pop, k=min(cfg.tournament_size, len(pop)))
            a = min(parents, key=lambda c: c.loss)
            if rng.random() < cfg.crossover_rate:
                others = rng.sample(pop, k=min(cfg.tournament_size, len(pop)))
                b = min(others, key=lambda c: c.loss)
                child = crossover_networks(a.network, b.network, rng)
            else:
                child = mutate_network(a.network, rng,
                                       enable_2d=cfg.enable_2d)
            loss = network_loss(child, train_ch, labels_train,
                                parsimony=cfg.parsimony, use_jax=use_jax,
                                complexity_divisor=cdiv)
            acc = network_accuracy(child, train_ch, labels_train,
                                    use_jax=use_jax)
            offspring.append(NetworkCandidate(network=child, loss=loss,
                                              accuracy=acc))

        combined = pop + offspring
        combined.sort(key=lambda c: c.loss)
        pop = combined[: cfg.pop_size]
        best = pop[0]

        if best.loss < best_ever.loss - 1e-9:
            best_ever = best
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        test_acc = float("nan")
        if test_ch is not None and labels_test is not None:
            test_acc = network_accuracy(best.network, test_ch, labels_test,
                                         use_jax=use_jax)

        mean_loss = float(np.mean([c.loss for c in pop
                                     if math.isfinite(c.loss)] or [float("inf")]))
        history.append(dict(
            gen=gen,
            best_loss=float(best.loss),
            best_train_acc=float(best.accuracy),
            best_test_acc=float(test_acc) if math.isfinite(test_acc) else float("nan"),
            best_cx=int(best.network.complexity),
            mean_loss=mean_loss,
        ))

        if cfg.verbose:
            tr_acc = best.accuracy
            te_str = f"{test_acc:.3f}" if math.isfinite(test_acc) else "—"
            print(f"[gen {gen:3d}] loss={best.loss:.4f}  "
                  f"train_acc={tr_acc:.3f}  test_acc={te_str}  "
                  f"cx={best.network.complexity}")

        if gens_no_improve >= cfg.early_stop_patience:
            if cfg.verbose:
                print(f"[gp] early stop at gen {gen} "
                      f"({gens_no_improve} gens without improvement)")
            break

    return best_ever, history


__all__ = [
    "SymbolicNetwork",
    "NetworkCandidate",
    "NetworkGPConfig",
    "DEFAULT_IMAGE_CHANNELS",
    "make_image_channels",
    "evaluate_network",
    "evaluate_network_batch",
    "evaluate_network_jax_batch",
    "clear_jax_tree_cache",
    "random_network",
    "warm_start_from_binary",
    "mutate_network",
    "crossover_networks",
    "network_loss",
    "network_accuracy",
    "run_network_gp",
]
