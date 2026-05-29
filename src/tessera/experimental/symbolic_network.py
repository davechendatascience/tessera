"""Symbolic network for image classification (Milestone A scaffold).

Provenance: `docs/research/hybrid_symbolic_networks.md` Milestone A.

Status: **UNTESTED**. Milestone-A MVP — first cut of the
SymbolicNetwork architecture, network-aware GP loop, and a
binary-classification fitness path. All choices are the simplest
possible per the milestone scope:

- Hardcoded 2-layer architecture (K layer-1 trees + 1 layer-2 tree)
- Binary classification (2-class)
- Layer-1 trees: image → array, mean-pooled to scalar (matches the
  existing single-tree MNIST benchmark convention)
- Layer-2 tree: K scalar features → 1 scalar score
- Mutation: pick one of (K+1) slots, mutate that tree
- Crossover: swap one slot between two networks
- Tournament selection + (μ + λ) survival

Graduation criterion
--------------------
On MNIST digit 0 vs digit 1: TEST accuracy > 0.95. The existing
single-tree benchmark hit TEST 0.80 on 0-vs-rest (a harder task);
the K-network on the easier 2-class problem should comfortably beat
this if the architecture is workable. If it doesn't, either K is too
small, the GP can't navigate network-space, or the mean-pool
aggregation is too lossy.

Removal criterion
-----------------
TEST acc ≤ baseline (0.80) even with K=8 and longer search budgets.
Then the compositional structure isn't helping at this scale and the
direction needs different infrastructure.

Initial commit: 2026-05-29
Last evaluation: never

What this module provides
-------------------------

    SymbolicNetwork
        Frozen dataclass holding K layer-1 trees + 1 layer-2 tree.

    evaluate_network(network, image) -> float
        Score one image.

    evaluate_network_batch(network, images) -> np.ndarray
        Score a batch of images. Returns N scores.

    random_network(rng, K, ...) -> SymbolicNetwork
        Construct a random network.

    mutate_network(network, rng) -> SymbolicNetwork
        Pick one slot, mutate that tree, return new network.

    crossover_networks(a, b, rng) -> SymbolicNetwork
        Swap one slot between a and b.

    network_loss(network, images, labels, parsimony) -> float
        MSE(sigmoid(scores), labels) + parsimony · total_complexity.

    network_accuracy(network, images, labels) -> float
        Threshold scores at 0; compare to labels.

    NetworkGPConfig (dataclass)
    run_network_gp(images_train, labels_train, cfg, images_test, labels_test)
        -> (best_candidate, history)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from tessera.expression.tree import (
    Node, Var, Const, BinOp, evaluate, complexity as tree_complexity,
)
from tessera.expression.mutation import (
    random_tree, mutate as tree_mutate, validate_tree,
)
from tessera.expression.simplify import simplify
from tessera.expression.jit import is_pure_pointwise


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


def _compile_image_tree_jax(tree: Node):
    """Compile a layer-1 tree to a JIT'd batched function:
       fn(images_batch, consts) -> features_batch

    images_batch: shape (N, H, W).
    consts: 1-D array of Const values in pre-order.
    features_batch: shape (N,) — mean-pooled per-image scalars.

    Returns None if the tree contains FunctionalOp / FunctionalOp2D
    (not supported by the per-tree JIT path).
    """
    if not is_pure_pointwise(tree):
        return None
    from tessera.expression.batched import (
        topology_key, _build_parametric_fn,
    )
    import jax
    import jax.numpy as jnp

    key = (topology_key(tree), ("image",), "image")
    if key in _JAX_TREE_CACHE:
        return _JAX_TREE_CACHE[key]

    var_idx = {"image": 0}
    counter = [0]
    raw_fn = _build_parametric_fn(tree, var_idx, counter)

    def _per_image(image, consts):
        out = raw_fn((image,), consts)
        out = jnp.asarray(out)
        if out.ndim == 0:
            return out
        return jnp.mean(out)

    vmapped = jax.vmap(_per_image, in_axes=(0, None))
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
    network: "SymbolicNetwork", images: np.ndarray,
) -> Optional[np.ndarray]:
    """JAX batched evaluation. Returns scores of shape (N, n_classes)
    as numpy array, or None if any tree in the network has FunctionalOp
    / 2D (caller should fall back to numpy).
    """
    if not _jax_available():
        return None
    from tessera.expression.batched import extract_constants
    import jax.numpy as jnp

    images_j = jnp.asarray(images)

    feature_cols = []
    for tree in network.layer_1_trees:
        fn = _compile_image_tree_jax(tree)
        if fn is None:
            return None
        consts_list = extract_constants(tree)
        consts = (jnp.asarray(consts_list, dtype=images_j.dtype)
                  if consts_list else jnp.zeros(0, dtype=images_j.dtype))
        feats = fn(images_j, consts)  # shape (N,)
        feature_cols.append(feats)
    features_matrix = jnp.stack(feature_cols, axis=1)  # shape (N, K)

    score_cols = []
    for tree in network.layer_2_trees:
        l2_fn = _compile_feature_tree_jax(tree, network.K)
        if l2_fn is None:
            return None
        l2_consts_list = extract_constants(tree)
        l2_consts = (jnp.asarray(l2_consts_list, dtype=images_j.dtype)
                     if l2_consts_list else jnp.zeros(0, dtype=images_j.dtype))
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
    """K-feature 2-layer symbolic network for N-class classification.

    Layer 1: K trees, each takes the image (named "image") and produces
    a value. The mean of the output array (or the value itself if
    already scalar) is taken as the feature.

    Layer 2: N trees (one per class). Each takes the K named scalar
    features "f0", "f1", ..., "f{K-1}" and produces a scalar score.

    Prediction: softmax(scores) for class probabilities; argmax(scores)
    for the predicted class. Binary is the N=2 special case.
    """
    layer_1_trees: tuple[Node, ...]
    layer_2_trees: tuple[Node, ...]   # one per class; len = n_classes

    @property
    def K(self) -> int:
        return len(self.layer_1_trees)

    @property
    def n_classes(self) -> int:
        return len(self.layer_2_trees)

    @property
    def complexity(self) -> int:
        """Total complexity = Σ tree complexities (all K+N trees)."""
        return (sum(tree_complexity(t) for t in self.layer_1_trees)
                + sum(tree_complexity(t) for t in self.layer_2_trees))

    def __str__(self) -> str:
        l1 = "\n    ".join(f"f{i} = {t}" for i, t in enumerate(self.layer_1_trees))
        l2 = "\n    ".join(f"class_{c} = {t}" for c, t in enumerate(self.layer_2_trees))
        return (f"SymbolicNetwork(K={self.K}, n_classes={self.n_classes}, "
                f"cx={self.complexity}):\n"
                f"  Layer 1:\n    {l1}\n"
                f"  Layer 2 (one tree per class):\n    {l2}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_network(network: SymbolicNetwork, image: np.ndarray) -> np.ndarray:
    """Score one image; returns array of shape (n_classes,).

    Layer-1 trees mean-pool to scalar features. Each layer-2 tree
    consumes the K features and emits one class score. Failures in
    individual evaluations contribute 0 to that slot (silent — candidates
    with broken trees lose to selection).
    """
    features = []
    for tree in network.layer_1_trees:
        try:
            out = evaluate(tree, {"image": image}, fill_warmup=0.0)
            out = np.asarray(out, dtype=np.float64)
            if out.ndim == 0:
                val = float(out)
            else:
                val = float(np.nanmean(out))
            if not math.isfinite(val):
                val = 0.0
            features.append(val)
        except Exception:
            features.append(0.0)

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
    network: SymbolicNetwork, images: np.ndarray,
) -> np.ndarray:
    """Evaluate on N images. Returns shape (N, n_classes)."""
    n = images.shape[0]
    out = np.zeros((n, network.n_classes), dtype=np.float64)
    for i in range(n):
        out[i] = evaluate_network(network, images[i])
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
    layer_1_max_depth: int = 3,
    layer_2_max_depth: int = 3,
    enable_2d: bool = True,
    max_attempts_per_slot: int = 30,
    seed_layer_2_sum: bool = True,
) -> SymbolicNetwork:
    """Construct a random K-feature network.

    Layer-1 trees use a restricted alphabet for non-degenerate image
    features:
    - pointwise + Measure2D (if enable_2d) for spatial structure
    - 1D FunctionalOp (L[], V2[], B[]) is DISABLED — these expect
      1D time-series, not 2D image fields; applying them to images
      produces degenerate (constant or NaN) outputs
    - reduce_* operators are FILTERED OUT — they collapse the whole
      image to a scalar that's barely image-dependent

    Layer-2 trees use a pointwise-only alphabet over the K scalar
    features f0..f{K-1}. By default the FIRST candidate is the sum
    `f0 + f1 + ... + f_{K-1}` (a known non-degenerate combination
    that uses all features). The GP refines from there. This is the
    detect-then-seed pattern at the within-network init level —
    instead of starting from a likely-degenerate random tree, start
    from a tree that actually consumes the features.

    Failures to generate a valid layer-1 tree fall back to Var("image").
    """
    # Layer 1
    l1_features = ["image"]
    l1_feature_set = set(l1_features)
    l1_trees: list[Node] = []
    for _ in range(K):
        chosen: Optional[Node] = None
        for _ in range(max_attempts_per_slot):
            try:
                # pointwise_only=False with enable_2d=True allows 2D
                # Measure but NOT 1D FunctionalOp. (random_tree's
                # default behavior splits these.)
                # Actually we use pointwise_only=True + enable_2d as
                # a separate signal — see filter below.
                t = random_tree(rng, l1_features, max_depth=layer_1_max_depth,
                                enable_2d=enable_2d,
                                pointwise_only=True)
            except Exception:
                continue
            if validate_tree(t, l1_feature_set) is not None:
                continue
            if _tree_uses_reduce(t):
                # Disallow reduce_* — collapses image to scalar.
                continue
            chosen = t
            break
        if chosen is None:
            chosen = Var("image")
        try:
            chosen = simplify(chosen)
        except Exception:
            pass
        l1_trees.append(chosen)

    # Layer 2 — one tree per class
    l2_features = [f"f{i}" for i in range(K)]
    l2_feature_set = set(l2_features)
    l2_trees: list[Node] = []

    # Build sum-of-features tree (used as the binary positive-class seed).
    sum_features: Node = Var("f0")
    for i in range(1, K):
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
                anchor = c % K
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
                for i in range(1, K):
                    l2_tree = BinOp("add", l2_tree, Var(f"f{i}"))
        try:
            l2_tree = simplify(l2_tree)
        except Exception:
            pass
        l2_trees.append(l2_tree)

    return SymbolicNetwork(
        layer_1_trees=tuple(l1_trees),
        layer_2_trees=tuple(l2_trees),
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
        for _ in range(5):
            try:
                new_tree = tree_mutate(
                    [parent], rng, ["image"],
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
        )

    # Layer-2 slot: 0..n_classes-1
    class_idx = slot - network.K
    parent = network.layer_2_trees[class_idx]
    feature_names = [f"f{i}" for i in range(network.K)]
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
    )


def crossover_networks(
    a: SymbolicNetwork, b: SymbolicNetwork, rng: random.Random,
) -> SymbolicNetwork:
    """Swap one slot from b into a. Networks must agree on (K, n_classes)."""
    if a.K != b.K or a.n_classes != b.n_classes:
        raise ValueError(
            f"crossover: shape mismatch ({a.K}/{a.n_classes} vs "
            f"{b.K}/{b.n_classes})"
        )
    n_slots = a.K + a.n_classes
    slot = rng.randrange(n_slots)
    if slot < a.K:
        new_l1 = list(a.layer_1_trees)
        new_l1[slot] = b.layer_1_trees[slot]
        return SymbolicNetwork(
            layer_1_trees=tuple(new_l1),
            layer_2_trees=a.layer_2_trees,
        )
    class_idx = slot - a.K
    new_l2 = list(a.layer_2_trees)
    new_l2[class_idx] = b.layer_2_trees[class_idx]
    return SymbolicNetwork(
        layer_1_trees=a.layer_1_trees,
        layer_2_trees=tuple(new_l2),
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


def network_loss(
    network: SymbolicNetwork,
    images: np.ndarray, labels: np.ndarray,
    *, parsimony: float = 0.001, use_jax: bool = False,
) -> float:
    """Cross-entropy loss (mean over samples) + parsimony · total_complexity.

    For each sample i with true class y_i:
        loss_i = -log_softmax(scores[i])[y_i]

    Returns +inf on non-finite scores (selection drops broken candidates).
    Binary case (n_classes=2) is just the special case of softmax over 2 logits.
    """
    scores = _scores(network, images, use_jax=use_jax)
    if not np.isfinite(scores).all():
        return float("inf")
    log_probs = _log_softmax(scores)
    n = scores.shape[0]
    true_log_probs = log_probs[np.arange(n), labels.astype(int)]
    ce = float(-np.mean(true_log_probs))
    return ce + parsimony * network.complexity


def network_accuracy(
    network: SymbolicNetwork,
    images: np.ndarray, labels: np.ndarray,
    *, use_jax: bool = False,
) -> float:
    """argmax(scores) vs labels."""
    scores = _scores(network, images, use_jax=use_jax)
    preds = scores.argmax(axis=-1)
    return float(np.mean(preds == labels.astype(int)))


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
    layer_1_max_depth: int = 3
    layer_2_max_depth: int = 3
    enable_2d: bool = True
    parsimony: float = 0.001
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
) -> tuple[NetworkCandidate, list[dict]]:
    """Run the network-aware GP. Returns (best_candidate_ever, history).

    History is a list of per-generation dicts:
      {gen, best_loss, best_train_acc, best_test_acc, best_cx, mean_loss}
    """
    rng = random.Random(cfg.seed)

    # Decide JAX path once per run.
    use_jax = cfg.use_jax_eval and _jax_available()
    if cfg.use_jax_eval and not _jax_available() and cfg.verbose:
        print("[gp] use_jax_eval=True requested but JAX not installed; "
              "falling back to numpy.")

    # Initialize.
    pop: list[NetworkCandidate] = []
    while len(pop) < cfg.pop_size:
        net = random_network(
            rng, K=cfg.K, n_classes=cfg.n_classes,
            layer_1_max_depth=cfg.layer_1_max_depth,
            layer_2_max_depth=cfg.layer_2_max_depth,
            enable_2d=cfg.enable_2d,
        )
        loss = network_loss(net, images_train, labels_train,
                            parsimony=cfg.parsimony, use_jax=use_jax)
        acc = network_accuracy(net, images_train, labels_train, use_jax=use_jax)
        pop.append(NetworkCandidate(network=net, loss=loss, accuracy=acc))

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
            loss = network_loss(child, images_train, labels_train,
                                parsimony=cfg.parsimony, use_jax=use_jax)
            acc = network_accuracy(child, images_train, labels_train,
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
        if images_test is not None and labels_test is not None:
            test_acc = network_accuracy(best.network, images_test, labels_test,
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
    "evaluate_network",
    "evaluate_network_batch",
    "evaluate_network_jax_batch",
    "clear_jax_tree_cache",
    "random_network",
    "mutate_network",
    "crossover_networks",
    "network_loss",
    "network_accuracy",
    "run_network_gp",
]
