"""tessera.expression — measure-theoretic operators and symbolic search primitives.

Public API
----------

Measures (Lebesgue-decomposed signed measures on non-negative lags):
    Atom, Measure, DENSITY_FAMILIES, register_density
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum

Functionals (n-ary wrappers; bilinear and Volterra-2):
    Functional, LinearFunctional, SeparableBilinear, Volterra2
    apply_with_cache

Caching:
    FunctionalCache

Numba-JIT kernels (internal, exposed for direct benchmarking):
    from tessera.expression._numba_kernels import (
        ema_recursive, atomic_apply, conv_causal, benchmark_apply_paths,
    )
"""
from __future__ import annotations

from .measure import (
    Atom, Measure, DENSITY_FAMILIES, register_density,
    measure_lag, measure_diff, measure_ema, measure_roll_mean,
    measure_power_law, measure_signed_sum,
)
from .cache import FunctionalCache
from .functional import (
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
    apply_with_cache,
)
from .tree import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D, Node,
    BIN_OPS, UN_OPS, BIN_OP_FNS, UN_OP_FNS,
    complexity, depth, used_features, iter_subtrees, replace_at, evaluate,
    collect_const_values, set_const_values,
)
from .simplify import simplify
from .mutation import (
    MAX_DEPTH, MAX_COMPLEXITY, MAX_CONST_MAGNITUDE,
    validate_tree,
    random_measure, random_measure_2d, random_functional, random_tree,
    subtree_swap, subtree_crossover, constant_jitter,
    term_insert, term_delete, op_swap, measure_mutate, measure_2d_mutate,
    OP_WEIGHTS, mutate,
)
from .gp import (
    GPConfig, Candidate, GP, mse_loss, pareto_front,
    optimize_constants,
)
from .measure_2d import (
    Atom2D, Measure2D,
    measure_2d_atomic, measure_2d_separable,
    measure_2d_laplacian_5pt, measure_2d_diff_t, measure_2d_grad_x,
    measure_2d_sobel_x, measure_2d_sobel_y,
)
from .axes import (
    Invariance, Axis, TypedVar,
    check_compatibility, OperatorAxisRule, OPERATOR_RULES,
)
from .jit import (
    compile_tree, evaluate_jit, is_pure_pointwise,
    clear_jit_cache, jit_cache_size,
)
from .batched import (
    topology_key, extract_constants, n_constants,
    compile_topology, evaluate_population,
    clear_topo_cache, topo_cache_size,
)

__all__ = [
    # measure
    "Atom", "Measure", "DENSITY_FAMILIES", "register_density",
    "measure_lag", "measure_diff", "measure_ema", "measure_roll_mean",
    "measure_power_law", "measure_signed_sum",
    # cache
    "FunctionalCache",
    # functional
    "Functional", "LinearFunctional", "SeparableBilinear", "Volterra2",
    "apply_with_cache",
    # tree
    "Var", "Const", "BinOp", "UnOp", "FunctionalOp", "FunctionalOp2D", "Node",
    "BIN_OPS", "UN_OPS", "BIN_OP_FNS", "UN_OP_FNS",
    "complexity", "depth", "used_features", "iter_subtrees", "replace_at", "evaluate",
    "simplify", "collect_const_values", "set_const_values",
    # jit
    "compile_tree", "evaluate_jit", "is_pure_pointwise",
    "clear_jit_cache", "jit_cache_size",
    # batched (Tier 3)
    "topology_key", "extract_constants", "n_constants",
    "compile_topology", "evaluate_population",
    "clear_topo_cache", "topo_cache_size",
    # mutation
    "MAX_DEPTH", "MAX_COMPLEXITY", "MAX_CONST_MAGNITUDE",
    "validate_tree",
    "random_measure", "random_measure_2d", "random_functional", "random_tree",
    "subtree_swap", "subtree_crossover", "constant_jitter",
    "term_insert", "term_delete", "op_swap", "measure_mutate", "measure_2d_mutate",
    "OP_WEIGHTS", "mutate",
    # gp
    "GPConfig", "Candidate", "GP", "mse_loss", "pareto_front",
    "optimize_constants",
    # 2D measures
    "Atom2D", "Measure2D",
    "measure_2d_atomic", "measure_2d_separable",
    "measure_2d_laplacian_5pt", "measure_2d_diff_t", "measure_2d_grad_x",
    "measure_2d_sobel_x", "measure_2d_sobel_y",
    # axes (axis-semantic type system)
    "Invariance", "Axis", "TypedVar",
    "check_compatibility", "OperatorAxisRule", "OPERATOR_RULES",
]
