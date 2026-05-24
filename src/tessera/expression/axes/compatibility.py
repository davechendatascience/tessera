"""Axis-operator compatibility rules.

For each tessera operator, declare which axis types its inputs can
have. Used to check whether a tree respects the axis-semantic
declarations attached to its environment variables.

This is a *non-enforcing* checker (for now). The GP search loop does
NOT yet refuse to generate trees that violate axis compatibility —
that's the future `tessera.axes`-aware mutation work. Today this
module gives users a tool to VALIDATE constraints after the fact.

Compatibility rules
-------------------
- **Pointwise ops** (add, sub, mul, div, min, max, gt/lt/ge/le,
  tanh, abs, sign, neg, step): preserve axes — they apply
  element-wise and the output has the same shape and axes as the
  input.
- **LinearFunctional** (1-D convolution with a causal Measure):
  REQUIRES the convolution axis to have invariance `TRANSLATION` or
  `CAUSAL_TRANSLATION`. Outputs the same axes.
- **SeparableBilinear** / **Volterra2** (bilinear / quadratic in
  convolutions): same axis requirement.
- **FunctionalOp2D** (2-D measure on a (T, X) field): requires the
  variable to have AT LEAST 2 axes, both of which must be
  `TRANSLATION` or `CAUSAL_TRANSLATION`. (The natural use case is
  a (time, space) field where time is causal and space is full
  translation.)
- **reduce_mean / reduce_max / reduce_sum / reduce_std**: legal on
  ANY axis type. The result has one fewer dimension (the reduced
  axis is eliminated).

What's NOT yet handled
----------------------
- **Permutation-axis-aware operators**: sum / mean / max are symmetric
  and work on permutation axes; but a `gt(x, y)` between two elements
  of a permutation axis is NOT permutation-equivariant. The current
  rules don't enforce this finer distinction.
- **CYCLIC axis**: tessera doesn't have circular-convolution ops; for
  now we accept LinearFunctional on cyclic axes but note that the
  result is NOT cyclically equivariant.
- **GRAPH axis**: tessera has no graph-aware ops. Defer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

from ..tree import (
    Node, Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    iter_subtrees,
)
from ..functional import (
    Functional, LinearFunctional, SeparableBilinear, Volterra2,
)

from .types import Axis, Invariance, TypedVar


# Set of invariance types a 1-D convolution-style measure can act on
_CONVOLUTIONAL_AXES = {
    Invariance.TRANSLATION,
    Invariance.CAUSAL_TRANSLATION,
}


@dataclass(frozen=True)
class OperatorAxisRule:
    """Rule for how an operator interacts with axis types.

    Attributes
    ----------
    name : str
        Operator identifier.
    is_pointwise : bool
        True iff the op is elementwise (preserves all axes).
    is_reduction : bool
        True iff the op collapses an array to a scalar.
    requires_invariance : set[Invariance] | None
        If not None, the operator requires AT LEAST ONE of its input
        axes to have one of these invariance types. Examples:
            FunctionalOp(LinearFunctional): requires TRANSLATION or
                                            CAUSAL_TRANSLATION
            FunctionalOp2D: requires (TRANSLATION/CAUSAL) × (TRANSLATION/CAUSAL)
    """
    name: str
    is_pointwise: bool = False
    is_reduction: bool = False
    requires_invariance: Optional[frozenset] = None


# Built-in compatibility rules
OPERATOR_RULES: dict[str, OperatorAxisRule] = {
    # Pointwise binary
    **{op: OperatorAxisRule(name=op, is_pointwise=True)
       for op in ("add", "sub", "mul", "div", "min", "max",
                  "gt", "lt", "ge", "le")},
    # Pointwise unary (non-reducing)
    **{op: OperatorAxisRule(name=op, is_pointwise=True)
       for op in ("tanh", "abs", "sign", "neg", "step")},
    # Reductions
    **{op: OperatorAxisRule(name=op, is_reduction=True)
       for op in ("reduce_mean", "reduce_max", "reduce_sum", "reduce_std")},
    # Functional ops
    "LinearFunctional": OperatorAxisRule(
        name="LinearFunctional",
        requires_invariance=frozenset(_CONVOLUTIONAL_AXES),
    ),
    "SeparableBilinear": OperatorAxisRule(
        name="SeparableBilinear",
        requires_invariance=frozenset(_CONVOLUTIONAL_AXES),
    ),
    "Volterra2": OperatorAxisRule(
        name="Volterra2",
        requires_invariance=frozenset(_CONVOLUTIONAL_AXES),
    ),
    "FunctionalOp2D": OperatorAxisRule(
        name="FunctionalOp2D",
        requires_invariance=frozenset(_CONVOLUTIONAL_AXES),
    ),
}


def _functional_class_name(f: Functional) -> str:
    """Return the rule key for a Functional instance."""
    if isinstance(f, LinearFunctional):
        return "LinearFunctional"
    if isinstance(f, SeparableBilinear):
        return "SeparableBilinear"
    if isinstance(f, Volterra2):
        return "Volterra2"
    return "Unknown"


def check_compatibility(
    tree: Node,
    typed_env: dict[str, TypedVar],
) -> Optional[str]:
    """Verify a tree respects the axis declarations of its env.

    Walks the tree; for every operator that touches a `Var(name)`,
    looks up `typed_env[name]` and checks compatibility against
    `OPERATOR_RULES`.

    Returns
    -------
    None if compatible; otherwise a short error string describing
    the first violation found.

    Caveats
    -------
    This checker is CONSERVATIVE. It checks per-Var operator
    constraints but doesn't track the FLOW of axes through the tree
    (i.e., it doesn't note that `reduce_mean(x)` should eliminate an
    axis from the downstream effective shape). For tessera's current
    typical SR trees (Var → FunctionalOp → pointwise compositions →
    scalar output), the conservative check is sufficient.

    Usage
    -----
        typed_env = {
            "image": TypedVar("image", axes=(
                Axis("h", 28, Invariance.TRANSLATION),
                Axis("w", 28, Invariance.TRANSLATION),
            )),
        }
        err = check_compatibility(tree, typed_env)
        if err is not None:
            print(f"axis-violation: {err}")
    """
    for sub in iter_subtrees(tree):
        err = _check_node(sub, typed_env)
        if err is not None:
            return err
    return None


def _check_node(
    node: Node,
    typed_env: dict[str, TypedVar],
) -> Optional[str]:
    """Check ONE node against the env's axis declarations."""
    # Helper: collect the TypedVar referenced (if any) by this node
    referenced_tvars: list[TypedVar] = []
    for sub in iter_subtrees(node):
        if isinstance(sub, Var) and sub.name in typed_env:
            referenced_tvars.append(typed_env[sub.name])

    if not referenced_tvars:
        return None  # untyped subtree; nothing to check

    if isinstance(node, FunctionalOp):
        rule_name = _functional_class_name(node.functional)
        rule = OPERATOR_RULES.get(rule_name)
        if rule is None or rule.requires_invariance is None:
            return None
        return _check_invariance_requirement(rule, referenced_tvars)

    if isinstance(node, FunctionalOp2D):
        rule = OPERATOR_RULES["FunctionalOp2D"]
        # FunctionalOp2D needs the arg to be at least 2-D
        for tv in referenced_tvars:
            if tv.ndim < 2:
                return (
                    f"FunctionalOp2D requires a 2-D variable; "
                    f"{tv.name!r} has only {tv.ndim} axis"
                )
            # Both top-2 axes should be convolutional
            if not all(a.invariance in _CONVOLUTIONAL_AXES
                       for a in tv.axes[:2]):
                bad = [a for a in tv.axes[:2]
                       if a.invariance not in _CONVOLUTIONAL_AXES]
                return (
                    f"FunctionalOp2D requires the first 2 axes of "
                    f"{tv.name!r} to be TRANSLATION/CAUSAL_TRANSLATION; "
                    f"got {[a.invariance.value for a in bad]}"
                )
        return None

    # Pointwise / reduction / etc. — accept (per the docstring's
    # "non-enforcing" promise we don't ban anything that wouldn't
    # NUMERICALLY crash)
    return None


def _check_invariance_requirement(
    rule: OperatorAxisRule,
    typed_vars: list[TypedVar],
) -> Optional[str]:
    """A functional-style rule requires AT LEAST ONE convolutional
    axis on the referenced variable."""
    if not rule.requires_invariance:
        return None
    for tv in typed_vars:
        if not tv.axes:
            return (
                f"{rule.name} requires the input variable to have at "
                f"least one declared axis; {tv.name!r} has none"
            )
        if not any(a.invariance in rule.requires_invariance
                   for a in tv.axes):
            return (
                f"{rule.name} requires at least one axis of {tv.name!r} "
                f"to have invariance in "
                f"{[i.value for i in rule.requires_invariance]}; "
                f"got {[a.invariance.value for a in tv.axes]}"
            )
    return None
