"""tessera.expression.axes — axis-semantic type system.

Lets users declare WHAT KIND OF DIMENSION each axis of a variable is,
so the SR search and constraint checker can respect the symmetry
group acting on that axis.

Public API
----------
    Invariance     — enum of invariance groups (TRANSLATION,
                     CAUSAL_TRANSLATION, PERMUTATION, CYCLIC,
                     LOG_TRANSLATION, ROTATION, GRAPH, NONE)
    Axis           — declares (name, size, invariance) for one axis
    TypedVar       — Var with an axis declaration tuple
    check_compatibility — verify an Expr tree respects the axis
                          constraints of its env

Status
------
Minimum-useful first version. Provides the TYPE SYSTEM and a
COMPATIBILITY CHECKER. Does NOT yet enforce in the GP search loop
(future work — see docs/research/invariance_in_sr.md §11).

Existing untyped `Var` continues to work; `TypedVar` is an OPTIONAL
overlay you adopt when you want explicit invariance declarations.
"""
from .types import Invariance, Axis, TypedVar
from .compatibility import (
    check_compatibility,
    OperatorAxisRule, OPERATOR_RULES,
)

__all__ = [
    "Invariance", "Axis", "TypedVar",
    "check_compatibility",
    "OperatorAxisRule", "OPERATOR_RULES",
]
