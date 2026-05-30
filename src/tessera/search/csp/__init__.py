"""tessera.search.csp — gradient-free, enumerative symbolic regression.

A different search paradigm from the population-based searchers (GP, SA):
ENUMERATE a CSP-generated, const-free expression dictionary (symmetry-broken,
deduped) and fit a sparse LINEAR combination of it — no gradient descent,
constants enter as closed-form least-squares coefficients. A top-down
DECOMPOSITION driver (outer-op peel + polynomial-STLSQ leaf + validation-gated
separability) recovers deep compositional forms a single enumeration can't
reach. This is the FFX / SINDy family with an AI-Feynman-style decomposition
prepass, in tessera's operator vocabulary.

Quickstart
----------
    import numpy as np
    from tessera.search.csp import discover_decompose

    env = {"v": np.random.uniform(0, 0.9, 2000)}
    y   = np.sqrt(1 - env["v"]**2)                 # relativistic factor
    res = discover_decompose(env, y)               # -> sqrt(1 - v^2), exact

Public API
----------
    discover            : single-pass CSP dictionary + sparse linear fit
    discover_decompose  : top-down decomposition around `discover`
    discover_boosted / discover_deep : stacked variants (documented as a
                          negative result in docs/research/deep_symbolic_csp.md)
    CSPSRConfig, CSPSRResult, DecomposeResult, expr_to_str
"""
from __future__ import annotations

from tessera.search.csp.csp_sr import (  # noqa: F401
    discover, discover_boosted, discover_deep,
    CSPSRConfig, CSPSRResult, BoostedResult, expr_to_str,
    DEFAULT_UNARY, DEFAULT_BINARY,
)
from tessera.search.csp.decompose import (  # noqa: F401
    discover_decompose, DecomposeResult,
)

__all__ = [
    "discover", "discover_decompose",
    "discover_boosted", "discover_deep",
    "CSPSRConfig", "CSPSRResult", "BoostedResult", "DecomposeResult",
    "expr_to_str", "DEFAULT_UNARY", "DEFAULT_BINARY",
]
