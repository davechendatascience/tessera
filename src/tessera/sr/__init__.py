"""tessera.sr — gradient-free symbolic regression.

A CSP-enumerated, const-free expression dictionary fit by a sparse LINEAR
combination (no gradient descent), with a top-down DECOMPOSITION driver
(outer-op peel + polynomial-STLSQ leaf + validation-gated separability) that
recovers deep compositional forms a single enumeration can't reach.

Quickstart
----------
    import numpy as np
    from tessera.sr import discover_decompose, CSPSRConfig

    env = {"v": np.random.uniform(0, 0.9, 2000)}
    y   = np.sqrt(1 - env["v"]**2)                 # relativistic factor
    res = discover_decompose(env, y)               # -> sqrt(1 - v^2), exact

Public API
----------
    discover            : single-pass CSP dictionary + sparse linear fit
    discover_decompose  : top-down decomposition around `discover`
    discover_boosted / discover_deep : stacked variants (see notes)
    CSPSRConfig, CSPSRResult, DecomposeResult, expr_to_str
"""
from __future__ import annotations

from tessera.sr.csp_sr import (  # noqa: F401
    discover, discover_boosted, discover_deep,
    CSPSRConfig, CSPSRResult, BoostedResult, expr_to_str,
    DEFAULT_UNARY, DEFAULT_BINARY,
)
from tessera.sr.decompose import (  # noqa: F401
    discover_decompose, DecomposeResult,
)

__all__ = [
    "discover", "discover_decompose",
    "discover_boosted", "discover_deep",
    "CSPSRConfig", "CSPSRResult", "BoostedResult", "DecomposeResult",
    "expr_to_str", "DEFAULT_UNARY", "DEFAULT_BINARY",
]
