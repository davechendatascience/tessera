"""Backward-compatibility shim.

`csp_sr` was promoted out of `experimental` into the stable package
`tessera.search.csp`. Import from there:

    from tessera.search.csp import discover, CSPSRConfig, discover_decompose

This shim keeps existing imports (`tessera.experimental.csp_sr`) working.
"""
from tessera.search.csp.csp_sr import *      # noqa: F401,F403
from tessera.search.csp.csp_sr import (      # noqa: F401
    discover, discover_boosted, discover_deep,
    CSPSRConfig, CSPSRResult, BoostedResult, expr_to_str,
    DEFAULT_UNARY, DEFAULT_BINARY,
)
