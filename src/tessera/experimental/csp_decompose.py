"""Backward-compatibility shim.

`csp_decompose` was promoted out of `experimental` into the stable package
`tessera.search.csp` (as `tessera.search.csp.decompose`). Import from there:

    from tessera.search.csp import discover_decompose

This shim keeps existing imports (`tessera.experimental.csp_decompose`) working.
"""
from tessera.search.csp.decompose import *   # noqa: F401,F403
from tessera.search.csp.decompose import (   # noqa: F401
    discover_decompose, DecomposeResult,
)
