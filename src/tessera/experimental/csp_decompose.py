"""Backward-compatibility shim.

`csp_decompose` was promoted out of `experimental` into the stable package
`tessera.sr` (as `tessera.sr.decompose`). Import from there:

    from tessera.sr import discover_decompose

This shim keeps existing imports (`tessera.experimental.csp_decompose`) working.
"""
from tessera.sr.decompose import *         # noqa: F401,F403
from tessera.sr.decompose import (         # noqa: F401
    discover_decompose, DecomposeResult,
)
