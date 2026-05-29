"""GP cross-process determinism tests.

Regression coverage for the PYTHONHASHSEED leak found in 2026-05-28:
`_OP_SWAP_GROUPS` was stored as `list[set[str]]`; set iteration order
depends on hash randomization, which propagated into `rng.choice(others)`
and made GP runs non-reproducible across Python invocations even with
identical config and seed.

These tests run the GP under different simulated hash seeds (by mocking
str hash or by direct construction) and verify that the result is
identical to a baseline single run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tessera.expression.mutation import _OP_SWAP_GROUPS
from tessera.search import GP, GPConfig


# ---------------- Static structure check ----------------

def test_op_swap_groups_are_tuples_not_sets():
    """`_OP_SWAP_GROUPS` must be tuples to keep iteration order stable
    across processes. Sets iterate in hash-randomized order; lists/tuples
    do not.
    """
    for group in _OP_SWAP_GROUPS:
        assert isinstance(group, tuple), (
            f"_OP_SWAP_GROUPS entry must be tuple for cross-process "
            f"determinism; got {type(group).__name__}: {group!r}. "
            "See test docstring for context."
        )


def test_op_swap_groups_are_sorted():
    """Tuples in `_OP_SWAP_GROUPS` should be sorted so the iteration
    order is canonical (independent of how a future contributor types
    them). Belt-and-suspenders to the tuple constraint above."""
    for group in _OP_SWAP_GROUPS:
        assert list(group) == sorted(group), (
            f"_OP_SWAP_GROUPS entry {group!r} should be sorted. "
            "Keeps iteration order canonical and easy to audit."
        )


# ---------------- Cross-process determinism ----------------

_CROSS_PROCESS_SCRIPT = '''
import sys; sys.path.insert(0, {bench_dir!r})
import numpy as np
from tessera.search import GP, GPConfig
from tessera.expression.tree import Var, Const, BinOp

# Tiny deterministic problem: y = 3*x + 2
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 200)
y = 3.0 * x + 2.0

cfg = GPConfig(
    pop_size=60, n_gens=15, seed=2026,
    pointwise_only=True, verbose=False,
    optimize_constants_every=3, optimize_constants_maxiter=20,
    early_stop_patience=100,
)
gp = GP(cfg)
front = gp.run({{"x": x}}, y, feature_names=["x"])
best = min(front, key=lambda c: c.train_loss)
# Print enough state to detect any divergence
print(f"{{best.complexity}}|{{best.train_loss:.10f}}|{{best.tree}}")
'''


@pytest.mark.parametrize("hashseed", ["0", "1", "42"])
def test_gp_deterministic_across_hashseeds(hashseed: str, tmp_path: Path):
    """Run a small GP problem in 3 subprocesses with different
    PYTHONHASHSEED values. Results must be bit-identical.

    This is the integration-level check that complements the static
    `test_op_swap_groups_are_tuples_not_sets` test above. Even if
    someone refactors `_OP_SWAP_GROUPS`, this test fails if a hash-
    randomized iteration leaks anywhere new.
    """
    bench_dir = Path(__file__).resolve().parent.parent.parent / "benchmarks"
    script = _CROSS_PROCESS_SCRIPT.format(bench_dir=str(bench_dir))

    results: list[str] = []
    for seed_attempt in (hashseed, "0", "999"):
        env = {"PYTHONHASHSEED": seed_attempt}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env={**__import__("os").environ, **env},
            timeout=60,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"subprocess failed (hashseed={seed_attempt}):\n"
                f"stdout={proc.stdout}\nstderr={proc.stderr}"
            )
        # Last non-empty line is the result.
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        assert lines, f"no output from subprocess (hashseed={seed_attempt})"
        results.append(lines[-1])

    assert len(set(results)) == 1, (
        f"GP results differ across PYTHONHASHSEED values: {results}. "
        "Some set/dict iteration order is leaking into search decisions. "
        "See test docstring for context."
    )
