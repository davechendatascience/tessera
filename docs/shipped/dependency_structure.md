# Tessera dependency structure — invariants and rationale

**Status:** ✓ DONE. Frozen contract enforced by `tests/test_dependency_structure.py`.

This document captures tessera's internal module-dependency contract: which modules may import which, why, and how the contract is enforced. Anyone proposing a new module or refactor should read this first.

## Why have a contract

The user's framing: "*See in each addition of feature whether our framework stays loose and built using an axiomatic structure. No circular dependency and anticipates future upgrades by allowing room.*"

The contract has three properties:

1. **The import graph is a DAG** — no cycles. Cycles produce import-order bugs that are very hard to diagnose and force `import inside function` workarounds.
2. **Layering is strictly bottom-up** — higher-level modules depend on lower-level ones, never the reverse. Specifically: `expression.measure` cannot import `expression.tree`, and no `expression.*` module (except a documented backward-compat shim) can import `search.*`.
3. **Extension points use predicate/strategy parameters** — new behaviour is added by passing a callable or strategy, not by editing the core. Example: `materialize_shared_subtrees(is_cacheable=..., canonical_key=..., cache=...)`.

## The layering (verified by `scripts/audit_deps.py`)

```
depth 0 (primitives, no internal deps):
  tessera.backend
  tessera.expression.measure
  tessera.expression.axes.types
  tessera.koopman.model
  tessera.search.losses
  tessera.search.losses_trading

depth 1 (depend on primitives):
  tessera.expression.cache
  tessera.expression.functional
  tessera.expression.measure_2d
  tessera.expression.simplify

depth 2 (the Node + evaluator):
  tessera.expression.tree

depth 3 (tree-walkers + GP primitives):
  tessera.expression.axes.compatibility
  tessera.expression.interval
  tessera.expression.jit
  tessera.expression.mutation
  tessera.expression.simplify.{ac,core}
  tessera.search.base
  tessera.search.scoring

depth 4 (compositions + bounds):
  tessera.expression.batched
  tessera.expression.materialize
  tessera.search.bounds
  tessera.search.const_opt
  tessera.search.pareto

depth 5 (Hall of Fame uses Pareto + Candidate):
  tessera.search.hall_of_fame

depth 6 (search drivers):
  tessera.search.gp
  tessera.search.random_search
  tessera.search.sa

depth 7 (backward-compat shims):
  tessera.expression.gp   (re-exports from tessera.search.gp; allowed
                            by explicit exception in the dependency-
                            structure tests)
```

## Forbidden imports (enforced by test_dependency_structure.py)

Each rule has the form *importer-prefix → forbidden-target-prefix*:

| Importer | Forbidden target | Why |
|---|---|---|
| `tessera.expression.measure` | `tessera.expression.tree` | Measures are primitives, the tree builds on them |
| `tessera.expression.measure_2d` | `tessera.expression.tree` | Same |
| `tessera.expression.functional` | `tessera.expression.tree` | Same |
| `tessera.expression.cache` | `tessera.expression.tree` | Cache is per-Measure, doesn't need trees |
| `tessera.expression.tree` | `tessera.search.*` | Tree is the *language* the search operates over; the search consumes trees, not the other way |
| `tessera.expression.*` | `tessera.search.*` | (Same; broader form.) Documented exception: `tessera.expression.gp` is a backward-compat shim. |
| `tessera.backend` | `tessera.expression.*` | Backend is below expression in the layering |
| `tessera.backend` | `tessera.search.*` | Same |

Adding a new module: trace its imports. If they violate these rules, the dependency-structure test fails. Either rethink the design or add an explicit exception with rationale.

## Extension-point pattern

When adding a feature with future-upgrade room, prefer **strategy parameters** over inheritance or module-level constants. Example from `materialize_shared_subtrees`:

```python
def materialize_shared_subtrees(
    trees,
    env,
    *,
    threshold=2,
    is_cacheable=None,       # callable: which subtrees are worth caching
    canonical_key=None,      # callable: subtree identity function
    cache=None,              # optional persistent dict for cross-call reuse
    ...
):
    ...
```

The defaults (`default_is_cacheable`, `default_canonical_key`) ship a working behaviour. Future upgrades plug in:
- Better identity via e-graph orbit-ID (when equality saturation ships) → swap `canonical_key=`
- Broader cacheability (compound pointwise) → swap `is_cacheable=`
- Cross-generation persistence → pass `cache=` dict

No subclassing, no inheritance, no module-level globals.

## Tools

- `scripts/audit_deps.py` — prints the full dependency graph + cycle check + depth layering. Run anytime.
- `tests/test_dependency_structure.py` — CI test that fails if a cycle or backwards-layering violation is introduced. Three tests:
  - `test_no_import_cycles`
  - `test_no_backwards_layering_violations`
  - `test_materialize_does_not_depend_on_search` (specific regression guard)

## Backward-compat shims

Some files exist purely as re-export shims for backward compatibility. They get an explicit exception in the test's `BACKCOMPAT_SHIMS` set. As of 2026-05-24:

- `tessera.expression.gp` — re-exports `GP`, `GPConfig`, etc. from `tessera.search.gp`. Old code that imports `from tessera.expression import GP` still works.

If we ever remove the shim, both the file and its exception in the test go away.

## What's NOT enforced

These are conventions, not test-enforced contracts:

1. **Module size.** No upper bound on lines per module.
2. **Naming.** No required prefix scheme.
3. **One responsibility per module.** Subjective; reviewed PR-by-PR.
4. **Test coverage.** Not enforced as a percentage; reviewed by feel.

These could become enforced contracts later. They're listed here so we know what's *not* in scope today.
