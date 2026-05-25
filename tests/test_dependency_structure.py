"""Structural-hardening tests for the tessera dependency graph.

Per the user's requirement: "see in each addition of feature whether
our framework stays loose and built using an axiomatic structure. No
circular dependency and anticipates future upgrades by allowing room."

These tests fail fast if a future change introduces:
- A cyclic import in tessera.* modules
- A backwards layering violation (e.g., `tessera.expression.tree`
  importing from `tessera.search.gp`)

The audit logic lives in `scripts/audit_deps.py`; this test wraps a
subset of it for CI enforcement.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _collect_deps() -> dict[str, set[str]]:
    root = Path(__file__).resolve().parent.parent / "src" / "tessera"
    deps: dict[str, set[str]] = {}
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or "_numba_kernels" in py.name:
            continue
        rel = py.relative_to(root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts.pop()
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        mod = "tessera." + ".".join(parts) if parts else "tessera"
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("tessera"):
                    imports.add(node.module)
                elif node.level and node.level > 0:
                    cur_parts = mod.split(".")
                    target = ".".join(cur_parts[:-node.level])
                    if node.module:
                        target = target + "." + node.module
                    if target.startswith("tessera"):
                        imports.add(target)
        deps[mod] = imports
    return deps


def test_no_import_cycles():
    """tessera.* import graph must be a DAG."""
    deps = _collect_deps()
    visited: set[str] = set()
    stack: list[str] = []
    stack_set: set[str] = set()
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        if node in stack_set:
            i = stack.index(node)
            cycles.append(stack[i:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.append(node)
        stack_set.add(node)
        for n in deps.get(node, set()):
            if n in deps:
                dfs(n)
        stack.pop()
        stack_set.discard(node)

    for mod in deps:
        if mod not in visited:
            dfs(mod)
    assert not cycles, f"cyclic dependencies: {cycles}"


def test_no_backwards_layering_violations():
    """High-level modules can depend on low-level modules; the
    reverse must never happen.

    The contract:
      tessera.backend                  (depth 0; foundational)
      tessera.expression.measure       (depth 0)
      tessera.expression.measure_2d    (depth 1; on measure)
      tessera.expression.functional    (depth 1)
      tessera.expression.cache         (depth 1)
      tessera.expression.tree          (depth 2; on the above)
      tessera.expression.{jit,
                          batched,
                          materialize, ...}   (depth 3-4; on tree)
      tessera.search.*                 (depth 5+; on expression)

    Specifically forbidden imports:
      - tessera.expression.measure       importing tessera.expression.tree
      - tessera.expression.tree          importing tessera.search.*
      - tessera.backend                  importing tessera.expression.*
    """
    deps = _collect_deps()
    forbidden = []

    # Documented exception: tessera.expression.gp is a backward-compat
    # shim that re-exports from tessera.search.gp. Listed here so the
    # exemption is explicit, not silent.
    BACKCOMPAT_SHIMS = {"tessera.expression.gp"}

    rules = [
        # (importer-prefix, forbidden-target-prefix)
        ("tessera.expression.measure", "tessera.expression.tree"),
        ("tessera.expression.measure_2d", "tessera.expression.tree"),
        ("tessera.expression.functional", "tessera.expression.tree"),
        ("tessera.expression.cache", "tessera.expression.tree"),
        ("tessera.expression.tree", "tessera.search"),
        ("tessera.expression", "tessera.search"),  # any expression mod
        ("tessera.backend", "tessera.expression"),
        ("tessera.backend", "tessera.search"),
    ]

    for mod, imports in deps.items():
        if mod in BACKCOMPAT_SHIMS:
            continue
        for prefix, target_prefix in rules:
            if not mod.startswith(prefix):
                continue
            for imp in imports:
                if imp.startswith(target_prefix):
                    # Allow self-imports within the same module package
                    if imp.startswith(prefix):
                        continue
                    forbidden.append((mod, imp))

    assert not forbidden, (
        f"backwards layering violations:\n"
        + "\n".join(f"  {a} imports {b}" for a, b in forbidden)
    )


def test_materialize_does_not_depend_on_search():
    """The materialize module is a primitive — search uses IT, not
    the other way around. Future regression guard."""
    deps = _collect_deps()
    imports = deps.get("tessera.expression.materialize", set())
    search_imports = [i for i in imports if i.startswith("tessera.search")]
    assert not search_imports, (
        f"materialize must not import from search; found {search_imports}"
    )


def test_no_production_code_imports_experimental():
    """`tessera.experimental.*` is a one-way subpackage: experimental
    code may import from anywhere (production layers are upstream),
    but no production code may import from experimental.

    See `tessera.experimental.__init__` docstring for the discipline.
    The motivation: experimental code implements untested conjectures
    from research notes. If production code depended on it, an
    experimental module's removal or graduation would break the rest
    of tessera.
    """
    deps = _collect_deps()
    violations = []
    for mod, imports in deps.items():
        # experimental code can import experimental — that's fine
        if mod.startswith("tessera.experimental"):
            continue
        for imp in imports:
            if imp.startswith("tessera.experimental"):
                violations.append((mod, imp))
    assert not violations, (
        f"production code imports from tessera.experimental (forbidden):\n"
        + "\n".join(f"  {a} imports {b}" for a, b in violations)
        + "\nSee tessera/experimental/__init__.py for the discipline."
    )


def test_experimental_subpackage_exists():
    """Smoke test: the experimental subpackage exists and has a
    docstring explaining the discipline.

    Documenting the discipline at the package level (rather than only
    in `docs/`) means anyone who lands inside the code via tooling
    (LSP, IDE) sees the contract immediately.
    """
    import tessera.experimental
    doc = tessera.experimental.__doc__
    assert doc is not None, "tessera.experimental needs a module docstring"
    assert "research-note conjectures" in doc, (
        "tessera.experimental docstring should explain it implements "
        "research-note conjectures"
    )
    assert "graduates" in doc or "graduation" in doc, (
        "tessera.experimental docstring should describe graduation lifecycle"
    )
