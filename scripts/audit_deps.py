"""Audit tessera's internal dependency graph: layering + cycle check.

Run from the tessera repo root:
    python scripts/audit_deps.py
"""
import ast
from pathlib import Path

ROOT = Path('src/tessera')


def module_path(rel: str) -> str:
    parts = rel.replace('\\', '/').split('/')
    if parts[-1] == '__init__.py':
        parts.pop()
    elif parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return 'tessera.' + '.'.join(parts)


def main():
    deps = {}
    for py in ROOT.rglob('*.py'):
        if '__pycache__' in str(py) or '_numba_kernels' in py.name:
            continue
        rel = py.relative_to(ROOT)
        mod = module_path(str(rel))
        try:
            tree = ast.parse(py.read_text(encoding='utf-8'))
        except SyntaxError:
            continue
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('tessera'):
                    imports.add(node.module)
                elif node.level and node.level > 0:
                    cur_parts = mod.split('.')
                    target = '.'.join(cur_parts[:-node.level])
                    if node.module:
                        target = target + '.' + node.module
                    if target.startswith('tessera'):
                        imports.add(target)
        deps[mod] = imports

    print('=== tessera dependency graph ===\n')
    for mod in sorted(deps):
        if not deps[mod]:
            continue
        print(f'{mod}:')
        for dep in sorted(deps[mod]):
            marker = '  *** ' if dep == mod else '    '
            print(f'{marker}-> {dep}')

    print('\n=== cycle check ===')
    visited = set()
    stack = []
    stack_set = set()
    cycles = []

    def dfs(node):
        if node in stack_set:
            i = stack.index(node)
            cycles.append(stack[i:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.append(node)
        stack_set.add(node)
        for n in deps.get(node, []):
            if n in deps:
                dfs(n)
        stack.pop()
        stack_set.discard(node)

    for mod in deps:
        if mod not in visited:
            dfs(mod)

    if cycles:
        for c in cycles:
            print(f'  CYCLE: {" -> ".join(c)}')
    else:
        print('  no cycles detected')

    print('\n=== layering depth ===')
    # Reverse topological depth: leaves (no internal deps) = depth 0;
    # depth = 1 + max(depth of internal deps).
    depth = {}

    def compute_depth(node):
        if node in depth:
            return depth[node]
        depth[node] = 0  # sentinel for cycle handling
        internal_deps = [d for d in deps.get(node, []) if d in deps]
        if internal_deps:
            depth[node] = 1 + max(compute_depth(d) for d in internal_deps)
        return depth[node]

    for mod in deps:
        compute_depth(mod)

    by_depth = {}
    for mod, d in depth.items():
        by_depth.setdefault(d, []).append(mod)
    for d in sorted(by_depth):
        print(f'  depth {d}:')
        for mod in sorted(by_depth[d]):
            print(f'    {mod}')


if __name__ == '__main__':
    main()
