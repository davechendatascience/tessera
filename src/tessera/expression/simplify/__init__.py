"""tessera.expression.simplify — algebraic simplification of Expr trees.

A growing collection of tree rewriters that preserve semantics but
reduce node count (or canonicalise structure). Used by the GP scoring
path to make complexity reflect *effective* tree size, not nominal
node count.

Available simplifiers
---------------------
    simplify(node)            — the default; rule-based folds from core.py
                                (constant folding, X−X→0, safe-divide,
                                 algebraic identities)

Future additions (see docs/research_notes/search_as_energy_min.md):
    simplify_ac(node)         — Associative-Commutative normalisation
                                (sort children of add/mul/min/max)
    simplify_polynomial(node) — SymPy-based polynomial reduction
                                (optional dep)
    simplify_egg(node)        — equality-saturation via egg/egglog
                                (optional dep; highest ceiling)
"""
from .core import simplify

__all__ = ["simplify"]
