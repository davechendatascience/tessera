"""tessera.expression.simplify — algebraic simplification of Expr trees.

A growing collection of tree rewriters that preserve semantics but
reduce node count or canonicalise structure. Used by the GP scoring
path to make complexity reflect *effective* tree size, not nominal
node count.

Available simplifiers
---------------------
    simplify(node)            — rule-based folds from core.py
                                (constant folding, X−X→0, safe-divide,
                                 algebraic identities)

    simplify_ac(node)         — Associative-Commutative normalisation
                                (sort children of add/mul/min/max into
                                 canonical left-leaning shape)

    simplify_canonical(node)  — RECOMMENDED default for SR scoring.
                                simplify(simplify_ac(node)). AC norm
                                first so constants cluster, then rules
                                fold them.

Future additions (see docs/research_notes/search_as_energy_min.md):
    simplify_polynomial(node) — SymPy-based polynomial reduction
                                (optional dep)
    simplify_egg(node)        — equality-saturation via egg/egglog
                                (optional dep; highest ceiling)

Composition convention
----------------------
Newer simplifiers should be **idempotent** (running twice == running
once) and **monotone in complexity** (output complexity ≤ input
complexity, with equality only on already-canonical inputs).
Composing two idempotent monotone simplifiers gives another
idempotent monotone simplifier — that's why
`simplify_canonical = simplify ∘ simplify_ac` is safe.
"""
from .core import simplify
from .ac import simplify_ac


def simplify_canonical(node):
    """AC normalise then rule-based simplify. Recommended for SR scoring.

    AC sort first so commutative subtrees become canonical and constants
    cluster together; then the rule-based pass folds adjacent constants
    and other algebraic identities. Result: maximally-reduced tree
    under tessera's current rewrite vocabulary.
    """
    return simplify(simplify_ac(node))


__all__ = ["simplify", "simplify_ac", "simplify_canonical"]
