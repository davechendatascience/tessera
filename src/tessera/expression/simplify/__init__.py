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

    simplify_polynomial(node) — additive polynomial canonicalisation:
                                collect like monomial terms in add/sub
                                chains. Hand-rolled (no sympy dep),
                                designed to absorb the output shape of
                                `tessera.search.sufficient_stats`. See
                                `polynomial.py` module docstring.

    simplify_full(node)       — simplify_polynomial(simplify_canonical(node))
                                The full pipeline: AC sort → rule-based
                                folds → polynomial like-term collection.

Future additions (see docs/research/search_as_energy_min.md):
    simplify_egg(node)        — equality-saturation via egg/egglog
                                (optional dep; highest ceiling)
    Vocabulary-aware compression — recognise that a polynomial is a
                                truncated Taylor series of an existing
                                primitive (sin/cos/exp/...). Genuinely
                                hard; not in any CAS off-the-shelf.

Composition convention
----------------------
Newer simplifiers should be **idempotent** (running twice == running
once) and **monotone in complexity** (output complexity ≤ input
complexity, with equality only on already-canonical inputs).
Composing idempotent monotone simplifiers gives another idempotent
monotone simplifier — that's why these pipelines are safe to chain.
"""
from .core import simplify
from .ac import simplify_ac
from .polynomial import simplify_polynomial
from .cas_fallback import (
    cas_simplify, simplify_front_with_cas,
    is_worth_cas_pass, get_backend as cas_backend,
)


def simplify_canonical(node):
    """AC normalise then rule-based simplify. Recommended for SR scoring.

    AC sort first so commutative subtrees become canonical and constants
    cluster together; then the rule-based pass folds adjacent constants
    and other algebraic identities. Result: maximally-reduced tree
    under tessera's current rewrite vocabulary.
    """
    return simplify(simplify_ac(node))


def simplify_full(node):
    """Full pipeline: simplify_canonical then simplify_polynomial.

    Use when the tree may contain redundant monomial terms that aren't
    caught by AC + rule-based folds — typically after the
    sufficient-stats polish step appends a `Σ c_k · x^k` subtree.
    """
    return simplify_polynomial(simplify_canonical(node))


__all__ = [
    "simplify", "simplify_ac", "simplify_polynomial",
    "simplify_canonical", "simplify_full",
    "cas_simplify", "simplify_front_with_cas",
    "is_worth_cas_pass", "cas_backend",
]
