"""Tests for measure canonicalisation at construction.

Per docs/research_notes/measure_theory_and_perfect_info.md §3.1
(Lebesgue decomposition uniqueness): two semantically identical
measures should be equal-by-value and hash-equal so the
FunctionalCache recognises them.
"""
import numpy as np
import pytest

from tessera.expression import (
    Atom, Measure, FunctionalCache, LinearFunctional, apply_with_cache,
    measure_signed_sum,
)


# ---------------- Canonical form at construction ----------------

def test_atoms_order_does_not_matter():
    """Same atoms in different order produce equal Measures."""
    m1 = Measure(atoms=(Atom(1.0, 0), Atom(2.0, 5)))
    m2 = Measure(atoms=(Atom(2.0, 5), Atom(1.0, 0)))
    assert m1 == m2
    assert hash(m1) == hash(m2)


def test_atoms_sorted_by_lag_in_canonical_form():
    """After construction, atoms are sorted by lag ascending."""
    m = Measure(atoms=(Atom(2.0, 5), Atom(1.0, 0), Atom(3.0, 10)))
    lags = [a.lag for a in m.atoms]
    assert lags == sorted(lags)
    assert lags == [0, 5, 10]


def test_duplicate_lag_atoms_are_merged():
    """Two atoms at the same lag combine into one with summed weight."""
    m = Measure(atoms=(Atom(1.0, 5), Atom(2.0, 5)))
    assert len(m.atoms) == 1
    assert m.atoms[0].lag == 5
    assert m.atoms[0].weight == pytest.approx(3.0)


def test_zero_weight_atoms_are_dropped():
    """Atoms with effectively-zero weight are removed."""
    m = Measure(atoms=(Atom(1.0, 0), Atom(0.0, 5), Atom(1e-15, 10)))
    assert len(m.atoms) == 1
    assert m.atoms[0].lag == 0


def test_atoms_that_cancel_to_zero_are_dropped():
    """w=+1 and w=-1 at same lag should cancel to zero ⇒ no atom remains."""
    m = Measure(atoms=(Atom(1.0, 5), Atom(-1.0, 5)))
    assert m.atoms == ()


def test_signed_sum_helper_produces_canonical_form():
    """measure_signed_sum constructs canonical-by-default."""
    m1 = measure_signed_sum([(2.0, 5), (1.0, 0)])
    m2 = measure_signed_sum([(1.0, 0), (2.0, 5)])
    assert m1 == m2


def test_canonicalisation_preserves_evaluation():
    """The canonicalised measure applies to data identically to the
    original (semantic equivalence, not just syntactic)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    m1 = Measure(atoms=(Atom(1.0, 0), Atom(2.0, 3), Atom(-1.0, 3)))
    m2 = Measure(atoms=(Atom(1.0, 0), Atom(1.0, 3)))
    # m1 should canonicalise to m2 (2 + -1 = 1 at lag 3, after merge)
    assert m1 == m2
    np.testing.assert_array_equal(
        m1.apply(x, fill_warmup=0.0),
        m2.apply(x, fill_warmup=0.0),
    )


# ---------------- FunctionalCache benefit ----------------

def test_cache_hits_on_mathematically_equivalent_measures():
    """Two distinct construction orders of the same measure should
    share a cache entry. Pre-canonicalisation this was a miss."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)

    m_forward = Measure(atoms=(Atom(1.0, 0), Atom(-1.0, 5)))
    m_reverse = Measure(atoms=(Atom(-1.0, 5), Atom(1.0, 0)))
    # Sanity: they're equal-by-value now
    assert m_forward == m_reverse

    cache = FunctionalCache(mem_size=100)
    f_forward = LinearFunctional(measure=m_forward)
    f_reverse = LinearFunctional(measure=m_reverse)

    # First call: cache miss + populate
    apply_with_cache(f_forward, cache, var_ids=("x",), xs=(x,))
    stats_after_first = dict(cache.stats)
    assert stats_after_first["mem_hits"] == 0

    # Second call with the "reverse" measure: SHOULD hit the cache
    # (because the canonicalised measure has the same hash)
    apply_with_cache(f_reverse, cache, var_ids=("x",), xs=(x,))
    stats_after_second = dict(cache.stats)
    assert stats_after_second["mem_hits"] == 1, (
        "expected cache hit after canonicalisation; "
        f"stats = {stats_after_second}"
    )


# ---------------- Density part is unaffected ----------------

def test_canonicalisation_does_not_touch_density():
    """The density family + params are not changed by canonicalisation."""
    from tessera.expression import measure_ema
    m = measure_ema(halflife=24)
    # No atoms; density preserved
    assert m.atoms == ()
    assert m.density_family == "exponential"
    assert m.density_params_dict["halflife"] == 24.0


def test_mixed_atomic_plus_density_canonicalisation():
    """Atomic part is canonicalised; density part is unchanged."""
    from tessera.expression import measure_ema
    base = measure_ema(halflife=24)
    # Manually add unordered atoms
    m = Measure(
        atoms=(Atom(2.0, 5), Atom(1.0, 0)),
        density_family=base.density_family,
        density_params=base.density_params,
        support_max=base.support_max,
    )
    # Atoms are sorted; density survives
    lags = [a.lag for a in m.atoms]
    assert lags == [0, 5]
    assert m.density_family == "exponential"
