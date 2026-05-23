"""Tests for measure convolution (Measure.compose) and the
measure-algebra identity L_μ(L_ν(x)) ≡ L_{μ*ν}(x)."""
import numpy as np
import pytest

from tessera.expression import (
    Atom, Measure,
    measure_lag, measure_diff, measure_ema, measure_signed_sum,
)


# ---------------- Basic compose semantics ----------------

def test_lag_compose_lag_sums_lags():
    """lag(k) ∗ lag(m) = lag(k + m)."""
    m = measure_lag(2).compose(measure_lag(3))
    assert m.atoms == (Atom(weight=1.0, lag=5),)


def test_diff_compose_diff_is_second_diff():
    """diff(1) ∗ diff(1) = second difference: 1, -2, 1."""
    m = measure_diff(1).compose(measure_diff(1))
    weights = tuple((a.weight, a.lag) for a in m.atoms)
    assert weights == ((1.0, 0), (-2.0, 1), (1.0, 2))


def test_compose_with_zero_measure_gives_zero():
    """Convolution with the zero measure (empty atoms, no density)
    yields the zero measure."""
    zero = Measure()
    m = measure_diff(1).compose(zero)
    assert m.atoms == ()


def test_compose_with_dirac_at_zero_is_identity():
    """μ ∗ δ_0 = μ. Identity element of the convolution algebra."""
    m_in = measure_diff(1)
    identity = measure_lag(0)   # δ at lag 0, weight 1
    out = m_in.compose(identity)
    # Same atoms as the input
    assert set((a.weight, a.lag) for a in out.atoms) == \
           set((a.weight, a.lag) for a in m_in.atoms)


def test_compose_is_commutative_on_atomic_measures():
    """μ ∗ ν = ν ∗ μ for atomic measures (convolution is commutative)."""
    m1 = measure_signed_sum([(2.0, 1), (-1.0, 3)])
    m2 = measure_signed_sum([(3.0, 0), (1.0, 5)])
    assert m1.compose(m2) == m2.compose(m1)


def test_compose_is_associative():
    """(μ ∗ ν) ∗ ρ = μ ∗ (ν ∗ ρ)."""
    m1 = measure_signed_sum([(2.0, 1), (-1.0, 3)])
    m2 = measure_signed_sum([(3.0, 0), (1.0, 5)])
    m3 = measure_diff(2)
    left = m1.compose(m2).compose(m3)
    right = m1.compose(m2.compose(m3))
    assert left == right


# ---------------- The headline identity: nested ≡ composed on interior ----------------

def test_nested_apply_equals_composed_apply_on_interior():
    """L_μ(L_ν(x)) and L_{μ*ν}(x) agree on the finite (non-warmup) rows."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    mu = measure_diff(1)
    nu = measure_diff(1)
    composed = mu.compose(nu)

    y_nested = mu.apply(nu.apply(x))   # default fill_warmup=NaN
    y_composed = composed.apply(x)

    mask = np.isfinite(y_nested) & np.isfinite(y_composed)
    np.testing.assert_allclose(y_nested[mask], y_composed[mask], rtol=1e-10)


def test_compose_with_density_via_kernel_convolution():
    """μ ∗ ν for non-atomic μ uses the convolved kernel.

    Note: the COMPOSED measure uses the truncated (support_max-bounded)
    kernel; the NESTED apply uses the recursive EMA which has effectively
    infinite memory. The mathematical identity holds in the limit of
    infinite support; here we just verify correlation > 0.99 to confirm
    the math is structurally right, not that the truncated kernel is
    bit-exact with the infinite recurrence.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    ema = measure_ema(halflife=10)
    diff = measure_diff(1)
    composed = diff.compose(ema)

    y_nested = diff.apply(ema.apply(x))
    y_composed = composed.apply(x)

    mask = np.isfinite(y_nested) & np.isfinite(y_composed)
    corr = float(np.corrcoef(y_nested[mask], y_composed[mask])[0, 1])
    assert corr > 0.99, (
        f"composed and nested apply should be highly correlated; "
        f"got corr={corr:.4f}"
    )


# ---------------- Edge cases ----------------

def test_compose_three_atomic_measures():
    """Triple composition associates correctly."""
    m1 = measure_lag(1)
    m2 = measure_lag(2)
    m3 = measure_lag(3)
    m = m1.compose(m2).compose(m3)
    assert m.atoms == (Atom(weight=1.0, lag=6),)


def test_compose_with_signed_cancellation():
    """Composing measures whose atoms cancel produces fewer atoms.
    (1·δ(0) + -1·δ(1)) ∗ (1·δ(0) + 1·δ(1)) = 1·δ(0) + 0·δ(1) − 1·δ(2)
    = 1·δ(0) − 1·δ(2)   (the zero at lag 1 is dropped by canonicalisation)
    """
    m1 = measure_signed_sum([(1.0, 0), (-1.0, 1)])
    m2 = measure_signed_sum([(1.0, 0), (1.0, 1)])
    out = m1.compose(m2)
    weights = tuple((a.weight, a.lag) for a in out.atoms)
    assert weights == ((1.0, 0), (-1.0, 2))   # lag-1 atom cancels


def test_compose_returns_canonical_form():
    """The output of compose is automatically canonical (sorted, merged,
    zero-free) because it constructs a new Measure which canonicalises
    in __post_init__."""
    m1 = measure_signed_sum([(1.0, 0), (1.0, 0)])  # constructs to (2.0, 0)
    m2 = measure_lag(0)
    out = m1.compose(m2)
    # Output should be (2.0, 0) — canonical
    assert out.atoms == (Atom(weight=2.0, lag=0),)
