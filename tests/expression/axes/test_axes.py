"""Tests for tessera.expression.axes — axis-semantic type system."""
import numpy as np
import pytest

from tessera.expression import (
    Var, Const, BinOp, UnOp, FunctionalOp, FunctionalOp2D,
    LinearFunctional, SeparableBilinear, Volterra2,
    measure_diff, measure_ema, measure_2d_diff_t,
)
from tessera.expression.axes import (
    Invariance, Axis, TypedVar,
    check_compatibility, OPERATOR_RULES,
)


# ---------------- Invariance enum + Axis ----------------

def test_invariance_enum_values():
    """All declared invariance types are accessible."""
    for inv in ("TRANSLATION", "CAUSAL_TRANSLATION", "PERMUTATION",
                "CYCLIC", "LOG_TRANSLATION", "ROTATION", "GRAPH", "NONE"):
        assert hasattr(Invariance, inv)


def test_axis_construction_validates_size():
    """Axis size must be >= 1."""
    with pytest.raises(ValueError):
        Axis("bad", size=0, invariance=Invariance.TRANSLATION)


def test_axis_repr_includes_invariance():
    a = Axis("time", 100, Invariance.CAUSAL_TRANSLATION)
    assert "causal_translation" in repr(a)


# ---------------- TypedVar ----------------

def test_typed_var_for_time_series():
    """A time series: one axis with causal-translation invariance."""
    tv = TypedVar("returns", axes=(
        Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),
    ))
    assert tv.ndim == 1
    assert tv.shape == (10000,)
    assert tv.axes[0].invariance == Invariance.CAUSAL_TRANSLATION


def test_typed_var_for_image():
    """An image: two axes with full translation invariance."""
    tv = TypedVar("image", axes=(
        Axis("h", 28, Invariance.TRANSLATION),
        Axis("w", 28, Invariance.TRANSLATION),
    ))
    assert tv.ndim == 2
    assert tv.shape == (28, 28)


def test_typed_var_for_multi_asset_basket():
    """Time × asset: time is causal, asset is permutation-invariant."""
    tv = TypedVar("prices", axes=(
        Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),
        Axis("asset", 8, Invariance.PERMUTATION),
    ))
    assert tv.ndim == 2
    assert tv.axes[1].invariance == Invariance.PERMUTATION


def test_typed_var_empty_name_raises():
    with pytest.raises(ValueError):
        TypedVar("", axes=())


# ---------------- Compatibility: pointwise ops ----------------

def test_pointwise_ops_always_compatible():
    """add, sub, mul, etc. preserve any axis structure."""
    tv = TypedVar("x", axes=(Axis("time", 100, Invariance.CAUSAL_TRANSLATION),))
    env = {"x": tv}
    # x + x — both pointwise — should be fine
    tree = BinOp("add", Var("x"), Var("x"))
    assert check_compatibility(tree, env) is None
    # abs(x) — pointwise unary
    assert check_compatibility(UnOp("abs", Var("x")), env) is None


def test_indicator_ops_compatible():
    """gt/lt/ge/le pointwise — accept anything."""
    tv = TypedVar("x", axes=(Axis("time", 100, Invariance.CAUSAL_TRANSLATION),))
    env = {"x": tv}
    assert check_compatibility(
        BinOp("gt", Var("x"), Const(0.0)), env
    ) is None


# ---------------- Compatibility: functional ops ----------------

def test_linear_functional_on_translation_axis_ok():
    """LinearFunctional on a CAUSAL_TRANSLATION axis: legal."""
    tv = TypedVar("ts", axes=(
        Axis("time", 1000, Invariance.CAUSAL_TRANSLATION),
    ))
    tree = FunctionalOp(
        LinearFunctional(measure=measure_diff(1)),
        (Var("ts"),),
    )
    assert check_compatibility(tree, {"ts": tv}) is None


def test_linear_functional_on_permutation_axis_rejected():
    """LinearFunctional on a PERMUTATION axis: violation."""
    tv = TypedVar("basket", axes=(
        Axis("asset", 8, Invariance.PERMUTATION),
    ))
    tree = FunctionalOp(
        LinearFunctional(measure=measure_diff(1)),
        (Var("basket"),),
    )
    err = check_compatibility(tree, {"basket": tv})
    assert err is not None and "LinearFunctional" in err
    assert "permutation" in err


def test_functional_op_2d_on_image_ok():
    """FunctionalOp2D on a TRANSLATION × TRANSLATION image: legal."""
    tv = TypedVar("img", axes=(
        Axis("h", 28, Invariance.TRANSLATION),
        Axis("w", 28, Invariance.TRANSLATION),
    ))
    tree = FunctionalOp2D(measure_2d_diff_t(lag_t=1), Var("img"))
    assert check_compatibility(tree, {"img": tv}) is None


def test_functional_op_2d_on_1d_var_rejected():
    """FunctionalOp2D requires a 2-D variable."""
    tv = TypedVar("ts", axes=(
        Axis("time", 1000, Invariance.CAUSAL_TRANSLATION),
    ))
    tree = FunctionalOp2D(measure_2d_diff_t(lag_t=1), Var("ts"))
    err = check_compatibility(tree, {"ts": tv})
    assert err is not None and "2-D" in err


def test_functional_op_2d_on_permutation_axis_rejected():
    """FunctionalOp2D on a PERMUTATION axis: violation."""
    tv = TypedVar("basket", axes=(
        Axis("time", 1000, Invariance.CAUSAL_TRANSLATION),
        Axis("asset", 8, Invariance.PERMUTATION),   # 2nd axis is wrong type
    ))
    tree = FunctionalOp2D(measure_2d_diff_t(lag_t=1), Var("basket"))
    err = check_compatibility(tree, {"basket": tv})
    assert err is not None
    assert "permutation" in err


# ---------------- Compatibility: reductions ----------------

def test_reduce_op_always_compatible():
    """reduce_mean / reduce_max / etc. work on any axis type."""
    for inv in (Invariance.TRANSLATION, Invariance.CAUSAL_TRANSLATION,
                Invariance.PERMUTATION, Invariance.CYCLIC):
        tv = TypedVar("x", axes=(Axis("dim", 100, inv),))
        tree = UnOp("reduce_mean", Var("x"))
        assert check_compatibility(tree, {"x": tv}) is None, (
            f"reduce_mean rejected on {inv}"
        )


# ---------------- Untyped Vars are silently accepted ----------------

def test_untyped_vars_are_accepted():
    """If a Var's name is not in typed_env, the checker doesn't
    complain — it's untyped (legacy) usage."""
    tree = BinOp("add", Var("untyped"), Var("typed"))
    typed_env = {"typed": TypedVar("typed", axes=(
        Axis("time", 100, Invariance.CAUSAL_TRANSLATION),
    ))}
    # No error because "untyped" has no declaration
    assert check_compatibility(tree, typed_env) is None


# ---------------- OPERATOR_RULES table ----------------

def test_operator_rules_table_has_expected_keys():
    """Sanity-check the rules table has entries for the main ops."""
    expected_keys = {
        # Pointwise
        "add", "sub", "mul", "div", "min", "max",
        "gt", "lt", "ge", "le",
        "tanh", "abs", "sign", "neg", "step",
        # Reductions
        "reduce_mean", "reduce_max", "reduce_sum", "reduce_std",
        # Functionals
        "LinearFunctional", "SeparableBilinear", "Volterra2",
        "FunctionalOp2D",
    }
    assert expected_keys.issubset(set(OPERATOR_RULES))


def test_reductions_marked_as_reduction():
    for op in ("reduce_mean", "reduce_max", "reduce_sum", "reduce_std"):
        assert OPERATOR_RULES[op].is_reduction


def test_functional_ops_have_invariance_requirement():
    """All four 1D functional ops require convolutional axes."""
    for op in ("LinearFunctional", "SeparableBilinear",
                "Volterra2", "FunctionalOp2D"):
        rule = OPERATOR_RULES[op]
        assert rule.requires_invariance is not None
        assert Invariance.TRANSLATION in rule.requires_invariance
        assert Invariance.CAUSAL_TRANSLATION in rule.requires_invariance
