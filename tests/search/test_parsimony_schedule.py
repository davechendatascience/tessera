"""Tests for the non-monotone parsimony schedule.

Per docs/planned/roadmap.md §2.3 (shipped 2026-05-24).
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.search import (
    GP, GPConfig, climb_then_anneal_parsimony,
)


# ---------------- Factory behaviour ----------------

def test_climb_then_anneal_constant_in_climb_phase():
    schedule = climb_then_anneal_parsimony(
        climb_until=0.3, climb_value=0.0001, final_value=0.005
    )
    # gen 0, 5, 9 out of 30 → all in the climb phase (< 30%)
    assert schedule(0, 30) == 0.0001
    assert schedule(5, 30) == 0.0001
    # gen 9 / 30 = 0.30 → already past the threshold (uses < strictly)
    # so schedule(9, 30) is in the anneal phase; that's fine.


def test_climb_then_anneal_final_at_end():
    schedule = climb_then_anneal_parsimony(
        climb_until=0.3, climb_value=0.0001, final_value=0.005
    )
    # At gen = n_gens, we want exactly final_value
    assert abs(schedule(30, 30) - 0.005) < 1e-12


def test_climb_then_anneal_linear_interpolation():
    schedule = climb_then_anneal_parsimony(
        climb_until=0.3, climb_value=0.0001, final_value=0.005
    )
    # At progress = 0.65 (midpoint of the anneal phase 0.3 → 1.0):
    # anneal_fraction = (0.65 - 0.3) / 0.7 = 0.5
    # Expected value: 0.0001 + (0.005 - 0.0001) * 0.5 = 0.00255
    v = schedule(65, 100)
    expected = 0.0001 + (0.005 - 0.0001) * (0.65 - 0.3) / 0.7
    assert abs(v - expected) < 1e-9


def test_climb_then_anneal_handles_zero_n_gens():
    """Edge case: n_gens=0 shouldn't divide by zero."""
    schedule = climb_then_anneal_parsimony()
    # Should return final_value without crashing
    v = schedule(0, 0)
    assert np.isfinite(v)


def test_climb_then_anneal_custom_values():
    schedule = climb_then_anneal_parsimony(
        climb_until=0.5, climb_value=0.0, final_value=0.01
    )
    assert schedule(0, 100) == 0.0
    assert schedule(49, 100) == 0.0
    # At gen=100 (progress=1), should be 0.01
    assert abs(schedule(100, 100) - 0.01) < 1e-12


# ---------------- GP integration ----------------

def test_gp_uses_schedule_when_set():
    """When parsimony_schedule is set, the GP's _current_parsimony
    updates each generation."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100).astype(np.float64)
    y = 2.0 * x + 0.5

    schedule = climb_then_anneal_parsimony(
        climb_until=0.3, climb_value=0.0001, final_value=0.005
    )
    cfg = GPConfig(
        pop_size=20, n_gens=10, seed=0, pointwise_only=True,
        verbose=False, parsimony=0.005, parsimony_schedule=schedule,
        optimize_constants_every=0,   # disable polish for speed
    )
    gp = GP(cfg)
    # Before run: _current_parsimony defaults to cfg.parsimony
    assert gp._current_parsimony == 0.005
    gp.run({"x": x}, y, ["x"])
    # After run: _current_parsimony should be at the end-of-schedule
    # value (gen=n_gens=10 → progress=1 → final_value=0.005)
    assert abs(gp._current_parsimony - 0.005) < 1e-9


def test_gp_static_parsimony_when_schedule_none():
    """Without a schedule, _current_parsimony stays at cfg.parsimony."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50).astype(np.float64)
    y = x + 1.0

    cfg = GPConfig(
        pop_size=15, n_gens=5, seed=0, pointwise_only=True,
        verbose=False, parsimony=0.01,
        # parsimony_schedule defaults to None
        optimize_constants_every=0,
    )
    gp = GP(cfg)
    gp.run({"x": x}, y, ["x"])
    # No schedule active → _current_parsimony unchanged from init
    assert gp._current_parsimony == 0.01


def test_gp_with_schedule_finds_signal():
    """End-to-end smoke: GP with climb-then-anneal schedule still
    finds a low-loss candidate on a simple problem."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200).astype(np.float64)
    y = x * x - 0.5 * x + 1.0

    schedule = climb_then_anneal_parsimony(
        climb_until=0.3, climb_value=0.0001, final_value=0.01
    )
    cfg = GPConfig(
        pop_size=40, n_gens=15, seed=42,
        pointwise_only=True, verbose=False,
        parsimony=0.01, parsimony_schedule=schedule,
        optimize_constants_every=3,
    )
    gp = GP(cfg)
    front = gp.run({"x": x}, y, ["x"])
    assert len(front) > 0
    best = min(front, key=lambda c: c.train_loss)
    # var(y) ~ 1; should beat that with the schedule active
    assert best.train_loss < 0.5
