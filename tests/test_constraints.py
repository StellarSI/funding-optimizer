"""Tests for regional-floor fairness constraints (issue P13).

Builds the allocation problem with :func:`build_allocation_problem` and solves
it with the deterministic HiGHS adapter from P14. Objectives and constraints
are tested as separate, testable pieces.
"""

from __future__ import annotations

import numpy as np
import pytest

from funding_optimizer.optimizer.constraints import (
    Program,
    RegionFloor,
    build_allocation_problem,
    maximize_reach_objective,
    regional_floor_constraints,
    solve_allocation,
)


def _programs() -> list[Program]:
    return [
        Program(id="edu", region="Africa", expected_reach=10.0, min_share=0.10),
        Program(id="health", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="cash", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


def _floors() -> list[RegionFloor]:
    return [
        RegionFloor(region="Africa", floor=0.30),
        RegionFloor(region="Asia", floor=0.20),
    ]


def test_maximize_reach_objective_is_negated_reach() -> None:
    programs = _programs()
    c = maximize_reach_objective(programs)
    assert np.allclose(c, [-10.0, -6.0, -8.0])


def test_regional_floor_constraints_shape() -> None:
    programs = _programs()
    a_ub, b_ub = regional_floor_constraints(programs, 100.0, _floors())
    # Two region floors + one program min_share (edu >= 0.10 * 100).
    assert a_ub.shape == (3, 3)
    assert b_ub.shape == (3,)


def test_regional_floor_allocation_satisfies_floors() -> None:
    result = solve_allocation(_programs(), 100.0, _floors())

    assert result.success is True
    assert result.allocation is not None
    edu, health, cash = result.allocation

    # Full budget allocated.
    assert np.isclose(edu + health + cash, 100.0, atol=1e-6)
    # Regional floors: Africa >= 30, Asia >= 20.
    assert edu + health >= 30.0 - 1e-6
    assert cash >= 20.0 - 1e-6
    # Program min share: edu >= 10.
    assert edu >= 10.0 - 1e-6

    # Known optimum: edu (reach 10) > cash (reach 8) > health (reach 6),
    # so Asia takes its minimum 20 and the rest goes to edu.
    assert np.isclose(edu, 80.0, atol=1e-6)
    assert np.isclose(health, 0.0, atol=1e-6)
    assert np.isclose(cash, 20.0, atol=1e-6)
    assert np.isclose(result.objective_value, -960.0, atol=1e-6)


def test_no_region_floors_still_allocates_full_budget() -> None:
    result = solve_allocation(_programs(), 100.0, [])
    assert result.success is True
    assert result.allocation is not None
    assert np.isclose(result.allocation.sum(), 100.0, atol=1e-6)
    # All budget goes to the highest-reach program (edu).
    assert np.isclose(result.allocation[0], 100.0, atol=1e-6)


def test_regional_floors_exceeding_budget_raise() -> None:
    programs = _programs()
    bad_floors = [
        RegionFloor(region="Africa", floor=0.60),
        RegionFloor(region="Asia", floor=0.60),
    ]
    with pytest.raises(ValueError, match="exceeds the whole budget"):
        build_allocation_problem(programs, 100.0, bad_floors)

    bad_program = Program(id="bad", region="Africa", expected_reach=1.0, min_share=1.5)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        build_allocation_problem([bad_program], 100.0, [])

    bad_floor = RegionFloor(region="Africa", floor=1.5)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        build_allocation_problem(programs, 100.0, [bad_floor])
