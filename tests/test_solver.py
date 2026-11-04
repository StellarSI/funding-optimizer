"""Tests for the deterministic LP solver adapter (issue P14).

SciPy/HiGHS is exercised only through the ``HiGHSolver`` adapter; no test
imports ``scipy`` directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from funding_optimizer.optimizer.solver import HiGHSolver, SolverResult


def _solver() -> HiGHSolver:
    return HiGHSolver()


def test_maximize_simple_objective_happy_path():
    """maximize 3x + 2y subject to x + y <= 10, x <= 6, x,y >= 0.

    Optimum: x = 6, y = 4, objective = 26 (linprog minimizes -26).
    """
    solver = _solver()
    c = np.array([-3.0, -2.0])
    a_ub = np.array([[1.0, 1.0], [1.0, 0.0]])
    b_ub = np.array([10.0, 6.0])

    result = solver.solve(c, A_ub=a_ub, b_ub=b_ub)

    assert isinstance(result, SolverResult)
    assert result.success is True
    assert result.allocation is not None
    assert np.allclose(result.allocation, [6.0, 4.0], atol=1e-6)
    assert result.objective_value is not None
    assert np.isclose(result.objective_value, -26.0, atol=1e-6)


def test_infeasible_problem_reports_failure():
    """x + y <= -1 with x, y >= 0 is infeasible."""
    solver = _solver()
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([-1.0])

    result = solver.solve(c, A_ub=a_ub, b_ub=b_ub)

    assert result.success is False
    assert result.status != 0


def test_deterministic_repeated_runs_agree():
    """Identical inputs must produce identical allocations (reproducibility)."""
    solver = _solver()
    c = np.array([-4.0, -1.0, -2.0])
    a_ub = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    b_ub = np.array([10.0, 12.0])

    first = solver.solve(c, A_ub=a_ub, b_ub=b_ub)
    second = solver.solve(c, A_ub=a_ub, b_ub=b_ub)

    assert first.success is True and second.success is True
    assert first.allocation is not None and second.allocation is not None
    assert np.allclose(first.allocation, second.allocation, atol=1e-9)
    assert np.isclose(first.objective_value, second.objective_value, atol=1e-9)


def test_variable_bounds_respected():
    """Upper bound x <= 3 is honored even though the objective wants more x."""
    solver = _solver()
    c = np.array([-10.0, -1.0])
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([100.0])
    bounds = [(0.0, 3.0), (0.0, None)]

    result = solver.solve(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds)

    assert result.success is True
    assert result.allocation is not None
    assert np.isclose(result.allocation[0], 3.0, atol=1e-6)
    assert np.isclose(result.allocation[1], 97.0, atol=1e-6)
    assert np.isclose(result.objective_value, -(10.0 * 3.0 + 1.0 * 97.0), atol=1e-6)
