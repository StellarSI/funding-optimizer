"""Tests for the deterministic LP solver adapter (issue P14).

SciPy/HiGHS is exercised only through the ``HiGHSolver`` adapter; no test
imports ``scipy`` directly.
"""

from __future__ import annotations

import numpy as np

from funding_optimizer.optimizer.solver import (
    HiGHSolver,
    SolverResult,
    solve_with_lexicographic_tie_break,
)


def _solver() -> HiGHSolver:
    return HiGHSolver()


def test_maximize_simple_objective_happy_path() -> None:
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


def test_infeasible_problem_reports_failure() -> None:
    """x + y <= -1 with x, y >= 0 is infeasible."""
    solver = _solver()
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([-1.0])

    result = solver.solve(c, A_ub=a_ub, b_ub=b_ub)

    assert result.success is False
    assert result.status != 0


def test_deterministic_repeated_runs_agree() -> None:
    """Identical inputs must produce identical allocations (reproducibility)."""
    solver = _solver()
    c = np.array([-4.0, -1.0, -2.0])
    a_ub = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    b_ub = np.array([10.0, 12.0])

    first = solver.solve(c, A_ub=a_ub, b_ub=b_ub)
    second = solver.solve(c, A_ub=a_ub, b_ub=b_ub)

    assert first.success is True and second.success is True
    assert first.allocation is not None and second.allocation is not None
    assert first.objective_value is not None
    assert second.objective_value is not None
    assert np.allclose(first.allocation, second.allocation, atol=1e-9)
    assert np.isclose(first.objective_value, second.objective_value, atol=1e-9)


def test_variable_bounds_respected() -> None:
    """Upper bound x <= 3 is honored even though the objective wants more x."""
    solver = _solver()
    c = np.array([-10.0, -1.0])
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([100.0])
    bounds: list[tuple[float | None, float | None]] = [(0.0, 3.0), (0.0, None)]

    result = solver.solve(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds)

    assert result.success is True
    assert result.allocation is not None
    assert result.objective_value is not None
    assert np.isclose(result.allocation[0], 3.0, atol=1e-6)
    assert np.isclose(result.allocation[1], 97.0, atol=1e-6)
    assert np.isclose(result.objective_value, -(10.0 * 3.0 + 1.0 * 97.0), atol=1e-6)


def test_tie_break_prefers_earlier_programs_on_equal_value() -> None:
    """Two identical programs: every split of 100 has the same value.

    The plain solver may return any split; lexicographic tie-breaking must
    return the same canonical split on every run, preferring earlier-listed
    programs.
    """
    solver = _solver()
    c = np.array([-1.0, -1.0])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([100.0])

    first = solve_with_lexicographic_tie_break(solver, c, A_eq=a_eq, b_eq=b_eq)
    second = solve_with_lexicographic_tie_break(solver, c, A_eq=a_eq, b_eq=b_eq)

    assert first.success is True
    assert first.tie_break_applied is True
    assert first.allocation is not None
    assert np.allclose(first.allocation, [100.0, 0.0], atol=1e-6)
    assert second.allocation is not None
    assert np.allclose(first.allocation, second.allocation, atol=1e-9)
    assert first.objective_value is not None
    assert np.isclose(first.objective_value, -100.0, atol=1e-6)


def test_tie_break_respects_floors() -> None:
    """With x_2 floored at 25, the tie-break leaves the floor in place."""
    solver = _solver()
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[0.0, -1.0]])
    b_ub = np.array([-25.0])
    a_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([100.0])

    result = solve_with_lexicographic_tie_break(
        solver, c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq
    )

    assert result.success is True
    assert result.tie_break_applied is True
    assert result.allocation is not None
    assert np.allclose(result.allocation, [75.0, 25.0], atol=1e-6)


def test_tie_break_three_way_split() -> None:
    """Three identical programs with a floor on the last: (60, 0, 30)."""
    solver = _solver()
    c = np.array([-1.0, -1.0, -1.0])
    a_ub = np.array([[0.0, 0.0, -1.0]])
    b_ub = np.array([-30.0])
    a_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([90.0])

    result = solve_with_lexicographic_tie_break(
        solver, c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq
    )

    assert result.success is True
    assert result.allocation is not None
    assert np.allclose(result.allocation, [60.0, 0.0, 30.0], atol=1e-6)


def test_tie_break_preserves_unique_optimum() -> None:
    """A unique optimum passes through the refinement unchanged."""
    solver = _solver()
    c = np.array([-3.0, -2.0])
    a_ub = np.array([[1.0, 1.0], [1.0, 0.0]])
    b_ub = np.array([10.0, 6.0])

    result = solve_with_lexicographic_tie_break(solver, c, A_ub=a_ub, b_ub=b_ub)

    assert result.success is True
    assert result.tie_break_applied is True
    assert result.allocation is not None
    assert np.allclose(result.allocation, [6.0, 4.0], atol=1e-6)
    assert result.objective_value is not None
    assert np.isclose(result.objective_value, -26.0, atol=1e-6)


def test_tie_break_returns_infeasible_result_unchanged() -> None:
    """Failed primary solves are not refined and report no tie-break."""
    solver = _solver()
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([-1.0])

    result = solve_with_lexicographic_tie_break(solver, c, A_ub=a_ub, b_ub=b_ub)

    assert result.success is False
    assert result.tie_break_applied is False
