"""Tests for actionable infeasibility diagnostics (issue P15).

Fixtures are toy linear programs with known feasibility outcomes, so each
diagnostic's numbers can be asserted exactly.
"""

from __future__ import annotations

import pytest

from funding_optimizer.optimizer.constraints import Program, RegionFloor, solve_allocation
from funding_optimizer.optimizer.diagnostics import (
    InfeasibilityDiagnostic,
    diagnose_infeasibility,
    explain_result,
    required_region_fractions,
    total_required_fraction,
)
from funding_optimizer.optimizer.solver import SolverResult


def _feasible_programs() -> list[Program]:
    return [
        Program(id="edu", region="Africa", expected_reach=10.0, min_share=0.10),
        Program(id="cash", region="Asia", expected_reach=8.0),
    ]


def test_feasible_problem_yields_no_diagnostics() -> None:
    programs = _feasible_programs()
    floors = [RegionFloor(region="Africa", floor=0.30)]

    result = solve_allocation(programs, 100.0, floors)

    assert result.success is True
    assert explain_result(result, programs, 100.0, floors) == []


def test_required_region_fractions_use_the_binding_requirement() -> None:
    programs = [
        Program(id="a1", region="Africa", expected_reach=1.0, min_share=0.20),
        Program(id="a2", region="Africa", expected_reach=1.0, min_share=0.15),
        Program(id="b", region="Asia", expected_reach=1.0),
    ]
    floors = [
        RegionFloor(region="Africa", floor=0.30),
        RegionFloor(region="Asia", floor=0.10),
    ]

    required = required_region_fractions(programs, floors)

    # Africa: max(0.30 floor, 0.20 + 0.15 program minimums) = 0.35.
    assert required["Africa"] == pytest.approx(0.35)
    # Asia: max(0.10 floor, no program minimums) = 0.10.
    assert required["Asia"] == pytest.approx(0.10)
    assert total_required_fraction(programs, floors) == pytest.approx(0.45)


def test_region_floors_exceeding_budget_are_diagnosed() -> None:
    programs = [
        Program(id="a", region="Africa", expected_reach=1.0),
        Program(id="b", region="Asia", expected_reach=1.0),
    ]
    floors = [
        RegionFloor(region="Africa", floor=0.60),
        RegionFloor(region="Asia", floor=0.70),
    ]

    diagnostics = diagnose_infeasibility(programs, 100.0, floors)

    codes = [diagnostic.code for diagnostic in diagnostics]
    assert "floors_exceed_budget" in codes
    summary = next(d for d in diagnostics if d.code == "floors_exceed_budget")
    assert summary.excess == pytest.approx(0.30)
    assert "1.3" in summary.message

    africa = next(
        d
        for d in diagnostics
        if d.code == "region_requirement_above_capacity" and d.region == "Africa"
    )
    # Africa needs 0.60 but only 1.0 - 0.70 = 0.30 remains for it.
    assert africa.excess == pytest.approx(0.30)
    assert "at most 0.3" in africa.suggestion


def test_program_minimums_exceeding_budget_are_diagnosed() -> None:
    """Two 60% minimums commit 120% — validate() passes, the solver fails."""
    programs = [
        Program(id="big-a", region="Africa", expected_reach=1.0, min_share=0.60),
        Program(id="big-b", region="Asia", expected_reach=1.0, min_share=0.60),
    ]

    diagnostics = diagnose_infeasibility(programs, 100.0, [])

    codes = [diagnostic.code for diagnostic in diagnostics]
    assert "floors_exceed_budget" in codes
    program_diagnostics = [
        d for d in diagnostics if d.code == "program_min_share_above_capacity"
    ]
    assert {d.program_id for d in program_diagnostics} == {"big-a", "big-b"}
    for diagnostic in program_diagnostics:
        # Each program needs 0.60 but only 1.0 - 0.60 = 0.40 remains for it.
        assert diagnostic.excess == pytest.approx(0.20)


def test_region_floor_without_programs_is_flagged() -> None:
    """A floor on an empty region makes the LP row unsatisfiable."""
    programs = [Program(id="a", region="Africa", expected_reach=1.0)]
    floors = [RegionFloor(region="Europe", floor=0.20)]

    diagnostics = diagnose_infeasibility(programs, 100.0, floors)

    assert any(
        diagnostic.code == "region_has_no_programs" and diagnostic.region == "Europe"
        for diagnostic in diagnostics
    )


def test_invalid_share_values_are_reported_not_raised() -> None:
    programs = [Program(id="bad", region="Africa", expected_reach=1.0, min_share=1.5)]
    floors = [RegionFloor(region="Africa", floor=-0.2)]

    diagnostics = diagnose_infeasibility(programs, 100.0, floors)

    assert {diagnostic.code for diagnostic in diagnostics} == {
        "invalid_program_min_share",
        "invalid_region_floor",
    }


def test_non_positive_budget_is_reported() -> None:
    diagnostics = diagnose_infeasibility([], 0.0, [])

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "invalid_budget"


def test_explain_result_reports_structural_causes_for_failed_solves() -> None:
    """The min-share conflict passes validate() and fails at the solver."""
    programs = [
        Program(id="big-a", region="Africa", expected_reach=1.0, min_share=0.60),
        Program(id="big-b", region="Asia", expected_reach=1.0, min_share=0.60),
    ]

    result = solve_allocation(programs, 100.0, [])

    assert result.success is False
    diagnostics = explain_result(result, programs, 100.0, [])
    assert diagnostics
    assert all(isinstance(diagnostic, InfeasibilityDiagnostic) for diagnostic in diagnostics)
    assert any(diagnostic.code == "floors_exceed_budget" for diagnostic in diagnostics)


def test_explain_result_is_empty_for_successful_solves() -> None:
    result = SolverResult(
        success=True,
        status=0,
        message="optimal",
        objective_value=-1.0,
        allocation=None,
    )
    programs = _feasible_programs()

    assert explain_result(result, programs, 100.0, []) == []


def test_explain_result_falls_back_to_the_solver_message() -> None:
    """No structural cause: the diagnostic carries the solver's own report."""
    result = SolverResult(
        success=False,
        status=2,
        message="Solver reports infeasible",
        objective_value=None,
        allocation=None,
    )
    programs = _feasible_programs()

    diagnostics = explain_result(result, programs, 100.0, [])

    assert [diagnostic.code for diagnostic in diagnostics] == ["solver_infeasible"]
    assert "Solver reports infeasible" in diagnostics[0].message


def test_explain_result_flags_unbounded_objectives() -> None:
    result = SolverResult(
        success=False,
        status=3,
        message="Solver reports unbounded",
        objective_value=None,
        allocation=None,
    )
    programs = _feasible_programs()

    diagnostics = explain_result(result, programs, 100.0, [])

    assert [diagnostic.code for diagnostic in diagnostics] == ["unbounded_objective"]
