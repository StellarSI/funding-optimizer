"""Tests for phased allocation over budget schedules (issue #20).

The phased allocator distributes a budget that arrives over time. Each phase
receives the scheduled amount plus any budget carried forward from earlier
phases whose programs could not absorb their full tranche. All expected
values in the hand-computed fixtures are exact; the scenario pinned in
``tests/fixtures/toy_budget_schedule.json`` is verified against them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from funding_optimizer.optimizer.constraints import Program, RegionFloor
from funding_optimizer.optimizer.schedule import (
    BudgetPhase,
    BudgetSchedule,
    solve_phased_allocation,
    validate_phase_caps,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "toy_budget_schedule.json"


def _fixture_programs() -> list[Program]:
    return [
        Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.1),
        Program(id="p2", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="p3", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


def _fixture_schedule() -> BudgetSchedule:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    phases = [BudgetPhase(id=phase["id"], amount=phase["amount"]) for phase in data["phases"]]
    return BudgetSchedule(phases)


def test_fixture_pins_phased_optimum() -> None:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    floors = [RegionFloor(region=f["region"], floor=f["floor"]) for f in data["region_floors"]]
    result = solve_phased_allocation(
        _fixture_programs(),
        _fixture_schedule(),
        floors,
        phase_caps=data["phase_caps"],
    )

    assert result.success is True
    assert result.total_budget == pytest.approx(200.0)
    assert result.total_allocated == pytest.approx(189.0, abs=1e-6)
    assert result.unallocated == pytest.approx(11.0, abs=1e-6)
    assert result.objective_value == pytest.approx(-1659.0, abs=1e-6)

    expected_phases = data["expected"]["phases"]
    assert [phase.phase_id for phase in result.phases] == ["q1", "q2"]
    for phase_result, expected in zip(result.phases, expected_phases, strict=True):
        assert phase_result.available == pytest.approx(expected["available"], abs=1e-6)
        assert phase_result.carry_in == pytest.approx(expected["carry_in"], abs=1e-6)
        assert phase_result.carry_out == pytest.approx(expected["carry_out"], abs=1e-6)
        assert phase_result.objective_value == pytest.approx(expected["objective_value"], abs=1e-6)
        for program_id, amount in expected["allocation"].items():
            assert phase_result.allocation is not None
            assert phase_result.allocation[program_id] == pytest.approx(amount, abs=1e-6)

    expected_cumulative = data["expected"]["cumulative_allocation"]
    for program_id, amount in expected_cumulative.items():
        assert result.cumulative_allocation[program_id] == pytest.approx(amount, abs=1e-6)


def test_phase_carry_rolls_into_next_phase() -> None:
    programs = _fixture_programs()
    schedule = _fixture_schedule()
    floors = [RegionFloor(region="Africa", floor=0.3)]
    result = solve_phased_allocation(
        programs, schedule, floors, phase_caps={"p1": 0.5, "p2": 0.15, "p3": 0.25}
    )
    assert result.phases[0].carry_out == pytest.approx(10.0, abs=1e-6)
    assert result.phases[1].carry_in == pytest.approx(10.0, abs=1e-6)
    assert result.phases[1].available == pytest.approx(110.0, abs=1e-6)


def test_without_caps_each_phase_is_fully_allocated() -> None:
    programs = [
        Program(id="p1", region="Africa", expected_reach=10.0),
        Program(id="p2", region="Asia", expected_reach=5.0),
    ]
    schedule = BudgetSchedule(
        [BudgetPhase(id="q1", amount=100.0), BudgetPhase(id="q2", amount=60.0)]
    )
    result = solve_phased_allocation(programs, schedule, [])

    assert result.success is True
    assert [phase.carry_out for phase in result.phases] == [0.0, 0.0]
    # p1 has the highest single-unit reach, so it absorbs every phase.
    assert result.cumulative_allocation["p1"] == pytest.approx(160.0, abs=1e-6)
    assert result.unallocated == pytest.approx(0.0, abs=1e-6)


def test_schedule_requires_at_least_one_phase() -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        BudgetSchedule([])


def test_schedule_rejects_duplicate_phase_ids() -> None:
    phases = [BudgetPhase(id="q1", amount=10.0), BudgetPhase(id="q1", amount=20.0)]
    with pytest.raises(ValueError, match="Duplicate phase id"):
        BudgetSchedule(phases)


def test_phase_amount_must_be_positive_finite() -> None:
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive, finite"):
            BudgetPhase(id="q1", amount=bad)


def test_phase_rejects_bool_amount() -> None:
    with pytest.raises(TypeError):
        BudgetPhase(id="q1", amount=True)  # type: ignore[arg-type]


def test_phase_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BudgetPhase(id="   ", amount=10.0)


def test_phase_caps_reject_unknown_program() -> None:
    programs = _fixture_programs()
    with pytest.raises(ValueError, match="unknown program"):
        validate_phase_caps(programs, {"nope": 0.5})


def test_phase_caps_must_be_fractions_in_unit_interval() -> None:
    programs = _fixture_programs()
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_phase_caps(programs, {"p1": bad})


def test_phase_caps_reject_bool_values() -> None:
    programs = _fixture_programs()
    with pytest.raises(TypeError):
        validate_phase_caps(programs, {"p1": True})  # type: ignore[dict-item]


def test_cap_below_min_share_fails_phase_and_stops() -> None:
    programs = [Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.9)]
    schedule = BudgetSchedule(
        [BudgetPhase(id="q1", amount=100.0), BudgetPhase(id="q2", amount=50.0)]
    )
    result = solve_phased_allocation(programs, schedule, [], phase_caps={"p1": 0.5})

    assert result.success is False
    assert len(result.phases) == 1  # the second phase was never attempted
    failing = result.phases[0]
    assert failing.success is False
    assert failing.allocation is None
    codes = [diagnostic.code for diagnostic in failing.diagnostics]
    assert "phase_cap_below_min_share" in codes
    assert result.unallocated is None


def test_structurally_infeasible_program_minimums_stop_the_run() -> None:
    programs = [
        Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.6),
        Program(id="p2", region="Asia", expected_reach=5.0, min_share=0.6),
    ]
    schedule = BudgetSchedule([BudgetPhase(id="q1", amount=100.0)])
    result = solve_phased_allocation(programs, schedule, [])

    assert result.success is False
    assert result.phases[0].diagnostics  # structural reasons are attached
    assert result.unallocated is None


def test_regional_floors_above_budget_raise() -> None:
    programs = _fixture_programs()
    schedule = _fixture_schedule()
    floors = [
        RegionFloor(region="Africa", floor=0.8),
        RegionFloor(region="Asia", floor=0.5),
    ]
    with pytest.raises(ValueError, match="exceeds the whole budget"):
        solve_phased_allocation(programs, schedule, floors)


def test_schedule_must_be_a_budget_schedule() -> None:
    with pytest.raises(TypeError, match="BudgetSchedule"):
        solve_phased_allocation(
            _fixture_programs(),
            cast("BudgetSchedule", [BudgetPhase(id="q1", amount=10.0)]),
        )


def test_at_least_one_program_required() -> None:
    with pytest.raises(ValueError, match="at least one program"):
        solve_phased_allocation([], _fixture_schedule(), [])


def test_tie_break_is_threaded_through() -> None:
    programs = _fixture_programs()
    schedule = _fixture_schedule()
    floors = [RegionFloor(region="Africa", floor=0.3)]
    result = solve_phased_allocation(programs, schedule, floors, tie_break=True)

    assert result.success is True
    assert result.tie_break_applied is True
    allocations = [
        phase.allocation for phase in result.phases if phase.allocation is not None
    ]
    assert np.isclose(sum(sum(a.values()) for a in allocations), result.total_allocated)
