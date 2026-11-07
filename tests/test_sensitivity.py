"""Tests for sensitivity analysis of caps, minimums and fairness (issue #21).

One-at-a-time sweeps of regional floors, program ``min_share`` minimums and
per-program budget caps are re-solved with the deterministic HiGHS adapter.
The toy fixture in ``tests/fixtures/toy_sensitivity_lp.json`` pins a known
optimum, including a binding cap whose perturbation moves the objective in
the expected direction. All expected values were computed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from funding_optimizer.optimizer.constraints import Program, RegionFloor
from funding_optimizer.optimizer.sensitivity import (
    binding_constraints,
    constraint_slacks,
    sensitivity_analysis,
    validate_caps,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "toy_sensitivity_lp.json"


def _fixture_data() -> dict:
    data: dict = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return data


def _fixture_programs() -> list[Program]:
    return [
        Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.1),
        Program(id="p2", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="p3", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


def _fixture_floors() -> list[RegionFloor]:
    return [RegionFloor(region="Africa", floor=0.3), RegionFloor(region="Asia", floor=0.2)]


def test_fixture_pins_sweep_entries() -> None:
    data = _fixture_data()
    report = sensitivity_analysis(
        _fixture_programs(),
        data["budget"],
        _fixture_floors(),
        caps=data["caps"],
        perturbation=data["perturbation"],
    )

    expected = data["expected"]
    assert report.base_objective_value == pytest.approx(expected["base_objective_value"], abs=1e-6)
    assert report.base_allocation is not None
    for program_id, amount in expected["base_allocation"].items():
        assert report.base_allocation[program_id] == pytest.approx(amount, abs=1e-6)
    assert list(report.binding_parameters) == expected["binding_parameters"]

    assert len(report.entries) == len(expected["entries"])
    for entry, want in zip(report.entries, expected["entries"], strict=True):
        assert entry.parameter == want["parameter"]
        assert entry.kind == want["kind"]
        assert entry.perturbation == pytest.approx(want["perturbation"], abs=1e-9)
        assert entry.perturbed_value == pytest.approx(want["perturbed_value"], abs=1e-9)
        assert entry.objective_value == pytest.approx(want["objective_value"], abs=1e-6)
        assert entry.objective_delta == pytest.approx(want["objective_delta"], abs=1e-6)
        assert entry.binding is want["binding"]
        assert entry.success is True


def test_sweep_order_is_deterministic() -> None:
    report = sensitivity_analysis(
        _fixture_programs(), 100.0, _fixture_floors(), caps={"p1": 0.5}, perturbation=0.1
    )
    assert [entry.parameter for entry in report.entries] == [
        "region_floor:Africa",
        "region_floor:Africa",
        "region_floor:Asia",
        "region_floor:Asia",
        "program_min_share:p1",
        "program_min_share:p1",
        "program_min_share:p2",
        "program_min_share:p2",
        "program_min_share:p3",
        "program_min_share:p3",
        "program_cap:p1",
        "program_cap:p1",
    ]


def test_uncapped_base_binds_the_asia_floor() -> None:
    report = sensitivity_analysis(_fixture_programs(), 100.0, _fixture_floors(), perturbation=0.1)

    assert report.base_objective_value == pytest.approx(-960.0, abs=1e-6)
    assert report.base_allocation == pytest.approx({"p1": 80.0, "p2": 0.0, "p3": 20.0}, abs=1e-6)
    assert report.binding_parameters == ("region_floor:Asia",)
    by_parameter = {(entry.parameter, entry.perturbation): entry for entry in report.entries}
    assert by_parameter[("region_floor:Asia", 0.1)].objective_delta == pytest.approx(20.0, abs=1e-6)
    asia_neg = by_parameter[("region_floor:Asia", -0.1)]
    p2_pos = by_parameter[("program_min_share:p2", 0.1)]
    assert asia_neg.objective_delta == pytest.approx(-20.0, abs=1e-6)
    assert p2_pos.objective_delta == pytest.approx(40.0, abs=1e-6)


def test_perturbation_clamped_into_unit_interval() -> None:
    programs = [Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.05)]
    report = sensitivity_analysis(programs, 100.0, [], perturbation=0.1)
    negative = next(entry for entry in report.entries if entry.perturbation < 0)
    assert negative.perturbed_value == 0.0


def test_floor_perturbation_that_breaks_budget_is_reported_not_raised() -> None:
    programs = _fixture_programs()
    floors = [RegionFloor(region="Africa", floor=0.8), RegionFloor(region="Asia", floor=0.2)]
    report = sensitivity_analysis(programs, 100.0, floors, perturbation=0.1)

    failing = next(
        entry
        for entry in report.entries
        if entry.parameter == "region_floor:Africa" and entry.perturbation > 0
    )
    assert failing.success is False
    assert failing.objective_value is None
    assert failing.objective_delta is None
    assert "exceeds" in failing.message
    # The base problem itself is feasible and solves fine.
    assert report.base_objective_value == pytest.approx(-960.0, abs=1e-6)


def test_base_problem_infeasible_returns_failed_report() -> None:
    programs = [Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.6)]
    report = sensitivity_analysis(programs, 100.0, [], caps={"p1": 0.3}, perturbation=0.1)

    assert report.base_objective_value is None
    assert report.base_allocation is None
    assert report.message.startswith("base problem is infeasible")
    assert report.binding_parameters == ()
    assert all(entry.success is False for entry in report.entries)


def test_caps_referencing_unknown_program_raise() -> None:
    with pytest.raises(ValueError, match="unknown program"):
        validate_caps(_fixture_programs(), {"nope": 0.5})


def test_caps_must_be_fractions_in_unit_interval() -> None:
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_caps(_fixture_programs(), {"p1": bad})


def test_caps_reject_bool_values() -> None:
    with pytest.raises(TypeError):
        validate_caps(_fixture_programs(), {"p1": True})  # type: ignore[dict-item]


def test_invalid_perturbation_raises() -> None:
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="perturbation"):
            sensitivity_analysis(_fixture_programs(), 100.0, _fixture_floors(), perturbation=bad)


def test_invalid_budget_raises() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="budget"):
            sensitivity_analysis(_fixture_programs(), bad, _fixture_floors())


def test_at_least_one_program_required() -> None:
    with pytest.raises(ValueError, match="at least one program"):
        sensitivity_analysis([], 100.0, _fixture_floors())


def test_floors_above_budget_raise() -> None:
    programs = _fixture_programs()
    floors = [RegionFloor(region="Africa", floor=0.9), RegionFloor(region="Asia", floor=0.4)]
    with pytest.raises(ValueError, match="exceeds the whole budget"):
        sensitivity_analysis(programs, 100.0, floors)


def test_constraint_slacks_and_binding_constraints() -> None:
    allocation = {"p1": 80.0, "p2": 0.0, "p3": 20.0}
    slacks = constraint_slacks(_fixture_programs(), 100.0, allocation, _fixture_floors())

    assert slacks["region_floor:Africa"] == pytest.approx(50.0)
    assert slacks["region_floor:Asia"] == pytest.approx(0.0)
    assert slacks["program_min_share:p1"] == pytest.approx(70.0)

    bound = binding_constraints(_fixture_programs(), 100.0, allocation, _fixture_floors())
    assert bound == ("region_floor:Asia",)


def test_constraint_slacks_require_full_allocation() -> None:
    with pytest.raises(ValueError, match="missing program"):
        constraint_slacks(_fixture_programs(), 100.0, {"p1": 80.0}, _fixture_floors())


def test_binding_constraints_with_caps() -> None:
    allocation = {"p1": 50.0, "p2": 0.0, "p3": 50.0}
    bound = binding_constraints(
        _fixture_programs(), 100.0, allocation, _fixture_floors(), {"p1": 0.5}
    )
    assert bound == ("program_cap:p1",)
