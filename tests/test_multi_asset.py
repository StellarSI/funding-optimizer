"""Tests for multi-asset budgets and per-asset constraints (issue #17).

Builds the multi-asset allocation problem with
:func:`build_multi_asset_allocation_problem` and solves it with the
deterministic HiGHS adapter from P14. Objectives, constraints and bounds are
tested as separate, testable pieces. The toy fixture in
``tests/fixtures/toy_multi_asset_lp.json`` pins a known optimum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from funding_optimizer.models import Budget
from funding_optimizer.optimizer.constraints import (
    AssetConstraint,
    Program,
    RegionFloor,
    build_multi_asset_allocation_problem,
    multi_asset_bounds,
    multi_asset_budget_constraints,
    multi_asset_constraints,
    multi_asset_objective,
    solve_multi_asset_allocation,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "toy_multi_asset_lp.json"


def _programs() -> list[Program]:
    return [
        Program(id="edu", region="Africa", expected_reach=10.0, min_share=0.10),
        Program(id="health", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="cash", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


def _budget() -> Budget:
    return Budget(amounts={"USDC": 100.0, "XLM": 50.0})


def _floors() -> list[RegionFloor]:
    return [
        RegionFloor(region="Africa", floor=0.30),
        RegionFloor(region="Asia", floor=0.20),
    ]


def test_objective_tiles_negated_reach_per_tranche() -> None:
    c = multi_asset_objective(_programs(), _budget())
    assert np.allclose(c, [-10.0, -6.0, -8.0, -10.0, -6.0, -8.0])


def test_budget_constraints_allocate_each_tranche_fully() -> None:
    a_eq, b_eq = multi_asset_budget_constraints(_programs(), _budget())
    assert a_eq.shape == (2, 6)
    assert np.allclose(a_eq[0], [1, 1, 1, 0, 0, 0])
    assert np.allclose(a_eq[1], [0, 0, 0, 1, 1, 1])
    assert np.allclose(b_eq, [100.0, 50.0])


def test_constraints_shape_and_per_asset_rows() -> None:
    constraints = [AssetConstraint(asset="XLM", program_id="edu", max_share=0.5)]
    a_ub, b_ub = multi_asset_constraints(_programs(), _budget(), _floors(), constraints)
    # Per tranche: two region floors + edu min_share (XLM constraint only caps, it adds no row).
    assert a_ub.shape == (6, 6)
    # edu cap appears as an upper bound, not an inequality row.
    assert np.allclose(b_ub[:3], [-30.0, -20.0, -10.0])
    assert np.allclose(b_ub[3:], [-15.0, -10.0, -5.0])


def test_asset_constraint_min_share_adds_row() -> None:
    programs = [
        Program(id="edu", region="Africa", expected_reach=10.0, min_share=0.0),
        Program(id="cash", region="Asia", expected_reach=8.0, min_share=0.0),
    ]
    budget = Budget(amounts={"USDC": 100.0, "XLM": 50.0})
    constraints = [AssetConstraint(asset="XLM", program_id="cash", min_share=0.4)]
    a_ub, b_ub = multi_asset_constraints(programs, budget, [], constraints)
    assert a_ub.shape == (1, 4)
    # Only the XLM tranche's cash variable is constrained (asset-major order).
    assert np.allclose(a_ub[0], [0, 0, 0, -1.0])
    assert np.isclose(b_ub[0], -0.4 * 50.0)


def test_bounds_cap_tranche_and_apply_max_share() -> None:
    constraints = [AssetConstraint(asset="XLM", program_id="edu", max_share=0.5)]
    bounds = multi_asset_bounds(_programs(), _budget(), constraints)
    assert len(bounds) == 6
    assert bounds[3] == (0.0, 25.0)  # XLM tranche, edu capped at 0.5 * 50
    assert bounds[0] == (0.0, 100.0)  # USDC tranche, uncapped (natural tranche cap)


def test_problem_dict_carries_assets_and_bounds() -> None:
    problem = build_multi_asset_allocation_problem(_programs(), _budget(), _floors())
    assert problem["assets"] == ("USDC", "XLM")
    assert problem["c"] is not None
    bounds = problem["bounds"]
    assert isinstance(bounds, list) and len(bounds) == 6


def test_toy_fixture_known_optimum() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    programs = [Program(**program) for program in fixture["programs"]]
    budget = Budget(amounts=fixture["budget"])
    floors = [RegionFloor(**floor) for floor in fixture["region_floors"]]
    constraints = [AssetConstraint(**constraint) for constraint in fixture["asset_constraints"]]

    result = solve_multi_asset_allocation(programs, budget, floors, asset_constraints=constraints)

    assert result.success is True
    assert result.allocation is not None
    expected = fixture["expected"]["allocation"]
    for asset, allocation in expected.items():
        for program_id, amount in allocation.items():
            assert np.isclose(result.allocation[asset][program_id], amount, atol=1e-6)
    for asset, tranche in budget.amounts.items():
        assert result.allocation is not None
        assert np.isclose(sum(result.allocation[asset].values()), tranche, atol=1e-6)
    assert result.objective_value is not None
    assert np.isclose(result.objective_value, fixture["expected"]["objective_value"], atol=1e-6)


def test_per_asset_constraints_apply_within_each_tranche_independently() -> None:
    constraints = [AssetConstraint(asset="USDC", program_id="cash", max_share=0.0)]
    result = solve_multi_asset_allocation(_programs(), _budget(), [], asset_constraints=constraints)
    assert result.success is True
    assert result.allocation is not None
    # USDC tranche: cash capped at zero, so it cannot receive USDC.
    assert np.isclose(result.allocation["USDC"]["cash"], 0.0, atol=1e-6)
    # XLM tranche: no constraint on cash, highest-reach program preferred.
    assert result.allocation["XLM"]["edu"] > 0.0


def test_infeasible_when_tranche_minimums_exceed_100_percent() -> None:
    constraints = [
        AssetConstraint(asset="XLM", program_id="edu", min_share=0.6),
        AssetConstraint(asset="XLM", program_id="health", min_share=0.6),
    ]
    result = solve_multi_asset_allocation(_programs(), _budget(), [], asset_constraints=constraints)
    assert result.success is False
    assert result.allocation is None


def test_no_floors_allocates_full_budget_per_tranche() -> None:
    result = solve_multi_asset_allocation(_programs(), _budget())
    assert result.success is True
    assert result.allocation is not None
    for asset, tranche in _budget().amounts.items():
        assert np.isclose(sum(result.allocation[asset].values()), tranche, atol=1e-6)
        assert np.isclose(result.allocation[asset]["edu"], tranche, atol=1e-6)


def test_empty_programs_raise() -> None:
    with pytest.raises(ValueError, match="at least one program"):
        build_multi_asset_allocation_problem([], _budget())


def test_non_budget_raises() -> None:
    with pytest.raises(TypeError, match="must be a Budget"):
        build_multi_asset_allocation_problem(_programs(), 100.0)  # type: ignore[arg-type]


def test_asset_constraint_validation() -> None:
    with pytest.raises(ValueError, match="not part of the budget"):
        solve_multi_asset_allocation(
            _programs(), _budget(), asset_constraints=[AssetConstraint("KSH", "edu")]
        )
    with pytest.raises(ValueError, match="does not match any program"):
        solve_multi_asset_allocation(
            _programs(), _budget(), asset_constraints=[AssetConstraint("USDC", "nope")]
        )
    with pytest.raises(ValueError, match="Duplicate AssetConstraint"):
        solve_multi_asset_allocation(
            _programs(),
            _budget(),
            asset_constraints=[
                AssetConstraint("USDC", "edu", max_share=0.5),
                AssetConstraint("USDC", "edu", max_share=0.9),
            ],
        )


def test_asset_constraint_share_bounds_validated() -> None:
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        AssetConstraint("USDC", "edu", min_share=1.5)
    with pytest.raises(ValueError, match="exceeds max_share"):
        AssetConstraint("USDC", "edu", min_share=0.8, max_share=0.2)
    with pytest.raises(ValueError, match="non-empty"):
        AssetConstraint("USDC", "   ")


def test_asset_codes_in_constraints_are_normalized() -> None:
    constraint = AssetConstraint(" usdc ", "edu")
    assert constraint.asset == "USDC"
