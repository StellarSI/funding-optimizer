"""Tests for the maximize_reach objective solver (issue #10).

Uses the toy linear-program fixture (``tests/fixtures/toy_lp.json``)
with a provably known optimum; no network access and no real recipient
data are involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from funding_optimizer.errors import AllocationError, ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import (
    MaximizeReachSolver,
    Optimizer,
    solve_maximize_reach,
)
from funding_optimizer.plan import AllocationPlan

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "toy_lp.json"


def load_fixture() -> dict[str, object]:
    """Load the toy LP fixture typed as a plain JSON object mapping."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    return cast(dict[str, object], json.loads(text))


def fixture_programs() -> list[Program]:
    """Build the fixture's two-program toy problem."""
    fixture = load_fixture()
    programs = cast(list[dict[str, object]], fixture["programs"])
    return [
        Program(
            id=cast(str, record["id"]),
            min_share=cast(float, record["min_share"]),
            expected_reach=cast(int, record["expected_reach"]),
            region=cast(str | None, record.get("region")),
        )
        for record in programs
    ]


def test_toy_lp_fixture_yields_the_known_optimum() -> None:
    budget = Budget(total=250_000.0, asset="USDC")
    programs = fixture_programs()

    result = MaximizeReachSolver().solve(budget, programs)

    assert result.plan.total == 250_000.0
    assert result.plan.asset == "USDC"
    assert result.expected_total_reach == 11_250
    shares = {allocation.program_id: allocation.share for allocation in result.plan.allocations}
    amounts = {allocation.program_id: allocation.amount for allocation in result.plan.allocations}
    assert shares == {"cash-transfers-ng": 0.9, "education-ke": 0.1}
    assert amounts == {"cash-transfers-ng": 225_000.0, "education-ke": 25_000.0}
    # cash-transfers-ng absorbs the whole remainder; education-ke binds at its floor.
    assert result.binding_program_ids == ("education-ke",)
    assert "binding_min_share_programs=[education-ke]" in result.diagnostic()


def test_solve_maximize_reach_returns_the_plan_only() -> None:
    budget = Budget(total=100.0, asset="XLM")
    programs = [
        Program(id="alpha", min_share=0.0, expected_reach=50),
        Program(id="beta", min_share=0.0, expected_reach=70),
    ]
    plan = solve_maximize_reach(budget, programs)
    assert isinstance(plan, AllocationPlan)
    assert plan.objective == "maximize_reach"
    assert {a.program_id: a.share for a in plan.allocations} == {"alpha": 0.0, "beta": 1.0}


def test_equal_reach_ties_break_by_ascending_program_id() -> None:
    budget = Budget(total=1_000.0, asset="USDC")
    programs = [
        Program(id="zulu", expected_reach=100),
        Program(id="alpha", expected_reach=100),
    ]
    result = MaximizeReachSolver().solve(budget, programs)
    amounts = {a.program_id: a.amount for a in result.plan.allocations}
    # Determinism: the tie goes to 'alpha', reproducibly.
    assert amounts["alpha"] == 1_000.0
    assert result.expected_total_reach == 100


def test_infeasible_min_shares_raise_allocation_error_with_diagnostic() -> None:
    budget = Budget(total=1_000.0, asset="USDC")
    programs = [
        Program(id="a", min_share=0.7, expected_reach=10),
        Program(id="b", min_share=0.6, expected_reach=10),
    ]
    with pytest.raises(AllocationError, match="infeasible"):
        MaximizeReachSolver().solve(budget, programs)


def test_empty_program_list_and_duplicate_ids_are_rejected() -> None:
    budget = Budget(total=1_000.0, asset="USDC")
    with pytest.raises(ValidationError, match="at least one program"):
        MaximizeReachSolver().solve(budget, [])

    duplicated = [Program(id="same", expected_reach=1), Program(id="same", expected_reach=1)]
    with pytest.raises(ValidationError, match="unique"):
        MaximizeReachSolver().solve(budget, duplicated)


def test_unsupported_fairness_mode_is_rejected_at_this_boundary() -> None:
    budget = Budget(total=1_000.0, asset="USDC")
    with pytest.raises(ValidationError, match="fairness"):
        MaximizeReachSolver().solve(
            budget,
            [Program(id="a", expected_reach=1)],
            fairness="regional_floor",
        )


def test_zero_reach_programs_still_distribute_the_full_budget() -> None:
    budget = Budget(total=900.0, asset="USDC")
    programs = [
        Program(id="a", min_share=0.25, expected_reach=0),
        Program(id="b", min_share=0.25, expected_reach=0),
    ]
    plan = solve_maximize_reach(budget, programs)
    assert sum(a.amount for a in plan.allocations) == pytest.approx(900.0)
    assert plan.unallocated == 0.0
    # All-zero reach: the deterministic tie-break hands the remainder to 'a'.
    assert {a.program_id: a.share for a in plan.allocations} == {"a": 0.75, "b": 0.25}


def test_readme_quick_example_problem_is_solvable_by_the_scoped_solver() -> None:
    """The README quick example must keep solving to a valid plan."""
    budget = Budget(total=250_000.0, asset="USDC")
    programs = [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
        Program(id="health-gh", min_share=0.15, expected_reach=7_800),
    ]
    result = MaximizeReachSolver().solve(budget, programs)
    amounts = {a.program_id: a.amount for a in result.plan.allocations}
    # The highest-reach program (cash-transfers-ng) absorbs the remainder.
    assert amounts["cash-transfers-ng"] == pytest.approx(187_500.0)
    # The engine (main Optimizer) still solves the same problem.
    engine_plan: AllocationPlan = Optimizer(objective="maximize_reach").solve(budget, programs)
    assert engine_plan.allocated_total == pytest.approx(250_000.0)
