"""Unit tests for the ``AllocationPlan`` model.

These tests exercise the allocation plan returned by the optimizer
(``README.md`` §Quick example): totals, metadata and the ``summary()``
output.
"""

from __future__ import annotations

from typing import Any

import pytest

from funding_optimizer import Budget, Optimizer, Program
from funding_optimizer.errors import ValidationError
from funding_optimizer.plan import Allocation, AllocationPlan


def test_allocation_plan_summary() -> None:
    budget = Budget(total=250_000, asset="USDC")
    programs = [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
        Program(id="health-gh", min_share=0.15, expected_reach=7_800),
    ]
    plan = Optimizer(objective="maximize_reach", fairness="regional_floor").solve(budget, programs)

    summary = plan.summary()
    assert "Allocation plan (objective=maximize_reach, fairness=regional_floor)" in summary
    assert "Total budget: 250,000.00 USDC" in summary
    assert "cash-transfers-ng" in summary
    assert "education-ke" in summary
    assert "health-gh" in summary
    assert "Allocated: 250,000.00 USDC" in summary
    assert "Unallocated: 0.00 USDC" in summary


def test_allocated_total_sums_allocation_amounts() -> None:
    plan = AllocationPlan(
        objective="maximize_reach",
        fairness="none",
        asset="USDC",
        total=1_000,
        allocations=(
            Allocation(program_id="a", region=None, share=0.5, amount=500.0, expected_reach=10),
            Allocation(program_id="b", region=None, share=0.5, amount=500.0, expected_reach=20),
        ),
    )
    assert plan.allocated_total == pytest.approx(1_000.0)


def test_unallocated_defaults_to_zero() -> None:
    plan = AllocationPlan(
        objective="equal_share",
        fairness="none",
        asset="USDC",
        total=100,
        allocations=(),
    )
    assert plan.unallocated == 0.0


def test_empty_allocations_summary() -> None:
    plan = AllocationPlan(
        objective="maximize_reach",
        fairness="none",
        asset="USDC",
        total=100,
        allocations=(),
    )
    summary = plan.summary()
    assert "No programs were allocated." in summary


def test_allocation_plan_is_frozen() -> None:
    plan = AllocationPlan(
        objective="maximize_reach",
        fairness="none",
        asset="USDC",
        total=100,
        allocations=(),
    )
    # mypy rejects direct assignment to a frozen dataclass field, so go
    # through an ``Any`` alias to keep the runtime assertion.
    mutable_alias: Any = plan
    with pytest.raises(AttributeError):
        mutable_alias.total = 200


def test_allocation_plan_validates_via_optimizer() -> None:
    with pytest.raises(ValidationError):
        Optimizer().solve(Budget(total=100, asset="USDC"), [])
