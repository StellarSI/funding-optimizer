"""Tests for the README quick example and input validation.

These tests exercise the public API documented in ``README.md``
(§Getting started / §Quick example) and the validation contract.
"""

from __future__ import annotations

import pytest

from funding_optimizer import Budget, Optimizer, Program
from funding_optimizer.errors import ValidationError


def test_readme_quick_example() -> None:
    budget = Budget(total=250_000, asset="USDC")
    programs = [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
        Program(id="health-gh", min_share=0.15, expected_reach=7_800),
    ]

    plan = Optimizer(objective="maximize_reach", fairness="regional_floor").solve(budget, programs)
    summary = plan.summary()

    assert "cash-transfers-ng" in summary
    assert "education-ke" in summary
    assert "health-gh" in summary
    assert "250,000.00 USDC" in summary
    assert plan.unallocated == 0.0
    assert len(plan.allocations) == 3


def test_min_share_is_respected() -> None:
    budget = Budget(total=1_000, asset="USDC")
    programs = [Program(id="a", min_share=0.30, expected_reach=1)]

    plan = Optimizer().solve(budget, programs)

    assert plan.allocations[0].share >= 0.30 - 1e-9


def test_allocated_amounts_sum_to_total() -> None:
    budget = Budget(total=1_000, asset="USDC")
    programs = [
        Program(id="a", min_share=0.2, expected_reach=10),
        Program(id="b", min_share=0.1, expected_reach=20),
        Program(id="c", min_share=0.0, expected_reach=30),
    ]

    plan = Optimizer().solve(budget, programs)

    assert plan.allocated_total == pytest.approx(budget.total)
    assert plan.unallocated == pytest.approx(0.0)


def test_regional_floor_binds() -> None:
    budget = Budget(total=100, asset="USDC")
    programs = [
        Program(id="east-a", min_share=0.0, expected_reach=1, region="east"),
        Program(id="west-a", min_share=0.0, expected_reach=100, region="west"),
    ]

    plan = Optimizer(objective="maximize_reach", fairness="regional_floor").solve(budget, programs)
    east_total = sum(a.share for a in plan.allocations if a.region == "east")

    assert east_total >= 0.5 - 1e-6


def test_equal_share_objective() -> None:
    budget = Budget(total=300, asset="USDC")
    programs = [
        Program(id="a", min_share=0.0, expected_reach=1000),
        Program(id="b", min_share=0.0, expected_reach=1),
    ]

    plan = Optimizer(objective="equal_share").solve(budget, programs)

    assert plan.allocations[0].share == pytest.approx(0.5)
    assert plan.allocations[1].share == pytest.approx(0.5)


def test_budget_validation() -> None:
    with pytest.raises(ValidationError):
        Budget(total=0, asset="USDC")
    with pytest.raises(ValidationError):
        Budget(total=-5, asset="USDC")
    with pytest.raises(ValidationError):
        Budget(total=100, asset="")


def test_program_validation() -> None:
    with pytest.raises(ValidationError):
        Program(id="", min_share=0.1, expected_reach=1)
    with pytest.raises(ValidationError):
        Program(id="a", min_share=1.5, expected_reach=1)
    with pytest.raises(ValidationError):
        Program(id="a", min_share=0.1, expected_reach=-1)
    with pytest.raises(ValidationError):
        Program(id="a", min_share=0.1, expected_reach=1, region="")


def test_empty_programs_rejected() -> None:
    with pytest.raises(ValidationError):
        Optimizer().solve(Budget(total=100, asset="USDC"), [])


def test_duplicate_program_ids_rejected() -> None:
    programs = [
        Program(id="a", min_share=0.1, expected_reach=1),
        Program(id="a", min_share=0.1, expected_reach=2),
    ]
    with pytest.raises(ValidationError):
        Optimizer().solve(Budget(total=100, asset="USDC"), programs)


def test_min_share_sum_over_one_rejected() -> None:
    programs = [
        Program(id="a", min_share=0.6, expected_reach=1),
        Program(id="b", min_share=0.6, expected_reach=1),
    ]
    with pytest.raises(ValidationError):
        Optimizer().solve(Budget(total=100, asset="USDC"), programs)


def test_unknown_objective_rejected() -> None:
    with pytest.raises(ValidationError):
        Optimizer(objective="nonsense")


def test_unknown_fairness_rejected() -> None:
    with pytest.raises(ValidationError):
        Optimizer(fairness="nonsense")


def test_validation_error_is_value_error() -> None:
    assert issubclass(ValidationError, ValueError)
