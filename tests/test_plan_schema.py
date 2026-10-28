"""Tests for plan JSON Schema publication and stable serialization (issue #8)."""

from __future__ import annotations

import pytest

from funding_optimizer.errors import ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import Optimizer
from funding_optimizer.plan import Allocation, AllocationPlan
from funding_optimizer.schemas import load_plan_schema, validate_plan_record

BUDGET = Budget(total=1_000.0, asset="USDC")
PROGRAMS = [
    Program(id="cash-transfers-ng", min_share=0.2, expected_reach=12_000),
    Program(id="education-ke", min_share=0.1, expected_reach=4_500),
    Program(id="health-gh", min_share=0.15, expected_reach=7_800),
]


def sample_plan() -> AllocationPlan:
    """Solve the README quick example for use in round-trip tests."""
    return Optimizer(objective="maximize_reach", fairness="regional_floor").solve(
        BUDGET, PROGRAMS
    )


def test_published_schema_loads_and_declares_plan_contract() -> None:
    schema = load_plan_schema()
    assert schema["title"] == "AllocationPlan"
    assert set(schema["required"]) == {
        "objective",
        "fairness",
        "asset",
        "total",
        "unallocated",
        "allocations",
    }
    assert schema["properties"]["objective"]["enum"] == [
        "maximize_reach",
        "equal_share",
        "minimize_gap",
    ]


def test_plan_that_matches_readme_quick_example_passes_the_schema() -> None:
    plan = sample_plan()
    validate_plan_record(plan.to_dict())


def test_to_json_from_json_round_trip_is_lossless_and_stable() -> None:
    plan = sample_plan()
    first = plan.to_json()
    again = AllocationPlan.from_json(first)
    assert again == plan
    # Stable serialization: identical inputs serialize identically.
    assert again.to_json() == first


def test_serialization_output_conforms_to_the_published_schema() -> None:
    plan = sample_plan()
    validate_plan_record(plan.to_dict())
    # The serialized JSON, re-parsed, still validates.
    validate_plan_record(AllocationPlan.from_json(plan.to_json()).to_dict())


def test_from_json_rejects_malformed_json() -> None:
    with pytest.raises(ValidationError):
        AllocationPlan.from_json("{not json at all")


def test_from_dict_rejects_unknown_objective() -> None:
    record = sample_plan().to_dict()
    record["objective"] = "overthrow_market"
    with pytest.raises(ValidationError):
        AllocationPlan.from_dict(record)


def test_from_dict_rejects_null_share() -> None:
    record = sample_plan().to_dict()
    record["allocations"][0]["share"] = None
    allocation = record["allocations"][0]
    with pytest.raises(ValidationError):
        Allocation.from_dict(allocation)


def test_from_dict_rejects_negative_expected_reach() -> None:
    record = sample_plan().to_dict()
    record["allocations"][0]["expected_reach"] = -1
    allocation = record["allocations"][0]
    with pytest.raises(ValidationError):
        Allocation.from_dict(allocation)


def test_validator_rejects_missing_required_keys() -> None:
    payload = sample_plan().to_dict()
    del payload["allocations"]
    with pytest.raises(ValidationError, match="allocations"):
        validate_plan_record(payload)


def test_validator_rejects_out_of_range_share() -> None:
    payload = sample_plan().to_dict()
    payload["allocations"][0]["share"] = 1.5
    with pytest.raises(ValidationError, match="share"):
        validate_plan_record(payload)


def test_validator_rejects_non_positive_total() -> None:
    payload = sample_plan().to_dict()
    payload["total"] = 0
    with pytest.raises(ValidationError, match="total"):
        validate_plan_record(payload)


def test_validator_rejects_unknown_properties() -> None:
    payload = sample_plan().to_dict()
    payload["malicious_extra"] = True
    with pytest.raises(ValidationError, match="malicious_extra"):
        validate_plan_record(payload)
