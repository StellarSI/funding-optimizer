"""Tests for optimization-run persistence models (issue #9).

No database is spun up: only the typed boundaries — mapping plans into
records and back — are exercised, so these tests stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from funding_optimizer.db import (
    AllocationRecord,
    Base,
    OptimizationRunRecord,
    plan_from_run,
)
from funding_optimizer.errors import ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import Optimizer
from funding_optimizer.plan import Allocation, AllocationPlan


def sample_plan() -> AllocationPlan:
    budget = Budget(total=250_000.0, asset="USDC")
    programs = [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
        Program(id="health-gh", min_share=0.15, expected_reach=7_800),
    ]
    return Optimizer(objective="maximize_reach", fairness="regional_floor").solve(
        budget, programs
    )


def test_plan_round_trips_through_records_without_loss() -> None:
    plan = sample_plan()
    run = OptimizationRunRecord.from_plan(plan)

    assert run.objective == plan.objective
    assert run.fairness == plan.fairness
    assert run.asset == plan.asset
    assert float(run.total) == plan.total
    assert len(run.allocations) == len(plan.allocations)
    for record, allocation in zip(run.allocations, plan.allocations, strict=True):
        assert record.program_id == allocation.program_id
        assert record.region == allocation.region
        assert record.expected_reach == allocation.expected_reach


def test_run_record_converts_back_through_public_plan_shape() -> None:
    plan = sample_plan()
    run = OptimizationRunRecord.from_plan(plan)
    # Simulate the flush the session would perform: consider id assigned.
    run.id = 1
    restored = plan_from_run(run)
    assert restored.total == plan.total
    assert restored.unallocated == plan.unallocated
    assert restored.asset == plan.asset
    assert [a.program_id for a in restored.allocations] == [
        a.program_id for a in plan.allocations
    ]


def test_map_tables_to_metadata() -> None:
    Base.metadata.create_all(sa.create_engine("sqlite://"))

    with sa.create_engine("sqlite://").connect() as connection:
        names = sorted(sa.inspect(connection).get_table_names())
    assert "optimization_runs" in names
    assert "allocation_records" in names


def test_persisted_records_reappear_in_a_session(tmp_path: Path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path}/state.db")
    Base.metadata.create_all(engine)
    plan = sample_plan()
    run = OptimizationRunRecord.from_plan(plan)

    with Session(engine) as session:
        session.add(run)
        session.commit()
        stored = session.get(OptimizationRunRecord, run.id)

    assert stored is not None
    assert len(stored.allocations) == len(plan.allocations)
    rebuilt = plan_from_run(stored)
    assert rebuilt == plan


def test_from_allocation_rejects_out_of_range_share() -> None:
    bad = Allocation(
        program_id="cash-transfers-ng",
        region=None,
        share=1.5,
        amount=10.0,
        expected_reach=0,
    )
    with pytest.raises(ValidationError):
        AllocationRecord.from_allocation(bad)


def test_from_allocation_rejects_negative_amount_and_reach() -> None:
    negative_amount = Allocation(
        program_id="cash-transfers-ng",
        region=None,
        share=0.5,
        amount=-1.0,
        expected_reach=10,
    )
    with pytest.raises(ValidationError):
        AllocationRecord.from_allocation(negative_amount)

    negative_reach = Allocation(
        program_id="cash-transfers-ng",
        region=None,
        share=0.5,
        amount=1.0,
        expected_reach=-5,
    )
    with pytest.raises(ValidationError):
        AllocationRecord.from_allocation(negative_reach)


def test_from_allocation_rejects_empty_program_id() -> None:
    anon = Allocation(
        program_id="  ",
        region=None,
        share=0.5,
        amount=1.0,
        expected_reach=10,
    )
    with pytest.raises(ValidationError):
        AllocationRecord.from_allocation(anon)
