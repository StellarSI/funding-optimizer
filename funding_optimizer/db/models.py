"""Persistence for optimization runs and allocation plans (issue #9).

Defines the SQL tables backing run history and the typed boundaries that
convert an :class:`~funding_optimizer.plan.AllocationPlan` — or an
allocation problem — into storable records, and records back into plans.

Boundary rules (per the issue scope):
- Only model/validation logic lives here. Solver, transport, and payout
  mechanics stay out of this layer: the optimizer recommends
  allocations and never signs or submits payouts.
- Postgres-flavored defaults (e.g. ``TIMESTAMPTZ``) are used via
  SQLAlchemy's ``DateTime(timezone=True)``; durability requires the
  caller to configure a connection to their PostgreSQL database.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from funding_optimizer.errors import ValidationError
from funding_optimizer.plan import Allocation, AllocationPlan

__all__ = ["AllocationRecord", "Base", "OptimizationRunRecord", "plan_from_run"]


class Base(DeclarativeBase):
    """Declarative base for all funding-optimizer tables."""


class OptimizationRunRecord(Base):
    """One solver invocation: the input objective and mode plus its plan."""

    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    objective: Mapped[str] = mapped_column(String(64), nullable=False)
    fairness: Mapped[str] = mapped_column(String(64), nullable=False)
    asset: Mapped[str] = mapped_column(String(69), nullable=False)
    total: Mapped[float] = mapped_column(Numeric(20, 4), nullable=False)
    unallocated: Mapped[float] = mapped_column(Numeric(20, 4), nullable=False, default=0.0)
    allocations: Mapped[list[AllocationRecord]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AllocationRecord.id",
    )

    @classmethod
    def from_plan(cls, plan: AllocationPlan) -> OptimizationRunRecord:
        """Build an unattached record from a solved :class:`AllocationPlan`.

        Raises:
            ValidationError: When the plan cannot be trusted as input
                (unknown objectives or fairness modes, negative totals).
        """
        _validate_plan_fields(plan)
        return cls(
            objective=plan.objective,
            fairness=plan.fairness,
            asset=plan.asset,
            total=plan.total,
            unallocated=plan.unallocated,
            allocations=[AllocationRecord.from_allocation(item) for item in plan.allocations],
        )


class AllocationRecord(Base):
    """The budget share assigned to one program inside a run."""

    __tablename__ = "allocation_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False
    )
    program_id: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str | None] = mapped_column(String(255), nullable=True)
    share: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(20, 4), nullable=False)
    expected_reach: Mapped[int] = mapped_column(default=0, nullable=False)
    run: Mapped[OptimizationRunRecord] = relationship(back_populates="allocations")

    @classmethod
    def from_allocation(cls, allocation: Allocation) -> AllocationRecord:
        """Build an unattached record from a single :class:`Allocation`."""
        if not isinstance(allocation.program_id, str) or not allocation.program_id:
            raise ValidationError("allocation.program_id must be a non-empty string")
        if not 0.0 <= allocation.share <= 1.0:
            raise ValidationError(
                f"allocation for {allocation.program_id!r}: share must be in [0, 1]"
            )
        if allocation.amount < 0:
            raise ValidationError(
                f"allocation for {allocation.program_id!r}: amount must be non-negative"
            )
        if allocation.expected_reach < 0:
            raise ValidationError(
                f"allocation for {allocation.program_id!r}: "
                "expected_reach must be non-negative"
            )
        return cls(
            program_id=allocation.program_id,
            region=allocation.region,
            share=allocation.share,
            amount=allocation.amount,
            expected_reach=allocation.expected_reach,
        )


def _validate_plan_fields(plan: AllocationPlan) -> None:
    """Validate plan envelopes before the record is constructed."""
    if not isinstance(plan.objective, str) or not plan.objective:
        raise ValidationError("plan.objective must be a non-empty string")
    if not isinstance(plan.fairness, str) or not plan.fairness:
        raise ValidationError("plan.fairness must be a non-empty string")
    if not isinstance(plan.asset, str) or not plan.asset:
        raise ValidationError("plan.asset must be a non-empty string")
    if plan.total <= 0:
        raise ValidationError("plan.total must be greater than zero")
    if plan.unallocated < 0:
        raise ValidationError("plan.unallocated must be non-negative")


def plan_from_run(record: OptimizationRunRecord) -> AllocationPlan:
    """Convert a persisted run back into an :class:`AllocationPlan`.

    Raises:
        ValidationError: When persisted values are inconsistent.
    """
    if record.id is None:
        raise ValidationError("run record must be flushed to the database first")
    allocations = tuple(_allocation_from_record(item) for item in record.allocations)
    return AllocationPlan(
        objective=record.objective,
        fairness=record.fairness,
        asset=record.asset,
        total=float(record.total),
        unallocated=float(record.unallocated),
        allocations=allocations,
    )


def _allocation_from_record(record: AllocationRecord) -> Allocation:
    """Materialize a single :class:`Allocation` from its persisted row."""
    program = record.program_id
    if not program:
        raise ValidationError("persisted allocation has an empty program_id")
    return Allocation(
        program_id=program,
        region=record.region,
        share=float(record.share),
        amount=float(record.amount),
        expected_reach=record.expected_reach,
    )
