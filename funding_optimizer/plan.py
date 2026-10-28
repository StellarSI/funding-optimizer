"""Allocation plan models.

The :class:`AllocationPlan` is the result of solving an allocation problem:
it records how a :class:`Budget` was split across :class:`Program`
candidates, together with run metadata and a human-readable summary.

The optimizer recommends allocations. It never signs or submits payouts;
callers retain control over execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from funding_optimizer.errors import ValidationError

__all__ = ["Allocation", "AllocationPlan"]


@dataclass(frozen=True)
class Allocation:
    """The share of the budget assigned to one program."""

    program_id: str
    region: str | None
    share: float
    amount: float
    expected_reach: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, schema-conformant dictionary (plan.schema.json)."""
        return {
            "program_id": self.program_id,
            "region": self.region,
            "share": self.share,
            "amount": self.amount,
            "expected_reach": self.expected_reach,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Allocation:
        """Rebuild an :class:`Allocation` from ``Allocation.to_dict()`` output.

        Raises:
            ValidationError: When the record is missing keys or types deviate.
        """
        missing = [
            key
            for key in ("program_id", "region", "share", "amount", "expected_reach")
            if key not in record
        ]
        if missing:
            raise ValidationError(f"allocation record is missing required keys: {missing}")
        if not isinstance(record["program_id"], str) or not record["program_id"]:
            raise ValidationError("allocation.program_id must be a non-empty string")
        if record["region"] is not None and not isinstance(record["region"], str):
            raise ValidationError("allocation.region must be a string or null")
        if not isinstance(record["share"], (int, float)) or isinstance(record["share"], bool):
            raise ValidationError("allocation.share must be a number")
        if not isinstance(record["amount"], (int, float)) or isinstance(record["amount"], bool):
            raise ValidationError("allocation.amount must be a number")
        reach = record["expected_reach"]
        if not isinstance(reach, int) or isinstance(reach, bool) or reach < 0:
            raise ValidationError("allocation.expected_reach must be a non-negative integer")
        return cls(
            program_id=record["program_id"],
            region=record["region"],
            share=record["share"],
            amount=record["amount"],
            expected_reach=reach,
        )


@dataclass(frozen=True)
class AllocationPlan:
    """Result of solving an allocation problem.

    ``total`` is the budget amount, ``asset`` the currency it is denominated
    in, and ``allocations`` the per-program breakdown. ``unallocated`` tracks
    any portion of the budget the solver chose not to distribute.
    """

    objective: str
    fairness: str
    asset: str
    total: float
    allocations: tuple[Allocation, ...]
    unallocated: float = 0.0

    @property
    def allocated_total(self) -> float:
        """Total amount actually allocated across programs."""
        return sum(allocation.amount for allocation in self.allocations)

    def summary(self) -> str:
        """Return a human-readable summary of the plan."""
        lines = [
            f"Allocation plan (objective={self.objective}, fairness={self.fairness})",
            f"Total budget: {self.total:,.2f} {self.asset}",
            "",
        ]
        if not self.allocations:
            lines.append("No programs were allocated.")
        for allocation in self.allocations:
            region = f" [{allocation.region}]" if allocation.region else ""
            lines.append(
                f"- {allocation.program_id}{region}: {allocation.share:.2%} -> "
                f"{allocation.amount:,.2f} {self.asset} "
                f"(expected reach {allocation.expected_reach:,})"
            )
        lines.append("")
        lines.append(f"Allocated: {self.allocated_total:,.2f} {self.asset}")
        lines.append(f"Unallocated: {self.unallocated:,.2f} {self.asset}")
        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Stable serialization (issue #8)
    # ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary that conforms to ``plan.schema.json``.

        The field set and key order are fixed, so two plans built from
        identical inputs always serialize to identical payloads — stable
        for diffs, snapshots, and content-addressed hashing.
        """
        return {
            "objective": self.objective,
            "fairness": self.fairness,
            "asset": self.asset,
            "total": self.total,
            "unallocated": self.unallocated,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the plan as JSON.

        Args:
            indent: Pretty-print with this indentation; ``None`` (default)
                produces single-line compact output.

        Raises:
            ValidationError: When the payload violates ``plan.schema.json``.
        """
        payload = self.to_dict()
        from funding_optimizer.schemas import load_plan_schema, validate_payload

        validate_payload(payload, load_plan_schema())
        return json.dumps(payload, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> AllocationPlan:
        """Rebuild a plan from ``AllocationPlan.to_dict()`` output.

        Raises:
            ValidationError: When the record does not conform to
                ``plan.schema.json``.
        """
        from funding_optimizer.schemas import load_plan_schema, validate_payload

        validate_payload(record, load_plan_schema())
        return cls(
            objective=record["objective"],
            fairness=record["fairness"],
            asset=record["asset"],
            total=record["total"],
            unallocated=record["unallocated"],
            allocations=tuple(Allocation.from_dict(item) for item in record["allocations"]),
        )

    @classmethod
    def from_json(cls, text: str) -> AllocationPlan:
        """Parse and validate a plan serialized with :meth:`to_json`.

        Raises:
            ValidationError: When the JSON is malformed or schema-invalid.
        """
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValidationError(f"plan JSON is not valid: {error}") from error
        if not isinstance(record, dict):
            raise ValidationError("plan JSON must be an object")
        return cls.from_dict(record)
