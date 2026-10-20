"""Core data models for funding-optimizer.

These models describe the allocation problem: a :class:`Budget` to split
across a set of :class:`Program` candidates, and the resulting
:class:`AllocationPlan`.

The optimizer recommends allocations. It never signs or submits payouts;
callers retain control over execution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from funding_optimizer.errors import ValidationError

__all__ = ["Allocation", "AllocationPlan", "Budget", "Program"]

# Stellar asset codes are 1-12 alphanumeric characters. A classic asset
# reference may append an ``:ISSUER`` account id.
_ASSET_CODE_PATTERN = re.compile(r"[A-Z0-9]{1,12}(?::[A-Z0-9]{1,56})?", re.ASCII)


@dataclass(frozen=True)
class Budget:
    """Total funds available to allocate across programs.

    ``total`` must be a finite, strictly positive amount of ``asset``.
    ``asset`` is normalized to uppercase and must be a Stellar asset code
    (1-12 alphanumeric characters), optionally suffixed with ``:ISSUER``.
    """

    total: float
    asset: str

    def __post_init__(self) -> None:
        if isinstance(self.total, bool) or not isinstance(self.total, (int, float)):
            raise ValidationError("budget.total must be a number")
        if not math.isfinite(self.total):
            raise ValidationError("budget.total must be a finite number")
        if self.total <= 0:
            raise ValidationError("budget.total must be greater than zero")
        if not isinstance(self.asset, str):
            raise ValidationError("budget.asset must be a non-empty string")
        asset = self.asset.strip().upper()
        if not asset:
            raise ValidationError("budget.asset must be a non-empty string")
        if _ASSET_CODE_PATTERN.fullmatch(asset) is None:
            raise ValidationError(
                "budget.asset must be a Stellar asset code "
                "(1-12 alphanumeric characters, optionally ':ISSUER')"
            )
        object.__setattr__(self, "asset", asset)


@dataclass(frozen=True)
class Program:
    """A candidate program that can receive part of the budget.

    ``min_share`` is the minimum fraction of the budget in ``[0, 1]`` that
    the optimizer must assign to the program. ``expected_reach`` is the
    non-negative number of people the program expects to reach; it drives the
    ``maximize_reach`` objective.
    """

    id: str
    min_share: float = 0.0
    expected_reach: int = 0
    region: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValidationError("program.id must be a non-empty string")
        object.__setattr__(self, "id", self.id.strip())

        if isinstance(self.min_share, bool) or not isinstance(self.min_share, (int, float)):
            raise ValidationError(f"program {self.id!r}: min_share must be a number")
        if not math.isfinite(self.min_share):
            raise ValidationError(f"program {self.id!r}: min_share must be a finite number")
        if not 0.0 <= self.min_share <= 1.0:
            raise ValidationError(f"program {self.id!r}: min_share must be between 0 and 1")

        if isinstance(self.expected_reach, bool) or not isinstance(self.expected_reach, int):
            raise ValidationError(f"program {self.id!r}: expected_reach must be an integer")
        if self.expected_reach < 0:
            raise ValidationError(f"program {self.id!r}: expected_reach must be non-negative")

        if self.region is not None and (
            not isinstance(self.region, str) or not self.region.strip()
        ):
            raise ValidationError(f"program {self.id!r}: region must be a non-empty string or None")
        if isinstance(self.region, str):
            object.__setattr__(self, "region", self.region.strip())


@dataclass(frozen=True)
class Allocation:
    """The share of the budget assigned to one program."""

    program_id: str
    region: str | None
    share: float
    amount: float
    expected_reach: int


@dataclass(frozen=True)
class AllocationPlan:
    """Result of solving an allocation problem."""

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
