"""Allocation plan models.

The :class:`AllocationPlan` is the result of solving an allocation problem:
it records how a :class:`Budget` was split across :class:`Program`
candidates, together with run metadata and a human-readable summary.

The optimizer recommends allocations. It never signs or submits payouts;
callers retain control over execution.
"""

from __future__ import annotations

from dataclasses import dataclass


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
