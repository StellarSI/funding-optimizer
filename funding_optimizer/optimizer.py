"""Allocation solver.

The optimizer recommends how to split a budget across programs. It never
signs or submits payouts; callers retain control over execution.
"""

from __future__ import annotations

from collections.abc import Iterable

from funding_optimizer.errors import ValidationError
from funding_optimizer.models import Allocation, AllocationPlan, Budget, Program

__all__ = ["Optimizer"]

OBJECTIVES = frozenset({"maximize_reach", "equal_share", "minimize_gap"})
FAIRNESS_MODES = frozenset({"none", "regional_floor"})

_EPSILON = 1e-9


class Optimizer:
    """Recommend an allocation of ``budget`` across ``programs``."""

    def __init__(self, objective: str = "maximize_reach", fairness: str = "none") -> None:
        if objective not in OBJECTIVES:
            raise ValidationError(
                f"unknown objective {objective!r}; expected one of {sorted(OBJECTIVES)}"
            )
        if fairness not in FAIRNESS_MODES:
            raise ValidationError(
                f"unknown fairness {fairness!r}; expected one of {sorted(FAIRNESS_MODES)}"
            )
        self.objective = objective
        self.fairness = fairness

    def solve(self, budget: Budget, programs: Iterable[Program]) -> AllocationPlan:
        program_list = list(programs)
        if not program_list:
            raise ValidationError("at least one program is required")
        if len({program.id for program in program_list}) != len(program_list):
            raise ValidationError("program ids must be unique")

        total_min = sum(program.min_share for program in program_list)
        if total_min > 1.0 + _EPSILON:
            raise ValidationError("program min_share values sum to more than 100%")

        shares = {program.id: program.min_share for program in program_list}
        remainder = 1.0 - sum(shares.values())
        if remainder > _EPSILON:
            shares = self._distribute_remainder(program_list, shares, remainder)
        if self.fairness == "regional_floor":
            shares = self._apply_regional_floor(program_list, shares)
        shares = self._sanitize(shares)

        allocations = self._build_allocations(budget, program_list, shares)
        allocated = round(sum(allocation.amount for allocation in allocations), 2)
        return AllocationPlan(
            objective=self.objective,
            fairness=self.fairness,
            asset=budget.asset,
            total=budget.total,
            allocations=tuple(allocations),
            unallocated=round(budget.total - allocated, 2),
        )

    def _distribute_remainder(
        self, programs: list[Program], shares: dict[str, float], remainder: float
    ) -> dict[str, float]:
        if self.objective != "maximize_reach":
            each = remainder / len(programs)
            return {program_id: share + each for program_id, share in shares.items()}

        total_reach = sum(program.expected_reach for program in programs)
        if total_reach <= 0:
            each = remainder / len(programs)
            return {program_id: share + each for program_id, share in shares.items()}

        out = dict(shares)
        for program in programs:
            out[program.id] = shares[program.id] + remainder * (
                program.expected_reach / total_reach
            )
        return out

    def _apply_regional_floor(
        self, programs: list[Program], shares: dict[str, float]
    ) -> dict[str, float]:
        """Ensure no region receives less than an equal share of the budget.

        The floor is ``1 / number_of_regions``. Regions below the floor are
        raised by redistributing from regions above it, proportional to their
        current share. Shares are re-normalized afterwards by :meth:`solve`.
        """
        regions = sorted({program.region for program in programs if program.region})
        if not regions:
            return shares

        floor = 1.0 / len(regions)
        region_totals = {
            region: sum(shares[program.id] for program in programs if program.region == region)
            for region in regions
        }
        below = {
            region: floor - total
            for region, total in region_totals.items()
            if total < floor - _EPSILON
        }
        if not below:
            return shares

        deficit = sum(below.values())
        above = [region for region in regions if region_totals[region] > floor + _EPSILON]
        excess_total = sum(region_totals[region] - floor for region in above)
        if excess_total <= _EPSILON:
            return shares

        out = dict(shares)
        for region, shortfall in below.items():
            members = [program for program in programs if program.region == region]
            weights = self._weights(shares, members, region_totals[region])
            for program in members:
                out[program.id] += shortfall * weights[program.id]
        for region in above:
            take = deficit * ((region_totals[region] - floor) / excess_total)
            members = [program for program in programs if program.region == region]
            weights = self._weights(shares, members, region_totals[region])
            for program in members:
                out[program.id] -= take * weights[program.id]
        return out

    @staticmethod
    def _weights(
        shares: dict[str, float], members: list[Program], region_total: float
    ) -> dict[str, float]:
        if region_total > _EPSILON:
            return {program.id: shares[program.id] / region_total for program in members}
        count = len(members)
        return {program.id: (1.0 / count if count else 0.0) for program in members}

    @staticmethod
    def _sanitize(shares: dict[str, float]) -> dict[str, float]:
        cleaned = {program_id: max(0.0, share) for program_id, share in shares.items()}
        total = sum(cleaned.values())
        if total <= 0:
            raise ValidationError("allocation produced no positive shares")
        return {program_id: share / total for program_id, share in cleaned.items()}

    @staticmethod
    def _build_allocations(
        budget: Budget, programs: list[Program], shares: dict[str, float]
    ) -> list[Allocation]:
        allocations: list[Allocation] = []
        running = 0.0
        for program in programs:
            amount = round(budget.total * shares[program.id], 2)
            running += amount
            allocations.append(
                Allocation(
                    program_id=program.id,
                    region=program.region,
                    share=round(shares[program.id], 6),
                    amount=amount,
                    expected_reach=program.expected_reach,
                )
            )

        # Correct rounding drift on the final allocation so amounts sum exactly.
        if allocations:
            drift = round(budget.total - running, 2)
            last = allocations[-1]
            allocations[-1] = Allocation(
                program_id=last.program_id,
                region=last.region,
                share=last.share,
                amount=round(last.amount + drift, 2),
                expected_reach=last.expected_reach,
            )
        return allocations
