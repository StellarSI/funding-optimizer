"""The ``maximize_reach`` objective (issue #10).

Implements the allocation-optimization capability promised by
``README.md:19`` and exercised in the README quick example
(``README.md:76``) at the named target path
``funding_optimizer/optimizer/solver.py``.

Model (the toy linear program, ``README §Features — Allocation
optimization``): maximize the total expected reach

    maximize  sum_i share_i * expected_reach_i
    s.t.      share_i >= min_share_i for every program i
              sum_i share_i = 1
              share_i >= 0

``expected_reach_i`` is the reach the program projects for the whole
budget; reach scales linearly with the share it receives. The optimum is
then straightforward: bind every ``min_share`` floor, and hand the
remaining budget to the program with the largest ``expected_reach``
(fractional-knapsack greedy, provably optimal for this LP). Ties between
equal-reach programs break by ascending program id so answers are
reproducible run over run (matching #11's determinism contract).

The solver produces a full :class:`~funding_optimizer.plan.AllocationPlan`
— a recommendation — and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable

from funding_optimizer.errors import AllocationError, ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.plan import Allocation, AllocationPlan

__all__ = [
    "MaximizeReachSolveResult",
    "MaximizeReachSolver",
    "solve_maximize_reach",
]

_EPSILON = 1e-9


class MaximizeReachSolveResult:
    """Diagnostic wrapper around a solved maximize_reach problem.

    Exposes the optimal plan plus solver reasoning: the binding
    constraints (programs stuck at their ``min_share`` floor) and the
    expected total reach of the plan.
    """

    def __init__(
        self,
        plan: AllocationPlan,
        binding_program_ids: tuple[str, ...],
        expected_total_reach: int,
    ) -> None:
        self.plan = plan
        self.binding_program_ids = binding_program_ids
        self.expected_total_reach = expected_total_reach

    def diagnostic(self) -> str:
        """Return a human-readable infeasibility/explanatory diagnostic."""
        binding = ", ".join(self.binding_program_ids) or "none"
        return (
            f"objective=maximize_reach, expected_total_reach={self.expected_total_reach:,}, "
            f"binding_min_share_programs=[{binding}]"
        )


class MaximizeReachSolver:
    """Solve the maximize_reach linear program for a budget and programs.

    Deterministic adapter: coroutine-free, pure, and reproducible — the
    same problem always yields the same plan (P14 contract).
    """

    def solve(
        self,
        budget: Budget,
        programs: Iterable[Program],
        *,
        fairness: str = "none",
    ) -> MaximizeReachSolveResult:
        """Compute the reach-maximizing allocation plan.

        Raises:
            ValidationError: When the problems have conflicting filters
                (e.g. fairness is not ``none``), no programs, duplicate
                ids, or infeasible floors — infeasible problems surface
                as :class:`~funding_optimizer.errors.AllocationError`
                with an explanatory diagnostic instead of ad-hoc numbers.
        """
        if fairness != "none":
            raise ValidationError(
                "maximize_reach supports fairness='none'; use the Optimizer engine "
                "with fairness='regional_floor' for fairness-constrained solves"
            )

        program_list = list(programs)
        if not program_list:
            raise ValidationError("at least one program is required")

        ids = {program.id for program in program_list}
        if len(ids) != len(program_list):
            raise ValidationError("program ids must be unique")

        total_min = sum(program.min_share for program in program_list)
        if total_min > 1.0 + _EPSILON:
            raise AllocationError(
                "maximize_reach problem is infeasible: min_share values sum to "
                f"{total_min:.4f}, more than 100% of the budget"
            )

        # Bind every min_share floor.
        shares = {program.id: program.min_share for program in program_list}
        remainder = 1.0 - sum(shares.values())

        # The rest goes to the highest-reach program (LP optimum for the
        # linear reach model); ties break by ascending program id.
        best_reach = max(program.expected_reach for program in program_list)
        leaders = [program for program in program_list if program.expected_reach == best_reach]
        lead = min(leaders, key=lambda program: program.id)
        shares[lead.id] += remainder

        binding = tuple(
            program.id
            for program in program_list
            if abs(shares[program.id] - program.min_share) < _EPSILON
        )
        expected_total_reach = sum(
            int(shares[program.id] * program.expected_reach) for program in program_list
        )
        return MaximizeReachSolveResult(
            _build_plan(budget, program_list, shares), binding, expected_total_reach
        )


def solve_maximize_reach(
    budget: Budget,
    programs: Iterable[Program],
) -> AllocationPlan:
    """Solve maximize_reach and return only the allocation plan."""
    return MaximizeReachSolver().solve(budget, programs).plan


def _build_plan(
    budget: Budget, programs: list[Program], shares: dict[str, float]
) -> AllocationPlan:
    """Assemble the deterministic AllocationPlan (same rules as the engine)."""
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

    allocated = round(sum(allocation.amount for allocation in allocations), 2)
    return AllocationPlan(
        objective="maximize_reach",
        fairness="none",
        asset=budget.asset,
        total=budget.total,
        allocations=tuple(allocations),
        unallocated=round(budget.total - allocated, 2),
    )
