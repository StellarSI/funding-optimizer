"""Cohort-level allocation for the funding optimizer.

Groups programs into cohorts and allocates budget across cohorts first, then
distributes each cohort's share to its member programs. The optimizer
recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .constraints import Program
from .solver import HiGHSolver, Solver, solve_with_lexicographic_tie_break


@dataclass(frozen=True)
class Cohort:
    """A group of programs that shares one slice of the budget.

    ``program_ids`` lists the member programs in priority order. A cohort's
    ``min_share`` and ``max_share`` bound the fraction of the total budget
    the cohort may receive, expressed as a fraction of ``budget``.
    """

    id: str
    program_ids: tuple[str, ...]
    min_share: float = 0.0
    max_share: float = 1.0


@dataclass(frozen=True)
class CohortResult:
    """Outcome of a cohort-level allocation run."""

    success: bool
    status: int
    message: str
    objective_value: float | None
    cohort_allocation: dict[str, float]
    """Cohort id -> total amount the cohort receives."""
    program_allocation: dict[str, float]
    """Program id -> amount the program receives within its cohort."""
    iterations: int | None = None


def _validate(
    programs: list[Program], cohorts: list[Cohort], budget: float
) -> None:
    if not isfinite(budget) or budget <= 0.0:
        raise ValueError(
            f"Budget must be a positive, finite number, got {budget!r}"
        )
    if not programs:
        raise ValueError("at least one program is required")
    if not cohorts:
        raise ValueError("at least one cohort is required")
    program_ids = {program.id for program in programs}
    assigned: set[str] = set()
    total_min = 0.0
    for cohort in cohorts:
        if not 0.0 <= cohort.min_share <= 1.0:
            raise ValueError(
                f"Cohort {cohort.id!r} min_share {cohort.min_share} is not in [0, 1]"
            )
        if not 0.0 <= cohort.max_share <= 1.0:
            raise ValueError(
                f"Cohort {cohort.id!r} max_share {cohort.max_share} is not in [0, 1]"
            )
        if cohort.min_share > cohort.max_share:
            raise ValueError(
                f"Cohort {cohort.id!r} min_share {cohort.min_share} exceeds "
                f"max_share {cohort.max_share}"
            )
        total_min += cohort.min_share
        for program_id in cohort.program_ids:
            if program_id not in program_ids:
                raise ValueError(
                    f"Cohort {cohort.id!r} references unknown program "
                    f"{program_id!r}"
                )
            if program_id in assigned:
                raise ValueError(
                    f"Program {program_id!r} is assigned to multiple cohorts"
                )
            assigned.add(program_id)
    if total_min > 1.0:
        raise ValueError(
            f"Cohort min_shares sum to {total_min}, which exceeds the whole "
            "budget (1.0)"
        )


def _cohort_effective_reach(
    cohort: Cohort, reach_by_id: dict[str, float]
) -> float:
    return max(reach_by_id[program_id] for program_id in cohort.program_ids)


def _cohort_objective(
    cohorts: list[Cohort], reach_by_id: dict[str, float]
) -> np.ndarray:
    return np.array(
        [-_cohort_effective_reach(cohort, reach_by_id) for cohort in cohorts],
        dtype=float,
    )


def _cohort_constraints(
    cohorts: list[Cohort], budget: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float | None]],
]:
    n = len(cohorts)
    # Equality: the cohort shares must sum to the whole budget.
    a_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([float(budget)], dtype=float)
    rows: list[list[float]] = []
    limits: list[float] = []
    for position, cohort in enumerate(cohorts):
        row = [0.0] * n
        row[position] = -1.0
        rows.append(row)
        limits.append(-float(cohort.min_share) * float(budget))
    for position, cohort in enumerate(cohorts):
        row = [0.0] * n
        row[position] = 1.0
        rows.append(row)
        limits.append(float(cohort.max_share) * float(budget))
    a_ub = np.array(rows, dtype=float)
    b_ub = np.array(limits, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n
    return a_ub, b_ub, a_eq, b_eq, bounds


def solve_cohort_allocation(
    programs: list[Program],
    cohorts: list[Cohort],
    budget: float,
    *,
    solver: Solver | None = None,
    tie_break: bool = False,
) -> CohortResult:
    """Build and solve the cohort-level allocation problem.

    The cohort-level program decides how much of ``budget`` each cohort
    receives (maximizing the cohort's best member reach, subject to cohort
    ``min_share`` / ``max_share``), then hands each cohort's share to its
    highest-reach member program, breaking reach ties by program order so
    the result is deterministic.
    """
    if solver is None:
        solver = HiGHSolver()
    _validate(programs, cohorts, budget)
    reach_by_id = {program.id: float(program.expected_reach) for program in programs}
    index_by_id = {program.id: position for position, program in enumerate(programs)}
    programs_by_id = {program.id: program for program in programs}
    c = _cohort_objective(cohorts, reach_by_id)
    a_ub, b_ub, a_eq, b_eq, bounds = _cohort_constraints(cohorts, budget)
    if tie_break:
        solved = solve_with_lexicographic_tie_break(
            solver,
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
        )
    else:
        solved = solver.solve(
            c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds
        )
    if not solved.success or solved.allocation is None:
        return CohortResult(
            success=False,
            status=solved.status,
            message=solved.message,
            objective_value=solved.objective_value,
            cohort_allocation={},
            program_allocation={},
            iterations=solved.iterations,
        )
    cohort_allocation = {
        cohort.id: float(solved.allocation[position])
        for position, cohort in enumerate(cohorts)
    }
    program_allocation: dict[str, float] = {
        program.id: 0.0 for program in programs
    }
    for position, cohort in enumerate(cohorts):
        amount = float(solved.allocation[position])
        members = [programs_by_id[program_id] for program_id in cohort.program_ids]
        best = max(
            members,
            key=lambda program: (
                program.expected_reach,
                -index_by_id[program.id],
            ),
        )
        program_allocation[best.id] += amount
    return CohortResult(
        success=True,
        status=solved.status,
        message=solved.message,
        objective_value=solved.objective_value,
        cohort_allocation=cohort_allocation,
        program_allocation=program_allocation,
        iterations=solved.iterations,
    )
