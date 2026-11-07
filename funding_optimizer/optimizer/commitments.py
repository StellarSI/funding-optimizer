"""Locked and already-committed funds for the funding optimizer.

Respects funds that are already committed to specific programs: committed
programs are locked to their commitment amount and the remaining budget is
allocated across the remaining programs. The optimizer recommends
allocations and never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .constraints import Program, RegionFloor
from .solver import HiGHSolver, Solver, solve_with_lexicographic_tie_break


@dataclass(frozen=True)
class Commitment:
    """Funds already committed (locked) to a single program."""

    program_id: str
    amount: float


@dataclass(frozen=True)
class CommitmentResult:
    """Outcome of a committed-funds allocation run."""

    success: bool
    status: int
    message: str
    objective_value: float | None
    program_allocation: dict[str, float]
    """Program id -> total amount the program receives."""
    iterations: int | None = None


def _validate(
    programs: list[Program],
    commitments: list[Commitment],
    budget: float,
    region_floors: list[RegionFloor],
) -> None:
    if not isfinite(budget) or budget <= 0.0:
        raise ValueError(
            f"Budget must be a positive, finite number, got {budget!r}"
        )
    if not programs:
        raise ValueError("at least one program is required")
    for region_floor in region_floors:
        if not 0.0 <= region_floor.floor <= 1.0:
            raise ValueError(
                f"Region {region_floor.region!r} floor {region_floor.floor} is not in [0, 1]"
            )
    program_ids = {program.id for program in programs}
    committed: dict[str, float] = {}
    total_committed = 0.0
    for commitment in commitments:
        if commitment.program_id not in program_ids:
            raise ValueError(
                f"Commitment references unknown program {commitment.program_id!r}"
            )
        if (
            not isfinite(commitment.amount)
            or commitment.amount < 0.0
        ):
            raise ValueError(
                f"Commitment for {commitment.program_id!r} must be a "
                f"non-negative, finite number, got {commitment.amount!r}"
            )
        if commitment.program_id in committed:
            raise ValueError(
                f"Duplicate commitment for program {commitment.program_id!r}"
            )
        committed[commitment.program_id] = float(commitment.amount)
        total_committed += float(commitment.amount)
    if total_committed > float(budget):
        raise ValueError(
            f"Commitments sum to {total_committed}, which exceeds the budget {budget}"
        )


def solve_committed_allocation(
    programs: list[Program],
    budget: float,
    commitments: list[Commitment],
    region_floors: list[RegionFloor] | None = None,
    *,
    solver: Solver | None = None,
    tie_break: bool = False,
) -> CommitmentResult:
    """Build and solve the allocation problem with locked commitments.

    Committed programs are fixed to their commitment amount and excluded from
    reallocation. The remaining budget is distributed across the uncommitted
    programs, subject to regional floors (committed amounts count toward the
    floor of their region) and program ``min_share`` requirements.
    """
    if solver is None:
        solver = HiGHSolver()
    region_floors = [] if region_floors is None else list(region_floors)
    _validate(programs, commitments, budget, region_floors)

    n = len(programs)
    index_by_id = {program.id: position for position, program in enumerate(programs)}
    reach_by_id = {program.id: float(program.expected_reach) for program in programs}

    committed: dict[str, float] = {
        commitment.program_id: float(commitment.amount) for commitment in commitments
    }

    # Objective: maximize total reach (linprog minimizes the negated reach).
    c = np.array([-reach_by_id[program.id] for program in programs], dtype=float)

    # Bounds: committed programs are locked to their commitment amount.
    bounds: list[tuple[float, float | None]] = []
    for program in programs:
        if program.id in committed:
            amount = committed[program.id]
            bounds.append((amount, amount))
        else:
            bounds.append((0.0, None))

    # Equality: the full budget is allocated.
    a_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([float(budget)], dtype=float)

    # Inequalities: regional floors and uncommitted program min_shares.
    rows: list[list[float]] = []
    limits: list[float] = []
    for region_floor in region_floors:
        row = [0.0] * n
        for program in programs:
            if program.region == region_floor.region:
                row[index_by_id[program.id]] = -1.0
        rows.append(row)
        limits.append(-float(region_floor.floor) * float(budget))
    for program in programs:
        if program.id in committed:
            continue
        if program.min_share > 0.0:
            row = [0.0] * n
            row[index_by_id[program.id]] = -1.0
            rows.append(row)
            limits.append(-float(program.min_share) * float(budget))
    if rows:
        a_ub = np.array(rows, dtype=float)
        b_ub = np.array(limits, dtype=float)
    else:
        a_ub = np.zeros((0, n), dtype=float)
        b_ub = np.zeros(0, dtype=float)

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
        solved = solver.solve(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)

    if not solved.success or solved.allocation is None:
        return CommitmentResult(
            success=False,
            status=solved.status,
            message=solved.message,
            objective_value=solved.objective_value,
            program_allocation={},
            iterations=solved.iterations,
        )
    program_allocation = {
        program.id: float(solved.allocation[index_by_id[program.id]])
        for program in programs
    }
    return CommitmentResult(
        success=True,
        status=solved.status,
        message=solved.message,
        objective_value=solved.objective_value,
        program_allocation=program_allocation,
        iterations=solved.iterations,
    )
