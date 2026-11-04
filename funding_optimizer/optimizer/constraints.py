"""Regional-floor fairness constraints for the allocation optimizer.

Builds the linear pieces (objective, inequality constraints, equality
constraints) of a budget-allocation linear program, and a small helper that
solves it using the deterministic :class:`~funding_optimizer.optimizer.solver.HiGHSolver`
adapter from issue P14.

All constraints keep the optimizer's boundary: it recommends allocations and
never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import HiGHSolver, Solver, SolverResult


@dataclass(frozen=True)
class Program:
    """A funded program or recipient cohort."""

    id: str
    region: str
    expected_reach: float
    """People reached per unit of budget allocated to this program."""
    min_share: float = 0.0
    """Minimum fraction of the total budget this program must receive."""


@dataclass(frozen=True)
class RegionFloor:
    """Fairness floor for a geographic region."""

    region: str
    floor: float
    """Minimum fraction of the total budget allocated to programs in this region."""


def validate(programs: list[Program], region_floors: list[RegionFloor]) -> None:
    """Validate that the allocation problem is structurally feasible.

    Raises
    ------
    ValueError
        If regional floors sum to more than the whole budget, or if any program
        ``min_share`` or region ``floor`` lies outside ``[0, 1]``.
    """
    for program in programs:
        if not 0.0 <= program.min_share <= 1.0:
            raise ValueError(
                f"Program {program.id!r} min_share {program.min_share} is not in [0, 1]"
            )
    for region_floor in region_floors:
        if not 0.0 <= region_floor.floor <= 1.0:
            raise ValueError(
                f"Region {region_floor.region!r} floor {region_floor.floor} is not in [0, 1]"
            )
    total_floor = sum(region_floor.floor for region_floor in region_floors)
    if total_floor > 1.0:
        raise ValueError(
            f"Regional floors sum to {total_floor}, which exceeds the whole budget (1.0)"
        )


def maximize_reach_objective(programs: list[Program]) -> np.ndarray:
    """Objective coefficients for maximizing total expected reach.

    linprog minimizes, so the per-unit reach values are negated. Returning the
    objective separately from the constraints keeps each piece testable.
    """
    return np.array([-float(p.expected_reach) for p in programs], dtype=float)


def regional_floor_constraints(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
) -> tuple[np.ndarray, np.ndarray]:
    """Inequality constraints ``A_ub @ x <= b_ub`` enforcing regional floors.

    Each region floor ``sum_{i in region} x_i >= floor * budget`` is expressed
    as ``-sum_{i in region} x_i <= -floor * budget``. Program ``min_share``
    requirements ``x_i >= min_share * budget`` are expressed as
    ``-x_i <= -min_share * budget``.
    """
    n = len(programs)
    index_by_id = {program.id: position for position, program in enumerate(programs)}
    rows: list[list[float]] = []
    lower_bounds: list[float] = []

    for region_floor in region_floors:
        row = [0.0] * n
        for program in programs:
            if program.region == region_floor.region:
                row[index_by_id[program.id]] = -1.0
        rows.append(row)
        lower_bounds.append(-float(region_floor.floor) * float(budget))

    for program in programs:
        if program.min_share > 0.0:
            row = [0.0] * n
            row[index_by_id[program.id]] = -1.0
            rows.append(row)
            lower_bounds.append(-float(program.min_share) * float(budget))

    if not rows:
        return (
            np.zeros((0, n), dtype=float),
            np.zeros(0, dtype=float),
        )
    return np.array(rows, dtype=float), np.array(lower_bounds, dtype=float)


def budget_constraint(programs: list[Program], budget: float) -> tuple[np.ndarray, np.ndarray]:
    """Equality constraint ``sum(x) == budget`` allocating the full budget."""
    n = len(programs)
    return np.ones((1, n), dtype=float), np.array([float(budget)], dtype=float)


def build_allocation_problem(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
) -> dict[str, object]:
    """Assemble the full linear program: objective, bounds, constraints.

    Returns a dictionary suitable for passing to :meth:`Solver.solve`.
    """
    validate(programs, region_floors)
    c = maximize_reach_objective(programs)
    A_ub, b_ub = regional_floor_constraints(programs, budget, region_floors)
    A_eq, b_eq = budget_constraint(programs, budget)
    bounds = [(0.0, None)] * len(programs)
    return {
        "c": c,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
    }


def solve_allocation(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
    *,
    solver: Solver | None = None,
) -> SolverResult:
    """Build and solve the regional-floor allocation problem.

    Uses the deterministic HiGHS solver adapter unless a different solver is
    supplied, demonstrating the P14 adapter consumed by the P13 constraints.
    """
    if solver is None:
        solver = HiGHSolver()
    problem = build_allocation_problem(programs, budget, region_floors)
    return solver.solve(
        problem["c"],  # type: ignore[arg-type]
        A_ub=problem["A_ub"],  # type: ignore[arg-type]
        b_ub=problem["b_ub"],  # type: ignore[arg-type]
        A_eq=problem["A_eq"],  # type: ignore[arg-type]
        b_eq=problem["b_eq"],  # type: ignore[arg-type]
        bounds=problem["bounds"],  # type: ignore[arg-type]
    )
