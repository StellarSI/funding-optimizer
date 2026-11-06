"""Fairness and budget constraints for the allocation optimizer.

Builds the linear pieces (objective, inequality constraints, equality
constraints) of a budget-allocation linear program, and small helpers that
solve it using the deterministic :class:`~funding_optimizer.optimizer.solver.HiGHSolver`
adapter from issue P14.

Supports both a single-asset float budget (:func:`build_allocation_problem`)
and a multi-asset :class:`~funding_optimizer.models.Budget`
(:func:`build_multi_asset_allocation_problem`) with optional per-asset
constraints (:class:`AssetConstraint`).

All constraints keep the optimizer's boundary: it recommends allocations and
never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from funding_optimizer.models import Budget, validate_asset_code

from .solver import HiGHSolver, Solver, SolverResult, solve_with_lexicographic_tie_break


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
    tie_break: bool = False,
) -> SolverResult:
    """Build and solve the regional-floor allocation problem.

    Uses the deterministic HiGHS solver adapter unless a different solver is
    supplied, demonstrating the P14 adapter consumed by the P13 constraints.

    With ``tie_break=True``, allocations that tie on objective value are
    canonicalized lexicographically (earlier-listed programs are funded
    first), so repeated runs return identical allocations even when the LP
    has multiple optima.
    """
    if solver is None:
        solver = HiGHSolver()
    problem = build_allocation_problem(programs, budget, region_floors)
    if tie_break:
        return solve_with_lexicographic_tie_break(
            solver,
            problem["c"],  # type: ignore[arg-type]
            A_ub=problem["A_ub"],  # type: ignore[arg-type]
            b_ub=problem["b_ub"],  # type: ignore[arg-type]
            A_eq=problem["A_eq"],  # type: ignore[arg-type]
            b_eq=problem["b_eq"],  # type: ignore[arg-type]
            bounds=problem["bounds"],  # type: ignore[arg-type]
        )
    return solver.solve(
        problem["c"],  # type: ignore[arg-type]
        A_ub=problem["A_ub"],  # type: ignore[arg-type]
        b_ub=problem["b_ub"],  # type: ignore[arg-type]
        A_eq=problem["A_eq"],  # type: ignore[arg-type]
        b_eq=problem["b_eq"],  # type: ignore[arg-type]
        bounds=problem["bounds"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class AssetConstraint:
    """Per-asset constraint for one tranche of a multi-asset budget.

    Bounds how much of ``asset``'s tranche ``program_id`` may or must receive,
    expressed as fractions of that tranche. ``min_share`` behaves like
    :attr:`Program.min_share` but is scoped to a single asset; ``max_share``
    caps the program's slice of the tranche. The default (``0.0``/``1.0``)
    leaves the program unconstrained within the tranche.
    """

    asset: str
    program_id: str
    min_share: float = 0.0
    max_share: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", validate_asset_code(self.asset))
        if not isinstance(self.program_id, str):
            raise TypeError(f"AssetConstraint program_id must be a string, got {self.program_id!r}")
        program_id = self.program_id.strip()
        if not program_id:
            raise ValueError("AssetConstraint program_id must be a non-empty string")
        object.__setattr__(self, "program_id", program_id)
        for label, value in (("min_share", self.min_share), ("max_share", self.max_share)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"AssetConstraint {label} must be a real number, got {value!r}")
            fraction = float(value)
            if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"AssetConstraint {label} {fraction} is not in [0, 1]")
            object.__setattr__(self, label, fraction)
        if self.min_share > self.max_share:
            raise ValueError(
                f"AssetConstraint min_share {self.min_share} exceeds max_share {self.max_share}"
            )


@dataclass(frozen=True)
class MultiAssetResult:
    """Outcome of a multi-asset allocation run."""

    success: bool
    status: int
    message: str
    assets: tuple[str, ...]
    objective_value: float | None
    allocation: dict[str, dict[str, float]] | None
    """Nested ``{asset: {program_id: amount}}`` map; ``None`` when the solve failed."""

    iterations: int | None = None


def validate_multi_asset(
    programs: list[Program],
    budget: Budget,
    region_floors: list[RegionFloor],
    asset_constraints: list[AssetConstraint] | tuple[AssetConstraint, ...] = (),
) -> None:
    """Validate a multi-asset allocation problem structurally.

    Raises :class:`ValueError` when the budget is not a :class:`Budget`, when
    no programs are given, when a region floor or program share is invalid,
    or when an asset constraint references an unknown asset or program or is
    duplicated.
    """
    validate(programs, region_floors)
    if not isinstance(budget, Budget):
        raise TypeError(f"budget must be a Budget instance, got {type(budget).__name__}")
    if not programs:
        raise ValueError("at least one program is required")

    program_ids = {program.id for program in programs}
    seen: set[tuple[str, str]] = set()
    for constraint in asset_constraints:
        if constraint.asset not in budget.amounts:
            raise ValueError(
                f"AssetConstraint asset {constraint.asset!r} is not part of the budget"
            )
        if constraint.program_id not in program_ids:
            raise ValueError(
                f"AssetConstraint program_id {constraint.program_id!r} does not match any program"
            )
        key = (constraint.asset, constraint.program_id)
        if key in seen:
            raise ValueError(
                f"Duplicate AssetConstraint for asset {constraint.asset!r} "
                f"and program {constraint.program_id!r}"
            )
        seen.add(key)


def multi_asset_objective(programs: list[Program], budget: Budget) -> np.ndarray:
    """Objective coefficients for maximizing reach across all asset tranches.

    Variables are ordered asset-major (every program for the first asset,
    then every program for the second, ...); the negated per-unit reach is
    tiled once per asset. ``expected_reach`` is treated as per unit of any
    asset - per-asset exchange rates are out of scope.
    """
    base = maximize_reach_objective(programs)
    return np.tile(base, len(budget.assets))


def multi_asset_budget_constraints(
    programs: list[Program],
    budget: Budget,
) -> tuple[np.ndarray, np.ndarray]:
    """Equality constraints ``A_eq @ x == b_eq`` allocating each tranche fully.

    One row per asset: ``sum_{i in programs} x[a, i] == budget[a]``.
    """
    n = len(programs)
    total = n * len(budget.assets)
    rows = np.zeros((len(budget.assets), total), dtype=float)
    limits = np.zeros(len(budget.assets), dtype=float)
    for tranche, asset in enumerate(budget.assets):
        offset = tranche * n
        rows[tranche, offset : offset + n] = 1.0
        limits[tranche] = float(budget.amounts[asset])
    return rows, limits


def multi_asset_constraints(
    programs: list[Program],
    budget: Budget,
    region_floors: list[RegionFloor],
    asset_constraints: list[AssetConstraint] | tuple[AssetConstraint, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Inequality constraints ``A_ub @ x <= b_ub`` per asset tranche.

    Enforces, within every tranche: regional floors (``sum_{i in region} x >=
    floor * tranche``), program ``min_share`` minimums, and
    :class:`AssetConstraint` ``min_share`` minimums. Each is expressed as
    ``<=`` rows, mirroring :func:`regional_floor_constraints`.
    """
    n = len(programs)
    total = n * len(budget.assets)
    index_by_id = {program.id: position for position, program in enumerate(programs)}
    rows: list[list[float]] = []
    limits: list[float] = []

    for tranche, asset in enumerate(budget.assets):
        scale = float(budget.amounts[asset])
        offset = tranche * n
        for region_floor in region_floors:
            row = [0.0] * total
            for program in programs:
                if program.region == region_floor.region:
                    row[offset + index_by_id[program.id]] = -1.0
            rows.append(row)
            limits.append(-float(region_floor.floor) * scale)
        for program in programs:
            if program.min_share > 0.0:
                row = [0.0] * total
                row[offset + index_by_id[program.id]] = -1.0
                rows.append(row)
                limits.append(-float(program.min_share) * scale)
        for constraint in asset_constraints:
            if constraint.asset != asset or constraint.min_share <= 0.0:
                continue
            row = [0.0] * total
            row[offset + index_by_id[constraint.program_id]] = -1.0
            rows.append(row)
            limits.append(-float(constraint.min_share) * scale)

    if not rows:
        return (
            np.zeros((0, total), dtype=float),
            np.zeros(0, dtype=float),
        )
    return np.array(rows, dtype=float), np.array(limits, dtype=float)


def multi_asset_bounds(
    programs: list[Program],
    budget: Budget,
    asset_constraints: list[AssetConstraint] | tuple[AssetConstraint, ...] = (),
) -> list[tuple[float, float | None]]:
    """Per-variable bounds, asset-major.

    Each variable is capped by its tranche amount; an :class:`AssetConstraint`
    ``max_share`` tightens that cap to ``max_share * tranche``.
    """
    caps: dict[tuple[str, str], float] = {}
    for constraint in asset_constraints:
        if constraint.max_share < 1.0:
            caps[(constraint.asset, constraint.program_id)] = constraint.max_share

    bounds: list[tuple[float, float | None]] = []
    for asset in budget.assets:
        scale = float(budget.amounts[asset])
        for program in programs:
            cap_share = caps.get((asset, program.id))
            upper = scale if cap_share is None else min(scale, cap_share * scale)
            bounds.append((0.0, upper))
    return bounds


def build_multi_asset_allocation_problem(
    programs: list[Program],
    budget: Budget,
    region_floors: list[RegionFloor] | None = None,
    asset_constraints: list[AssetConstraint] | tuple[AssetConstraint, ...] = (),
) -> dict[str, object]:
    """Assemble the multi-asset linear program: objective, bounds, constraints.

    Returns a dictionary suitable for passing to :meth:`Solver.solve`, plus an
    ``assets`` key carrying the tranche order of the variables.
    """
    floors = [] if region_floors is None else region_floors
    validate_multi_asset(programs, budget, floors, asset_constraints)
    c = multi_asset_objective(programs, budget)
    a_ub, b_ub = multi_asset_constraints(programs, budget, floors, asset_constraints)
    a_eq, b_eq = multi_asset_budget_constraints(programs, budget)
    bounds = multi_asset_bounds(programs, budget, asset_constraints)
    return {
        "c": c,
        "A_ub": a_ub,
        "b_ub": b_ub,
        "A_eq": a_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "assets": budget.assets,
    }


def solve_multi_asset_allocation(
    programs: list[Program],
    budget: Budget,
    region_floors: list[RegionFloor] | None = None,
    asset_constraints: list[AssetConstraint] | tuple[AssetConstraint, ...] = (),
    *,
    solver: Solver | None = None,
    tie_break: bool = False,
) -> MultiAssetResult:
    """Build and solve the multi-asset allocation problem.

    Tranches are balanced by independent equality constraints; floors,
    minimums and per-asset bounds apply within each tranche. Uses the
    deterministic HiGHS adapter unless a different solver is supplied.

    With ``tie_break=True``, tied optima are canonicalized lexicographically
    in variable order (asset-major, earlier-listed programs first).
    """
    floors = [] if region_floors is None else region_floors
    validate_multi_asset(programs, budget, floors, asset_constraints)
    if solver is None:
        solver = HiGHSolver()
    problem = build_multi_asset_allocation_problem(programs, budget, floors, asset_constraints)
    if tie_break:
        solved = solve_with_lexicographic_tie_break(
            solver,
            problem["c"],  # type: ignore[arg-type]
            A_ub=problem["A_ub"],  # type: ignore[arg-type]
            b_ub=problem["b_ub"],  # type: ignore[arg-type]
            A_eq=problem["A_eq"],  # type: ignore[arg-type]
            b_eq=problem["b_eq"],  # type: ignore[arg-type]
            bounds=problem["bounds"],  # type: ignore[arg-type]
        )
    else:
        solved = solver.solve(
            problem["c"],  # type: ignore[arg-type]
            A_ub=problem["A_ub"],  # type: ignore[arg-type]
            b_ub=problem["b_ub"],  # type: ignore[arg-type]
            A_eq=problem["A_eq"],  # type: ignore[arg-type]
            b_eq=problem["b_eq"],  # type: ignore[arg-type]
            bounds=problem["bounds"],  # type: ignore[arg-type]
        )

    assets = budget.assets
    if not solved.success or solved.allocation is None:
        return MultiAssetResult(
            success=False,
            status=solved.status,
            message=solved.message,
            assets=assets,
            objective_value=solved.objective_value,
            allocation=None,
            iterations=solved.iterations,
        )

    n = len(programs)
    allocation: dict[str, dict[str, float]] = {}
    for tranche, asset in enumerate(assets):
        offset = tranche * n
        allocation[asset] = {
            program.id: float(solved.allocation[offset + position])
            for position, program in enumerate(programs)
        }
    return MultiAssetResult(
        success=True,
        status=solved.status,
        message=solved.message,
        assets=assets,
        objective_value=solved.objective_value,
        allocation=allocation,
        iterations=solved.iterations,
    )
