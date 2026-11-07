"""Phased allocation for budget schedules.

Grant budgets rarely arrive as one lump sum: disbursement schedules deliver
the budget in phases (monthly, quarterly, ...). This module allocates each
phase's tranche with the same deterministic optimizer used for one-shot
allocations, and threads the result across phases so that absorption-limited
phases roll their unallocated remainder forward into the next phase.

The model, per phase, is a linear program over the variables
``x_0 .. x_{n-1}`` (amounts to send to each program) and ``s`` (budget
carried out of the phase):

* ``sum(x) + s == available`` where ``available = phase amount + carried in``
  (the full available amount must be either allocated or carried).
* regional floors and program ``min_share`` minimums as fractions of
  ``available`` (same constraints as the single-asset problem).
* optional per-program absorption caps (``phase_caps``): a program may take at
  most ``cap * available`` in a phase. Caps below the budget force the
  remainder into ``s``.

Phases are solved in schedule order; ``s`` from phase *t* becomes budget
available in phase *t + 1*. The optimizer recommends allocations and never
signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .constraints import (
    Program,
    RegionFloor,
    maximize_reach_objective,
    regional_floor_constraints,
    validate,
)
from .diagnostics import InfeasibilityDiagnostic, diagnose_infeasibility
from .solver import HiGHSolver, Solver, solve_with_lexicographic_tie_break

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BudgetPhase:
    """One phase of a budget schedule: an id and the amount that arrives."""

    id: str
    amount: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError(f"BudgetPhase id must be a string, got {self.id!r}")
        phase_id = self.id.strip()
        if not phase_id:
            raise ValueError("BudgetPhase id must be a non-empty string")
        object.__setattr__(self, "id", phase_id)
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)):
            raise TypeError(f"BudgetPhase amount must be a real number, got {self.amount!r}")
        amount = float(self.amount)
        if not isfinite(amount) or amount <= 0.0:
            raise ValueError(f"BudgetPhase amount must be a positive, finite number, got {amount!r}")
        object.__setattr__(self, "amount", amount)


@dataclass(frozen=True)
class BudgetSchedule:
    """A validated, ordered sequence of budget phases.

    Phases keep the order they are declared in, which is the order in which
    the optimizer allocates them and carries budget forward. The ``total``
    property sums the scheduled amounts (not the carried-forward remainder).
    """

    phases: tuple[BudgetPhase, ...]

    def __init__(self, phases: Iterable[BudgetPhase]) -> None:
        materialized = tuple(phases)
        if not materialized:
            raise ValueError("BudgetSchedule requires at least one phase")
        seen: set[str] = set()
        for phase in materialized:
            if not isinstance(phase, BudgetPhase):
                raise TypeError(f"BudgetSchedule phases must be BudgetPhase, got {phase!r}")
            if phase.id in seen:
                raise ValueError(f"Duplicate phase id {phase.id!r} in budget schedule")
            seen.add(phase.id)
        object.__setattr__(self, "phases", materialized)

    @property
    def total(self) -> float:
        """Total scheduled amount across all phases."""
        return float(sum(phase.amount for phase in self.phases))

    @property
    def phase_ids(self) -> tuple[str, ...]:
        """Phase ids in schedule order."""
        return tuple(phase.id for phase in self.phases)


@dataclass(frozen=True)
class PhaseAllocation:
    """Outcome of allocating one phase of the schedule."""

    phase_id: str
    available: float
    """Budget available in this phase: the scheduled amount plus any budget
    carried in from the previous phase."""
    carry_in: float
    carry_out: float
    """Amount not allocated in this phase and carried into the next one."""
    allocation: dict[str, float] | None
    """Program id -> amount allocated in this phase; ``None`` on failure."""
    objective_value: float | None
    success: bool
    message: str
    diagnostics: tuple[InfeasibilityDiagnostic, ...] = ()


@dataclass(frozen=True)
class PhasedAllocationResult:
    """Outcome of a phased allocation run across a whole schedule."""

    success: bool
    phases: tuple[PhaseAllocation, ...]
    cumulative_allocation: dict[str, float]
    """Program id -> total amount allocated across all solved phases."""
    total_budget: float
    """Sum of the scheduled phase amounts."""
    total_allocated: float
    """Sum of the amounts allocated across all solved phases."""
    unallocated: float | None
    """Budget carried out of the final phase; ``None`` when a phase failed."""
    objective_value: float | None
    """Sum of the per-phase solver objectives (negated reach, as elsewhere)."""
    message: str
    tie_break_applied: bool = False


def validate_phase_caps(
    programs: Sequence[Program],
    phase_caps: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate per-program absorption caps and normalize them to floats.

    Raises :class:`ValueError` when a cap references an unknown program or a
    fraction outside ``[0, 1]``, or :class:`TypeError` for non-numeric caps.
    """
    caps: dict[str, float] = {}
    if phase_caps is None:
        return caps
    program_ids = {program.id for program in programs}
    for program_id, value in phase_caps.items():
        if program_id not in program_ids:
            raise ValueError(f"phase_caps references unknown program {program_id!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"phase_caps[{program_id!r}] must be a real number, got {value!r}")
        cap = float(value)
        if not isfinite(cap) or not 0.0 <= cap <= 1.0:
            raise ValueError(f"phase_caps[{program_id!r}] {cap} is not in [0, 1]")
        caps[program_id] = cap
    return caps


def build_phase_problem(
    programs: Sequence[Program],
    available: float,
    region_floors: Sequence[RegionFloor],
    phase_caps: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Assemble the per-phase linear program: objective, bounds, constraints.

    Variables are ordered program-major, with the carry variable last. The
    returned dictionary is suitable for passing to :meth:`Solver.solve`.
    """
    caps = validate_phase_caps(programs, phase_caps)
    n = len(programs)
    c = np.concatenate([maximize_reach_objective(list(programs)), np.zeros(1)])

    a_ub, b_ub = regional_floor_constraints(list(programs), float(available), list(region_floors))
    if a_ub.shape[0]:
        a_ub = np.hstack([a_ub, np.zeros((a_ub.shape[0], 1), dtype=float)])
    else:
        a_ub = np.zeros((0, n + 1), dtype=float)

    a_eq = np.ones((1, n + 1), dtype=float)
    b_eq = np.array([float(available)], dtype=float)

    bounds: list[tuple[float, float]] = [
        (0.0, caps.get(program.id, 1.0) * float(available)) for program in programs
    ]
    bounds.append((0.0, float(available)))  # carry variable

    return {"c": c, "A_ub": a_ub, "b_ub": b_ub, "A_eq": a_eq, "b_eq": b_eq, "bounds": bounds}


def _cap_diagnostics(
    programs: Sequence[Program],
    phase_caps: Mapping[str, float],
) -> list[InfeasibilityDiagnostic]:
    diagnostics: list[InfeasibilityDiagnostic] = []
    for program in programs:
        cap = phase_caps.get(program.id)
        if cap is not None and program.min_share > cap + _TOLERANCE:
            diagnostics.append(
                InfeasibilityDiagnostic(
                    code="phase_cap_below_min_share",
                    message=(
                        f"Program {program.id!r} requires at least {program.min_share:.4g} "
                        f"of the phase but its absorption cap is {cap:.4g}."
                    ),
                    suggestion=(
                        f"Raise program {program.id!r}'s phase cap to at least "
                        f"{program.min_share:.4g}, or lower its min_share."
                    ),
                    program_id=program.id,
                    excess=program.min_share - cap,
                )
            )
    return diagnostics


def solve_phased_allocation(
    programs: list[Program],
    schedule: BudgetSchedule,
    region_floors: list[RegionFloor] | None = None,
    *,
    phase_caps: Mapping[str, float] | None = None,
    solver: Solver | None = None,
    tie_break: bool = False,
) -> PhasedAllocationResult:
    """Solve a phased allocation across a budget schedule.

    Programs absorb at most ``phase_caps[program_id]`` (a fraction of the
    phase's available budget) per phase; the remainder is carried into the
    next phase. Regional floors and program ``min_share`` minimums apply to
    each phase's available budget. Uses the deterministic HiGHS adapter
    unless a different solver is supplied.

    If a phase is infeasible, the run stops at that phase: later phases
    depend on the carried budget, so they are not attempted. The returned
    result carries ``success=False`` with the per-phase diagnostics attached
    to the failing phase.
    """
    floors = [] if region_floors is None else list(region_floors)
    if not isinstance(schedule, BudgetSchedule):
        raise TypeError(f"schedule must be a BudgetSchedule, got {type(schedule).__name__}")
    if not programs:
        raise ValueError("at least one program is required")
    validate(list(programs), floors)
    caps = validate_phase_caps(programs, phase_caps)
    if solver is None:
        solver = HiGHSolver()

    phases: list[PhaseAllocation] = []
    cumulative = {program.id: 0.0 for program in programs}
    total_allocated = 0.0
    objective_total = 0.0
    tie_break_seen = False
    carry_in = 0.0

    for phase in schedule.phases:
        available = phase.amount + carry_in
        problem = build_phase_problem(programs, available, floors, caps)
        kwargs: dict[str, object] = {
            "A_ub": problem["A_ub"],
            "b_ub": problem["b_ub"],
            "A_eq": problem["A_eq"],
            "b_eq": problem["b_eq"],
            "bounds": problem["bounds"],
        }
        if tie_break:
            solved = solve_with_lexicographic_tie_break(
                solver,
                problem["c"],  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
            tie_break_seen = tie_break_seen or solved.tie_break_applied
        else:
            solved = solver.solve(
                problem["c"],  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )

        if not solved.success or solved.allocation is None:
            diagnostics = diagnose_infeasibility(list(programs), available, floors)
            diagnostics.extend(_cap_diagnostics(programs, caps))
            phases.append(
                PhaseAllocation(
                    phase_id=phase.id,
                    available=available,
                    carry_in=carry_in,
                    carry_out=carry_in + phase.amount,
                    allocation=None,
                    objective_value=None,
                    success=False,
                    message=solved.message,
                    diagnostics=tuple(diagnostics),
                )
            )
            return PhasedAllocationResult(
                success=False,
                phases=tuple(phases),
                cumulative_allocation=cumulative,
                total_budget=schedule.total,
                total_allocated=total_allocated,
                unallocated=None,
                objective_value=None,
                message=(
                    f"Phase {phase.id!r} is infeasible; later phases were not attempted."
                ),
                tie_break_applied=tie_break_seen,
            )

        n = len(programs)
        allocation = {
            program.id: float(solved.allocation[position]) for position, program in enumerate(programs)
        }
        carry_out = float(solved.allocation[n])
        for program_id, amount in allocation.items():
            cumulative[program_id] += amount
        total_allocated += sum(allocation.values())
        if solved.objective_value is not None:
            objective_total += float(solved.objective_value)
        phases.append(
            PhaseAllocation(
                phase_id=phase.id,
                available=available,
                carry_in=carry_in,
                carry_out=carry_out,
                allocation=allocation,
                objective_value=solved.objective_value,
                success=True,
                message=solved.message,
            )
        )
        carry_in = carry_out

    return PhasedAllocationResult(
        success=True,
        phases=tuple(phases),
        cumulative_allocation=cumulative,
        total_budget=schedule.total,
        total_allocated=total_allocated,
        unallocated=carry_in,
        objective_value=objective_total,
        message="optimal",
        tie_break_applied=tie_break_seen,
    )
