"""Sensitivity analysis for caps, minimums, and fairness constraints.

One-at-a-time sweep of every lever in the allocation linear program:

* ``region_floor`` — the fairness floors from :class:`RegionFloor`
  (parameter name ``region_floor:<region>``),
* ``program_min_share`` — each program's minimum share of the budget
  (``program_min_share:<program id>``),
* ``program_cap`` — optional per-program absorption caps supplied as a
  mapping of program id to the maximum fraction of the budget the program
  may receive (``program_cap:<program id>``). Caps are passed as an upper
  bound on the program's allocation variable; they do not change the
  objective or the constraints.

For every parameter the analysis re-solves the problem at ``base - d`` and
``base + d`` (clamped to ``[0, 1]``, with the clamped value reported in the
entry) and records the objective change. Solver objectives are negated
reach, as elsewhere in the package: a positive ``objective_delta`` means the
perturbation cost you reach, a negative one means it bought reach.

A constraint is *binding* at the base optimum when, within solver tolerance,
it holds with equality at the base allocation. Binding parameters are the
ones worth watching: loosening one does nothing, tightening one moves the
objective. Non-binding parameters move the objective by zero until they
become binding.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from .constraints import (
    Program,
    RegionFloor,
    build_allocation_problem,
    solve_allocation,
    validate,
)
from .diagnostics import diagnose_infeasibility
from .solver import HiGHSolver, Solver, solve_with_lexicographic_tie_break

_TOLERANCE = 1e-9

_KIND_FLOOR = "region_floor"
_KIND_MIN_SHARE = "program_min_share"
_KIND_CAP = "program_cap"


@dataclass(frozen=True)
class SensitivityEntry:
    """How one parameter perturbation moved the optimum."""

    parameter: str
    """``<kind>:<name>``, e.g. ``"region_floor:Africa"``."""
    kind: str
    base_value: float
    perturbed_value: float
    perturbation: float
    """The signed delta applied (``-perturbation`` or ``+perturbation``)."""
    objective_value: float | None
    objective_delta: float | None
    """Perturbed minus base objective (negated reach); ``None`` on failure."""
    success: bool
    binding: bool
    """True when the constraint holds with equality at the base optimum."""
    message: str = ""


@dataclass(frozen=True)
class SensitivityReport:
    """Full sensitivity sweep of a base allocation problem."""

    budget: float
    base_objective_value: float | None
    base_allocation: dict[str, float] | None
    perturbation: float
    entries: tuple[SensitivityEntry, ...]
    message: str = ""

    @property
    def binding_parameters(self) -> tuple[str, ...]:
        """Parameters binding at the base optimum, in sweep order, deduplicated."""
        return tuple(dict.fromkeys(entry.parameter for entry in self.entries if entry.binding))


def validate_caps(
    programs: Sequence[Program],
    caps: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate per-program budget caps and normalize them to floats.

    Raises :class:`ValueError` when a cap references an unknown program or a
    fraction outside ``[0, 1]``, or :class:`TypeError` for non-numeric caps.
    """
    normalized: dict[str, float] = {}
    if caps is None:
        return normalized
    program_ids = {program.id for program in programs}
    for program_id, value in caps.items():
        if program_id not in program_ids:
            raise ValueError(f"caps references unknown program {program_id!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"caps[{program_id!r}] must be a real number, got {value!r}")
        cap = float(value)
        if not isfinite(cap) or not 0.0 <= cap <= 1.0:
            raise ValueError(f"caps[{program_id!r}] {cap} is not in [0, 1]")
        normalized[program_id] = cap
    return normalized


def _allocate(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
    caps: Mapping[str, float],
    *,
    solver: Solver,
    tie_break: bool,
) -> tuple[bool, float | None, dict[str, float] | None, str]:
    """Solve the (possibly capped) problem; return (success, objective, allocation, message)."""
    try:
        problem = build_allocation_problem(programs, budget, region_floors)
    except ValueError as exc:
        return False, None, None, str(exc)

    if caps:
        bounds = list(problem["bounds"])  # type: ignore[arg-type]
        for position, program in enumerate(programs):
            cap = caps.get(program.id)
            if cap is None:
                continue
            lower, _upper = bounds[position]
            bounds[position] = (lower if lower is not None else 0.0, cap * float(budget))
        problem["bounds"] = bounds
        if tie_break:
            result = solve_with_lexicographic_tie_break(
                solver,
                problem["c"],  # type: ignore[arg-type]
                A_ub=problem["A_ub"],  # type: ignore[arg-type]
                b_ub=problem["b_ub"],  # type: ignore[arg-type]
                A_eq=problem["A_eq"],  # type: ignore[arg-type]
                b_eq=problem["b_eq"],  # type: ignore[arg-type]
                bounds=problem["bounds"],  # type: ignore[arg-type]
            )
        else:
            result = solver.solve(
                problem["c"],  # type: ignore[arg-type]
                A_ub=problem["A_ub"],  # type: ignore[arg-type]
                b_ub=problem["b_ub"],  # type: ignore[arg-type]
                A_eq=problem["A_eq"],  # type: ignore[arg-type]
                b_eq=problem["b_eq"],  # type: ignore[arg-type]
                bounds=problem["bounds"],  # type: ignore[arg-type]
            )
    else:
        solved = solve_allocation(
            programs, budget, region_floors, solver=solver, tie_break=tie_break
        )
        result = solved

    if not result.success or result.allocation is None:
        reasons = diagnose_infeasibility(programs, budget, region_floors)
        if reasons:
            detail = "; ".join(f"{diagnostic.code}: {diagnostic.message}" for diagnostic in reasons)
            return False, None, None, detail
        return False, None, None, result.message

    allocation = {
        program.id: float(result.allocation[position]) for position, program in enumerate(programs)
    }
    objective = None if result.objective_value is None else float(result.objective_value)
    return True, objective, allocation, result.message


def constraint_slacks(
    programs: Sequence[Program],
    budget: float,
    allocation: Mapping[str, float],
    region_floors: Sequence[RegionFloor],
    caps: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Slack of every swept constraint at a given allocation.

    For floors and minimums, slack is ``allocated - required`` (positive
    means headroom). For caps, slack is ``cap * budget - allocated`` (unused
    capacity). Slack near zero means the constraint is tight. Raises
    :class:`ValueError` when a program is missing from ``allocation``.
    """
    allocation_by_id: dict[str, float] = {}
    for program in programs:
        if program.id not in allocation:
            raise ValueError(f"allocation is missing program {program.id!r}")
        allocation_by_id[program.id] = float(allocation[program.id])

    budget_value = float(budget)
    slacks: dict[str, float] = {}
    for region_floor in region_floors:
        allocated = sum(
            allocation_by_id[program.id]
            for program in programs
            if program.region == region_floor.region
        )
        slacks[f"{_KIND_FLOOR}:{region_floor.region}"] = (
            allocated - region_floor.floor * budget_value
        )
    for program in programs:
        slacks[f"{_KIND_MIN_SHARE}:{program.id}"] = (
            allocation_by_id[program.id] - program.min_share * budget_value
        )
    for program_id, cap in (caps or {}).items():
        slacks[f"{_KIND_CAP}:{program_id}"] = cap * budget_value - allocation_by_id[program_id]
    return slacks


def binding_constraints(
    programs: Sequence[Program],
    budget: float,
    allocation: Mapping[str, float],
    region_floors: Sequence[RegionFloor],
    caps: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Names of the swept constraints that are tight at ``allocation``.

    Minimums and floors bind only when the requirement is positive (a zero
    requirement with a zero allocation is slack, not a binding constraint);
    caps bind whenever allocation reaches the cap.
    """
    caps_normalized = validate_caps(programs, caps)
    slacks = constraint_slacks(programs, budget, allocation, region_floors, caps_normalized)
    bound: list[str] = []
    for region_floor in region_floors:
        key = f"{_KIND_FLOOR}:{region_floor.region}"
        if region_floor.floor > _TOLERANCE and abs(slacks[key]) <= 1e-6:
            bound.append(key)
    for program in programs:
        key = f"{_KIND_MIN_SHARE}:{program.id}"
        if program.min_share > _TOLERANCE and abs(slacks[key]) <= 1e-6:
            bound.append(key)
    for program_id in caps_normalized:
        key = f"{_KIND_CAP}:{program_id}"
        if abs(slacks[key]) <= 1e-6:
            bound.append(key)
    return tuple(bound)


def sensitivity_analysis(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor] | None = None,
    *,
    caps: Mapping[str, float] | None = None,
    perturbation: float = 0.1,
    solver: Solver | None = None,
    tie_break: bool = False,
) -> SensitivityReport:
    """Sweep every swept parameter and report how the optimum moves.

    Re-solves the allocation problem at ``parameter ± perturbation`` (clamped
    to ``[0, 1]``) and records the objective change for each. Regional floors
    that push the combined requirements over 100% of the budget, or minimums
    that conflict with caps, yield ``success=False`` entries with the
    diagnostic attached instead of raising.
    """
    floors = [] if region_floors is None else list(region_floors)
    budget_value = float(budget)
    if not isfinite(budget_value) or budget_value <= 0.0:
        raise ValueError(f"budget must be a positive, finite number, got {budget_value!r}")
    if not isfinite(perturbation) or perturbation <= 0.0:
        raise ValueError(f"perturbation must be a positive, finite number, got {perturbation!r}")
    if not programs:
        raise ValueError("at least one program is required")
    validate(list(programs), floors)
    caps_normalized = validate_caps(programs, caps)
    if solver is None:
        solver = HiGHSolver()

    base_success, base_objective, base_allocation, base_message = _allocate(
        list(programs),
        budget_value,
        floors,
        caps_normalized,
        solver=solver,
        tie_break=tie_break,
    )
    if base_allocation is None:
        binding: tuple[str, ...] = ()
    else:
        binding = binding_constraints(
            programs, budget_value, base_allocation, floors, caps_normalized
        )

    # Parameter order is deterministic: floors in input order, then programs
    # in input order, then caps ordered by program position.
    parameters: list[tuple[str, str, float]] = [
        (f"{_KIND_FLOOR}:{floor.region}", _KIND_FLOOR, floor.floor) for floor in floors
    ]
    parameters += [
        (f"{_KIND_MIN_SHARE}:{program.id}", _KIND_MIN_SHARE, program.min_share)
        for program in programs
    ]
    program_order = {program.id: position for position, program in enumerate(programs)}
    parameters += [
        (f"{_KIND_CAP}:{program_id}", _KIND_CAP, caps_normalized[program_id])
        for program_id in sorted(caps_normalized, key=lambda pid: program_order[pid])
    ]

    entries: list[SensitivityEntry] = []
    for parameter, kind, base_value in parameters:
        binding_here = parameter in binding
        for signed_delta in (-perturbation, perturbation):
            perturbed_value = min(1.0, max(0.0, base_value + signed_delta))
            perturbed_floors = [
                RegionFloor(floor.region, perturbed_value)
                if kind == _KIND_FLOOR and f"{_KIND_FLOOR}:{floor.region}" == parameter
                else floor
                for floor in floors
            ]
            perturbed_programs = [
                Program(
                    id=program.id,
                    region=program.region,
                    expected_reach=program.expected_reach,
                    min_share=(
                        perturbed_value
                        if kind == _KIND_MIN_SHARE
                        and f"{_KIND_MIN_SHARE}:{program.id}" == parameter
                        else program.min_share
                    ),
                )
                for program in programs
            ]
            perturbed_caps = dict(caps_normalized)
            if kind == _KIND_CAP:
                program_id = parameter.split(":", 1)[1]
                perturbed_caps[program_id] = perturbed_value

            success, objective, _allocation, message = _allocate(
                perturbed_programs,
                budget_value,
                perturbed_floors,
                perturbed_caps,
                solver=solver,
                tie_break=tie_break,
            )
            if success and objective is not None and base_objective is not None:
                delta: float | None = objective - base_objective
            else:
                delta = None
            entries.append(
                SensitivityEntry(
                    parameter=parameter,
                    kind=kind,
                    base_value=base_value,
                    perturbed_value=perturbed_value,
                    perturbation=signed_delta,
                    objective_value=objective if success else None,
                    objective_delta=delta,
                    success=success,
                    binding=binding_here,
                    message="" if success else message,
                )
            )

    message = "optimal" if base_success else f"base problem is infeasible: {base_message}"
    return SensitivityReport(
        budget=budget_value,
        base_objective_value=base_objective,
        base_allocation=base_allocation,
        perturbation=float(perturbation),
        entries=tuple(entries),
        message=message,
    )
