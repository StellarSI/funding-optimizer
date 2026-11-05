"""Actionable infeasibility diagnostics for the allocation optimizer.

When the regional-floor allocation problem cannot be solved, callers need to
know *why* and *what to change*. This module inspects the constraint set
structurally and returns diagnostics that name the offending region or
program, quantify the excess, and suggest a concrete relaxation. It never
raises for constraint data it can describe; raising on malformed inputs
remains :func:`funding_optimizer.optimizer.constraints.validate`'s job.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .constraints import Program, RegionFloor
from .solver import SolverResult

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class InfeasibilityDiagnostic:
    """One actionable reason the allocation problem has no solution."""

    code: str
    """Machine-readable identifier, e.g. ``"floors_exceed_budget"``."""

    message: str
    """Human-readable explanation of the conflict."""

    suggestion: str
    """Concrete change that would restore feasibility."""

    region: str | None = None
    """Region concerned, when the diagnostic is region-scoped."""

    program_id: str | None = None
    """Program concerned, when the diagnostic is program-scoped."""

    excess: float | None = None
    """Budget fraction by which the requirement overshoots capacity, if any."""


def required_region_fractions(
    programs: list[Program],
    region_floors: list[RegionFloor],
) -> dict[str, float]:
    """Minimum budget fraction each region must receive.

    A region's requirement is the larger of its explicit floor and the sum of
    its programs' ``min_share`` values — both constraints apply, so the
    binding one wins. Regions appear only when they have a floor or at least
    one program.
    """
    regions = sorted(
        {program.region for program in programs} | {floor.region for floor in region_floors}
    )
    floor_by_region = {floor.region: floor.floor for floor in region_floors}
    required: dict[str, float] = {}
    for region in regions:
        program_minimum = sum(program.min_share for program in programs if program.region == region)
        required[region] = max(floor_by_region.get(region, 0.0), program_minimum)
    return required


def total_required_fraction(
    programs: list[Program],
    region_floors: list[RegionFloor],
) -> float:
    """Total budget fraction the floors and program minimums commit."""
    return sum(required_region_fractions(programs, region_floors).values())


def diagnose_infeasibility(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
) -> list[InfeasibilityDiagnostic]:
    """Explain why the constraint set admits no allocation.

    Returns one diagnostic per actionable finding. An empty list means the
    structural analysis found nothing wrong — the infeasibility then lives
    elsewhere (see :func:`explain_result` for the solver-reported fallback).
    """
    budget_value = float(budget)
    if not isfinite(budget_value) or budget_value <= 0.0:
        return [
            InfeasibilityDiagnostic(
                code="invalid_budget",
                message=f"Budget {budget_value} is not a positive, finite amount.",
                suggestion="Use a strictly positive, finite budget.",
            )
        ]

    for program in programs:
        if not isfinite(program.min_share) or not 0.0 <= program.min_share <= 1.0:
            return [
                InfeasibilityDiagnostic(
                    code="invalid_program_min_share",
                    message=(
                        f"Program {program.id!r} min_share {program.min_share} " "is not in [0, 1]."
                    ),
                    suggestion=(
                        f"Set program {program.id!r} min_share to a fraction between 0 and 1."
                    ),
                    program_id=program.id,
                )
            ]
    for floor in region_floors:
        if not isfinite(floor.floor) or not 0.0 <= floor.floor <= 1.0:
            return [
                InfeasibilityDiagnostic(
                    code="invalid_region_floor",
                    message=(f"Region {floor.region!r} floor {floor.floor} is not in [0, 1]."),
                    suggestion=(
                        f"Set region {floor.region!r}'s floor to a fraction between 0 and 1."
                    ),
                    region=floor.region,
                )
            ]

    diagnostics: list[InfeasibilityDiagnostic] = []

    program_regions = {program.region for program in programs}
    for floor in region_floors:
        if floor.floor > 0.0 and floor.region not in program_regions:
            diagnostics.append(
                InfeasibilityDiagnostic(
                    code="region_has_no_programs",
                    message=(
                        f"Region {floor.region!r} has a floor of {floor.floor:.4g} "
                        "but no programs to allocate to."
                    ),
                    suggestion=(f"Add a program in region {floor.region!r} or drop its floor."),
                    region=floor.region,
                )
            )

    required = required_region_fractions(programs, region_floors)
    total = sum(required.values())
    if total > 1.0 + _TOLERANCE:
        excess = total - 1.0
        diagnostics.append(
            InfeasibilityDiagnostic(
                code="floors_exceed_budget",
                message=(
                    "Regional floors and program minimums commit at least "
                    f"{total:.4g} of the budget, but only 1.0 is available."
                ),
                suggestion=(
                    "Lower floors and program minimums by a total of at least "
                    f"{excess:.4g} of the budget."
                ),
                excess=excess,
            )
        )
        for region, requirement in required.items():
            max_allowed = 1.0 - (total - requirement)
            if requirement > max_allowed + _TOLERANCE:
                diagnostics.append(
                    InfeasibilityDiagnostic(
                        code="region_requirement_above_capacity",
                        message=(
                            f"Region {region!r} requires at least {requirement:.4g} of "
                            f"the budget, but at most {max_allowed:.4g} remains after "
                            "every other region's minimums."
                        ),
                        suggestion=(
                            f"Lower region {region!r}'s requirement to at most "
                            f"{max_allowed:.4g}, or lower the other regions' floors "
                            f"by at least {excess:.4g} in total."
                        ),
                        region=region,
                        excess=requirement - max_allowed,
                    )
                )

    # Program-level view: a single min_share can overshoot the room left by
    # every other requirement, which the region summary alone does not name.
    floor_by_region = {floor.region: floor.floor for floor in region_floors}
    for program in programs:
        own_region = program.region
        others_in_region = sum(
            other.min_share
            for other in programs
            if other.region == own_region and other.id != program.id
        )
        own_region_without = max(floor_by_region.get(own_region, 0.0), others_in_region)
        other_regions = sum(
            requirement for region, requirement in required.items() if region != own_region
        )
        max_allowed = 1.0 - own_region_without - other_regions
        if program.min_share > max_allowed + _TOLERANCE:
            diagnostics.append(
                InfeasibilityDiagnostic(
                    code="program_min_share_above_capacity",
                    message=(
                        f"Program {program.id!r} requires at least "
                        f"{program.min_share:.4g} of the budget, but at most "
                        f"{max_allowed:.4g} remains after every other minimum."
                    ),
                    suggestion=(
                        f"Lower program {program.id!r}'s min_share to at most "
                        f"{max_allowed:.4g}, or lower the other floors and minimums."
                    ),
                    program_id=program.id,
                    excess=program.min_share - max_allowed,
                )
            )

    return diagnostics


def explain_result(
    result: SolverResult,
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
) -> list[InfeasibilityDiagnostic]:
    """Diagnostics for a failed solve; empty when the solve succeeded.

    Structural causes are reported first; when the solver fails without an
    identifiable structural cause, a fallback diagnostic carries the solver's
    own status and message.
    """
    if result.success:
        return []

    diagnostics = diagnose_infeasibility(programs, budget, region_floors)
    if diagnostics:
        return diagnostics

    if result.status == 3:
        return [
            InfeasibilityDiagnostic(
                code="unbounded_objective",
                message=f"The solver reported an unbounded objective: {result.message}",
                suggestion=(
                    "Bound the allocations with max_share caps or per-variable "
                    "upper bounds and retry."
                ),
            )
        ]
    return [
        InfeasibilityDiagnostic(
            code="solver_infeasible",
            message=f"The solver reported status {result.status}: {result.message}",
            suggestion=(
                "Relax a constraint (region floor or program min_share) or "
                "increase the budget and retry."
            ),
        )
    ]
