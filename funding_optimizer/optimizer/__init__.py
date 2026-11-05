"""Deterministic allocation optimizer building blocks.

The optimizer recommends allocations and never signs or submits payouts.
"""

from .constraints import (
    Program,
    RegionFloor,
    build_allocation_problem,
    maximize_reach_objective,
    regional_floor_constraints,
    solve_allocation,
)
from .diagnostics import (
    InfeasibilityDiagnostic,
    diagnose_infeasibility,
    explain_result,
)
from .solver import HiGHSolver, Solver, SolverResult, solve_with_lexicographic_tie_break

__all__ = [
    "HiGHSolver",
    "InfeasibilityDiagnostic",
    "Program",
    "RegionFloor",
    "Solver",
    "SolverResult",
    "build_allocation_problem",
    "diagnose_infeasibility",
    "explain_result",
    "maximize_reach_objective",
    "regional_floor_constraints",
    "solve_allocation",
    "solve_with_lexicographic_tie_break",
]
