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
from .solver import HiGHSolver, Solver, SolverResult

__all__ = [
    "Solver",
    "HiGHSolver",
    "SolverResult",
    "Program",
    "RegionFloor",
    "maximize_reach_objective",
    "regional_floor_constraints",
    "build_allocation_problem",
    "solve_allocation",
]
