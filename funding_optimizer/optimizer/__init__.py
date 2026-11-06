"""Deterministic allocation optimizer building blocks.

The optimizer recommends allocations and never signs or submits payouts.
"""

from .constraints import (
    AssetConstraint,
    MultiAssetResult,
    Program,
    RegionFloor,
    build_allocation_problem,
    build_multi_asset_allocation_problem,
    maximize_reach_objective,
    multi_asset_bounds,
    multi_asset_budget_constraints,
    multi_asset_constraints,
    multi_asset_objective,
    regional_floor_constraints,
    solve_allocation,
    solve_multi_asset_allocation,
    validate_multi_asset,
)
from .diagnostics import (
    InfeasibilityDiagnostic,
    diagnose_infeasibility,
    explain_result,
)
from .solver import HiGHSolver, Solver, SolverResult, solve_with_lexicographic_tie_break

__all__ = [
    "AssetConstraint",
    "HiGHSolver",
    "InfeasibilityDiagnostic",
    "MultiAssetResult",
    "Program",
    "RegionFloor",
    "Solver",
    "SolverResult",
    "build_allocation_problem",
    "build_multi_asset_allocation_problem",
    "diagnose_infeasibility",
    "explain_result",
    "maximize_reach_objective",
    "multi_asset_bounds",
    "multi_asset_budget_constraints",
    "multi_asset_constraints",
    "multi_asset_objective",
    "regional_floor_constraints",
    "solve_allocation",
    "solve_multi_asset_allocation",
    "solve_with_lexicographic_tie_break",
    "validate_multi_asset",
]
