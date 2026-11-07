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
from .schedule import (
    BudgetPhase,
    BudgetSchedule,
    PhaseAllocation,
    PhasedAllocationResult,
    build_phase_problem,
    solve_phased_allocation,
    validate_phase_caps,
)
from .sensitivity import (
    SensitivityEntry,
    SensitivityReport,
    binding_constraints,
    constraint_slacks,
    sensitivity_analysis,
    validate_caps,
)
from .solver import HiGHSolver, Solver, SolverResult, solve_with_lexicographic_tie_break

__all__ = [
    "AssetConstraint",
    "BudgetPhase",
    "BudgetSchedule",
    "HiGHSolver",
    "InfeasibilityDiagnostic",
    "MultiAssetResult",
    "PhaseAllocation",
    "PhasedAllocationResult",
    "Program",
    "RegionFloor",
    "SensitivityEntry",
    "SensitivityReport",
    "Solver",
    "SolverResult",
    "binding_constraints",
    "build_allocation_problem",
    "build_multi_asset_allocation_problem",
    "build_phase_problem",
    "constraint_slacks",
    "diagnose_infeasibility",
    "explain_result",
    "maximize_reach_objective",
    "multi_asset_bounds",
    "multi_asset_budget_constraints",
    "multi_asset_constraints",
    "multi_asset_objective",
    "regional_floor_constraints",
    "sensitivity_analysis",
    "solve_allocation",
    "solve_multi_asset_allocation",
    "solve_phased_allocation",
    "solve_with_lexicographic_tie_break",
    "validate_caps",
    "validate_multi_asset",
    "validate_phase_caps",
]
