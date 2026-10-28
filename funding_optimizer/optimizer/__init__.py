"""Allocation optimizer package.

Re-exports the shared :class:`Optimizer` engine (now
:mod:`funding_optimizer.optimizer.engine`) plus objective-specific
solvers such as :mod:`funding_optimizer.optimizer.solver`.

The optimizer recommends allocations. It never signs or submits payouts.
"""

from __future__ import annotations

from funding_optimizer.optimizer.engine import FAIRNESS_MODES, OBJECTIVES, Optimizer
from funding_optimizer.optimizer.solver import (
    MaximizeReachSolveResult,
    MaximizeReachSolver,
    solve_maximize_reach,
)

__all__ = [
    "FAIRNESS_MODES",
    "MaximizeReachSolveResult",
    "MaximizeReachSolver",
    "OBJECTIVES",
    "Optimizer",
    "solve_maximize_reach",
]
