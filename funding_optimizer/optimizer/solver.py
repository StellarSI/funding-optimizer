"""Deterministic linear-programming solver adapter.

Wraps SciPy's HiGHS LP solver behind a narrow, deterministic adapter interface.
Solver options are pinned so that repeated calls with identical inputs produce
identical outputs, making constrained allocation runs reproducible and auditable.

SciPy/HiGHS is fully isolated behind this adapter: no other module imports
``scipy`` directly. The optimizer recommends allocations and never signs or
submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a bounded linear program."""

    success: bool
    status: int
    message: str
    objective_value: Optional[float]
    allocation: Optional[np.ndarray]
    iterations: Optional[int] = None


class Solver:
    """Narrow adapter interface for deterministic linear programs."""

    def solve(
        self,
        c: np.ndarray,
        *,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    ) -> SolverResult:
        """Solve ``min c^T x`` subject to the supplied linear constraints.

        Parameters
        ----------
        c:
            Linear objective coefficients (minimized). To maximize reach, pass
            the negative of the per-unit reach values.
        A_ub, b_ub:
            Inequality constraints ``A_ub @ x <= b_ub``.
        A_eq, b_eq:
            Equality constraints ``A_eq @ x == b_eq``.
        bounds:
            Per-variable ``(min, max)`` bounds. Use ``None`` for an unbounded
            end. When omitted, every variable defaults to ``[0, None)``, which
            is appropriate for allocation amounts.
        """
        raise NotImplementedError


class HiGHSolver(Solver):
    """Deterministic HiGHS-backed linear-programming solver.

    Pins HiGHS options so repeated runs are reproducible. SciPy/HiGHS is
    isolated behind the :class:`Solver` adapter interface.
    """

    def __init__(
        self,
        *,
        mip_rel_gap: float = 0.0,
        presolve: bool = True,
        time_limit: Optional[float] = None,
        dual_feasibility_tolerance: float = 1e-9,
        primal_feasibility_tolerance: float = 1e-9,
        ipm_optimality_tolerance: float = 1e-9,
    ) -> None:
        self._options: dict[str, object] = {
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": bool(presolve),
        }
        if time_limit is not None:
            self._options["time_limit"] = float(time_limit)
        self._options["dual_feasibility_tolerance"] = float(dual_feasibility_tolerance)
        self._options["primal_feasibility_tolerance"] = float(primal_feasibility_tolerance)
        self._options["ipm_optimality_tolerance"] = float(ipm_optimality_tolerance)

    def solve(
        self,
        c: np.ndarray,
        *,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    ) -> SolverResult:
        if bounds is None:
            bounds = [(0.0, None)] * int(np.asarray(c).size)

        result = linprog(
            c=np.asarray(c, dtype=float),
            A_ub=None if A_ub is None else np.asarray(A_ub, dtype=float),
            b_ub=None if b_ub is None else np.asarray(b_ub, dtype=float),
            A_eq=None if A_eq is None else np.asarray(A_eq, dtype=float),
            b_eq=None if b_eq is None else np.asarray(b_eq, dtype=float),
            bounds=bounds,
            method="highs",
            options=self._options,
        )

        allocation: Optional[np.ndarray] = None
        if result.x is not None:
            allocation = np.asarray(result.x, dtype=float)

        objective_value: Optional[float] = None
        if result.fun is not None:
            objective_value = float(result.fun)

        return SolverResult(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            objective_value=objective_value,
            allocation=allocation,
            iterations=getattr(result, "nit", None),
        )
