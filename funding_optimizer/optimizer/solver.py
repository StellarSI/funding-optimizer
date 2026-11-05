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

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a bounded linear program."""

    success: bool
    status: int
    message: str
    objective_value: float | None
    allocation: np.ndarray | None
    iterations: int | None = None
    tie_break_applied: bool = False
    """True when the allocation was canonicalized by lexicographic tie-breaking."""


class Solver:
    """Narrow adapter interface for deterministic linear programs."""

    def solve(
        self,
        c: np.ndarray,
        *,
        A_ub: np.ndarray | None = None,
        b_ub: np.ndarray | None = None,
        A_eq: np.ndarray | None = None,
        b_eq: np.ndarray | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None,
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
        time_limit: float | None = None,
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
        A_ub: np.ndarray | None = None,
        b_ub: np.ndarray | None = None,
        A_eq: np.ndarray | None = None,
        b_eq: np.ndarray | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None,
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

        allocation: np.ndarray | None = None
        if result.x is not None:
            allocation = np.asarray(result.x, dtype=float)

        objective_value: float | None = None
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


def solve_with_lexicographic_tie_break(
    solver: Solver,
    c: np.ndarray,
    *,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    tol: float = 1e-9,
) -> SolverResult:
    """Solve the LP, then canonicalize the allocation among equal-value optima.

    A linear program can admit many allocations with the same objective value,
    and the underlying solver may return any of them. This function refines
    the optimum deterministically: among all optimal allocations it prefers
    earlier-listed variables — maximizing ``x_0`` first, then ``x_1`` subject
    to that, and so on. Repeated calls with identical inputs therefore return
    identical allocations even when the LP has ties.

    The primary optimum is preserved up to ``tol``: an auxiliary constraint
    ``c @ x <= z + slack`` (with ``slack = tol * max(1, |z|)``) keeps every
    refinement step at optimality, and each already-chosen variable is held
    within the same tolerance. The returned allocation satisfies the original
    constraints to within solver tolerances.

    Failed or infeasible primary solves are returned unchanged with
    ``tie_break_applied=False``.
    """
    base = solver.solve(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not base.success or base.allocation is None or base.objective_value is None:
        return base

    coefficients = np.asarray(c, dtype=float)
    n = int(coefficients.size)
    optimum = float(base.objective_value)
    slack = tol * max(1.0, abs(optimum))
    bounds_list: list[tuple[float | None, float | None]] = (
        [(0.0, None)] * n if bounds is None else bounds
    )

    # Inequality system for refinement: original rows plus the optimality cut.
    rows: list[np.ndarray] = []
    limits: list[float] = []
    if A_ub is not None:
        a_original = np.asarray(A_ub, dtype=float)
        b_original = np.asarray(b_ub, dtype=float)
        for position in range(a_original.shape[0]):
            rows.append(a_original[position])
            limits.append(float(b_original[position]))
    rows.append(coefficients)
    limits.append(optimum + slack)

    fixed: list[tuple[int, float]] = []
    values = np.zeros(n, dtype=float)
    for i in range(n):
        # Maximize x_i: linprog minimizes, so the refinement objective is -e_i.
        unit = np.zeros(n, dtype=float)
        unit[i] = -1.0
        step_rows = list(rows)
        step_limits = list(limits)
        for j, chosen in fixed:
            eps = tol * max(1.0, abs(chosen))
            hold = np.zeros(n, dtype=float)
            hold[j] = 1.0
            step_rows.append(hold)
            step_limits.append(chosen + eps)
            step_rows.append(-hold)
            step_limits.append(-chosen + eps)
        refined = solver.solve(
            unit,
            A_ub=np.vstack(step_rows),
            b_ub=np.asarray(step_limits, dtype=float),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
        )
        if not refined.success or refined.allocation is None:
            # Refinement failed unexpectedly; fall back to the primary optimum.
            return base
        lower, upper = bounds_list[i]
        value = float(refined.allocation[i])
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        fixed.append((i, value))
        values[i] = value

    return SolverResult(
        success=True,
        status=0,
        message="optimal (lexicographic tie-break applied)",
        objective_value=float(coefficients @ values),
        allocation=values,
        tie_break_applied=True,
    )
