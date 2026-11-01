"""Per-program allocation constraints (issues #11 and #12).

Enforces:

- Maximum share caps (issue #11): no program receives more than its
  ``max_share`` fraction of the budget.
- Minimum share floors (issue #12): every program receives at least its
  ``min_share`` fraction of the budget.

Constraints are separate, testable pieces that sit on top of the
deterministic solver from issue #10. The optimizer recommends allocations;
it never signs or submits payouts.
"""

from __future__ import annotations

from dataclasses import dataclass

from funding_optimizer.errors import AllocationError, ValidationError
from funding_optimizer.models import Program

_EPSILON = 1e-9


@dataclass(frozen=True)
class CapConstraint:
    """Maximum share cap for a single program (issue #11).

    ``max_share`` is the largest fraction of the budget the optimizer may
    assign to ``program_id``. Must be in ``[0, 1]`` and >= the program's
    ``min_share``.
    """

    program_id: str
    max_share: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_share <= 1.0:
            raise ValidationError(
                f"program {self.program_id!r}: max_share must be in [0, 1], "
                f"got {self.max_share}"
            )


@dataclass(frozen=True)
class MinShareConstraint:
    """Minimum share floor for a single program (issue #12).

    ``min_share`` is the smallest fraction of the budget the optimizer must
    assign to ``program_id``. Must be in ``[0, 1]``.
    """

    program_id: str
    min_share: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_share <= 1.0:
            raise ValidationError(
                f"program {self.program_id!r}: min_share must be in [0, 1], "
                f"got {self.min_share}"
            )


class Constraints:
    """Collection of per-program caps and min-share floors.

    Holds :class:`CapConstraint` and :class:`MinShareConstraint` instances
    keyed by program id. Provides validation against a program list and
    application to a share allocation.
    """

    def __init__(
        self,
        caps: dict[str, CapConstraint] | None = None,
        min_shares: dict[str, MinShareConstraint] | None = None,
    ) -> None:
        self.caps = dict(caps or {})
        self.min_shares = dict(min_shares or {})

    def validate(self, programs: list[Program]) -> None:
        """Validate all constraints against the program list.

        Raises:
            ValidationError: When a constraint references an unknown
                program, a cap is below its program's min_share, or a cap
                and min_share disagree.
            AllocationError: When the combined min_share floors exceed
                100% of the budget (infeasible).
        """
        program_ids = {p.id for p in programs}

        # Constraints must reference known programs.
        for pid in self.caps:
            if pid not in program_ids:
                raise ValidationError(f"cap references unknown program {pid!r}")
        for pid in self.min_shares:
            if pid not in program_ids:
                raise ValidationError(f"min_share constraint references unknown program {pid!r}")

        # Cap must not be below the program's min_share.
        for pid in self.caps:
            if pid in self.min_shares:
                if self.caps[pid].max_share < self.min_shares[pid].min_share:
                    raise ValidationError(
                        f"program {pid!r}: max_share {self.caps[pid].max_share} "
                        f"< min_share {self.min_shares[pid].min_share}: infeasible"
                    )
            # Also check against the program's own min_share.
            program = next(p for p in programs if p.id == pid)
            if self.caps[pid].max_share < program.min_share:
                raise ValidationError(
                    f"program {pid!r}: max_share {self.caps[pid].max_share} "
                    f"< program min_share {program.min_share}: infeasible"
                )

        # Min-share constraint must not disagree with program's min_share.
        for pid in self.min_shares:
            program = next(p for p in programs if p.id == pid)
            if abs(self.min_shares[pid].min_share - program.min_share) > _EPSILON:
                raise ValidationError(
                    f"program {pid!r}: constraint min_share "
                    f"{self.min_shares[pid].min_share} != program min_share "
                    f"{program.min_share}"
                )

        # Combined min_share floors must not exceed 100%.
        total_min = sum(p.min_share for p in programs)
        if total_min > 1.0 + _EPSILON:
            raise AllocationError(
                f"min_share values sum to {total_min:.4f}, more than 100% " "of the budget"
            )

    def apply(self, shares: dict[str, float], programs: list[Program]) -> dict[str, float]:
        """Apply min-share floors then caps to a share allocation.

        Min shares are applied first (raising any share below its floor,
        taking the deficit from programs above their floor), then caps are
        applied (lowering any share above its cap, redistributing the excess
        proportionally to programs below their cap). The total share always
        sums to 1.0.

        Args:
            shares: Current per-program share allocation.
            programs: Program list, used to look up per-program min_share
                floors when no explicit MinShareConstraint exists.

        Returns:
            A new share dictionary with all constraints satisfied.
        """
        result = dict(shares)

        # 1. Apply min_share floors, taking deficit from programs above floor.
        for program in programs:
            pid = program.id
            floor = self.min_shares.get(pid, None)
            min_val = floor.min_share if floor else program.min_share
            current = result.get(pid, 0.0)
            if current < min_val:
                deficit = min_val - current
                result[pid] = min_val
                self._take_deficit(result, deficit, pid, programs)

        # 2. Apply caps, redistributing excess proportionally.
        for pid, cap in self.caps.items():
            if pid not in result:
                continue
            if result[pid] > cap.max_share:
                excess = result[pid] - cap.max_share
                result[pid] = cap.max_share
                self._redistribute_excess(result, excess, pid, programs)

        return result

    def _take_deficit(
        self,
        shares: dict[str, float],
        deficit: float,
        protected_pid: str,
        programs: list[Program],
    ) -> None:
        """Take ``deficit`` share from programs above their floor.

        Deficit is taken proportionally from each program's headroom above
        its own min_share floor, never from the program receiving the floor.
        """
        if deficit <= _EPSILON:
            return

        eligible: dict[str, float] = {}
        for program in programs:
            pid = program.id
            if pid == protected_pid:
                continue
            floor = self.min_shares.get(pid, None)
            min_val = floor.min_share if floor else program.min_share
            headroom = shares.get(pid, 0.0) - min_val
            if headroom > _EPSILON:
                eligible[pid] = max(headroom, 0.0)

        if not eligible:
            return

        total_headroom = sum(eligible.values())
        for pid, headroom in eligible.items():
            shares[pid] -= deficit * (headroom / total_headroom)

    def _redistribute_excess(
        self,
        shares: dict[str, float],
        excess: float,
        capped_pid: str,
        programs: list[Program],
    ) -> None:
        """Redistribute ``excess`` share from a capped program to others.

        Excess goes to programs that are not at their cap and not the
        program that was just capped, proportional to their headroom below
        their cap.
        """
        if excess <= _EPSILON:
            return

        eligible: dict[str, float] = {}
        for pid in shares:
            if pid == capped_pid:
                continue
            cap = self.caps.get(pid, None)
            headroom = (cap.max_share - shares[pid]) if cap else (1.0 - shares[pid])
            if headroom > _EPSILON:
                eligible[pid] = max(headroom, 0.0)

        if not eligible:
            return

        total_headroom = sum(eligible.values())
        for pid, headroom in eligible.items():
            shares[pid] += excess * (headroom / total_headroom)


__all__ = ["CapConstraint", "Constraints", "MinShareConstraint"]
