"""Tests for per-program constraints (issues #11 and #12).

Covers CapConstraint (max share caps), MinShareConstraint (minimum share
floors), and the Constraints collection (validate + apply with
proportional redistribution). No network access, no real recipient data.
"""

from __future__ import annotations

import pytest

from funding_optimizer.errors import AllocationError, ValidationError
from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import (
    CapConstraint,
    Constraints,
    MinShareConstraint,
    Optimizer,
)


def _programs() -> list[Program]:
    return [
        Program(id="a", min_share=0.0, max_share=1.0, expected_reach=100, region="r1"),
        Program(id="b", min_share=0.0, max_share=1.0, expected_reach=50, region="r1"),
        Program(id="c", min_share=0.0, max_share=1.0, expected_reach=25, region="r2"),
    ]


# ---------------------------------------------------------------
# CapConstraint (issue #11)
# ---------------------------------------------------------------


def test_cap_constraint_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="max_share must be in"):
        CapConstraint(program_id="a", max_share=1.5)
    with pytest.raises(ValidationError, match="max_share must be in"):
        CapConstraint(program_id="a", max_share=-0.1)


def test_cap_constraint_is_applied_by_optimizer() -> None:
    budget = Budget(total=1000.0, asset="USDC")
    programs = _programs()
    constraints = Constraints(caps={"a": CapConstraint(program_id="a", max_share=0.2)})
    plan = Optimizer(objective="maximize_reach").solve(budget, programs, constraints=constraints)
    amounts = {a.program_id: a.amount for a in plan.allocations}
    # Program a capped at 20% of 1000 = 200
    assert amounts["a"] == pytest.approx(200.0, abs=1.0)
    assert plan.allocated_total == pytest.approx(1000.0, abs=1.0)


# ---------------------------------------------------------------
# MinShareConstraint (issue #12)
# ---------------------------------------------------------------


def test_min_share_constraint_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="min_share must be in"):
        MinShareConstraint(program_id="a", min_share=1.5)
    with pytest.raises(ValidationError, match="min_share must be in"):
        MinShareConstraint(program_id="a", min_share=-0.1)


def test_min_share_constraint_is_applied_by_optimizer() -> None:
    budget = Budget(total=1000.0, asset="USDC")
    programs = [
        Program(id="a", min_share=0.4, expected_reach=10, region="r1"),
        Program(id="b", min_share=0.0, expected_reach=100, region="r1"),
        Program(id="c", min_share=0.0, expected_reach=25, region="r2"),
    ]
    constraints = Constraints(min_shares={"a": MinShareConstraint(program_id="a", min_share=0.4)})
    plan = Optimizer(objective="maximize_reach").solve(budget, programs, constraints=constraints)
    amounts = {a.program_id: a.amount for a in plan.allocations}
    assert amounts["a"] >= 400.0  # floor is a minimum; optimizer may give more


# ---------------------------------------------------------------
# Constraints validation
# ---------------------------------------------------------------


def test_constraints_reject_unknown_program() -> None:
    constraints = Constraints(caps={"zzz": CapConstraint(program_id="zzz", max_share=0.5)})
    with pytest.raises(ValidationError, match="unknown program"):
        constraints.validate(_programs())


def test_constraints_cap_below_min_share_rejected() -> None:
    programs = [
        Program(id="a", min_share=0.5, max_share=1.0, expected_reach=100),
    ]
    constraints = Constraints(caps={"a": CapConstraint(program_id="a", max_share=0.2)})
    with pytest.raises(ValidationError, match="infeasible"):
        constraints.validate(programs)


def test_constraints_combined_floors_exceed_100_percent() -> None:
    programs = [
        Program(id="a", min_share=0.6, max_share=1.0, expected_reach=100),
        Program(id="b", min_share=0.6, max_share=1.0, expected_reach=50),
    ]
    constraints = Constraints()
    with pytest.raises(AllocationError, match="more than 100%"):
        constraints.validate(programs)


# ---------------------------------------------------------------
# Constraints.apply redistribution
# ---------------------------------------------------------------


def test_apply_redistributes_excess_from_capped_program() -> None:
    constraints = Constraints(caps={"a": CapConstraint(program_id="a", max_share=0.2)})
    shares = {"a": 0.8, "b": 0.1, "c": 0.1}
    result = constraints.apply(shares, _programs())
    assert result["a"] == pytest.approx(0.2)
    # Excess 0.6 redistributed to b and c proportionally to headroom
    assert result["b"] > 0.1
    assert result["c"] > 0.1
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_raises_floors_then_caps() -> None:
    constraints = Constraints(
        caps={"a": CapConstraint(program_id="a", max_share=0.2)},
        min_shares={"c": MinShareConstraint(program_id="c", min_share=0.3)},
    )
    shares = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = constraints.apply(shares, _programs())
    assert result["a"] == pytest.approx(0.2)
    assert result["c"] >= 0.3  # floor is a minimum, capping may add more
    assert sum(result.values()) == pytest.approx(1.0)
