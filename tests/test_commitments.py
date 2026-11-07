"""Tests for committed / locked funds (issue #19).

Committed programs are locked to their commitment amount and the remaining
budget is allocated across the remaining programs. All expected values are
computed by hand from the small toy fixtures below.
"""

from __future__ import annotations

import numpy as np
import pytest

from funding_optimizer.optimizer.commitments import (
    Commitment,
    solve_committed_allocation,
)
from funding_optimizer.optimizer.constraints import Program, RegionFloor


def _programs() -> list[Program]:
    return [
        Program(id="a", region="Africa", expected_reach=10.0, min_share=0.0),
        Program(id="b", region="Africa", expected_reach=5.0, min_share=0.0),
        Program(id="c", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


def test_committed_allocation_respects_locked_funds() -> None:
    programs = _programs()
    commitments = [Commitment(program_id="a", amount=30.0)]
    region_floors = [
        RegionFloor(region="Africa", floor=0.40),
        RegionFloor(region="Asia", floor=0.20),
    ]
    result = solve_committed_allocation(programs, 100.0, commitments, region_floors)

    assert result.success is True
    assert result.program_allocation is not None

    # Program a is locked to its commitment of 30.
    assert np.isclose(result.program_allocation["a"], 30.0, atol=1e-6)

    # Remaining 70 is split between b and c. The Africa floor (40) is already
    # 30 covered by a's commitment, so b must contribute at least 10. c has
    # the higher reach (8 vs 5), so c takes the rest: b = 10, c = 60.
    assert np.isclose(result.program_allocation["b"], 10.0, atol=1e-6)
    assert np.isclose(result.program_allocation["c"], 60.0, atol=1e-6)

    # The full budget is allocated.
    assert np.isclose(sum(result.program_allocation.values()), 100.0, atol=1e-6)


def test_committed_allocation_region_floor_covered_by_commitment() -> None:
    programs = _programs()
    # a's commitment of 50 already covers the Africa floor (40), so b is free
    # to take 0 and c takes the remaining 50.
    commitments = [Commitment(program_id="a", amount=50.0)]
    region_floors = [RegionFloor(region="Africa", floor=0.40)]
    result = solve_committed_allocation(programs, 100.0, commitments, region_floors)
    assert result.success is True
    assert np.isclose(result.program_allocation["a"], 50.0, atol=1e-6)
    assert np.isclose(result.program_allocation["b"], 0.0, atol=1e-6)
    assert np.isclose(result.program_allocation["c"], 50.0, atol=1e-6)


def test_commitments_exceeding_budget_raise() -> None:
    programs = _programs()
    commitments = [
        Commitment(program_id="a", amount=60.0),
        Commitment(program_id="b", amount=50.0),
    ]
    with pytest.raises(ValueError, match="exceeds the budget"):
        solve_committed_allocation(programs, 100.0, commitments)


def test_commitment_to_unknown_program_raises() -> None:
    programs = _programs()
    commitments = [Commitment(program_id="nope", amount=10.0)]
    with pytest.raises(ValueError, match="unknown program"):
        solve_committed_allocation(programs, 100.0, commitments)


def test_negative_commitment_raises() -> None:
    programs = _programs()
    commitments = [Commitment(program_id="a", amount=-5.0)]
    with pytest.raises(ValueError, match="non-negative"):
        solve_committed_allocation(programs, 100.0, commitments)
