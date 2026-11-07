"""Tests for cohort-level allocation (issue #18).

The cohort allocator splits a budget across program cohorts, then hands each
cohort's share to its highest-reach member program. All expected values are
computed by hand from the small toy fixtures below.
"""

from __future__ import annotations

import numpy as np
import pytest

from funding_optimizer.optimizer.cohorts import Cohort, solve_cohort_allocation
from funding_optimizer.optimizer.constraints import Program


def _programs() -> list[Program]:
    return [
        Program(id="p1", region="Africa", expected_reach=10.0),
        Program(id="p2", region="Africa", expected_reach=5.0),
        Program(id="p3", region="Asia", expected_reach=8.0),
    ]


def _cohorts() -> list[Cohort]:
    return [
        Cohort(id="c1", program_ids=("p1", "p2"), min_share=0.20, max_share=0.80),
        Cohort(id="c2", program_ids=("p3",), min_share=0.0, max_share=1.0),
    ]


def test_cohort_allocation_respects_cohort_shares() -> None:
    programs = _programs()
    cohorts = _cohorts()
    result = solve_cohort_allocation(programs, cohorts, 100.0)

    assert result.success is True
    assert result.cohort_allocation is not None
    assert result.program_allocation is not None

    # c1 has the higher effective reach (10 vs 8) so it takes its maximum
    # share of 80; c2 receives the remaining 20.
    assert np.isclose(result.cohort_allocation["c1"], 80.0, atol=1e-6)
    assert np.isclose(result.cohort_allocation["c2"], 20.0, atol=1e-6)

    # Within each cohort, all funds go to the highest-reach member program.
    assert np.isclose(result.program_allocation["p1"], 80.0, atol=1e-6)
    assert np.isclose(result.program_allocation["p2"], 0.0, atol=1e-6)
    assert np.isclose(result.program_allocation["p3"], 20.0, atol=1e-6)

    # The full budget is allocated.
    assert np.isclose(sum(result.program_allocation.values()), 100.0, atol=1e-6)


def test_cohort_allocation_respects_min_shares() -> None:
    programs = _programs()
    cohorts = [
        # c1 must receive at least 40 even though c2 is equally reachable.
        Cohort(id="c1", program_ids=("p1",), min_share=0.40, max_share=1.0),
        Cohort(id="c2", program_ids=("p3",), min_share=0.0, max_share=1.0),
    ]
    result = solve_cohort_allocation(programs, cohorts, 100.0)
    assert result.success is True
    assert result.cohort_allocation is not None
    # c1 (reach 10) is preferred, but c2's min_share of 0 is non-binding, so
    # c1 still takes everything: both cohorts can reach 100 but c1's max is
    # 1.0 and c2 has no minimum, so the solver puts all 100 into c1.
    assert np.isclose(result.cohort_allocation["c1"], 100.0, atol=1e-6)
    assert np.isclose(result.cohort_allocation["c2"], 0.0, atol=1e-6)


def test_cohort_min_shares_exceeding_budget_raise() -> None:
    programs = _programs()
    cohorts = [
        Cohort(id="c1", program_ids=("p1",), min_share=0.60, max_share=0.80),
        Cohort(id="c2", program_ids=("p3",), min_share=0.60, max_share=1.0),
    ]
    with pytest.raises(ValueError, match="exceeds the whole budget"):
        solve_cohort_allocation(programs, cohorts, 100.0)


def test_cohort_unknown_program_raises() -> None:
    programs = _programs()
    cohorts = [Cohort(id="c1", program_ids=("p1", "nope"), min_share=0.0, max_share=1.0)]
    with pytest.raises(ValueError, match="unknown program"):
        solve_cohort_allocation(programs, cohorts, 100.0)


def test_cohort_duplicate_program_raises() -> None:
    programs = _programs()
    cohorts = [
        Cohort(id="c1", program_ids=("p1",), min_share=0.0, max_share=1.0),
        Cohort(id="c2", program_ids=("p1",), min_share=0.0, max_share=1.0),
    ]
    with pytest.raises(ValueError, match="multiple cohorts"):
        solve_cohort_allocation(programs, cohorts, 100.0)
