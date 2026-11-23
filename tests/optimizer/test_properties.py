"""Property-based tests for linear program allocation conservation, caps, and fairness invariants.

Fulfills issue #84 (Tests: add property-based tests for optimizer conservation, caps, and fairness).
Uses Hypothesis to generate hundreds of pseudo-randomized problem configurations, ensuring
that the solver strictly preserves:
1. Total Budget Conservation: sum(allocations) == budget (within precision tolerance).
2. Non-Negativity: All allocations >= 0.
3. Minimum Share Invariance: Each program receives at least min_share * budget.
4. Regional Floor Compliance: Sum of allocations per region >= floor * budget.
5. Infeasible Input Rejection: Over-constrained problems fail gracefully without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from funding_optimizer.models import Budget
from funding_optimizer.optimizer.constraints import (
    AssetConstraint,
    Program,
    RegionFloor,
    solve_allocation,
    solve_multi_asset_allocation,
)

REGIONS = ["Africa", "Asia", "LatinAmerica", "Europe", "MiddleEast"]


@st.composite
def allocation_problem_strategy(draw: st.DrawFn) -> tuple[list[Program], float, list[RegionFloor]]:
    """Generates valid, non-trivial allocation problems with random budget and regional floors."""
    budget = draw(
        st.floats(min_value=1_000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
    )
    num_programs = draw(st.integers(min_value=2, max_value=6))

    # Generate distinct programs across known regions
    programs = []
    used_regions = set()
    for i in range(num_programs):
        region = draw(st.sampled_from(REGIONS))
        used_regions.add(region)
        reach = draw(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        )
        # Keep individual min_share modest so the sum across programs is feasible
        min_share = draw(
            st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
        )
        programs.append(
            Program(id=f"prog-{i}", region=region, expected_reach=reach, min_share=min_share)
        )

    # Generate feasible regional floors for a subset of used regions
    floors = []
    available_regions = list(used_regions)
    if available_regions:
        num_floors = draw(st.integers(min_value=0, max_value=min(2, len(available_regions))))
        selected_regions = draw(
            st.lists(
                st.sampled_from(available_regions),
                min_size=num_floors,
                max_size=num_floors,
                unique=True,
            )
        )
        for r in selected_regions:
            floor_val = draw(
                st.floats(min_value=0.05, max_value=0.25, allow_nan=False, allow_infinity=False)
            )
            floors.append(RegionFloor(region=r, floor=floor_val))

    return programs, budget, floors


@settings(max_examples=100, deadline=None)
@given(allocation_problem_strategy())
def test_property_budget_conservation(data: tuple[list[Program], float, list[RegionFloor]]) -> None:
    """Property: If problem is feasible, the total allocated capital must strictly equal total budget."""
    programs, budget, floors = data
    try:
        result = solve_allocation(programs, budget, floors)
    except ValueError:
        # Infeasible floor combinations correctly raised by build_allocation_problem
        return

    if result.success:
        assert result.allocation is not None
        total_allocated = float(np.sum(result.allocation))
        assert np.isclose(total_allocated, budget, rtol=1e-5, atol=1e-4)


@settings(max_examples=100, deadline=None)
@given(allocation_problem_strategy())
def test_property_non_negativity_and_min_shares(
    data: tuple[list[Program], float, list[RegionFloor]],
) -> None:
    """Property: No allocation is negative, and each program receives at least its min_share * budget."""
    programs, budget, floors = data
    try:
        result = solve_allocation(programs, budget, floors)
    except ValueError:
        return

    if result.success and result.allocation is not None:
        for p, alloc in zip(programs, result.allocation, strict=True):
            assert alloc >= -1e-6, f"Negative allocation detected for program {p.id}: {alloc}"
            min_expected = p.min_share * budget
            assert (
                alloc >= min_expected - 1e-4
            ), f"Program {p.id} received {alloc}, less than min_share {p.min_share} * {budget} = {min_expected}"


@settings(max_examples=100, deadline=None)
@given(allocation_problem_strategy())
def test_property_regional_fairness_floors(
    data: tuple[list[Program], float, list[RegionFloor]],
) -> None:
    """Property: Sum of allocations in any region with a floor must meet or exceed floor * budget."""
    programs, budget, floors = data
    try:
        result = solve_allocation(programs, budget, floors)
    except ValueError:
        return

    if result.success and result.allocation is not None:
        regional_allocations: dict[str, float] = {}
        for p, alloc in zip(programs, result.allocation, strict=True):
            regional_allocations[p.region] = regional_allocations.get(p.region, 0.0) + float(alloc)

        for f in floors:
            required_floor_amount = f.floor * budget
            actual_region_amount = regional_allocations.get(f.region, 0.0)
            assert (
                actual_region_amount >= required_floor_amount - 1e-4
            ), f"Region {f.region} received {actual_region_amount}, below floor {required_floor_amount}"


def test_adversarial_impossible_floors_fail_closed() -> None:
    """Edge Case: Impossibly high regional floors (> 1.0) must be rejected fail-closed."""
    programs = [
        Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.0),
        Program(id="p2", region="Asia", expected_reach=5.0, min_share=0.0),
    ]
    floors = [
        RegionFloor(region="Africa", floor=0.70),
        RegionFloor(region="Asia", floor=0.45),  # Sum = 1.15 > 1.0 (Infeasible)
    ]
    with pytest.raises(ValueError, match="exceeds the whole budget"):
        solve_allocation(programs, 1000.0, floors)


@settings(max_examples=50, deadline=None)
@given(
    allocation_problem_strategy(),
    st.floats(min_value=0.20, max_value=0.80, allow_nan=False, allow_infinity=False),
)
def test_property_multi_asset_upper_caps_respected(
    data: tuple[list[Program], float, list[RegionFloor]],
    cap_share: float,
) -> None:
    """Property: Per-asset upper caps (max_share) are never exceeded when solving multi-asset allocations."""
    programs, budget_val, floors = data
    budget = Budget(amounts={"USDC": budget_val, "XLM": budget_val * 0.5})
    capped_program = programs[0]
    if cap_share < capped_program.min_share:
        return

    constraints = [
        AssetConstraint(asset="USDC", program_id=capped_program.id, max_share=cap_share),
    ]
    try:
        result = solve_multi_asset_allocation(
            programs, budget, floors, asset_constraints=constraints
        )
    except ValueError:
        return

    if result.success and result.allocation is not None and "USDC" in result.allocation:
        usdc_alloc = float(result.allocation["USDC"][capped_program.id])
        max_allowed = cap_share * budget_val
        assert (
            usdc_alloc <= max_allowed + 1e-4
        ), f"Program {capped_program.id} USDC allocation {usdc_alloc} exceeded cap {max_allowed}"
