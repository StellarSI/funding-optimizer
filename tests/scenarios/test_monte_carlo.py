"""Scenario comparison and Monte Carlo determinism tests (issue #87).

Verifies:
1. Scenario comparison between baseline and policy/budget variants.
2. Seeded Monte Carlo determinism: repeated runs with the same seed yield identical outputs.
3. Monte Carlo invariant preservation (budget conservation across all sampled runs).
4. Edge cases and fail-closed validation for scenario configurations.
"""

from __future__ import annotations

import math

import pytest

from funding_optimizer.optimizer.constraints import Program, RegionFloor
from funding_optimizer.scenarios import (
    MonteCarloConfig,
    ScenarioConfig,
    compare_scenarios,
    run_monte_carlo_simulation,
)


@pytest.fixture
def base_programs() -> list[Program]:
    return [
        Program(id="p-edu", region="Africa", expected_reach=20.0, min_share=0.10),
        Program(id="p-health", region="Africa", expected_reach=15.0, min_share=0.05),
        Program(id="p-solar", region="Asia", expected_reach=18.0, min_share=0.15),
    ]


@pytest.fixture
def base_floors() -> list[RegionFloor]:
    return [
        RegionFloor(region="Africa", floor=0.40),
        RegionFloor(region="Asia", floor=0.30),
    ]


def test_scenario_comparison_evaluates_correctly(
    base_programs: list[Program], base_floors: list[RegionFloor]
) -> None:
    """Verifies comparative evaluation across multiple budget and constraint scenarios."""
    scenarios = [
        ScenarioConfig(
            name="conservative",
            budget=100_000.0,
            programs=base_programs,
            region_floors=base_floors,
        ),
        ScenarioConfig(
            name="expanded",
            budget=200_000.0,
            programs=base_programs,
            region_floors=base_floors,
        ),
    ]

    comparison = compare_scenarios(scenarios)
    assert len(comparison.scenarios) == 2
    assert comparison.scenarios[0].success is True
    assert comparison.scenarios[1].success is True

    # Higher budget achieves strictly positive additional reach gain
    delta = comparison.delta_reach("conservative", "expanded")
    assert delta > 0
    assert comparison.scenarios[1].budget > comparison.scenarios[0].budget


def test_monte_carlo_strict_determinism(
    base_programs: list[Program], base_floors: list[RegionFloor]
) -> None:
    """Verifies that running Monte Carlo with the same seed produces identical results."""
    mc_cfg = MonteCarloConfig(num_trials=50, seed=42, reach_volatility=0.15)

    run_1 = run_monte_carlo_simulation(base_programs, 100_000.0, base_floors, mc_cfg)
    run_2 = run_monte_carlo_simulation(base_programs, 100_000.0, base_floors, mc_cfg)

    assert run_1.num_trials == run_2.num_trials
    assert run_1.success_rate == run_2.success_rate
    assert run_1.mean_objective == pytest.approx(run_2.mean_objective, abs=1e-9)
    assert run_1.std_objective == pytest.approx(run_2.std_objective, abs=1e-9)
    assert run_1.p05_objective == pytest.approx(run_2.p05_objective, abs=1e-9)
    assert run_1.p50_objective == pytest.approx(run_2.p50_objective, abs=1e-9)
    assert run_1.p95_objective == pytest.approx(run_2.p95_objective, abs=1e-9)

    for prog_id in ["p-edu", "p-health", "p-solar"]:
        assert run_1.per_program_mean_allocation[prog_id] == pytest.approx(
            run_2.per_program_mean_allocation[prog_id], abs=1e-6
        )


def test_monte_carlo_budget_conservation_invariants(
    base_programs: list[Program], base_floors: list[RegionFloor]
) -> None:
    """Verifies that across all trials, mean allocations strictly sum to total budget."""
    total_budget = 150_000.0
    mc_cfg = MonteCarloConfig(num_trials=40, seed=123, reach_volatility=0.20)

    summary = run_monte_carlo_simulation(base_programs, total_budget, base_floors, mc_cfg)
    assert summary.success_rate == 1.0

    total_allocated = sum(summary.per_program_mean_allocation.values())
    assert total_allocated == pytest.approx(total_budget, rel=1e-4)


def test_scenario_validation_boundaries() -> None:
    """Verifies fail-closed behavior for malformed scenario configs."""
    with pytest.raises(ValueError, match="Scenario name cannot be empty"):
        ScenarioConfig(name="", budget=1000.0, programs=[], region_floors=[])

    with pytest.raises(ValueError, match="Scenario budget must be positive"):
        ScenarioConfig(name="test", budget=-10.0, programs=[], region_floors=[])

    with pytest.raises(ValueError, match="Scenario must contain at least one program"):
        ScenarioConfig(name="test", budget=100.0, programs=[], region_floors=[])

    with pytest.raises(ValueError, match="Must provide at least one scenario"):
        compare_scenarios([])

    from funding_optimizer.scenarios.engine import ScenarioComparison, ScenarioResult

    infeasible_comparison = ScenarioComparison(
        scenarios=[
            ScenarioResult(
                name="valid",
                budget=100.0,
                success=True,
                objective_value=50.0,
                allocations={},
                regional_totals={},
            ),
            ScenarioResult(
                name="failed",
                budget=100.0,
                success=False,
                objective_value=None,
                allocations={},
                regional_totals={},
            ),
        ]
    )
    with pytest.raises(ValueError, match="is infeasible or unoptimized"):
        infeasible_comparison.delta_reach("valid", "failed")
    with pytest.raises(ValueError, match="is infeasible or unoptimized"):
        infeasible_comparison.delta_reach("failed", "valid")


def test_monte_carlo_validation_boundaries(base_programs: list[Program]) -> None:
    """Verifies parameter validation on Monte Carlo configuration."""
    with pytest.raises(ValueError, match="num_trials must be positive"):
        MonteCarloConfig(num_trials=0, seed=1, reach_volatility=0.1)

    with pytest.raises(ValueError, match="reach_volatility cannot be negative"):
        MonteCarloConfig(num_trials=10, seed=1, reach_volatility=-0.5)

    cfg = MonteCarloConfig(num_trials=10, seed=1, reach_volatility=0.1)
    with pytest.raises(ValueError, match="Must provide at least one program"):
        run_monte_carlo_simulation([], 1000.0, [], cfg)

    with pytest.raises(ValueError, match="Budget must be positive and finite"):
        run_monte_carlo_simulation(base_programs, -500.0, [], cfg)


def test_monte_carlo_handles_intermittent_and_total_solver_failures(
    base_programs: list[Program],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies Monte Carlo accurately aggregates successful trials and returns NaN on total failure."""
    import numpy as np

    import funding_optimizer.scenarios.engine as engine_mod
    from funding_optimizer.optimizer import SolverResult

    call_count = 0

    def mock_solve(*args: object, **kwargs: object) -> SolverResult:
        nonlocal call_count
        call_count += 1
        # Alternate: Trial 1 succeeds, Trial 2 fails, Trial 3 succeeds
        if call_count % 2 == 1:
            return SolverResult(
                success=True,
                status=0,
                message="Optimal",
                objective_value=-100.0,
                allocation=np.array([400.0, 350.0, 250.0]),
            )
        return SolverResult(
            success=False,
            status=1,
            message="Infeasible",
            objective_value=None,
            allocation=None,
        )

    monkeypatch.setattr(engine_mod, "solve_allocation", mock_solve)

    cfg = MonteCarloConfig(num_trials=4, seed=42, reach_volatility=0.1)
    summary = run_monte_carlo_simulation(base_programs, 1000.0, [], cfg)

    assert summary.num_trials == 4
    assert summary.success_rate == 0.5
    assert summary.mean_objective == 100.0
    assert summary.per_program_mean_allocation["p-edu"] == 400.0
    assert summary.per_program_mean_allocation["p-health"] == 350.0
    assert summary.per_program_mean_allocation["p-solar"] == 250.0

    # Total failure case -> objectives are NaN
    def mock_all_fail(*args: object, **kwargs: object) -> SolverResult:
        return SolverResult(
            success=False, status=1, message="Infeasible", objective_value=None, allocation=None
        )

    monkeypatch.setattr(engine_mod, "solve_allocation", mock_all_fail)
    fail_summary = run_monte_carlo_simulation(base_programs, 1000.0, [], cfg)
    assert fail_summary.success_rate == 0.0
    assert math.isnan(fail_summary.mean_objective)
