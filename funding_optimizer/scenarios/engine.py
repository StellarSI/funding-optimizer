"""Scenario comparison and Monte Carlo simulation engine.

Enables funders to simulate 'what-if' budget shocks, reach uncertainty,
and policy shifts under deterministic pseudo-random seeds before committing capital.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from funding_optimizer.optimizer.constraints import (
    Program,
    RegionFloor,
    solve_allocation,
)
from funding_optimizer.optimizer.solver import SolverResult


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a specific 'what-if' allocation scenario."""

    name: str
    budget: float
    programs: list[Program]
    region_floors: list[RegionFloor]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Scenario name cannot be empty")
        if not isfinite(self.budget) or self.budget <= 0:
            raise ValueError(f"Scenario budget must be positive and finite: {self.budget}")
        if not self.programs:
            raise ValueError("Scenario must contain at least one program")


@dataclass(frozen=True)
class ScenarioResult:
    """Deterministic result of evaluating a single scenario."""

    name: str
    budget: float
    success: bool
    objective_value: float | None
    allocations: dict[str, float]
    regional_totals: dict[str, float]


@dataclass(frozen=True)
class ScenarioComparison:
    """Comparative analysis across multiple evaluated scenarios."""

    scenarios: list[ScenarioResult]

    def delta_reach(self, baseline_name: str, candidate_name: str) -> float:
        """Calculate the positive reach delta between a baseline and candidate scenario."""
        scenario_map = {s.name: s for s in self.scenarios}
        if baseline_name not in scenario_map:
            raise KeyError(f"Baseline scenario not found: {baseline_name}")
        if candidate_name not in scenario_map:
            raise KeyError(f"Candidate scenario not found: {candidate_name}")

        base_s = scenario_map[baseline_name]
        cand_s = scenario_map[candidate_name]

        if not base_s.success or base_s.objective_value is None:
            raise ValueError(f"Baseline scenario '{baseline_name}' is infeasible or unoptimized")
        if not cand_s.success or cand_s.objective_value is None:
            raise ValueError(f"Candidate scenario '{candidate_name}' is infeasible or unoptimized")

        return float(cand_s.objective_value - base_s.objective_value)


def evaluate_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Evaluate a single allocation scenario deterministically."""
    result: SolverResult = solve_allocation(config.programs, config.budget, config.region_floors)

    alloc_map: dict[str, float] = {}
    region_map: dict[str, float] = {}
    positive_reach: float | None = None

    if result.success and result.allocation is not None:
        for p, amt_val in zip(config.programs, result.allocation, strict=True):
            amt = float(amt_val)
            alloc_map[p.id] = amt
            region_map[p.region] = region_map.get(p.region, 0.0) + amt
        if result.objective_value is not None:
            positive_reach = float(abs(result.objective_value))

    return ScenarioResult(
        name=config.name,
        budget=config.budget,
        success=result.success,
        objective_value=positive_reach,
        allocations=alloc_map,
        regional_totals=region_map,
    )


def compare_scenarios(configs: Sequence[ScenarioConfig]) -> ScenarioComparison:
    """Evaluate and compare a series of allocation scenarios."""
    if not configs:
        raise ValueError("Must provide at least one scenario configuration")

    results = [evaluate_scenario(cfg) for cfg in configs]
    return ScenarioComparison(scenarios=results)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Parameters for running a seeded Monte Carlo reach uncertainty simulation."""

    num_trials: int
    seed: int
    reach_volatility: float  # Gaussian std-dev fraction (e.g. 0.10 for 10% uncertainty)

    def __post_init__(self) -> None:
        if self.num_trials <= 0:
            raise ValueError(f"num_trials must be positive: {self.num_trials}")
        if self.reach_volatility < 0.0:
            raise ValueError(f"reach_volatility cannot be negative: {self.reach_volatility}")


@dataclass(frozen=True)
class MonteCarloSummary:
    """Aggregate statistics from Monte Carlo simulation trials."""

    num_trials: int
    seed: int
    success_rate: float
    mean_objective: float
    std_objective: float
    p05_objective: float
    p50_objective: float
    p95_objective: float
    per_program_mean_allocation: dict[str, float]


def run_monte_carlo_simulation(
    programs: list[Program],
    budget: float,
    region_floors: list[RegionFloor],
    config: MonteCarloConfig,
) -> MonteCarloSummary:
    """Execute a fully deterministic, seeded Monte Carlo simulation over reach perturbations."""
    if not programs:
        raise ValueError("Must provide at least one program")
    if budget <= 0 or not isfinite(budget):
        raise ValueError(f"Budget must be positive and finite: {budget}")

    rng = np.random.default_rng(config.seed)
    num_programs = len(programs)

    objectives: list[float] = []
    successful_allocations: list[list[float]] = []
    successes = 0

    for _ in range(config.num_trials):
        # Deterministic log-normal / Gaussian relative shock to expected reach
        shocks = rng.normal(1.0, config.reach_volatility, size=num_programs)
        # Prevent non-positive reach
        shocks = np.clip(shocks, 0.05, 5.0)

        perturbed_programs = [
            Program(
                id=programs[j].id,
                region=programs[j].region,
                expected_reach=float(programs[j].expected_reach * shocks[j]),
                min_share=programs[j].min_share,
            )
            for j in range(num_programs)
        ]

        result = solve_allocation(perturbed_programs, budget, region_floors)
        if result.success and result.allocation is not None:
            successes += 1
            obj = float(abs(result.objective_value)) if result.objective_value is not None else 0.0
            objectives.append(obj)
            successful_allocations.append([float(x) for x in result.allocation])

    success_rate = successes / config.num_trials
    if objectives:
        obj_array = np.array(objectives)
        mean_obj = float(np.mean(obj_array))
        std_obj = float(np.std(obj_array))
        p05_obj = float(np.percentile(obj_array, 5))
        p50_obj = float(np.percentile(obj_array, 50))
        p95_obj = float(np.percentile(obj_array, 95))
    else:
        mean_obj = float("nan")
        std_obj = float("nan")
        p05_obj = float("nan")
        p50_obj = float("nan")
        p95_obj = float("nan")

    if successful_allocations:
        mean_allocs = np.mean(np.array(successful_allocations), axis=0)
    else:
        mean_allocs = np.zeros(num_programs)
    per_program_map = {programs[j].id: float(mean_allocs[j]) for j in range(num_programs)}

    return MonteCarloSummary(
        num_trials=config.num_trials,
        seed=config.seed,
        success_rate=success_rate,
        mean_objective=mean_obj,
        std_objective=std_obj,
        p05_objective=p05_obj,
        p50_objective=p50_obj,
        p95_objective=p95_obj,
        per_program_mean_allocation=per_program_map,
    )
