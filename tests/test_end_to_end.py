"""End-to-end integration tests for the full optimize-to-report pipeline.

Fulfills issue #90 (Tests: add an end-to-end optimize-to-report integration test).
Simulates the complete production lifecycle without external network calls:
1. Candidate cohort, multi-asset budget, and regional constraint ingestion.
2. High-precision LP allocation solving (SciPy/HiGHS deterministic adapter).
3. Budget conservation and regional floor verification.
4. Summary impact metrics reporting (total reach, coverage, per-region breakdown).
5. Deterministic JSON report serialization and infeasibility diagnostic verification.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from funding_optimizer.models import Budget
from funding_optimizer.optimizer import (
    AssetConstraint,
    Program,
    RegionFloor,
    diagnose_infeasibility,
    explain_result,
    solve_allocation,
    solve_multi_asset_allocation,
)


def test_end_to_end_optimize_to_report_and_export_lifecycle() -> None:
    """Executes the full pipeline: Input Programs -> LP Solver -> Impact Report -> JSON Export."""
    # 1. Define Budget and Candidate Programs
    budget = Budget(total=500_000.0, asset="USDC")
    programs = [
        Program(id="prog-edu-01", region="Africa", expected_reach=15.0, min_share=0.10),
        Program(id="prog-health-02", region="Africa", expected_reach=10.0, min_share=0.05),
        Program(id="prog-cleanwater-03", region="Asia", expected_reach=12.0, min_share=0.15),
    ]

    # 2. Define Regional Fairness Policy
    region_floors = [
        RegionFloor(region="Africa", floor=0.40),
        RegionFloor(region="Asia", floor=0.30),
    ]

    # 3. Solve Optimal Allocation
    result = solve_allocation(programs, budget.total, region_floors)
    assert result.success is True, f"Optimization failed: {result.message}"
    assert result.allocation is not None

    allocations = result.allocation
    assert len(allocations) == 3

    # 4. Impact Reporting & Invariant Audit
    total_allocated = float(np.sum(allocations))
    assert np.isclose(total_allocated, budget.total, atol=1e-4)

    africa_sum = float(allocations[0] + allocations[1])
    asia_sum = float(allocations[2])

    assert africa_sum >= 0.40 * budget.total - 1e-4
    assert asia_sum >= 0.30 * budget.total - 1e-4

    # Compute positive total expected reach from allocations
    total_reach = float(
        sum(programs[i].expected_reach * allocations[i] for i in range(len(programs)))
    )
    assert np.isclose(total_reach, -float(result.objective_value or 0.0), atol=1e-4)

    diagnostics = explain_result(result, programs, budget.total, region_floors)
    assert diagnostics == []

    # 5. Deterministic Report & Export Payload Serialization
    report_payload = {
        "asset": budget.asset,
        "budget": budget.total,
        "total_allocated": round(total_allocated, 2),
        "total_expected_reach": round(total_reach, 2),
        "allocations": [
            {
                "program_id": programs[i].id,
                "region": programs[i].region,
                "amount": round(float(allocations[i]), 7),
                "share": round(float(allocations[i]) / budget.total, 6),
                "projected_reach": round(float(allocations[i]) * programs[i].expected_reach, 2),
            }
            for i in range(len(programs))
        ],
        "regional_breakdown": {
            "Africa": round(africa_sum, 2),
            "Asia": round(asia_sum, 2),
        },
    }

    payload_json = json.dumps(report_payload, sort_keys=True)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:32]
    idempotency_key = f"sdp:alloc:{digest}"

    parsed = json.loads(payload_json)
    assert parsed["total_allocated"] == 500_000.0
    assert len(parsed["allocations"]) == 3
    assert idempotency_key.startswith("sdp:alloc:")


def test_end_to_end_multi_asset_pipeline() -> None:
    """Verifies end-to-end multi-asset optimization and per-tranche conservation."""
    budget = Budget(amounts={"USDC": 200_000.0, "XLM": 100_000.0})
    programs = [
        Program(id="p-alpha", region="Africa", expected_reach=20.0, min_share=0.10),
        Program(id="p-beta", region="LatAm", expected_reach=15.0, min_share=0.10),
    ]
    constraints = [
        AssetConstraint(asset="USDC", program_id="p-alpha", max_share=0.60),
    ]

    ma_result = solve_multi_asset_allocation(programs, budget, asset_constraints=constraints)
    assert ma_result.success is True
    assert ma_result.allocation is not None
    assert np.isclose(sum(ma_result.allocation["USDC"].values()), 200_000.0, atol=1e-4)
    assert np.isclose(sum(ma_result.allocation["XLM"].values()), 100_000.0, atol=1e-4)
    assert ma_result.allocation["USDC"]["p-alpha"] <= 0.60 * 200_000.0 + 1e-4


def test_end_to_end_infeasible_pipeline_fails_closed() -> None:
    """Verifies that infeasible input combinations fail closed and produce diagnostics."""
    programs = [
        Program(id="p1", region="Africa", expected_reach=5.0, min_share=0.10),
        Program(id="p2", region="Asia", expected_reach=5.0, min_share=0.10),
    ]
    # Regional floors sum to 0.70 + 0.45 = 1.15 > 1.0 (Infeasible)
    region_floors = [
        RegionFloor(region="Africa", floor=0.70),
        RegionFloor(region="Asia", floor=0.45),
    ]
    diagnostics = diagnose_infeasibility(programs, 100_000.0, region_floors)
    assert len(diagnostics) > 0
    assert any(d.code == "floors_exceed_budget" for d in diagnostics)

    with pytest.raises(ValueError, match="exceeds the whole budget"):
        solve_allocation(programs, 100_000.0, region_floors)
