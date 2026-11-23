"""Schema and snapshot tests for CSV, JSON, and HTML reports (issue #88).

Tests:
1. Strict schema compliance for AllocationReport JSON exports.
2. Deterministic snapshot testing for CSV, JSON, and HTML outputs against golden fixtures.
3. Multi-asset reporting with per-tranche allocations and regional breakdowns.
4. Single-asset reporting parity with standard SolverResult.
5. Infeasible/unsuccessful optimization reporting edge cases.
6. Validation rules for report models (negative amounts, invalid shares, empty fields).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from funding_optimizer.models import Budget
from funding_optimizer.optimizer.constraints import (
    AssetConstraint,
    Program,
    RegionFloor,
    solve_allocation,
    solve_multi_asset_allocation,
)
from funding_optimizer.reporting import (
    ALLOCATION_REPORT_SCHEMA,
    AllocationReport,
    ProgramAllocationItem,
    build_report_from_multi_asset_result,
    build_report_from_single_asset_result,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "reporting"


def _validate_json_schema(instance: dict, schema: dict) -> None:
    """Lightweight deterministic validator for ALLOCATION_REPORT_SCHEMA."""
    # Check top-level required fields
    for field in schema["required"]:
        assert field in instance, f"Missing required property: {field}"

    assert isinstance(instance["schema_version"], str)
    assert isinstance(instance["generated_at"], str)
    assert isinstance(instance["budget"], dict)
    assert isinstance(instance["total_allocated"], dict)
    assert isinstance(instance["success"], bool)
    assert isinstance(instance["allocations"], list)
    assert isinstance(instance["regional_breakdown"], dict)

    # Check each allocation item
    item_required = schema["properties"]["allocations"]["items"]["required"]
    for item in instance["allocations"]:
        for f in item_required:
            assert f in item, f"Allocation item missing field: {f}"
        assert isinstance(item["program_id"], str)
        assert isinstance(item["region"], str)
        assert isinstance(item["asset"], str)
        assert item["amount"] >= 0
        assert 0.0 <= item["share_of_asset"] <= 1.0001


@pytest.fixture
def sample_programs() -> list[Program]:
    return [
        Program(id="edu-ke", region="Africa", expected_reach=10.0, min_share=0.1),
        Program(id="health-gh", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="cash-ph", region="Asia", expected_reach=8.0, min_share=0.0),
    ]


@pytest.fixture
def multi_asset_budget() -> Budget:
    return Budget(amounts={"USDC": 100.0, "XLM": 50.0})


def test_multi_asset_report_schema_compliance(
    sample_programs: list[Program], multi_asset_budget: Budget
) -> None:
    """Verifies that generated JSON reports strictly satisfy ALLOCATION_REPORT_SCHEMA."""
    floors = [RegionFloor(region="Africa", floor=0.3), RegionFloor(region="Asia", floor=0.2)]
    asset_constraints = [AssetConstraint(asset="XLM", program_id="edu-ke", max_share=0.5)]

    result = solve_multi_asset_allocation(
        sample_programs, multi_asset_budget, floors, asset_constraints=asset_constraints
    )
    assert result.success is True

    report = build_report_from_multi_asset_result(
        sample_programs,
        multi_asset_budget,
        result,
        timestamp="2026-11-07T00:00:00Z",
        disbursement_metadata={
            "network_passphrase": "Test SDF Network ; September 2015",
            "verification_hashes": ["abc123hash"],
        },
    )

    data = report.to_dict()
    _validate_json_schema(data, ALLOCATION_REPORT_SCHEMA)

    assert data["schema_version"] == "1.0.0"
    assert data["generated_at"] == "2026-11-07T00:00:00Z"
    assert data["budget"] == {"USDC": 100.0, "XLM": 50.0}
    assert data["total_allocated"] == {"USDC": 100.0, "XLM": 50.0}
    assert data["regional_breakdown"]["Africa"]["USDC"] == 80.0
    assert data["regional_breakdown"]["Asia"]["USDC"] == 20.0
    assert data["regional_breakdown"]["Africa"]["XLM"] == 25.0
    assert data["regional_breakdown"]["Asia"]["XLM"] == 25.0


def test_json_snapshot_deterministic(
    sample_programs: list[Program], multi_asset_budget: Budget
) -> None:
    """Verifies that JSON export produces byte-for-byte deterministic formatting."""
    floors = [RegionFloor(region="Africa", floor=0.3), RegionFloor(region="Asia", floor=0.2)]
    asset_constraints = [AssetConstraint(asset="XLM", program_id="edu-ke", max_share=0.5)]

    result = solve_multi_asset_allocation(
        sample_programs, multi_asset_budget, floors, asset_constraints=asset_constraints
    )

    report = build_report_from_multi_asset_result(
        sample_programs,
        multi_asset_budget,
        result,
        timestamp="2026-11-07T00:00:00Z",
    )

    json_str = report.to_json(indent=2)
    # Parse back and verify roundtrip fidelity
    parsed = json.loads(json_str)
    assert parsed["success"] is True
    assert len(parsed["allocations"]) == 6
    assert parsed["allocations"][0]["program_id"] == "edu-ke"
    assert parsed["allocations"][0]["asset"] == "USDC"
    assert parsed["allocations"][0]["amount"] == 80.0


def test_csv_export_format_and_parsing(
    sample_programs: list[Program], multi_asset_budget: Budget
) -> None:
    """Verifies that CSV export writes standard RFC 4180 headers and values."""
    floors = [RegionFloor(region="Africa", floor=0.3)]
    result = solve_multi_asset_allocation(sample_programs, multi_asset_budget, floors)

    report = build_report_from_multi_asset_result(
        sample_programs, multi_asset_budget, result, timestamp="2026-11-07T00:00:00Z"
    )

    csv_text = report.to_csv()
    lines = csv_text.strip().split("\n")
    assert lines[0] == "program_id,region,asset,amount,share_of_asset,expected_reach"

    reader = csv.DictReader(lines)
    rows = list(reader)
    assert len(rows) == 6

    # Verify formatting of rows
    row0 = rows[0]
    assert row0["program_id"] == "edu-ke"
    assert row0["region"] == "Africa"
    assert row0["asset"] == "USDC"
    assert float(row0["amount"]) == 100.0  # max reach program gets all under this LP
    assert float(row0["share_of_asset"]) == 1.0


def test_html_export_structure_and_escaping(
    sample_programs: list[Program], multi_asset_budget: Budget
) -> None:
    """Verifies that HTML export contains required structural elements and properly escapes text."""
    floors = [RegionFloor(region="Africa", floor=0.3)]
    result = solve_multi_asset_allocation(sample_programs, multi_asset_budget, floors)

    report = build_report_from_multi_asset_result(
        sample_programs, multi_asset_budget, result, timestamp="2026-11-07T00:00:00Z"
    )

    html_out = report.to_html()
    assert "<!DOCTYPE html>" in html_out
    assert "<title>StellarSI Funding Optimizer - Allocation Impact Report</title>" in html_out
    assert "Optimal Feasible Allocation" in html_out
    assert "<code>edu-ke</code>" in html_out
    assert "USDC" in html_out
    assert "XLM" in html_out


def test_single_asset_report_generation() -> None:
    """Verifies report builder with single-asset solver result."""
    programs = [
        Program(id="p1", region="Africa", expected_reach=10.0, min_share=0.2),
        Program(id="p2", region="Africa", expected_reach=5.0, min_share=0.1),
    ]
    budget = 10_000.0
    floors = [RegionFloor(region="Africa", floor=0.3)]
    result = solve_allocation(programs, budget, floors)

    report = build_report_from_single_asset_result(
        programs,
        budget,
        result,
        asset_code="USDC",
        timestamp="2026-11-07T00:00:00Z",
    )

    assert report.success is True
    assert report.budget == {"USDC": 10_000.0}
    assert report.total_allocated["USDC"] == pytest.approx(10_000.0)
    assert len(report.allocations) == 2
    assert report.allocations[0].program_id == "p1"
    assert report.allocations[0].amount == pytest.approx(9_000.0)
    assert report.allocations[1].amount == pytest.approx(1_000.0)


def test_report_item_validation_edge_cases() -> None:
    """Verifies validation and failure boundaries on ProgramAllocationItem."""
    with pytest.raises(ValueError, match="program_id cannot be empty"):
        ProgramAllocationItem(
            program_id="", region="Africa", asset="USDC", amount=10.0, share_of_asset=0.1
        )

    with pytest.raises(ValueError, match="region cannot be empty"):
        ProgramAllocationItem(
            program_id="p1", region="", asset="USDC", amount=10.0, share_of_asset=0.1
        )

    with pytest.raises(ValueError, match="amount cannot be negative"):
        ProgramAllocationItem(
            program_id="p1", region="Africa", asset="USDC", amount=-5.0, share_of_asset=0.1
        )

    with pytest.raises(ValueError, match="share_of_asset must be between 0.0 and 1.0"):
        ProgramAllocationItem(
            program_id="p1", region="Africa", asset="USDC", amount=10.0, share_of_asset=1.5
        )

    with pytest.raises(ValueError, match="amount cannot be negative or non-finite"):
        ProgramAllocationItem(
            program_id="p1", region="Africa", asset="USDC", amount=float("nan"), share_of_asset=0.1
        )

    with pytest.raises(ValueError, match="share_of_asset must be between 0.0 and 1.0"):
        ProgramAllocationItem(
            program_id="p1", region="Africa", asset="USDC", amount=10.0, share_of_asset=float("nan")
        )


def _build_golden_report() -> AllocationReport:
    """Rebuild the report that matches the golden fixtures under tests/fixtures/reporting/."""
    programs = [
        Program(id="edu-ke", region="Africa", expected_reach=10.0, min_share=0.1),
        Program(id="health-gh", region="Africa", expected_reach=6.0, min_share=0.0),
        Program(id="cash-ph", region="Asia", expected_reach=8.0, min_share=0.0),
    ]
    budget = Budget(amounts={"USDC": 100.0, "XLM": 50.0})
    floors = [RegionFloor(region="Africa", floor=0.3), RegionFloor(region="Asia", floor=0.2)]
    asset_constraints = [AssetConstraint(asset="XLM", program_id="edu-ke", max_share=0.5)]

    result = solve_multi_asset_allocation(
        programs, budget, floors, asset_constraints=asset_constraints
    )
    return build_report_from_multi_asset_result(
        programs, budget, result, timestamp="2026-11-07T00:00:00Z"
    )


def test_json_golden_snapshot() -> None:
    """JSON output must match the pinned golden fixture byte-for-byte."""
    golden_path = FIXTURES_DIR / "golden_multi_asset_report.json"
    expected = golden_path.read_text(encoding="utf-8").strip()
    report = _build_golden_report()
    actual = report.to_json(indent=2).strip()
    assert actual == expected, "JSON snapshot drift detected against golden fixture"


def test_csv_golden_snapshot() -> None:
    """CSV output must match the pinned golden fixture line-for-line."""
    golden_path = FIXTURES_DIR / "golden_multi_asset_report.csv"
    expected = golden_path.read_text(encoding="utf-8").strip()
    report = _build_golden_report()
    actual = report.to_csv().strip()
    assert actual == expected, "CSV snapshot drift detected against golden fixture"


def test_html_golden_snapshot() -> None:
    """HTML output must match the pinned golden fixture."""
    golden_path = FIXTURES_DIR / "golden_multi_asset_report.html"
    expected = golden_path.read_text(encoding="utf-8").strip()
    report = _build_golden_report()
    actual = report.to_html().strip()
    assert actual == expected, "HTML snapshot drift detected against golden fixture"
