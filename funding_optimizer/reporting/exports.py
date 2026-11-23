"""Impact and allocation reporting models and formatters.

Generates structured, verifiable reports in CSV, JSON, and HTML formats
linking allocation decisions, regional distributions, and on-chain
verification metadata.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from funding_optimizer.models import Budget
from funding_optimizer.optimizer.constraints import MultiAssetResult, Program
from funding_optimizer.optimizer.solver import SolverResult

# Canonical JSON Schema for Impact and Allocation Reports (Issue #88)
ALLOCATION_REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AllocationImpactReport",
    "type": "object",
    "required": [
        "schema_version",
        "generated_at",
        "budget",
        "total_allocated",
        "success",
        "allocations",
        "regional_breakdown",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "generated_at": {"type": "string"},
        "budget": {"type": "object"},
        "total_allocated": {"type": "object"},
        "success": {"type": "boolean"},
        "objective_value": {"type": ["number", "null"]},
        "allocations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["program_id", "region", "asset", "amount", "share_of_asset"],
                "properties": {
                    "program_id": {"type": "string"},
                    "region": {"type": "string"},
                    "asset": {"type": "string"},
                    "amount": {"type": "number", "minimum": 0},
                    "share_of_asset": {"type": "number", "minimum": 0, "maximum": 1.0001},
                    "expected_reach": {"type": ["number", "null"]},
                },
            },
        },
        "regional_breakdown": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
        "stellar_disbursement_metadata": {
            "type": "object",
            "properties": {
                "network_passphrase": {"type": "string"},
                "verification_hashes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class ProgramAllocationItem:
    """Individual program allocation breakdown row."""

    program_id: str
    region: str
    asset: str
    amount: float
    share_of_asset: float
    expected_reach: float | None = None

    def __post_init__(self) -> None:
        if not self.program_id:
            raise ValueError("program_id cannot be empty")
        if not self.region:
            raise ValueError("region cannot be empty")
        if not self.asset:
            raise ValueError("asset cannot be empty")
        if not math.isfinite(self.amount) or self.amount < -1e-9:
            raise ValueError(f"amount cannot be negative or non-finite: {self.amount}")
        if (
            not math.isfinite(self.share_of_asset)
            or self.share_of_asset < -1e-9
            or self.share_of_asset > 1.0001
        ):
            raise ValueError(f"share_of_asset must be between 0.0 and 1.0: {self.share_of_asset}")


@dataclass(frozen=True)
class AllocationReport:
    """Consolidated deterministic impact and allocation report."""

    schema_version: str
    generated_at: str
    budget: dict[str, float]
    total_allocated: dict[str, float]
    success: bool
    allocations: list[ProgramAllocationItem]
    regional_breakdown: dict[str, dict[str, float]]
    objective_value: float | None = None
    stellar_disbursement_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary matching ALLOCATION_REPORT_SCHEMA."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "budget": self.budget,
            "total_allocated": self.total_allocated,
            "success": self.success,
            "objective_value": self.objective_value,
            "allocations": [asdict(item) for item in self.allocations],
            "regional_breakdown": self.regional_breakdown,
            "stellar_disbursement_metadata": self.stellar_disbursement_metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Export deterministic JSON representation."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_csv(self) -> str:
        """Export standardized CSV table of allocations."""
        output = io.StringIO()
        fieldnames = [
            "program_id",
            "region",
            "asset",
            "amount",
            "share_of_asset",
            "expected_reach",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in self.allocations:
            writer.writerow(
                {
                    "program_id": item.program_id,
                    "region": item.region,
                    "asset": item.asset,
                    "amount": f"{item.amount:.7f}",
                    "share_of_asset": f"{item.share_of_asset:.6f}",
                    "expected_reach": (
                        f"{item.expected_reach:.2f}" if item.expected_reach is not None else ""
                    ),
                }
            )
        return output.getvalue()

    def to_html(self) -> str:
        """Export standalone HTML report with summary metrics, allocation table, and metadata."""
        esc = html.escape
        rows = []
        for item in self.allocations:
            reach_str = f"{item.expected_reach:.1f}" if item.expected_reach is not None else "N/A"
            rows.append(
                f"      <tr>\n"
                f"        <td><code>{esc(item.program_id)}</code></td>\n"
                f"        <td>{esc(item.region)}</td>\n"
                f'        <td><span class="badge">{esc(item.asset)}</span></td>\n'
                f'        <td class="num">{item.amount:,.2f}</td>\n'
                f'        <td class="num">{item.share_of_asset * 100:.2f}%</td>\n'
                f'        <td class="num">{reach_str}</td>\n'
                f"      </tr>"
            )
        table_rows = "\n".join(rows)

        # Budget summary rows
        budget_items = []
        for asset, total in sorted(self.budget.items()):
            allocated = self.total_allocated.get(asset, 0.0)
            budget_items.append(
                f"<li><strong>{esc(asset)}:</strong> {allocated:,.2f} / {total:,.2f} allocated</li>"
            )
        budget_summary = "\n      ".join(budget_items)

        # Regional summary rows
        region_items = []
        for reg, assets in sorted(self.regional_breakdown.items()):
            asset_strs = [f"{a}: {amt:,.2f}" for a, amt in sorted(assets.items())]
            region_items.append(f"<li><strong>{esc(reg)}:</strong> {', '.join(asset_strs)}</li>")
        region_summary = "\n      ".join(region_items)

        status_class = "status-success" if self.success else "status-failure"
        status_text = "Optimal Feasible Allocation" if self.success else "Optimization Infeasible"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>StellarSI Funding Optimizer - Allocation Impact Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2rem; color: #1f2937; line-height: 1.5; }}
    h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
    .badge {{ background-color: #e0e7ff; color: #3730a3; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }}
    .status-success {{ color: #065f46; background-color: #d1fae5; padding: 0.3rem 0.6rem; border-radius: 4px; font-weight: bold; }}
    .status-failure {{ color: #991b1b; background-color: #fee2e2; padding: 0.3rem 0.6rem; border-radius: 4px; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 0.75rem; text-align: left; }}
    th {{ background-color: #f9fafb; font-weight: 600; }}
    td.num {{ text-align: right; }}
    .meta-box {{ background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 1rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>StellarSI Allocation & Impact Report</h1>
  <div class="meta-box">
    <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
    <p><strong>Generated At:</strong> {esc(self.generated_at)}</p>
    <p><strong>Schema Version:</strong> {esc(self.schema_version)}</p>
    <h3>Budget Summary</h3>
    <ul>
      {budget_summary}
    </ul>
    <h3>Regional Distribution</h3>
    <ul>
      {region_summary}
    </ul>
  </div>

  <h2>Disbursement Allocation Plan</h2>
  <table>
    <thead>
      <tr>
        <th>Program ID</th>
        <th>Region</th>
        <th>Asset</th>
        <th style="text-align: right;">Allocated Amount</th>
        <th style="text-align: right;">Asset Share</th>
        <th style="text-align: right;">Expected Reach</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def build_report_from_multi_asset_result(
    programs: list[Program],
    budget: Budget,
    result: MultiAssetResult,
    *,
    timestamp: str | None = None,
    disbursement_metadata: dict[str, Any] | None = None,
) -> AllocationReport:
    """Build an :class:`AllocationReport` from a :class:`MultiAssetResult` and input programs."""
    gen_time = timestamp or datetime.now(UTC).isoformat()
    program_map = {p.id: p for p in programs}

    raw_allocs = result.allocation or {}
    total_allocated: dict[str, float] = {}
    items: list[ProgramAllocationItem] = []
    regional_breakdown: dict[str, dict[str, float]] = {}

    for asset_code, tranche_total in budget.amounts.items():
        tranche_allocs = raw_allocs.get(asset_code, {})
        allocated_sum = 0.0

        for prog_id, amount in tranche_allocs.items():
            amt = float(amount)
            allocated_sum += amt
            prog = program_map.get(prog_id)
            region = prog.region if prog else "Unknown"
            expected_reach = prog.expected_reach if prog else None
            share = amt / tranche_total if tranche_total > 0 else 0.0

            items.append(
                ProgramAllocationItem(
                    program_id=prog_id,
                    region=region,
                    asset=asset_code,
                    amount=amt,
                    share_of_asset=share,
                    expected_reach=expected_reach,
                )
            )

            if region not in regional_breakdown:
                regional_breakdown[region] = {}
            regional_breakdown[region][asset_code] = (
                regional_breakdown[region].get(asset_code, 0.0) + amt
            )

        total_allocated[asset_code] = allocated_sum

    return AllocationReport(
        schema_version="1.0.0",
        generated_at=gen_time,
        budget={k: float(v) for k, v in budget.amounts.items()},
        total_allocated=total_allocated,
        success=result.success,
        objective_value=result.objective_value,
        allocations=items,
        regional_breakdown=regional_breakdown,
        stellar_disbursement_metadata=disbursement_metadata or {},
    )


def build_report_from_single_asset_result(
    programs: list[Program],
    budget: float,
    result: SolverResult,
    *,
    asset_code: str = "USDC",
    timestamp: str | None = None,
    disbursement_metadata: dict[str, Any] | None = None,
) -> AllocationReport:
    """Build an :class:`AllocationReport` from a single-asset :class:`SolverResult`."""
    gen_time = timestamp or datetime.now(UTC).isoformat()
    raw_allocs = result.allocation
    items: list[ProgramAllocationItem] = []
    regional_breakdown: dict[str, dict[str, float]] = {}
    allocated_sum = 0.0

    if raw_allocs is not None:
        for p, amt_val in zip(programs, raw_allocs, strict=True):
            amt = float(amt_val)
            allocated_sum += amt
            share = amt / budget if budget > 0 else 0.0
            items.append(
                ProgramAllocationItem(
                    program_id=p.id,
                    region=p.region,
                    asset=asset_code,
                    amount=amt,
                    share_of_asset=share,
                    expected_reach=p.expected_reach,
                )
            )
            if p.region not in regional_breakdown:
                regional_breakdown[p.region] = {}
            regional_breakdown[p.region][asset_code] = (
                regional_breakdown[p.region].get(asset_code, 0.0) + amt
            )

    return AllocationReport(
        schema_version="1.0.0",
        generated_at=gen_time,
        budget={asset_code: float(budget)},
        total_allocated={asset_code: allocated_sum},
        success=result.success,
        objective_value=result.objective_value,
        allocations=items,
        regional_breakdown=regional_breakdown,
        stellar_disbursement_metadata=disbursement_metadata or {},
    )
