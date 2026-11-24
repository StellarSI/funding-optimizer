"""Tests for unclaimed-funds balances and aging buckets (issue #37)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from funding_optimizer.analytics.unclaimed import (
    UnclaimedFundsError,
    compute_unclaimed_funds,
)
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "unclaimed_funds.json"


def load_fixture() -> list[NormalizedEvent]:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        NormalizedEvent(
            event_id=row["event_id"],
            source=EventSource(row["source"]),
            occurred_at=row["occurred_at"],
            event_type=row["event_type"],
            asset=row["asset"],
            amount=float(row["amount"]),
            direction=EventDirection(row["direction"]),
            account=row["account"],
            counterparty=row.get("counterparty"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def test_unclaimed_known_fixture() -> None:
    report = compute_unclaimed_funds(
        load_fixture(),
        disbursement_account="GFUND",
        as_of="2026-02-20T00:00:00Z",
    )

    # Hand-checkable as of 2026-02-20:
    # d1 (100) -> r1 on Jan 2  => claimed, excluded.
    # d4 (10)  Jan 1  => 50 days => 31-90.
    # d2 (50)  Jan 10 => 41 days => 31-90.
    # d3 (25)  Feb 15 =>  5 days => 0-7.
    assert report.as_of.isoformat().replace("+00:00", "Z") == "2026-02-20T00:00:00Z"
    assert report.disbursements == 3
    assert report.total_amount == pytest.approx(85.0)
    assert report.by_asset == {"USDC:GISSUER": pytest.approx(85.0)}
    assert [b.disbursement_event_id for b in report.balances] == [
        "horizon:d4:5",
        "horizon:d2:3",
        "horizon:d3:4",
    ]
    assert [b.aging_bucket for b in report.balances] == ["31-90", "31-90", "0-7"]
    assert [b.age_days for b in report.balances] == pytest.approx([50.0, 41.0, 5.0])
    assert [(b.bucket, b.total_amount, b.disbursements) for b in report.buckets] == [
        ("0-7", pytest.approx(25.0), 1),
        ("8-30", pytest.approx(0.0), 0),
        ("31-90", pytest.approx(60.0), 2),
        ("90+", pytest.approx(0.0), 0),
    ]
    serialized = report.as_dict()
    assert serialized["units"] == {
        "amount": "asset units",
        "age": "days",
        "timestamp": "utc-iso8601",
    }
    assert serialized["total_amount"] == pytest.approx(85.0)


def test_as_of_boundary_excludes_future_disbursements() -> None:
    report = compute_unclaimed_funds(
        load_fixture(),
        disbursement_account="GFUND",
        as_of="2026-01-05T00:00:00Z",
    )
    assert report.disbursements == 1
    assert report.total_amount == pytest.approx(10.0)
    assert report.balances[0].disbursement_event_id == "horizon:d4:5"
    assert report.balances[0].aging_bucket == "0-7"


def test_custom_aging_buckets() -> None:
    report = compute_unclaimed_funds(
        load_fixture(),
        as_of="2026-02-20T00:00:00Z",
        aging_buckets=[("short", 0.0, 10.0), ("long", 10.0, None)],
    )
    assert [b.aging_bucket for b in report.balances] == ["long", "long", "short"]
    assert [b.bucket for b in report.buckets] == ["short", "long"]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(UnclaimedFundsError):
        compute_unclaimed_funds("not-events")  # type: ignore[arg-type]
    with pytest.raises(UnclaimedFundsError):
        compute_unclaimed_funds([], disbursement_account=" ")
    with pytest.raises(UnclaimedFundsError):
        compute_unclaimed_funds(load_fixture(), as_of="not-a-date")
    with pytest.raises(UnclaimedFundsError):
        compute_unclaimed_funds(load_fixture(), aging_buckets=[("bad", 5.0, 1.0)])
