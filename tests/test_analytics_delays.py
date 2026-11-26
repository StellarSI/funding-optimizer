"""Tests for delivery-delay distributions by corridor and program (issue #39)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from funding_optimizer.analytics.delays import (
    DeliveryDelayError,
    compute_delivery_delay_distributions,
)
from funding_optimizer.analytics.delivery import ObservationWindow
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "delivery_delay_distributions.json"


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


def test_delay_distributions_known_fixture() -> None:
    report = compute_delivery_delay_distributions(
        load_fixture(),
        disbursement_account="GFUND",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        ),
    )

    # Hand-checkable:
    # US-KE/cash-ng: 1h (3600s) and 6h (21600s) -> mean 12600s.
    # EU-NG/edu-ke: 30m (1800s). One disbursement is unmatched.
    assert report.matched_receipts == 3
    assert [group.corridor for group in report.groups] == ["EU-NG", "US-KE"]
    assert [group.program for group in report.groups] == ["edu-ke", "cash-ng"]

    first, second = report.groups
    assert first.observations == 1
    assert first.mean_seconds == pytest.approx(1800.0)
    assert first.median_seconds == pytest.approx(1800.0)
    assert first.min_seconds == pytest.approx(1800.0)
    assert first.max_seconds == pytest.approx(1800.0)
    assert first.histogram["0-1h"] == 1

    assert second.observations == 2
    assert second.mean_seconds == pytest.approx(12600.0)
    assert second.median_seconds == pytest.approx(12600.0)
    assert second.p90_seconds == pytest.approx(19800.0)
    assert second.p95_seconds == pytest.approx(20700.0)
    assert second.min_seconds == pytest.approx(3600.0)
    assert second.max_seconds == pytest.approx(21600.0)
    assert second.histogram == {
        "0-1h": 1,
        "1h-6h": 1,
        "6h-1d": 0,
        "1d-7d": 0,
        "7d+": 0,
    }
    assert second.by_asset[0].asset == "USDC"
    assert second.by_asset[0].observations == 2
    assert second.by_asset[0].mean_seconds == pytest.approx(12600.0)


def test_missing_dimensions_are_reported_as_unknown() -> None:
    report = compute_delivery_delay_distributions(load_fixture())
    unknown = [group for group in report.groups if group.corridor == "unknown"]
    assert unknown == []  # every disbursement in the fixture is tagged

    events = [event for event in load_fixture() if event.event_id in {"d3", "r3"}]
    stripped = [NormalizedEvent(**{**event.__dict__, "metadata": {}}) for event in events]
    report = compute_delivery_delay_distributions(stripped)
    assert len(report.groups) == 1
    assert report.groups[0].corridor == "unknown"
    assert report.groups[0].program == "unknown"


def test_no_matches_produces_an_empty_report() -> None:
    report = compute_delivery_delay_distributions(
        [event for event in load_fixture() if event.direction.value == "out"]
    )
    assert report.groups == ()
    assert report.matched_receipts == 0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(DeliveryDelayError):
        compute_delivery_delay_distributions("not-events")  # type: ignore[arg-type]
    with pytest.raises(DeliveryDelayError):
        compute_delivery_delay_distributions(load_fixture(), bucket_bounds_seconds=(3600.0, 3600.0))
    with pytest.raises(DeliveryDelayError):
        compute_delivery_delay_distributions(
            load_fixture(), observation_window={"start": "bad", "end": "2026-01-31T00:00:00Z"}
        )
