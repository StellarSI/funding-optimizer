"""Tests for delivery time-to-receipt analytics (issue #36)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from funding_optimizer.analytics.delivery import (
    DeliveryAnalyticsError,
    ObservationWindow,
    compute_time_to_receipt_metrics,
)
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent


FIXTURE = Path(__file__).parent / "fixtures" / "delivery_time_to_receipt.json"


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


def test_time_to_receipt_known_fixture() -> None:
    metrics = compute_time_to_receipt_metrics(
        load_fixture(),
        disbursement_account="GFUND",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        ),
    )

    # Hand-checkable:
    # d1 2026-01-01T00:00:00Z -> r1 2026-01-01T01:00:00Z = 3600s
    # d2 2026-01-02T00:00:00Z -> r2 2026-01-02T04:00:00Z = 14400s
    # d3 unmatched
    assert metrics.disbursements == 3
    assert metrics.matched_receipts == 2
    assert metrics.unmatched_disbursements == 1
    assert metrics.match_rate == pytest.approx(2 / 3)
    assert metrics.mean_receipt_seconds == pytest.approx(9000.0)
    assert metrics.median_receipt_seconds == pytest.approx(9000.0)
    assert metrics.p95_receipt_seconds == pytest.approx(13860.0)
    assert metrics.min_receipt_seconds == pytest.approx(3600.0)
    assert metrics.max_receipt_seconds == pytest.approx(14400.0)
    assert metrics.observation_window is not None
    assert metrics.matches[0].receipt_seconds == pytest.approx(3600.0)
    assert metrics.matches[1].receipt_seconds == pytest.approx(14400.0)


def test_observation_window_filters_disbursements() -> None:
    metrics = compute_time_to_receipt_metrics(
        load_fixture(),
        disbursement_account="GFUND",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 2, tzinfo=UTC),
            end=datetime(2026, 1, 2, 23, 59, 59, tzinfo=UTC),
        ),
    )
    assert metrics.disbursements == 1
    assert metrics.matched_receipts == 1
    assert metrics.mean_receipt_seconds == pytest.approx(14400.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(DeliveryAnalyticsError):
        compute_time_to_receipt_metrics("not-events")  # type: ignore[arg-type]
    with pytest.raises(DeliveryAnalyticsError):
        compute_time_to_receipt_metrics([], disbursement_account=" ")
