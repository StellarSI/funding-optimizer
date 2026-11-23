"""Tests for trustline-readiness delivery analytics (issue #35)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from funding_optimizer.analytics.trustlines import (
    ObservationWindow,
    TrustlineReadinessError,
    compute_trustline_readiness_rates,
)
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "trustline_readiness.json"


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


def test_trustline_readiness_known_fixture() -> None:
    report = compute_trustline_readiness_rates(
        load_fixture(),
        {
            "cohort-a": ["GRECIPIENT1", "GRECIPIENT2", "GRECIPIENT3"],
            "cohort-b": ["GRECIPIENT4"],
        },
        asset="USDC:GISSUER",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        ),
    )

    # Hand-checkable:
    # cohort-a: GRECIPIENT1 has change_trust, GRECIPIENT2 has inbound payment,
    # GRECIPIENT3 only outbound => 2/3 ready.
    # cohort-b: GRECIPIENT4 has no events => 0/1 ready.
    assert report.asset == "USDC:GISSUER"
    assert len(report.cohorts) == 2
    a, b = report.cohorts
    assert a.cohort == "cohort-a"
    assert a.total_accounts == 3
    assert a.ready_accounts == 2
    assert a.readiness_rate == pytest.approx(2 / 3)
    assert list(a.ready_account_ids) == ["GRECIPIENT1", "GRECIPIENT2"]
    assert b.cohort == "cohort-b"
    assert b.total_accounts == 1
    assert b.ready_accounts == 0
    assert b.readiness_rate == pytest.approx(0.0)


def test_observation_window_filters_events() -> None:
    report = compute_trustline_readiness_rates(
        load_fixture(),
        {"cohort-a": ["GRECIPIENT1", "GRECIPIENT2"]},
        asset="USDC:GISSUER",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
            end=datetime(2026, 1, 1, 23, 59, tzinfo=UTC),
        ),
    )
    # GRECIPIENT1 change_trust at 00:00:00Z is outside window; GRECIPIENT2 inbound at 01:00:00Z is inside.
    cohort = report.cohorts[0]
    assert cohort.ready_accounts == 1
    assert list(cohort.ready_account_ids) == ["GRECIPIENT2"]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(TrustlineReadinessError):
        compute_trustline_readiness_rates("not-events", {}, asset="USDC:GISSUER")  # type: ignore[arg-type]
    with pytest.raises(TrustlineReadinessError):
        compute_trustline_readiness_rates([], {}, asset="")
