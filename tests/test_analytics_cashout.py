"""Tests for realized cash-out rates from on-chain flows (issue #38)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from funding_optimizer.analytics.cashout import (
    CashoutAnalyticsError,
    compute_realized_cashout_rates,
)
from funding_optimizer.analytics.delivery import ObservationWindow
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "cashout_rates.json"


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


JANUARY = ObservationWindow(
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
)


def test_cashout_known_fixture() -> None:
    report = compute_realized_cashout_rates(
        load_fixture(),
        disbursement_account="GFUND",
        observation_window=JANUARY,
    )

    # Hand-checkable:
    # USDC: 100 -> r1 (matched), 50 -> no receipt, 25 -> r3 (matched).
    # XLM: 1000 -> r4 (matched); realization counts the disbursed amount
    # even when the observed receipt amount is smaller (400).
    assert report.disbursements == 4
    assert report.matched_disbursements == 3
    assert report.disbursed_amount == pytest.approx(1175.0)
    assert report.realized_amount == pytest.approx(1125.0)
    assert report.realization_rate == pytest.approx(1125.0 / 1175.0)
    assert report.recipient_cashout_rate == pytest.approx(3 / 4)

    usdc, xlm = report.assets
    assert usdc.asset == "USDC:GISSUER"
    assert usdc.disbursements == 3
    assert usdc.matched_disbursements == 2
    assert usdc.disbursed_amount == pytest.approx(175.0)
    assert usdc.realized_amount == pytest.approx(125.0)
    assert usdc.realization_rate == pytest.approx(125.0 / 175.0)
    assert usdc.recipients == 3
    assert usdc.cashed_out_recipients == 2
    assert usdc.recipient_cashout_rate == pytest.approx(2 / 3)

    assert xlm.asset == "XLM"
    assert xlm.disbursements == 1
    assert xlm.matched_disbursements == 1
    assert xlm.realized_amount == pytest.approx(1000.0)
    assert xlm.realization_rate == pytest.approx(1.0)

    serialized = report.as_dict()
    assert serialized["units"]["realization_rate"] == "fraction"
    assert serialized["assets"][0]["asset"] == "USDC:GISSUER"


def test_window_filters_legs_and_mapping_form_is_accepted() -> None:
    report = compute_realized_cashout_rates(
        load_fixture(),
        disbursement_account="GFUND",
        observation_window={
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-01T23:59:59Z",
        },
    )
    assert report.disbursements == 1
    assert report.matched_disbursements == 1
    assert report.realization_rate == pytest.approx(1.0)
    assert [a.asset for a in report.assets] == ["USDC:GISSUER"]


def test_empty_events_produce_zero_rates() -> None:
    report = compute_realized_cashout_rates([])
    assert report.disbursements == 0
    assert report.realization_rate == pytest.approx(0.0)
    assert report.recipient_cashout_rate == pytest.approx(0.0)
    assert report.assets == ()


def test_invalid_inputs_raise() -> None:
    with pytest.raises(CashoutAnalyticsError):
        compute_realized_cashout_rates("not-events")  # type: ignore[arg-type]
    with pytest.raises(CashoutAnalyticsError):
        compute_realized_cashout_rates([], disbursement_account=" ")
    with pytest.raises(CashoutAnalyticsError):
        compute_realized_cashout_rates([], observation_window={"start": "bad", "end": "bad"})
