"""Tests for delivery aggregates by program, region, cohort and asset (issue #40)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from funding_optimizer.analytics.aggregates import (
    DeliveryAggregatesError,
    ObservationWindow,
    compute_delivery_aggregates,
)
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "delivery_aggregates.json"


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


def test_delivery_aggregates_known_fixture() -> None:
    report = compute_delivery_aggregates(load_fixture())

    assert report.events == 6
    assert report.disbursements == 4
    assert report.receipts == 2
    assert report.recipients == 4
    assert report.by_asset == {"USDC": pytest.approx(175.0), "XLM": pytest.approx(1000.0)}
    assert [(g.program, g.region, g.cohort, g.asset) for g in report.groups] == [
        ("cash-ng", "Africa", "C1", "USDC"),
        ("cash-ng", "Africa", "C2", "USDC"),
        ("edu-ke", "Africa", "C1", "XLM"),
        ("edu-ke", "Asia", "C1", "USDC"),
    ]

    first = report.groups[0]
    assert first.events == 2
    assert first.disbursements == 1
    assert first.receipts == 1
    assert first.recipients == 1
    assert first.disbursed_amount == pytest.approx(100.0)
    assert first.received_amount == pytest.approx(100.0)
    assert first.first_event_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert first.last_event_at == datetime(2026, 1, 1, 1, tzinfo=UTC)

    last = report.groups[-1]
    assert last.events == 1
    assert last.disbursed_amount == pytest.approx(25.0)
    assert last.received_amount == pytest.approx(0.0)

    as_dict = report.as_dict()
    assert as_dict["units"]["amount"] == "asset units"
    assert as_dict["dimension_keys"]["program"] == "program"
    assert as_dict["groups"][0]["first_event_at"].endswith("Z")


def test_observation_window_filters_all_events() -> None:
    report = compute_delivery_aggregates(
        load_fixture(),
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC),
        ),
    )
    assert report.events == 5
    assert report.disbursements == 3
    assert report.recipients == 3
    assert report.by_asset == {"USDC": pytest.approx(150.0), "XLM": pytest.approx(1000.0)}
    assert len(report.groups) == 3


def test_missing_dimensions_are_reported_as_unknown() -> None:
    event = load_fixture()[0]
    stripped = NormalizedEvent(**{**event.__dict__, "metadata": {}})
    report = compute_delivery_aggregates([stripped])

    assert len(report.groups) == 1
    assert (
        report.groups[0].program,
        report.groups[0].region,
        report.groups[0].cohort,
    ) == ("unknown", "unknown", "unknown")
    assert report.groups[0].disbursed_amount == pytest.approx(100.0)


def test_custom_dimension_keys() -> None:
    event = load_fixture()[0]
    renamed = NormalizedEvent(
        **{
            **event.__dict__,
            "metadata": {"initiative": "cash-ng", "geo": "Africa", "batch": "C1"},
        }
    )
    report = compute_delivery_aggregates(
        [renamed], program_key="initiative", region_key="geo", cohort_key="batch"
    )
    assert report.groups[0].program == "cash-ng"
    assert report.groups[0].region == "Africa"
    assert report.groups[0].cohort == "C1"


def test_invalid_inputs_raise() -> None:
    with pytest.raises(DeliveryAggregatesError):
        compute_delivery_aggregates("not-events")  # type: ignore[arg-type]
    with pytest.raises(DeliveryAggregatesError):
        compute_delivery_aggregates(load_fixture(), program_key="")
    with pytest.raises(DeliveryAggregatesError):
        compute_delivery_aggregates(load_fixture(), disbursement_account=" ")
    with pytest.raises(DeliveryAggregatesError):
        compute_delivery_aggregates(
            load_fixture(),
            observation_window={"start": "bad", "end": "2026-01-31T00:00:00Z"},
        )
