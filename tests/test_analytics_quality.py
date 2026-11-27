"""Tests for ingested-data freshness and completeness (issue #41)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from funding_optimizer.analytics.delivery import ObservationWindow
from funding_optimizer.analytics.quality import (
    DataQualityError,
    compute_data_quality,
)
from funding_optimizer.ingest.normalize import EventDirection, EventSource, NormalizedEvent

FIXTURE = Path(__file__).parent / "fixtures" / "data_quality.json"

EXPECTED = {
    "program": ["cash-ng", "edu-ke", "health-gh"],
    "region": ["Africa", "Asia"],
    "cohort": ["C1", "C2"],
}


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


def test_data_quality_known_fixture() -> None:
    # Hand-checkable: newest event is 2026-01-10, so lag at 2026-01-11 is
    # exactly 86400s. Programs are missing "health-gh" (6 of 7 expected
    # dimension values present).
    report = compute_data_quality(
        load_fixture(),
        as_of="2026-01-11T00:00:00Z",
        expected_dimensions=EXPECTED,
    )

    assert report.freshness.lag_seconds == pytest.approx(86400.0)
    assert report.freshness.fresh is True
    assert report.freshness.events_in_window == 4
    assert report.completeness.missing["program"] == ("health-gh",)
    assert report.completeness.missing["region"] == ()
    assert report.completeness.missing["cohort"] == ()
    assert report.completeness.coverage == pytest.approx(6 / 7)
    assert report.completeness.unattributed_events == 0

    as_dict = report.as_dict()
    units = cast(dict[str, str], as_dict["units"])
    assert units["lag"] == "seconds"
    assert units["coverage"] == "fraction"


def test_freshness_threshold_and_observation_window() -> None:
    report = compute_data_quality(
        load_fixture(), as_of="2026-01-11T00:00:00Z", max_lag_seconds=3600
    )
    assert report.freshness.fresh is False

    windowed = compute_data_quality(
        load_fixture(),
        as_of="2026-01-11T00:00:00Z",
        observation_window=ObservationWindow(
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 5, tzinfo=UTC),
        ),
    )
    assert windowed.freshness.events_in_window == 3
    assert windowed.freshness.lag_seconds == pytest.approx(8 * 86400.0)
    assert windowed.freshness.fresh is False


def test_missing_metadata_is_counted_as_unattributed() -> None:
    event = load_fixture()[0]
    untagged = NormalizedEvent(**{**event.__dict__, "metadata": {}})
    report = compute_data_quality([untagged], expected_dimensions=EXPECTED)

    assert report.completeness.unattributed_events == 1
    assert report.completeness.coverage == pytest.approx(0.0)


def test_empty_batch_is_not_fresh_and_has_full_coverage() -> None:
    report = compute_data_quality([], expected_dimensions=EXPECTED)
    assert report.freshness.fresh is False
    assert report.freshness.newest_event_at is None
    assert report.completeness.coverage == pytest.approx(0.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(DataQualityError):
        compute_data_quality("not-events")  # type: ignore[arg-type]
    with pytest.raises(DataQualityError):
        compute_data_quality(load_fixture(), max_lag_seconds=-1)
    with pytest.raises(DataQualityError):
        compute_data_quality(
            load_fixture(),
            observation_window={"start": "bad", "end": "2026-01-11T00:00:00Z"},
        )
