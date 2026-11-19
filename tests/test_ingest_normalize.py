"""Tests for the unified ingestion event model (issue #34).

Sanitized fixtures only. Both upstream shapes map onto the same frozen
event type, and every malformed record raises a deterministic, indexed
NormalizationError. No test performs a live network call.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from funding_optimizer.ingest.normalize import (
    EventDirection,
    EventSource,
    NormalizationError,
    normalize_analytics_record,
    normalize_analytics_records,
    normalize_horizon_record,
    normalize_horizon_records,
    normalize_record,
    normalize_records,
)

ISSUER = "G" + "A" * 55


def _horizon_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "123",
        "paging_token": "123",
        "transaction_hash": "deadbeef",
        "created_at": "2026-01-01T00:00:00Z",
        "type": "payment",
        "source_account": "GSENDER",
        "from": "GSENDER",
        "to": "GRECIPIENT",
        "asset_type": "native",
        "amount": "12.5000000",
    }
    record.update(overrides)
    return record


def _analytics_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "bucket-1",
        "timestamp": "2026-01-01T01:00:00+01:00",
        "type": "flow",
        "asset": "usdc:" + ISSUER,
        "amount": "150.25",
        "account": "GRECIPIENT",
        "counterparty": "GSENDER",
        "inflow": "150.25",
        "outflow": "20.00",
        "interval": "1h",
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------- #
# Horizon normalization                                                   #
# --------------------------------------------------------------------- #


def test_normalize_horizon_record_happy_path() -> None:
    event = normalize_horizon_record(_horizon_record(), index=2)

    assert event.source is EventSource.HORIZON
    assert event.event_id == "horizon:deadbeef:123"
    assert event.occurred_at == "2026-01-01T00:00:00Z"
    assert event.event_type == "payment"
    assert event.asset == "XLM"
    assert event.amount == pytest.approx(12.5)
    assert event.direction is EventDirection.IN
    assert event.account == "GRECIPIENT"
    assert event.counterparty == "GSENDER"
    assert event.metadata["transaction_hash"] == "deadbeef"


def test_normalize_horizon_record_issued_asset() -> None:
    event = normalize_horizon_record(
        _horizon_record(
            asset_type="credit_alphanum4",
            asset_code="usdc",
            asset_issuer=ISSUER,
        )
    )
    assert event.asset == "USDC:" + ISSUER


def test_normalize_horizon_record_subject_sender_flips_direction() -> None:
    event = normalize_horizon_record(_horizon_record(), subject="GSENDER")

    assert event.direction is EventDirection.OUT
    assert event.account == "GSENDER"
    assert event.counterparty == "GRECIPIENT"


def test_normalize_horizon_record_unknown_subject() -> None:
    event = normalize_horizon_record(_horizon_record(), subject="GUNRELATED")
    assert event.direction is EventDirection.UNKNOWN
    assert event.account == "GRECIPIENT"


def test_normalize_horizon_record_malformed_names_index() -> None:
    with pytest.raises(NormalizationError, match="record 7"):
        normalize_horizon_record(_horizon_record(amount="not-a-number"), index=7)


def test_normalize_horizon_record_rejects_non_mapping() -> None:
    with pytest.raises(NormalizationError, match="expected an object"):
        normalize_horizon_record(["not", "a", "record"])


# --------------------------------------------------------------------- #
# Analytics normalization                                                 #
# --------------------------------------------------------------------- #


def test_normalize_analytics_record_happy_path() -> None:
    event = normalize_analytics_record(_analytics_record(), index=3)

    assert event.source is EventSource.ANALYTICS
    assert event.event_id == "analytics:bucket-1"
    assert event.occurred_at == "2026-01-01T00:00:00Z"  # +01:00 -> UTC
    assert event.asset == "USDC:" + ISSUER
    assert event.amount == pytest.approx(150.25)
    assert event.direction is EventDirection.IN
    assert event.account == "GRECIPIENT"
    assert event.metadata["interval"] == "1h"


def test_normalize_analytics_record_outflow_direction() -> None:
    event = normalize_analytics_record(
        _analytics_record(inflow="10", outflow="40", amount="40")
    )
    assert event.direction is EventDirection.OUT
    assert event.amount == pytest.approx(40.0)


def test_normalize_analytics_record_subject_based_direction() -> None:
    record = _analytics_record()
    record.pop("inflow")
    record.pop("outflow")
    event = normalize_analytics_record(record, subject="GSENDER")
    assert event.direction is EventDirection.OUT


def test_normalize_analytics_record_synthesizes_stable_id() -> None:
    record = _analytics_record()
    record.pop("id")
    first = normalize_analytics_record(record, index=5)
    second = normalize_analytics_record(record, index=5)
    assert first.event_id == second.event_id
    assert first.event_id.startswith("analytics:5:")


def test_normalize_analytics_record_defaults_native_asset() -> None:
    record = _analytics_record()
    record.pop("asset")
    event = normalize_analytics_record(record)
    assert event.asset == "XLM"


def test_normalize_analytics_record_canonicalizes_naive_timestamp() -> None:
    event = normalize_analytics_record(_analytics_record(timestamp="2026-01-01 12:00:00"))
    assert event.occurred_at == "2026-01-01T12:00:00Z"


def test_normalize_analytics_record_malformed_fields() -> None:
    with pytest.raises(NormalizationError, match="amount"):
        normalize_analytics_record(_analytics_record(amount="bad"), index=1)
    with pytest.raises(NormalizationError, match="timestamp"):
        normalize_analytics_record(_analytics_record(timestamp="later"), index=2)
    with pytest.raises(NormalizationError, match="record 3"):
        normalize_analytics_record(_analytics_record(asset="!!!"), index=3)


# --------------------------------------------------------------------- #
# Unified entry points                                                    #
# --------------------------------------------------------------------- #


def test_normalize_record_dispatches_on_source() -> None:
    horizon = normalize_record(_horizon_record(), source=EventSource.HORIZON)
    analytics = normalize_record(_analytics_record(), source=EventSource.ANALYTICS)
    assert horizon.source is EventSource.HORIZON
    assert analytics.source is EventSource.ANALYTICS
    assert set(horizon.__dict__) == set(analytics.__dict__)

    with pytest.raises(NormalizationError, match="EventSource"):
        normalize_record(_horizon_record(), source=cast(EventSource, "horizon"))


def test_normalize_records_preserves_order_and_indexes_errors() -> None:
    horizon = normalize_horizon_records([_horizon_record(), _horizon_record()])
    assert [event.event_id for event in horizon] == [
        "horizon:deadbeef:123",
        "horizon:deadbeef:123",
    ]

    analytics = normalize_analytics_records([_analytics_record(), _analytics_record()])
    assert [event.event_type for event in analytics] == ["flow", "flow"]

    with pytest.raises(NormalizationError, match="record 1"):
        normalize_records(
            [_analytics_record(), _analytics_record(amount="bad")],
            source=EventSource.ANALYTICS,
        )


def test_normalize_records_rejects_non_iterable() -> None:
    with pytest.raises(NormalizationError, match="iterable"):
        normalize_records(cast(Any, 42), source=EventSource.HORIZON)
