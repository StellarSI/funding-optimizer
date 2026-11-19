"""Normalize Horizon and analytics-api records into one event model (#34).

The optimizer's feature and forecasting layers should not need to know which
upstream produced a record. This module maps both ingestion paths onto a
single frozen :class:`NormalizedEvent`:

- **Horizon** payment-like records are parsed with the strict Horizon loader
  (#32) and re-attributed into the shared model.
- **Analytics** flow records (from ``transaction-analytics-api``, #33) are
  parsed defensively from flat mappings; a summary becomes one event with a
  canonical timestamp and asset id.

Contract:
- ``occurred_at`` is normalized to a UTC ISO-8601 ``...Z`` string.
- ``asset`` is ``XLM`` for the native asset, otherwise a validated
  ``CODE:ISSUER`` identifier.
- ``amount`` is a non-negative, finite float; ``direction`` is ``in``, ``out``
  or ``unknown`` relative to ``subject`` when one is supplied.
- Every malformed record fails deterministically with a
  :class:`NormalizationError` naming the record index and field.
- Only scalar, sanitized values are carried in ``metadata``; no keys or
  recipient personal data are added. The normalizer never signs or submits
  payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any

from funding_optimizer.ingest.horizon import HorizonError, parse_payment_record
from funding_optimizer.models import validate_asset_code


class NormalizationError(ValueError):
    """Raised when a source record cannot be mapped to the event model."""


class EventSource(str, Enum):
    """Upstream that produced a normalized event."""

    HORIZON = "horizon"
    ANALYTICS = "analytics"


class EventDirection(str, Enum):
    """Direction of value movement relative to the event's account."""

    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizedEvent:
    """One value-movement event, independent of its upstream source."""

    event_id: str
    source: EventSource
    occurred_at: str
    event_type: str
    asset: str
    amount: float
    direction: EventDirection
    account: str
    counterparty: str | None
    metadata: Mapping[str, str]


def _require_mapping(record: Any, *, index: int, source: EventSource) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise NormalizationError(
            f"{source.value} record {index}: expected an object, got {record!r}"
        )
    return record


def _text(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    index: int,
    source: EventSource,
    default: str | None = None,
) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            raise NormalizationError(
                f"{source.value} record {index}: {key!r} must be a string, got {value!r}"
            )
    return default


def _amount(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    index: int,
    source: EventSource,
) -> float:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            raise NormalizationError(
                f"{source.value} record {index}: {key!r} must be a number, got {value!r}"
            )
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                raise NormalizationError(
                    f"{source.value} record {index}: {key!r} must be a number, got {value!r}"
                ) from None
        else:
            raise NormalizationError(
                f"{source.value} record {index}: {key!r} must be a number, got {value!r}"
            )
        if not isfinite(number) or number < 0:
            raise NormalizationError(
                f"{source.value} record {index}: {key!r} must be a non-negative, "
                f"finite number, got {value!r}"
            )
        return number
    raise NormalizationError(
        f"{source.value} record {index}: missing amount field "
        f"(expected one of {', '.join(keys)})"
    )


def _optional_amount(
    value: Any,
    key: str,
    *,
    index: int,
    source: EventSource,
) -> float | None:
    if value is None:
        return None
    return _amount({key: value}, (key,), index=index, source=source)


def _occurred_at(
    record: Mapping[str, Any], *, index: int, source: EventSource
) -> str:
    raw = _text(
        record,
        ("occurred_at", "created_at", "timestamp", "start"),
        index=index,
        source=source,
    )
    if raw is None:
        raise NormalizationError(
            f"{source.value} record {index}: missing timestamp "
            "(expected occurred_at, created_at, timestamp or start)"
        )
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise NormalizationError(
            f"{source.value} record {index}: invalid timestamp {raw!r}"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _asset(record: Mapping[str, Any], *, index: int, source: EventSource) -> str:
    code = _text(record, ("asset", "asset_code"), index=index, source=source)
    if code is None or code.upper() == "XLM":
        return "XLM"
    issuer = _text(record, ("asset_issuer", "issuer"), index=index, source=source)
    candidate = f"{code}:{issuer}" if issuer is not None else code
    try:
        return validate_asset_code(candidate)
    except (ValueError, TypeError) as exc:
        raise NormalizationError(f"{source.value} record {index}: {exc}") from None


def _direction(
    subject: str | None,
    account: str,
    counterparty: str | None,
    *,
    inflow: float | None,
    outflow: float | None,
) -> EventDirection:
    if inflow is not None and outflow is not None:
        if inflow > outflow:
            return EventDirection.IN
        if outflow > inflow:
            return EventDirection.OUT
        return EventDirection.UNKNOWN
    if subject is not None:
        if account == subject and counterparty != subject:
            return EventDirection.IN
        if counterparty == subject and account != subject:
            return EventDirection.OUT
        return EventDirection.UNKNOWN
    if account and counterparty and account != counterparty:
        return EventDirection.IN
    return EventDirection.UNKNOWN


def _metadata(record: Mapping[str, Any], *, source: EventSource) -> dict[str, str]:
    keys = (
        "transaction_hash",
        "paging_token",
        "interval",
        "payments",
        "counterparties",
        "net_flow",
        "type",
    )
    result: dict[str, str] = {}
    for key in keys:
        value = record.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            result[key] = str(value)
    return result


def normalize_horizon_record(
    record: Mapping[str, Any],
    *,
    subject: str | None = None,
    index: int = 0,
) -> NormalizedEvent:
    """Map one raw Horizon payment-like record to a :class:`NormalizedEvent`."""
    source = EventSource.HORIZON
    mapping = _require_mapping(record, index=index, source=source)
    try:
        payment = parse_payment_record(mapping, index=index)
    except HorizonError as exc:
        raise NormalizationError(f"horizon record {index}: {exc}") from exc

    account = payment.to_account
    counterparty = payment.from_account
    direction = _direction(subject, account, counterparty, inflow=None, outflow=None)
    if subject is not None and subject == counterparty and subject != account:
        account, counterparty = counterparty, account

    occurred_at = _occurred_at(
        {"created_at": payment.created_at}, index=index, source=source
    )
    metadata = {
        "transaction_hash": payment.transaction_hash,
        "paging_token": payment.paging_token,
        "type": payment.type,
    }
    return NormalizedEvent(
        event_id=f"horizon:{payment.transaction_hash}:{payment.id}",
        source=source,
        occurred_at=occurred_at,
        event_type=payment.type,
        asset=payment.asset,
        amount=payment.amount,
        direction=direction,
        account=account,
        counterparty=counterparty,
        metadata=metadata,
    )


def normalize_analytics_record(
    record: Mapping[str, Any],
    *,
    subject: str | None = None,
    index: int = 0,
) -> NormalizedEvent:
    """Map one flat analytics flow record to a :class:`NormalizedEvent`."""
    source = EventSource.ANALYTICS
    mapping = _require_mapping(record, index=index, source=source)

    account = (
        _text(mapping, ("account", "to", "destination"), index=index, source=source) or ""
    )
    counterparty = _text(
        mapping, ("from", "counterparty", "source_account"), index=index, source=source
    )
    inflow = _optional_amount(
        mapping.get("inflow"), "inflow", index=index, source=source
    )
    outflow = _optional_amount(
        mapping.get("outflow"), "outflow", index=index, source=source
    )
    amount = (
        _amount(mapping, ("amount", "volume"), index=index, source=source)
        if inflow is None and outflow is None
        else max(inflow or 0.0, outflow or 0.0)
    )
    direction = _direction(subject, account, counterparty, inflow=inflow, outflow=outflow)

    occurred_at = _occurred_at(mapping, index=index, source=source)
    raw_id = _text(mapping, ("id", "bucket_id", "event_id"), index=index, source=source)
    event_id = f"analytics:{raw_id}" if raw_id is not None else f"analytics:{index}:{occurred_at}"
    event_type = (
        _text(mapping, ("type", "event_type"), index=index, source=source) or "flow"
    )

    return NormalizedEvent(
        event_id=event_id,
        source=source,
        occurred_at=occurred_at,
        event_type=event_type,
        asset=_asset(mapping, index=index, source=source),
        amount=amount,
        direction=direction,
        account=account,
        counterparty=counterparty,
        metadata=_metadata(mapping, source=source),
    )


def normalize_record(
    record: Mapping[str, Any],
    *,
    source: EventSource,
    subject: str | None = None,
    index: int = 0,
) -> NormalizedEvent:
    """Normalize one record from the named upstream source."""
    if not isinstance(source, EventSource):
        raise NormalizationError(f"source must be an EventSource, got {source!r}")
    if source is EventSource.HORIZON:
        return normalize_horizon_record(record, subject=subject, index=index)
    return normalize_analytics_record(record, subject=subject, index=index)


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source: EventSource,
    subject: str | None = None,
) -> list[NormalizedEvent]:
    """Normalize a batch of records, preserving input order."""
    if not isinstance(records, Iterable):
        raise NormalizationError("records must be an iterable of mappings")
    return [
        normalize_record(record, source=source, subject=subject, index=index)
        for index, record in enumerate(records)
    ]


def normalize_horizon_records(
    records: Iterable[Mapping[str, Any]],
    *,
    subject: str | None = None,
) -> list[NormalizedEvent]:
    """Normalize a batch of Horizon payment records."""
    return normalize_records(records, source=EventSource.HORIZON, subject=subject)


def normalize_analytics_records(
    records: Iterable[Mapping[str, Any]],
    *,
    subject: str | None = None,
) -> list[NormalizedEvent]:
    """Normalize a batch of transaction-analytics-api flow records."""
    return normalize_records(records, source=EventSource.ANALYTICS, subject=subject)
