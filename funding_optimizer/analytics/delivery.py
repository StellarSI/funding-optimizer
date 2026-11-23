"""Time-to-receipt delivery analytics (issue #36).

Computes how long it took recipients to receive funds after a disbursement,
using normalized events from P34. The optimizer recommends allocations and
never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from funding_optimizer.ingest.normalize import NormalizedEvent


class DeliveryAnalyticsError(ValueError):
    """Raised when delivery analytics inputs are invalid."""


@dataclass(frozen=True)
class ObservationWindow:
    """Inclusive UTC observation window for delivery analytics."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _as_utc(self.start, "start")
        end = _as_utc(self.end, "end")
        if start > end:
            raise DeliveryAnalyticsError(
                f"observation window start {start.isoformat()} is after end {end.isoformat()}"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, moment: datetime) -> bool:
        ts = _as_utc(moment, "moment")
        return self.start <= ts <= self.end


@dataclass(frozen=True)
class ReceiptMatch:
    """One matched disbursement-to-receipt observation."""

    disbursement_event_id: str
    receipt_event_id: str
    account: str
    counterparty: str | None
    asset: str
    amount: float
    disbursed_at: datetime
    received_at: datetime
    receipt_seconds: float

    @property
    def receipt_hours(self) -> float:
        return self.receipt_seconds / 3600.0

    @property
    def receipt_days(self) -> float:
        return self.receipt_seconds / 86400.0


@dataclass(frozen=True)
class TimeToReceiptMetrics:
    """Time-to-receipt metrics computed from normalized events.

    Units:
    - observation window bounds are UTC datetimes
    - receipt durations are measured in seconds
    - summary metrics are reported in seconds
    """

    observation_window: ObservationWindow | None
    disbursements: int
    matched_receipts: int
    unmatched_disbursements: int
    match_rate: float
    mean_receipt_seconds: float | None
    median_receipt_seconds: float | None
    p95_receipt_seconds: float | None
    min_receipt_seconds: float | None
    max_receipt_seconds: float | None
    matches: Sequence[ReceiptMatch]

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_window": (
                None
                if self.observation_window is None
                else {
                    "start": self.observation_window.start.isoformat().replace("+00:00", "Z"),
                    "end": self.observation_window.end.isoformat().replace("+00:00", "Z"),
                    "unit": "utc-iso8601",
                }
            ),
            "units": {"receipt_duration": "seconds"},
            "disbursements": self.disbursements,
            "matched_receipts": self.matched_receipts,
            "unmatched_disbursements": self.unmatched_disbursements,
            "match_rate": self.match_rate,
            "mean_receipt_seconds": self.mean_receipt_seconds,
            "median_receipt_seconds": self.median_receipt_seconds,
            "p95_receipt_seconds": self.p95_receipt_seconds,
            "min_receipt_seconds": self.min_receipt_seconds,
            "max_receipt_seconds": self.max_receipt_seconds,
            "matches": [
                {
                    "disbursement_event_id": m.disbursement_event_id,
                    "receipt_event_id": m.receipt_event_id,
                    "account": m.account,
                    "counterparty": m.counterparty,
                    "asset": m.asset,
                    "amount": m.amount,
                    "disbursed_at": m.disbursed_at.isoformat().replace("+00:00", "Z"),
                    "received_at": m.received_at.isoformat().replace("+00:00", "Z"),
                    "receipt_seconds": m.receipt_seconds,
                }
                for m in self.matches
            ],
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DeliveryAnalyticsError(
                f"{name} must be an ISO-8601 datetime, got {value!r}"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise DeliveryAnalyticsError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise DeliveryAnalyticsError("cannot compute percentile of empty values")
    if not 0.0 <= p <= 1.0:
        raise DeliveryAnalyticsError("percentile must be between 0 and 1")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise DeliveryAnalyticsError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for i, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise DeliveryAnalyticsError(
                f"events[{i}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


def compute_time_to_receipt_metrics(
    events: Iterable[NormalizedEvent],
    *,
    disbursement_account: str | None = None,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
) -> TimeToReceiptMetrics:
    """Compute time-to-receipt metrics from normalized P34 events.

    Matching rule:
    - A disbursement is an event with direction OUT. If `disbursement_account`
      is provided, only OUT events from that account are considered.
    - A receipt is the first IN event for the same counterparty account and
      asset with occurred_at >= disbursement occurred_at.
    - Durations are computed in seconds.

    The observation window, when provided, filters disbursements by their
    occurred_at timestamp. Receipt may occur after the disbursement inside the
    provided event set.
    """
    normalized = _coerce_events(events)
    if isinstance(observation_window, Mapping):
        observation_window = ObservationWindow(
            start=_as_utc(observation_window["start"], "observation_window.start"),
            end=_as_utc(observation_window["end"], "observation_window.end"),
        )
    elif observation_window is not None and not isinstance(observation_window, ObservationWindow):
        raise DeliveryAnalyticsError(
            "observation_window must be an ObservationWindow, mapping, or None"
        )

    subject = disbursement_account.strip() if isinstance(disbursement_account, str) else None
    if disbursement_account is not None and not subject:
        raise DeliveryAnalyticsError(
            "disbursement_account must be a non-empty string when provided"
        )

    ordered = sorted(normalized, key=lambda e: (_event_time(e), e.event_id))
    disbursements = [
        event
        for event in ordered
        if event.direction.value == "out"
        and event.counterparty
        and (subject is None or event.account == subject)
        and (observation_window is None or observation_window.contains(_event_time(event)))
    ]

    matches: list[ReceiptMatch] = []
    used_receipts: set[str] = set()
    for disbursement in disbursements:
        disbursed_at = _event_time(disbursement)
        receipt = None
        for candidate in ordered:
            if candidate.event_id in used_receipts:
                continue
            if candidate.direction.value != "in":
                continue
            if candidate.asset != disbursement.asset:
                continue
            if candidate.account != disbursement.counterparty:
                continue
            if subject is not None and candidate.counterparty not in (None, subject):
                # If counterparty is known, prefer receipts from the disbursement account.
                continue
            received_at = _event_time(candidate)
            if received_at < disbursed_at:
                continue
            receipt = candidate
            break
        if receipt is None:
            continue
        received_at = _event_time(receipt)
        seconds = (received_at - disbursed_at).total_seconds()
        if not isfinite(seconds) or seconds < 0:
            raise DeliveryAnalyticsError(
                f"invalid receipt duration for disbursement {disbursement.event_id}"
            )
        used_receipts.add(receipt.event_id)
        matches.append(
            ReceiptMatch(
                disbursement_event_id=disbursement.event_id,
                receipt_event_id=receipt.event_id,
                account=disbursement.account,
                counterparty=disbursement.counterparty,
                asset=disbursement.asset,
                amount=disbursement.amount,
                disbursed_at=disbursed_at,
                received_at=received_at,
                receipt_seconds=float(seconds),
            )
        )

    durations = sorted(m.receipt_seconds for m in matches)
    matched = len(matches)
    total = len(disbursements)
    unmatched = total - matched
    return TimeToReceiptMetrics(
        observation_window=observation_window,
        disbursements=total,
        matched_receipts=matched,
        unmatched_disbursements=unmatched,
        match_rate=(matched / total) if total else 0.0,
        mean_receipt_seconds=(sum(durations) / len(durations)) if durations else None,
        median_receipt_seconds=_percentile(durations, 0.5) if durations else None,
        p95_receipt_seconds=_percentile(durations, 0.95) if durations else None,
        min_receipt_seconds=min(durations) if durations else None,
        max_receipt_seconds=max(durations) if durations else None,
        matches=tuple(matches),
    )
