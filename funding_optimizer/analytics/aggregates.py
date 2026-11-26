"""Delivery aggregates by program, region, cohort and asset (issue #40).

Aggregates normalized P34 events into delivery metrics split by the four
dimensions funders report on: program, region, cohort and asset. Program,
region and cohort are read from event metadata; asset comes from the event
itself. The four dimensions are always reported together so a number can
never be silently attributed to the wrong group.

Units and conventions:

- observation windows are inclusive UTC datetimes
- amounts are in asset units and are **never summed across assets**
- event/count fields are counts; recipients are distinct account ids
- first/last timestamps are UTC ISO-8601 strings in ``as_dict()``

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from funding_optimizer.ingest.normalize import NormalizedEvent

DEFAULT_PROGRAM_KEY = "program"
DEFAULT_REGION_KEY = "region"
DEFAULT_COHORT_KEY = "cohort"
UNKNOWN_DIMENSION = "unknown"


class DeliveryAggregatesError(ValueError):
    """Raised when delivery aggregate inputs are invalid."""


@dataclass(frozen=True)
class AggregateMetrics:
    """Delivery metrics for one (program, region, cohort, asset) group.

    Units: amounts in asset units; counts are integers; timestamps are UTC.
    """

    program: str
    region: str
    cohort: str
    asset: str
    events: int
    disbursements: int
    receipts: int
    recipients: int
    disbursed_amount: float
    received_amount: float
    first_event_at: datetime
    last_event_at: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "program": self.program,
            "region": self.region,
            "cohort": self.cohort,
            "asset": self.asset,
            "events": self.events,
            "disbursements": self.disbursements,
            "receipts": self.receipts,
            "recipients": self.recipients,
            "disbursed_amount": self.disbursed_amount,
            "received_amount": self.received_amount,
            "first_event_at": self.first_event_at.isoformat().replace("+00:00", "Z"),
            "last_event_at": self.last_event_at.isoformat().replace("+00:00", "Z"),
        }


@dataclass(frozen=True)
class AggregateReport:
    """Delivery aggregates across all program/region/cohort/asset groups."""

    observation_window: ObservationWindow | None
    dimension_keys: Mapping[str, str]
    groups: Sequence[AggregateMetrics]
    events: int
    disbursements: int
    receipts: int
    recipients: int
    by_asset: Mapping[str, float]

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
            "units": {
                "amount": "asset units",
                "timestamp": "utc-iso8601",
                "counts": "integers",
            },
            "dimension_keys": dict(self.dimension_keys),
            "events": self.events,
            "disbursements": self.disbursements,
            "receipts": self.receipts,
            "recipients": self.recipients,
            "by_asset": dict(self.by_asset),
            "groups": [group.as_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class ObservationWindow:
    """Inclusive UTC observation window for delivery aggregates."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _as_utc(self.start, "start")
        end = _as_utc(self.end, "end")
        if start > end:
            raise DeliveryAggregatesError(
                f"observation window start {start.isoformat()} is after end {end.isoformat()}"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, moment: datetime) -> bool:
        ts = _as_utc(moment, "moment")
        return self.start <= ts <= self.end


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DeliveryAggregatesError(
                f"{name} must be an ISO-8601 datetime, got {value!r}"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise DeliveryAggregatesError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _dimension(event: NormalizedEvent, key: str) -> str:
    value = event.metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNKNOWN_DIMENSION


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise DeliveryAggregatesError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise DeliveryAggregatesError(
                f"events[{index}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


@dataclass
class _Accumulator:
    events: int = 0
    disbursements: int = 0
    receipts: int = 0
    recipients: set[str] | None = None
    disbursed_amount: float = 0.0
    received_amount: float = 0.0
    first_event_at: datetime | None = None
    last_event_at: datetime | None = None

    def add(self, event: NormalizedEvent, moment: datetime) -> None:
        self.events += 1
        if self.recipients is None:
            self.recipients = set()
        if event.direction.value == "out":
            self.disbursements += 1
            if event.counterparty:
                self.recipients.add(event.counterparty)
            self.disbursed_amount += event.amount
        elif event.direction.value == "in":
            self.receipts += 1
            self.recipients.add(event.account)
            self.received_amount += event.amount
        if self.first_event_at is None or moment < self.first_event_at:
            self.first_event_at = moment
        if self.last_event_at is None or moment > self.last_event_at:
            self.last_event_at = moment


def compute_delivery_aggregates(
    events: Iterable[NormalizedEvent],
    *,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
    disbursement_account: str | None = None,
    program_key: str = DEFAULT_PROGRAM_KEY,
    region_key: str = DEFAULT_REGION_KEY,
    cohort_key: str = DEFAULT_COHORT_KEY,
) -> AggregateReport:
    """Aggregate delivery metrics by program, region, cohort and asset.

    Every event inside the inclusive observation window contributes to exactly
    one group. OUT events contribute to ``disbursements`` and
    ``disbursed_amount``; IN events contribute to ``receipts`` and
    ``received_amount``. ``disbursement_account``, when provided, restricts
    OUT events to that account (the normalizer already orients directions to
    a subject, so most callers do not need it).

    Amounts are tracked per asset and never summed across assets; the report's
    ``by_asset`` mapping keeps disbursed totals separated. Missing or blank
    metadata values are reported as ``"unknown"`` instead of dropping events.
    """
    normalized = _coerce_events(events)

    window: ObservationWindow | None
    if isinstance(observation_window, Mapping):
        window = ObservationWindow(
            start=_as_utc(observation_window["start"], "observation_window.start"),
            end=_as_utc(observation_window["end"], "observation_window.end"),
        )
    elif observation_window is not None and not isinstance(observation_window, ObservationWindow):
        raise DeliveryAggregatesError(
            "observation_window must be an ObservationWindow, mapping, or None"
        )
    else:
        window = observation_window

    subject = disbursement_account.strip() if isinstance(disbursement_account, str) else None
    if disbursement_account is not None and not subject:
        raise DeliveryAggregatesError(
            "disbursement_account must be a non-empty string when provided"
        )

    keys = {
        "program": program_key,
        "region": region_key,
        "cohort": cohort_key,
    }
    for name, key in keys.items():
        if not isinstance(key, str) or not key.strip():
            raise DeliveryAggregatesError(f"{name}_key must be a non-empty string")
    keys = {name: key.strip() for name, key in keys.items()}

    accumulators: dict[tuple[str, str, str, str], _Accumulator] = {}
    for event in normalized:
        moment = _event_time(event)
        if window is not None and not window.contains(moment):
            continue
        if subject is not None and event.direction.value == "out" and event.account != subject:
            continue
        group_key = (
            _dimension(event, keys["program"]),
            _dimension(event, keys["region"]),
            _dimension(event, keys["cohort"]),
            event.asset,
        )
        accumulators.setdefault(group_key, _Accumulator()).add(event, moment)

    groups: list[AggregateMetrics] = []
    for (program, region, cohort, asset), acc in sorted(accumulators.items()):
        first = acc.first_event_at
        last = acc.last_event_at
        if first is None or last is None:  # pragma: no cover - groups always have an event
            continue
        if not isfinite(acc.disbursed_amount) or not isfinite(acc.received_amount):
            raise DeliveryAggregatesError(f"non-finite amount aggregated for asset {asset}")
        groups.append(
            AggregateMetrics(
                program=program,
                region=region,
                cohort=cohort,
                asset=asset,
                events=acc.events,
                disbursements=acc.disbursements,
                receipts=acc.receipts,
                recipients=len(acc.recipients or ()),
                disbursed_amount=acc.disbursed_amount,
                received_amount=acc.received_amount,
                first_event_at=first,
                last_event_at=last,
            )
        )

    total_recipients: set[str] = set()
    for acc in accumulators.values():
        total_recipients.update(acc.recipients or ())

    by_asset: dict[str, float] = {}
    for group in groups:
        by_asset[group.asset] = by_asset.get(group.asset, 0.0) + group.disbursed_amount

    return AggregateReport(
        observation_window=window,
        dimension_keys=dict(keys),
        groups=tuple(groups),
        events=sum(group.events for group in groups),
        disbursements=sum(group.disbursements for group in groups),
        receipts=sum(group.receipts for group in groups),
        recipients=len(total_recipients),
        by_asset=by_asset,
    )


__all__ = [
    "DEFAULT_COHORT_KEY",
    "DEFAULT_PROGRAM_KEY",
    "DEFAULT_REGION_KEY",
    "UNKNOWN_DIMENSION",
    "AggregateMetrics",
    "AggregateReport",
    "DeliveryAggregatesError",
    "ObservationWindow",
    "compute_delivery_aggregates",
]
