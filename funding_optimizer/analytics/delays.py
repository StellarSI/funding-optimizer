"""Delivery-delay distributions by corridor and program (issue #39).

Builds on the time-to-receipt matcher (issue #36) and the normalized event
model (P34) to report how long delivery actually took, grouped by corridor
and program. Distributions are summarized with percentiles and a histogram
so funders can see both the typical delay and the long tail.

Units and conventions:

- observation windows are inclusive UTC datetimes
- all delays are measured in **seconds**
- `share` values are fractions in [0, 1]

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from funding_optimizer.analytics.delivery import (
    DeliveryAnalyticsError,
    ObservationWindow,
    compute_time_to_receipt_metrics,
)
from funding_optimizer.ingest.normalize import NormalizedEvent

DEFAULT_DELAY_BUCKETS_SECONDS: tuple[float, ...] = (
    3600.0,    # 1 hour
    21600.0,   # 6 hours
    86400.0,   # 1 day
    604800.0,  # 7 days
)

DEFAULT_CORRIDOR_KEY = "corridor"
DEFAULT_PROGRAM_KEY = "program"
UNKNOWN_DIMENSION = "unknown"


class DeliveryDelayError(ValueError):
    """Raised when delivery-delay analytics inputs are invalid."""


@dataclass(frozen=True)
class AssetDelaySummary:
    """Delay summary for one asset inside a corridor/program group.

    Units: amounts are not included; every duration is in seconds.
    """

    asset: str
    observations: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float


@dataclass(frozen=True)
class DeliveryDelayDistribution:
    """Delivery-delay distribution for one (corridor, program) pair."""

    corridor: str
    program: str
    observations: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    p90_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    histogram: Mapping[str, int]
    by_asset: Sequence[AssetDelaySummary]

    def as_dict(self) -> dict[str, object]:
        return {
            "corridor": self.corridor,
            "program": self.program,
            "observations": self.observations,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "p90_seconds": self.p90_seconds,
            "p95_seconds": self.p95_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "histogram": dict(self.histogram),
            "by_asset": [
                {
                    "asset": entry.asset,
                    "observations": entry.observations,
                    "mean_seconds": entry.mean_seconds,
                    "median_seconds": entry.median_seconds,
                    "p95_seconds": entry.p95_seconds,
                }
                for entry in self.by_asset
            ],
        }


@dataclass(frozen=True)
class DeliveryDelayReport:
    """Delivery-delay distributions across corridors and programs."""

    observation_window: ObservationWindow | None
    disbursement_account: str | None
    bucket_bounds_seconds: Sequence[float]
    groups: Sequence[DeliveryDelayDistribution]
    matched_receipts: int

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
            "units": {"delay": "seconds"},
            "disbursement_account": self.disbursement_account,
            "bucket_bounds_seconds": list(self.bucket_bounds_seconds),
            "matched_receipts": self.matched_receipts,
            "groups": [group.as_dict() for group in self.groups],
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DeliveryDelayError(
                f"{name} must be an ISO-8601 datetime, got {value!r}"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise DeliveryDelayError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise DeliveryDelayError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise DeliveryDelayError(
                f"events[{index}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


def _resolve_window(
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None,
) -> ObservationWindow | None:
    if isinstance(observation_window, Mapping):
        return ObservationWindow(
            start=_as_utc(observation_window["start"], "observation_window.start"),
            end=_as_utc(observation_window["end"], "observation_window.end"),
        )
    if observation_window is not None and not isinstance(observation_window, ObservationWindow):
        raise DeliveryDelayError(
            "observation_window must be an ObservationWindow, mapping, or None"
        )
    return observation_window


def _resolve_buckets(buckets: Sequence[float]) -> tuple[float, ...]:
    if isinstance(buckets, (str, bytes)) or not isinstance(buckets, Sequence):
        raise DeliveryDelayError("bucket_bounds_seconds must be a sequence of seconds")
    parsed = tuple(float(bound) for bound in buckets)
    previous = 0.0
    for bound in parsed:
        if not isfinite(bound) or bound <= previous:
            raise DeliveryDelayError(
                "bucket_bounds_seconds must be finite and strictly increasing"
            )
        previous = bound
    return parsed


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise DeliveryDelayError("cannot compute a percentile of empty delays")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _format_bound(seconds: float) -> str:
    if seconds >= 86400 and seconds % 86400 == 0:
        return f"{int(seconds / 86400)}d"
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{int(seconds / 3600)}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{int(seconds / 60)}m"
    return f"{seconds:g}s"


def _histogram_labels(bounds: Sequence[float]) -> list[str]:
    labels = [f"0-{_format_bound(bounds[0])}"]
    labels.extend(
        f"{_format_bound(lower)}-{_format_bound(upper)}"
        for lower, upper in zip(bounds, bounds[1:])
    )
    labels.append(f"{_format_bound(bounds[-1])}+")
    return labels


def _dimension(event: NormalizedEvent, key: str) -> str:
    value = event.metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNKNOWN_DIMENSION


def compute_delivery_delay_distributions(
    events: Iterable[NormalizedEvent],
    *,
    disbursement_account: str | None = None,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
    corridor_key: str = DEFAULT_CORRIDOR_KEY,
    program_key: str = DEFAULT_PROGRAM_KEY,
    bucket_bounds_seconds: Sequence[float] = DEFAULT_DELAY_BUCKETS_SECONDS,
) -> DeliveryDelayReport:
    """Compute delivery-delay distributions from normalized P34 events.

    Disbursement/receipt pairs come from :func:`compute_time_to_receipt_metrics`,
    so the same matching rule applies: an OUT event with a counterparty is
    paired with the earliest unused IN event for the same asset received by
    that counterparty at or after disbursement. The observation window filters
    disbursements; as with time-to-receipt, a receipt may occur after it.

    Corridor and program are read from ``event.metadata`` using
    ``corridor_key`` and ``program_key``; missing or blank values are reported
    as ``"unknown"`` instead of dropping the observation. Asset is retained as
    a sub-dimension so delays from different assets are never averaged into
    one misleading number.
    """
    normalized = _coerce_events(events)
    window = _resolve_window(observation_window)
    bounds = _resolve_buckets(bucket_bounds_seconds)
    labels = _histogram_labels(bounds)
    if not isinstance(corridor_key, str) or not corridor_key.strip():
        raise DeliveryDelayError("corridor_key must be a non-empty string")
    if not isinstance(program_key, str) or not program_key.strip():
        raise DeliveryDelayError("program_key must be a non-empty string")

    matches = compute_time_to_receipt_metrics(
        normalized,
        disbursement_account=disbursement_account,
        observation_window=window,
    ).matches
    events_by_id = {event.event_id: event for event in normalized}

    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for match in matches:
        disbursement = events_by_id.get(match.disbursement_event_id)
        if disbursement is None:
            continue
        corridor = _dimension(disbursement, corridor_key.strip())
        program = _dimension(disbursement, program_key.strip())
        grouped.setdefault((corridor, program), []).append((match.asset, match.receipt_seconds))

    groups: list[DeliveryDelayDistribution] = []
    for (corridor, program), observations in sorted(grouped.items()):
        delays = sorted(seconds for _, seconds in observations)
        histogram = {label: 0 for label in labels}
        for delay in delays:
            index = 0
            while index < len(bounds) and delay > bounds[index]:
                index += 1
            histogram[labels[index]] += 1

        by_asset: dict[str, list[float]] = {}
        for asset, seconds in observations:
            by_asset.setdefault(asset, []).append(seconds)
        asset_summaries = tuple(
            AssetDelaySummary(
                asset=asset,
                observations=len(values),
                mean_seconds=sum(values) / len(values),
                median_seconds=_percentile(sorted(values), 0.5),
                p95_seconds=_percentile(sorted(values), 0.95),
            )
            for asset, values in sorted(by_asset.items())
        )

        groups.append(
            DeliveryDelayDistribution(
                corridor=corridor,
                program=program,
                observations=len(delays),
                total_seconds=sum(delays),
                mean_seconds=sum(delays) / len(delays),
                median_seconds=_percentile(delays, 0.5),
                p90_seconds=_percentile(delays, 0.9),
                p95_seconds=_percentile(delays, 0.95),
                min_seconds=delays[0],
                max_seconds=delays[-1],
                histogram=histogram,
                by_asset=asset_summaries,
            )
        )

    return DeliveryDelayReport(
        observation_window=window,
        disbursement_account=disbursement_account.strip() if isinstance(disbursement_account, str) else None,
        bucket_bounds_seconds=bounds,
        groups=tuple(groups),
        matched_receipts=len(matches),
    )


__all__ = [
    "DEFAULT_CORRIDOR_KEY",
    "DEFAULT_DELAY_BUCKETS_SECONDS",
    "DEFAULT_PROGRAM_KEY",
    "UNKNOWN_DIMENSION",
    "AssetDelaySummary",
    "DeliveryDelayDistribution",
    "DeliveryDelayError",
    "DeliveryDelayReport",
    "compute_delivery_delay_distributions",
]
