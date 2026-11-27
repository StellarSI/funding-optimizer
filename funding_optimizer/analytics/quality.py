"""Freshness and completeness checks for ingested delivery data (issue #41).

Consumes normalized P34 events and reports whether the data is recent enough
to act on and whether the expected programs, regions and cohorts are present.
The optimizer recommends allocations and never signs or submits payouts.

Units and conventions:

- observation windows are inclusive UTC datetimes
- ``as_of`` is an inclusive UTC reference moment (defaults to the newest event)
- freshness lag is measured in **seconds**
- coverage and rates are fractions in [0, 1]; counts are integers
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from funding_optimizer.analytics.delivery import ObservationWindow
from funding_optimizer.ingest.normalize import NormalizedEvent

DEFAULT_PROGRAM_KEY = "program"
DEFAULT_REGION_KEY = "region"
DEFAULT_COHORT_KEY = "cohort"
DEFAULT_MAX_LAG_SECONDS = 86_400.0
UNKNOWN_DIMENSION = "unknown"


class DataQualityError(ValueError):
    """Raised when data-quality inputs are invalid."""


@dataclass(frozen=True)
class FreshnessCheck:
    """How recent the ingested events are relative to ``as_of``.

    Units: all timestamps are UTC; ``lag_seconds`` is in seconds.
    """

    as_of: datetime
    newest_event_at: datetime | None
    lag_seconds: float | None
    max_lag_seconds: float
    fresh: bool
    events_in_window: int

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat().replace("+00:00", "Z"),
            "newest_event_at": (
                None
                if self.newest_event_at is None
                else self.newest_event_at.isoformat().replace("+00:00", "Z")
            ),
            "lag_seconds": self.lag_seconds,
            "max_lag_seconds": self.max_lag_seconds,
            "fresh": self.fresh,
            "events_in_window": self.events_in_window,
        }


@dataclass(frozen=True)
class CompletenessCheck:
    """Expected versus observed dimension values in the ingested events.

    Units: ``coverage`` is a fraction in [0, 1]; counts are integers.
    """

    expected: Mapping[str, tuple[str, ...]]
    observed: Mapping[str, tuple[str, ...]]
    missing: Mapping[str, tuple[str, ...]]
    coverage: float
    unattributed_events: int
    events_checked: int

    def as_dict(self) -> dict[str, object]:
        return {
            "expected": {key: list(values) for key, values in self.expected.items()},
            "observed": {key: list(values) for key, values in self.observed.items()},
            "missing": {key: list(values) for key, values in self.missing.items()},
            "coverage": self.coverage,
            "unattributed_events": self.unattributed_events,
            "events_checked": self.events_checked,
        }


@dataclass(frozen=True)
class DataQualityReport:
    """Freshness and completeness results for one ingested event batch."""

    observation_window: ObservationWindow | None
    freshness: FreshnessCheck
    completeness: CompletenessCheck

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
            "units": {"lag": "seconds", "coverage": "fraction", "counts": "integers"},
            "freshness": self.freshness.as_dict(),
            "completeness": self.completeness.as_dict(),
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DataQualityError(f"{name} must be an ISO-8601 datetime, got {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise DataQualityError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise DataQualityError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise DataQualityError(
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
        raise DataQualityError("observation_window must be an ObservationWindow, mapping, or None")
    return observation_window


def _resolve_max_lag(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataQualityError("max_lag_seconds must be a number")
    lag = float(value)
    if not isfinite(lag) or lag < 0:
        raise DataQualityError("max_lag_seconds must be a non-negative finite number")
    return lag


def _normalize_expected(
    expected_dimensions: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if expected_dimensions is None:
        return {}
    if not isinstance(expected_dimensions, Mapping):
        raise DataQualityError("expected_dimensions must be a mapping of name to values")
    normalized: dict[str, tuple[str, ...]] = {}
    for name, values in expected_dimensions.items():
        if not isinstance(name, str) or not name.strip():
            raise DataQualityError("expected_dimensions keys must be non-empty strings")
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise DataQualityError(f"expected_dimensions[{name!r}] must be an iterable of strings")
        cleaned = tuple(
            sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})
        )
        normalized[name.strip()] = cleaned
    return normalized


def _dimension(event: NormalizedEvent, key: str) -> str:
    value = event.metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNKNOWN_DIMENSION


def compute_data_quality(
    events: Iterable[NormalizedEvent],
    *,
    as_of: datetime | str | None = None,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    expected_dimensions: Mapping[str, Iterable[str]] | None = None,
    program_key: str = DEFAULT_PROGRAM_KEY,
    region_key: str = DEFAULT_REGION_KEY,
    cohort_key: str = DEFAULT_COHORT_KEY,
) -> DataQualityReport:
    """Compute freshness and completeness metrics from normalized P34 events.

    The observation window filters events for both checks. ``as_of`` defaults
    to the newest event in the *unfiltered* batch, so an empty window can
    still be described without inventing an external clock. Freshness is
    measured as ``as_of - newest_event_at`` in seconds and reported as fresh
    when the lag is at most ``max_lag_seconds`` (an empty window is not
    fresh). Completeness compares the observed metadata values for each
    expected dimension with the supplied expectations; events whose metadata
    is missing for a checked dimension are counted as unattributed.
    """
    normalized = _coerce_events(events)
    window = _resolve_window(observation_window)
    lag_limit = _resolve_max_lag(max_lag_seconds)
    expected = _normalize_expected(expected_dimensions)

    keys = {
        "program": program_key,
        "region": region_key,
        "cohort": cohort_key,
    }
    for name, key in keys.items():
        if not isinstance(key, str) or not key.strip():
            raise DataQualityError(f"{name}_key must be a non-empty string")
    keys = {name: key.strip() for name, key in keys.items()}

    all_times = sorted(_event_time(event) for event in normalized)
    newest_overall = all_times[-1] if all_times else None
    reference = _as_utc(as_of, "as_of") if as_of is not None else newest_overall

    in_window = [
        event for event in normalized if window is None or window.contains(_event_time(event))
    ]
    window_times = sorted(_event_time(event) for event in in_window)
    newest_in_window = window_times[-1] if window_times else None

    if reference is None or newest_in_window is None:
        freshness = FreshnessCheck(
            as_of=reference or _as_utc("1970-01-01T00:00:00Z", "as_of"),
            newest_event_at=newest_in_window,
            lag_seconds=None,
            max_lag_seconds=lag_limit,
            fresh=False,
            events_in_window=len(in_window),
        )
    else:
        seconds = (reference - newest_in_window).total_seconds()
        if not isfinite(seconds) or seconds < 0:
            raise DataQualityError("freshness lag must be a non-negative finite duration")
        freshness = FreshnessCheck(
            as_of=reference,
            newest_event_at=newest_in_window,
            lag_seconds=float(seconds),
            max_lag_seconds=lag_limit,
            fresh=float(seconds) <= lag_limit,
            events_in_window=len(in_window),
        )

    checked_names: list[str] = []
    for name, key in keys.items():
        if name in expected:
            checked_names.append(name)
    checked_names = sorted(set(checked_names) | set(expected))

    observed: dict[str, tuple[str, ...]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    total_expected = 0
    total_present = 0
    for name in checked_names:
        values = {_dimension(event, keys.get(name, name)) for event in in_window}
        observed[name] = tuple(sorted(values - {UNKNOWN_DIMENSION}))
        expected_values = expected.get(name, observed[name])
        missing[name] = tuple(sorted(set(expected_values) - set(observed[name])))
        total_expected += len(expected_values)
        total_present += len(expected_values) - len(missing[name])

    unattributed = sum(
        1
        for event in in_window
        if any(
            _dimension(event, keys.get(name, name)) == UNKNOWN_DIMENSION for name in checked_names
        )
    )

    completeness = CompletenessCheck(
        expected={name: expected.get(name, observed[name]) for name in checked_names},
        observed=observed,
        missing=missing,
        coverage=(total_present / total_expected) if total_expected else 1.0,
        unattributed_events=unattributed,
        events_checked=len(in_window),
    )

    return DataQualityReport(
        observation_window=window,
        freshness=freshness,
        completeness=completeness,
    )


__all__ = [
    "DEFAULT_COHORT_KEY",
    "DEFAULT_MAX_LAG_SECONDS",
    "DEFAULT_PROGRAM_KEY",
    "DEFAULT_REGION_KEY",
    "UNKNOWN_DIMENSION",
    "CompletenessCheck",
    "DataQualityError",
    "DataQualityReport",
    "FreshnessCheck",
    "compute_data_quality",
]
