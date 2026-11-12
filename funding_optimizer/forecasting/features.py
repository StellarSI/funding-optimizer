"""Reusable forecasting features (issue #26).

Builds typed, reusable features from forecast observations keyed by
program, region, asset, and calendar attributes of the observation window.
These features give downstream forecast models (cash-out baseline,
delivery delays, funding-gap models) a common, transparent representation
without embedding model-specific logic.

The module consumes validated :class:`~.schema.ForecastDataset` /
:class:`~.schema.ForecastRow` / :class:`~.schema.TimeWindow` values and
never performs network calls or moves funds.

Examples
--------
>>> from datetime import date
>>> from funding_optimizer.forecasting import (
...     ForecastDataset, ForecastTarget, ObservationWindow, build_feature_rows,
... )
>>> dataset = ForecastDataset.from_records(
...     [{
...         "program": "cash-transfers-ng",
...         "region": "Africa",
...         "asset": "USDC",
...         "window": {"start": "2025-10-01", "end": "2025-12-31"},
...         "value": 0.82,
...     }],
...     target=ForecastTarget.CASH_OUT_RATE,
...     observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
... )
>>> row = build_feature_rows(dataset)[0]
>>> row.program, row.region, row.asset
('cash-transfers-ng', 'Africa', 'USDC')
>>> row.calendar.window_days
92
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Any

from .schema import ForecastDataset, ForecastKey, ForecastRow, TimeWindow

__all__ = [
    "CalendarFeatures",
    "FeatureRow",
    "KeyFeatures",
    "build_calendar_features",
    "build_feature_rows",
    "build_key_features",
    "calendar_feature_map",
    "identifier_feature_map",
]


def _require_dataset(dataset: object) -> ForecastDataset:
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(
            f"dataset must be a ForecastDataset, got {type(dataset).__name__}"
        )
    return dataset


def _require_row(row: object) -> ForecastRow:
    if not isinstance(row, ForecastRow):
        raise TypeError(f"row must be a ForecastRow, got {type(row).__name__}")
    return row


def _require_window(window: object) -> TimeWindow:
    if not isinstance(window, TimeWindow):
        raise TypeError(f"window must be a TimeWindow, got {type(window).__name__}")
    return window


def _require_key(key: object) -> ForecastKey:
    if not isinstance(key, ForecastKey):
        raise TypeError(f"key must be a ForecastKey, got {type(key).__name__}")
    return key


def _quarter(month: int) -> int:
    return (month - 1) // 3 + 1


@dataclass(frozen=True)
class CalendarFeatures:
    """Calendar attributes derived from a :class:`~.schema.TimeWindow`.

    Features describe the observation window's position and span on the
    calendar so models can reuse seasonality/time signals consistently.
    Start/end features come from the window bounds; span features describe
    the covered period.
    """

    start_date: date
    end_date: date
    start_year: int
    start_month: int
    start_quarter: int
    start_week: int
    start_day_of_year: int
    end_year: int
    end_month: int
    end_quarter: int
    end_week: int
    end_day_of_year: int
    window_days: int
    spans_month_boundary: bool
    spans_quarter_boundary: bool
    spans_year_boundary: bool

    @property
    def label(self) -> str:
        """Stable ``YYYY-MM-DD..YYYY-MM-DD`` label for the window."""
        return f"{self.start_date.isoformat()}..{self.end_date.isoformat()}"


def build_calendar_features(window: TimeWindow) -> CalendarFeatures:
    """Extract reusable calendar features from a time window.

    Parameters
    ----------
    window:
        A validated :class:`~.schema.TimeWindow` (or
        :class:`~.schema.ObservationWindow` / :class:`~.schema.ForecastHorizon`).

    Returns
    -------
    CalendarFeatures
        Deterministic calendar attributes for the window bounds and span.
    """
    validated = _require_window(window)
    start = validated.start
    end = validated.end
    start_iso = start.isocalendar()
    end_iso = end.isocalendar()
    return CalendarFeatures(
        start_date=start,
        end_date=end,
        start_year=start.year,
        start_month=start.month,
        start_quarter=_quarter(start.month),
        start_week=int(start_iso.week),
        start_day_of_year=int(start.timetuple().tm_yday),
        end_year=end.year,
        end_month=end.month,
        end_quarter=_quarter(end.month),
        end_week=int(end_iso.week),
        end_day_of_year=int(end.timetuple().tm_yday),
        window_days=validated.days,
        spans_month_boundary=(start.year, start.month) != (end.year, end.month),
        spans_quarter_boundary=(start.year, _quarter(start.month))
        != (end.year, _quarter(end.month)),
        spans_year_boundary=start.year != end.year,
    )


def identifier_feature_map(key: ForecastKey) -> dict[str, str]:
    """Return the program/region/asset identifier features for a key."""
    validated = _require_key(key)
    return {
        "program": validated.program,
        "region": validated.region,
        "asset": validated.asset,
        "key": validated.label,
    }


def calendar_feature_map(features: CalendarFeatures) -> dict[str, int | bool | str]:
    """Flatten calendar features into a reusable scalar feature map."""
    if not isinstance(features, CalendarFeatures):
        raise TypeError(
            f"features must be CalendarFeatures, got {type(features).__name__}"
        )
    return {
        "window_start": features.start_date.isoformat(),
        "window_end": features.end_date.isoformat(),
        "window_label": features.label,
        "start_year": features.start_year,
        "start_month": features.start_month,
        "start_quarter": features.start_quarter,
        "start_week": features.start_week,
        "start_day_of_year": features.start_day_of_year,
        "end_year": features.end_year,
        "end_month": features.end_month,
        "end_quarter": features.end_quarter,
        "end_week": features.end_week,
        "end_day_of_year": features.end_day_of_year,
        "window_days": features.window_days,
        "spans_month_boundary": features.spans_month_boundary,
        "spans_quarter_boundary": features.spans_quarter_boundary,
        "spans_year_boundary": features.spans_year_boundary,
    }


@dataclass(frozen=True)
class FeatureRow:
    """Reusable features for one observed forecast row.

    Combines the schema identifiers (program, region, asset) with calendar
    features derived from the row's time window and the observed target
    value.
    """

    key: ForecastKey
    program: str
    region: str
    asset: str
    target: str
    value: float
    window: TimeWindow
    calendar: CalendarFeatures

    def to_feature_map(self) -> dict[str, Any]:
        """Flatten identifiers and calendar features into one feature map."""
        feature_map: dict[str, Any] = {}
        feature_map.update(identifier_feature_map(self.key))
        feature_map.update(calendar_feature_map(self.calendar))
        feature_map["target"] = self.target
        feature_map["value"] = self.value
        return feature_map


@dataclass(frozen=True)
class KeyFeatures:
    """Aggregated reusable features for one (program, region, asset) key.

    Summarizes all rows in a dataset for the identifier group, including
    calendar coverage and observed-value statistics, so models can reuse a
    consistent key-level representation.
    """

    key: ForecastKey
    program: str
    region: str
    asset: str
    target: str
    n_observations: int
    mean_value: float
    min_value: float
    max_value: float
    mean_window_days: float
    min_window_days: int
    max_window_days: int
    earliest_start: date
    latest_end: date
    start_years: tuple[int, ...]
    start_months: tuple[int, ...]
    start_quarters: tuple[int, ...]

    def to_feature_map(self) -> dict[str, Any]:
        """Flatten key-level identifier, calendar, and summary features."""
        feature_map: dict[str, Any] = {}
        feature_map.update(identifier_feature_map(self.key))
        feature_map.update(
            {
                "target": self.target,
                "n_observations": self.n_observations,
                "mean_value": self.mean_value,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "mean_window_days": self.mean_window_days,
                "min_window_days": self.min_window_days,
                "max_window_days": self.max_window_days,
                "earliest_start": self.earliest_start.isoformat(),
                "latest_end": self.latest_end.isoformat(),
                "coverage_days": (self.latest_end - self.earliest_start).days + 1,
                "start_years": list(self.start_years),
                "start_months": list(self.start_months),
                "start_quarters": list(self.start_quarters),
            }
        )
        return feature_map


def _row_features(row: ForecastRow) -> FeatureRow:
    validated = _require_row(row)
    return FeatureRow(
        key=validated.key,
        program=validated.key.program,
        region=validated.key.region,
        asset=validated.key.asset,
        target=validated.target.value,
        value=float(validated.value),
        window=validated.window,
        calendar=build_calendar_features(validated.window),
    )


def build_feature_rows(dataset: ForecastDataset) -> tuple[FeatureRow, ...]:
    """Build reusable per-row program/region/asset/calendar features.

    Parameters
    ----------
    dataset:
        A validated :class:`~.schema.ForecastDataset`.

    Returns
    -------
    tuple[FeatureRow, ...]
        One feature row per dataset row, in dataset row order.
    """
    data = _require_dataset(dataset)
    return tuple(_row_features(row) for row in data.rows)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of an empty sequence")
    total = float(sum(values))
    result = total / float(len(values))
    if not isfinite(result):
        raise ValueError("feature mean must be finite")
    return result


def build_key_features(dataset: ForecastDataset) -> tuple[KeyFeatures, ...]:
    """Build aggregated reusable features for each identifier group.

    Groups rows by :class:`~.schema.ForecastKey` in first-seen order and
    summarizes program, region, asset, calendar coverage, and observed
    values for each group.

    Parameters
    ----------
    dataset:
        A validated :class:`~.schema.ForecastDataset`.

    Returns
    -------
    tuple[KeyFeatures, ...]
        One aggregated feature record per identifier, in first-seen order.
    """
    data = _require_dataset(dataset)
    features: list[KeyFeatures] = []
    for key, rows in data.group_by_key().items():
        row_features = [_row_features(row) for row in rows]
        values = [feature.value for feature in row_features]
        window_days = [feature.calendar.window_days for feature in row_features]
        starts = [feature.calendar.start_date for feature in row_features]
        ends = [feature.calendar.end_date for feature in row_features]
        features.append(
            KeyFeatures(
                key=key,
                program=key.program,
                region=key.region,
                asset=key.asset,
                target=data.target.value,
                n_observations=len(rows),
                mean_value=_mean(values),
                min_value=min(values),
                max_value=max(values),
                mean_window_days=_mean([float(days) for days in window_days]),
                min_window_days=min(window_days),
                max_window_days=max(window_days),
                earliest_start=min(starts),
                latest_end=max(ends),
                start_years=tuple(
                    sorted({feature.calendar.start_year for feature in row_features})
                ),
                start_months=tuple(
                    sorted({feature.calendar.start_month for feature in row_features})
                ),
                start_quarters=tuple(
                    sorted({feature.calendar.start_quarter for feature in row_features})
                ),
            )
        )
    return tuple(features)
