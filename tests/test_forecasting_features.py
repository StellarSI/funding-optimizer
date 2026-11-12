"""Tests for reusable forecasting features (issue #26).

Features are pure data transformations over validated forecast datasets:
they must expose program, region, asset, and calendar attributes, preserve
dataset ordering, and reject malformed inputs at the boundary.
"""

from __future__ import annotations

from datetime import date

import pytest

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
    build_calendar_features,
    build_feature_rows,
    build_key_features,
    calendar_feature_map,
    identifier_feature_map,
)


def _dataset() -> ForecastDataset:
    return ForecastDataset.from_records(
        [
            {
                "program": "cash-transfers-ng",
                "region": "Africa",
                "asset": "USDC",
                "window": {"start": "2025-10-01", "end": "2025-12-31"},
                "value": 0.82,
            },
            {
                "program": "cash-transfers-ng",
                "region": "Africa",
                "asset": "USDC",
                "window": {"start": "2025-07-01", "end": "2025-09-30"},
                "value": 0.78,
            },
            {
                "program": "education-ke",
                "region": "Africa",
                "asset": "XLM",
                "window": {"start": "2025-04-01", "end": "2025-06-30"},
                "value": 0.64,
            },
        ],
        target=ForecastTarget.CASH_OUT_RATE,
        observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )


def test_build_calendar_features_extracts_window_calendar_attributes() -> None:
    features = build_calendar_features(
        TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31))
    )

    assert features.start_year == 2025
    assert features.start_month == 10
    assert features.start_quarter == 4
    assert features.start_day_of_year == 274
    assert features.end_year == 2025
    assert features.end_month == 12
    assert features.end_quarter == 4
    assert features.end_day_of_year == 365
    assert features.window_days == 92
    assert features.spans_month_boundary is True
    assert features.spans_quarter_boundary is False
    assert features.spans_year_boundary is False
    assert features.label == "2025-10-01..2025-12-31"


def test_build_calendar_features_tracks_quarter_and_year_boundaries() -> None:
    horizon = ForecastHorizon(
        start=date(2025, 12, 15), end=date(2026, 1, 14), frequency="day"
    )
    features = build_calendar_features(horizon)

    assert features.start_quarter == 4
    assert features.end_quarter == 1
    assert features.spans_month_boundary is True
    assert features.spans_quarter_boundary is True
    assert features.spans_year_boundary is True


def test_build_feature_rows_preserves_order_and_exposes_reusable_features() -> None:
    rows = build_feature_rows(_dataset())

    assert len(rows) == 3
    assert [row.program for row in rows] == [
        "cash-transfers-ng",
        "cash-transfers-ng",
        "education-ke",
    ]
    assert [row.region for row in rows] == ["Africa", "Africa", "Africa"]
    assert [row.asset for row in rows] == ["USDC", "USDC", "XLM"]
    assert rows[0].calendar.window_days == 92
    assert rows[1].calendar.start_quarter == 3

    feature_map = rows[0].to_feature_map()
    assert feature_map["program"] == "cash-transfers-ng"
    assert feature_map["region"] == "Africa"
    assert feature_map["asset"] == "USDC"
    assert feature_map["key"] == "cash-transfers-ng/Africa/USDC"
    assert feature_map["target"] == "cash_out_rate"
    assert feature_map["value"] == pytest.approx(0.82)
    assert feature_map["start_quarter"] == 4
    assert feature_map["window_days"] == 92


def test_build_key_features_aggregates_program_region_asset_calendar_groups() -> None:
    features = build_key_features(_dataset())

    assert len(features) == 2
    main, edu = features
    assert main.program == "cash-transfers-ng"
    assert main.region == "Africa"
    assert main.asset == "USDC"
    assert main.n_observations == 2
    assert main.mean_value == pytest.approx(0.8)
    assert main.min_value == pytest.approx(0.78)
    assert main.max_value == pytest.approx(0.82)
    assert main.mean_window_days == pytest.approx(92.0)
    assert main.earliest_start == date(2025, 7, 1)
    assert main.latest_end == date(2025, 12, 31)
    assert main.start_quarters == (3, 4)

    assert edu.key.label == "education-ke/Africa/XLM"
    assert edu.n_observations == 1
    assert edu.start_months == (4,)

    key_map = main.to_feature_map()
    assert key_map["program"] == "cash-transfers-ng"
    assert key_map["region"] == "Africa"
    assert key_map["asset"] == "USDC"
    assert key_map["n_observations"] == 2
    assert key_map["coverage_days"] == 184
    assert key_map["start_quarters"] == [3, 4]


def test_identifier_and_calendar_feature_maps_are_reusable() -> None:
    dataset = _dataset()
    key = dataset.rows[0].key
    calendar = build_calendar_features(dataset.rows[0].window)

    assert identifier_feature_map(key) == {
        "program": "cash-transfers-ng",
        "region": "Africa",
        "asset": "USDC",
        "key": "cash-transfers-ng/Africa/USDC",
    }
    cal_map = calendar_feature_map(calendar)
    assert cal_map["start_year"] == 2025
    assert cal_map["window_days"] == 92
    assert cal_map["spans_month_boundary"] is True


def test_feature_builders_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="dataset must be a ForecastDataset"):
        build_feature_rows(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dataset must be a ForecastDataset"):
        build_key_features("not-a-dataset")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window must be a TimeWindow"):
        build_calendar_features("2025-01-01")  # type: ignore[arg-type]
