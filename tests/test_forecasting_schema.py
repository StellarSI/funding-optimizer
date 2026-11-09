"""Tests for the forecast dataset and time-window schema (issue #22).

The schema is pure data description: it validates time windows, targets,
identifiers, and typed predictions without touching the network or moving
funds. Every expected value is hand-computed.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pytest

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
    missing_identifiers,
)


def _observation() -> ObservationWindow:
    """Calendar year 2025 as the observation window."""
    return ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _records() -> list[dict[str, object]]:
    """Two hand-written observations, both inside the 2025 window."""
    return [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 0.82,
        },
        {
            "program": "education-ke",
            "region": "Africa",
            "asset": "XLM",
            "window": {"start": "2025-07-01", "end": "2025-09-30"},
            "value": 0.64,
        },
    ]


def _key() -> ForecastKey:
    return ForecastKey(program="cash-transfers-ng", region="Africa", asset="USDC")


def _row(
    target: ForecastTarget = ForecastTarget.CASH_OUT_RATE,
    value: float = 0.5,
) -> ForecastRow:
    return ForecastRow(
        key=_key(),
        window=TimeWindow(start=date(2025, 1, 1), end=date(2025, 3, 31)),
        target=target,
        value=value,
    )


def test_time_window_orders_bounds_and_counts_days() -> None:
    window = TimeWindow(start=date(2025, 1, 1), end=date(2025, 1, 31))

    assert window.days == 31
    assert window.contains(date(2025, 1, 1))
    assert window.contains(date(2025, 1, 31))
    assert not window.contains(date(2025, 2, 1))
    assert window.overlaps(TimeWindow(start=date(2025, 1, 20), end=date(2025, 2, 10)))
    assert not window.overlaps(TimeWindow(start=date(2025, 2, 1), end=date(2025, 2, 28)))


def test_time_window_normalizes_iso_strings_and_datetimes() -> None:
    iso = TimeWindow(start="2025-01-01", end="2025-01-31")  # type: ignore[arg-type]
    assert iso.start == date(2025, 1, 1)
    assert iso.end == date(2025, 1, 31)
    assert iso.days == 31

    stamped = TimeWindow(start=datetime(2025, 1, 1, 9, 30), end=date(2025, 1, 31))
    assert stamped.start == date(2025, 1, 1)


def test_time_window_rejects_inverted_and_malformed_bounds() -> None:
    with pytest.raises(ValueError, match="after"):
        TimeWindow(start=date(2025, 2, 1), end=date(2025, 1, 1))
    with pytest.raises(ValueError, match="ISO-8601"):
        TimeWindow(start="01/01/2025", end="2025-01-31")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window end"):
        TimeWindow(start=date(2025, 1, 1), end=None)  # type: ignore[arg-type]


def test_observation_window_covers_nested_windows() -> None:
    observation = _observation()
    inside = TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31))

    assert observation.days == 365
    assert observation.covers(inside)
    assert observation.covers(observation)
    assert not inside.covers(observation)


def test_horizon_counts_periods_per_frequency() -> None:
    daily = ForecastHorizon(start=date(2025, 1, 1), end=date(2025, 1, 14))
    assert daily.frequency == "day"
    assert daily.periods == 14

    weekly = ForecastHorizon(start=date(2025, 1, 1), end=date(2025, 1, 14), frequency="WEEK")
    assert weekly.frequency == "week"
    assert weekly.periods == 2

    monthly = ForecastHorizon(start=date(2025, 1, 1), end=date(2025, 12, 31), frequency="month")
    assert monthly.periods == 12


def test_horizon_rejects_unsupported_frequencies() -> None:
    with pytest.raises(ValueError, match="frequency"):
        ForecastHorizon(start=date(2025, 1, 1), end=date(2025, 1, 7), frequency="hourly")
    with pytest.raises(TypeError, match="frequency"):
        ForecastHorizon(start=date(2025, 1, 1), end=date(2025, 1, 7), frequency=None)  # type: ignore[arg-type]


def test_forecast_target_parses_known_values() -> None:
    assert ForecastTarget.parse("cash_out_rate") is ForecastTarget.CASH_OUT_RATE
    assert ForecastTarget.parse(ForecastTarget.DELIVERY_DELAY) is ForecastTarget.DELIVERY_DELAY
    assert ForecastTarget.parse(" funding_gap ") is ForecastTarget.FUNDING_GAP
    assert ForecastTarget.CASH_OUT_RATE == "cash_out_rate"
    assert sorted(member.value for member in ForecastTarget) == [
        "cash_out_rate",
        "delivery_delay",
        "funding_gap",
    ]


def test_forecast_target_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown forecast target"):
        ForecastTarget.parse("expected_reach")
    with pytest.raises(TypeError, match="forecast target"):
        ForecastTarget.parse(3)


def test_forecast_key_normalizes_identifiers() -> None:
    key = ForecastKey(program=" cash-transfers-ng ", region=" Africa ", asset="usdc")

    assert key.program == "cash-transfers-ng"
    assert key.region == "Africa"
    assert key.asset == "USDC"
    assert key.label == "cash-transfers-ng/Africa/USDC"
    assert key == ForecastKey(program="cash-transfers-ng", region="Africa", asset="USDC")
    assert len({key, ForecastKey(program="p2", region="Asia", asset="XLM")}) == 2


def test_forecast_key_rejects_missing_identifiers() -> None:
    with pytest.raises(ValueError, match="region"):
        ForecastKey(program="p1", region="   ", asset="USDC")
    with pytest.raises(ValueError, match="Asset code"):
        ForecastKey(program="p1", region="Africa", asset="not an asset!")
    with pytest.raises(TypeError, match="program"):
        ForecastKey(program=None, region="Africa", asset="USDC")  # type: ignore[arg-type]


def test_row_carries_its_identifiers_window_and_value() -> None:
    row = _row(value=0.82)

    assert row.program == "cash-transfers-ng"
    assert row.region == "Africa"
    assert row.asset == "USDC"
    assert row.window == TimeWindow(start=date(2025, 1, 1), end=date(2025, 3, 31))
    assert row.target == ForecastTarget.CASH_OUT_RATE
    assert row.value == pytest.approx(0.82)


def test_row_rejects_rows_without_identifiers_or_a_window() -> None:
    with pytest.raises(TypeError, match="key"):
        ForecastRow(
            key={"program": "p1"},  # type: ignore[arg-type]
            window=TimeWindow(start=date(2025, 1, 1), end=date(2025, 3, 31)),
            target=ForecastTarget.CASH_OUT_RATE,
            value=0.5,
        )
    with pytest.raises(TypeError, match="window"):
        ForecastRow(
            key=_key(),
            window=None,  # type: ignore[arg-type]
            target=ForecastTarget.CASH_OUT_RATE,
            value=0.5,
        )


def test_row_bounds_values_by_target() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _row(value=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        _row(target=ForecastTarget.DELIVERY_DELAY, value=-1.0)
    with pytest.raises(ValueError, match="finite"):
        _row(target=ForecastTarget.FUNDING_GAP, value=float("nan"))
    with pytest.raises(TypeError, match="value"):
        _row(value="0.5")  # type: ignore[arg-type]

    gap = _row(target=ForecastTarget.FUNDING_GAP, value=-500.0)
    assert gap.value == -500.0
    delay = _row(target=ForecastTarget.DELIVERY_DELAY, value=3.5)
    assert delay.value == 3.5


def test_missing_identifiers_reports_only_what_is_absent() -> None:
    assert missing_identifiers(_records()[0]) == ()
    incomplete = dict(_records()[0])
    del incomplete["asset"]
    del incomplete["window"]
    assert missing_identifiers(incomplete) == ("asset", "window")
    with pytest.raises(TypeError, match="mapping"):
        missing_identifiers(["not a record"])  # type: ignore[arg-type]


def test_from_records_builds_a_validated_dataset() -> None:
    dataset = ForecastDataset.from_records(
        _records(),
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
    )

    assert dataset.target is ForecastTarget.CASH_OUT_RATE
    assert len(dataset) == 2
    assert [row.target for row in dataset.rows] == [
        ForecastTarget.CASH_OUT_RATE,
        ForecastTarget.CASH_OUT_RATE,
    ]
    assert dataset.rows[0].value == pytest.approx(0.82)
    assert dataset.rows[0].window.start == date(2025, 10, 1)
    assert dataset.rows[0].window.end == date(2025, 12, 31)
    assert dataset.rows[0].asset == "USDC"

    grouped = dataset.group_by_key()
    assert set(grouped) == {
        ForecastKey(program="cash-transfers-ng", region="Africa", asset="USDC"),
        ForecastKey(program="education-ke", region="Africa", asset="XLM"),
    }
    assert [row.value for row in grouped[_key()]] == [pytest.approx(0.82)]


def test_from_records_accepts_windows_and_rows_inside_the_observation_window() -> None:
    window = TimeWindow(start=date(2025, 1, 1), end=date(2025, 12, 31))
    records: list[dict[str, object]] = [
        {
            "program": "p1",
            "region": "Africa",
            "asset": "USDC",
            "window": window,
            "value": 0.9,
        }
    ]
    dataset = ForecastDataset.from_records(
        records, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
    )
    assert len(dataset) == 1
    assert dataset.rows[0].window is window


@pytest.mark.parametrize("missing", ["program", "region", "asset", "window"])
def test_from_records_rejects_rows_missing_an_identifier(missing: str) -> None:
    records = _records()
    del records[0][missing]

    with pytest.raises(ValueError) as excinfo:
        ForecastDataset.from_records(
            records, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
        )

    message = str(excinfo.value)
    assert "must identify a program, region, asset and time window" in message
    assert f"missing: {missing}" in message
    assert missing in missing_identifiers(records[0])


def test_from_records_rejects_rows_without_a_value() -> None:
    records = _records()
    del records[0]["value"]

    with pytest.raises(ValueError, match="observed value"):
        ForecastDataset.from_records(
            records, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
        )


def test_from_records_rejects_rows_outside_the_observation_window() -> None:
    records = _records()
    records[0]["window"] = {"start": "2024-01-01", "end": "2024-12-31"}

    with pytest.raises(ValueError, match="outside the observation window"):
        ForecastDataset.from_records(
            records, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
        )


def test_from_records_rejects_malformed_windows() -> None:
    incomplete = _records()
    incomplete[0]["window"] = {"start": "2025-10-01"}
    with pytest.raises(ValueError, match="must provide both"):
        ForecastDataset.from_records(
            incomplete, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
        )

    unparseable = _records()
    unparseable[0]["window"] = "2025-10-01"
    with pytest.raises(ValueError, match="TimeWindow"):
        ForecastDataset.from_records(
            unparseable, target=ForecastTarget.CASH_OUT_RATE, observation=_observation()
        )


def test_dataset_rejects_empty_and_mixed_row_sets() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        ForecastDataset(
            target=ForecastTarget.CASH_OUT_RATE,
            observation=_observation(),
            rows=(),
        )

    mixed = ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(_row(), _row(value=0.6)),
    )
    assert len(mixed) == 2
    assert len(mixed.group_by_key()) == 1

    with pytest.raises(ValueError, match="targets"):
        ForecastDataset(
            target=ForecastTarget.CASH_OUT_RATE,
            observation=_observation(),
            rows=(_row(target=ForecastTarget.DELIVERY_DELAY, value=3.5),),
        )


def test_dataset_rejects_rows_outside_the_observation_window() -> None:
    outside = ForecastRow(
        key=_key(),
        window=TimeWindow(start=date(2024, 1, 1), end=date(2024, 12, 31)),
        target=ForecastTarget.CASH_OUT_RATE,
        value=0.4,
    )

    with pytest.raises(ValueError, match="outside the observation window"):
        ForecastDataset(
            target=ForecastTarget.CASH_OUT_RATE,
            observation=_observation(),
            rows=(outside,),
        )


def test_evaluation_metadata_is_validated() -> None:
    evaluation = ForecastEvaluation(
        n_observations=90,
        mae=0.03,
        rmse=0.05,
        pinball_loss=0.02,
        evaluated_at=date(2025, 12, 31),
    )

    assert evaluation.n_observations == 90
    assert evaluation.mae == pytest.approx(0.03)
    assert evaluation.rmse == pytest.approx(0.05)
    assert evaluation.pinball_loss == pytest.approx(0.02)
    assert evaluation.evaluated_at == date(2025, 12, 31)
    assert ForecastEvaluation(n_observations=0).mae is None

    with pytest.raises(ValueError, match="non-negative"):
        ForecastEvaluation(n_observations=-1)
    with pytest.raises(ValueError, match="finite"):
        ForecastEvaluation(n_observations=10, mae=float("inf"))
    with pytest.raises(TypeError, match="n_observations"):
        ForecastEvaluation(n_observations=True)
    with pytest.raises(ValueError, match="evaluated_at"):
        ForecastEvaluation(n_observations=10, evaluated_at="31/12/2025")  # type: ignore[arg-type]


def test_prediction_carries_horizon_method_and_evaluation() -> None:
    horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")
    prediction = ForecastPrediction(
        key=_key(),
        target=ForecastTarget.CASH_OUT_RATE,
        horizon=horizon,
        method="historical-baseline",
        value=0.77,
        evaluation=ForecastEvaluation(n_observations=90, mae=0.03),
    )

    assert prediction.key == _key()
    assert prediction.horizon == horizon
    assert prediction.horizon.periods == 3
    assert prediction.method == "historical-baseline"
    assert prediction.value == pytest.approx(0.77)
    evaluation = prediction.evaluation
    assert evaluation is not None
    assert evaluation.n_observations == 90
    assert evaluation.mae == pytest.approx(0.03)

    without_metadata = ForecastPrediction(
        key=_key(),
        target=ForecastTarget.DELIVERY_DELAY,
        horizon=ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 1, 7)),
        method="baseline",
        value=2.5,
    )
    assert without_metadata.evaluation is None
    assert without_metadata.horizon.frequency == "day"


def test_prediction_rejects_incomplete_or_invalid_predictions() -> None:
    horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31))

    with pytest.raises(ValueError, match="method"):
        ForecastPrediction(
            key=_key(),
            target=ForecastTarget.CASH_OUT_RATE,
            horizon=horizon,
            method="   ",
            value=0.5,
        )
    with pytest.raises(TypeError, match="horizon"):
        ForecastPrediction(
            key=_key(),
            target=ForecastTarget.CASH_OUT_RATE,
            horizon=TimeWindow(start=date(2026, 1, 1), end=date(2026, 3, 31)),  # type: ignore[arg-type]
            method="baseline",
            value=0.5,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ForecastPrediction(
            key=_key(),
            target=ForecastTarget.CASH_OUT_RATE,
            horizon=horizon,
            method="baseline",
            value=4.2,
        )
    with pytest.raises(TypeError, match="evaluation"):
        ForecastPrediction(
            key=_key(),
            target=ForecastTarget.CASH_OUT_RATE,
            horizon=horizon,
            method="baseline",
            value=0.5,
            evaluation={"n_observations": 90},  # type: ignore[arg-type]
        )


def test_schema_objects_are_immutable() -> None:
    window = TimeWindow(start=date(2025, 1, 1), end=date(2025, 1, 31))

    with pytest.raises(FrozenInstanceError):
        window.start = date(2025, 2, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        window.days = 1  # type: ignore[misc]
