"""Disbursement forecasting: schema, baselines, delay, gap, backtest, and metrics.

The forecasting package describes what funders need before committing
funds: observed cash-out rates, delivery delays, and funding gaps keyed by
program, region, asset and time window, plus the typed predictions those
observations produce.

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from .backtest import BacktestFold, BacktestResult, Forecaster, run_backtest
from .baseline import cash_out_rate_baseline
from .cold_start import cold_start_baseline
from .delays import delivery_delay_quantiles
from .features import (
    CalendarFeatures,
    FeatureRow,
    KeyFeatures,
    build_calendar_features,
    build_feature_rows,
    build_key_features,
    calendar_feature_map,
    identifier_feature_map,
)
from .gaps import funding_gap_baseline
from .metrics import (
    CalibrationBin,
    CalibrationResult,
    mean_absolute_error,
    pinball_loss,
    quantile_calibration,
)
from .schema import (
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
from .uncertainty import IntervalPrediction, prediction_intervals

__all__ = [
    "BacktestFold",
    "BacktestResult",
    "CalendarFeatures",
    "CalibrationBin",
    "CalibrationResult",
    "FeatureRow",
    "ForecastDataset",
    "ForecastEvaluation",
    "ForecastHorizon",
    "ForecastKey",
    "ForecastPrediction",
    "ForecastRow",
    "ForecastTarget",
    "Forecaster",
    "IntervalPrediction",
    "KeyFeatures",
    "ObservationWindow",
    "TimeWindow",
    "build_calendar_features",
    "build_feature_rows",
    "build_key_features",
    "calendar_feature_map",
    "cash_out_rate_baseline",
    "cold_start_baseline",
    "delivery_delay_quantiles",
    "funding_gap_baseline",
    "identifier_feature_map",
    "mean_absolute_error",
    "missing_identifiers",
    "pinball_loss",
    "prediction_intervals",
    "quantile_calibration",
    "run_backtest",
]
