"""Disbursement forecasting: schema, baselines, delay, and gap predictions.

The forecasting package describes what funders need before committing
funds: observed cash-out rates, delivery delays, and funding gaps keyed by
program, region, asset and time window, plus the typed predictions those
observations produce.

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from .baseline import cash_out_rate_baseline
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

__all__ = [
    "CalendarFeatures",
    "FeatureRow",
    "ForecastDataset",
    "ForecastEvaluation",
    "ForecastHorizon",
    "ForecastKey",
    "ForecastPrediction",
    "ForecastRow",
    "ForecastTarget",
    "KeyFeatures",
    "ObservationWindow",
    "TimeWindow",
    "build_calendar_features",
    "build_feature_rows",
    "build_key_features",
    "calendar_feature_map",
    "cash_out_rate_baseline",
    "delivery_delay_quantiles",
    "funding_gap_baseline",
    "identifier_feature_map",
    "missing_identifiers",
]
