"""Disbursement forecasting: the forecast dataset and time-window schema.

The forecasting package describes what funders need before committing
funds: observed cash-out rates, delivery delays, and funding gaps keyed by
program, region, asset and time window, plus the typed predictions those
observations produce.

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

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
    "ForecastDataset",
    "ForecastEvaluation",
    "ForecastHorizon",
    "ForecastKey",
    "ForecastPrediction",
    "ForecastRow",
    "ForecastTarget",
    "ObservationWindow",
    "TimeWindow",
    "missing_identifiers",
]
