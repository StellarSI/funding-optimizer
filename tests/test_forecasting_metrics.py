"""Tests for forecast evaluation metrics (issue #28).

Metrics are pure computations over actuals and predictions. Every expected
value below is hand-computed.
"""

from __future__ import annotations

import pytest

from funding_optimizer.forecasting.metrics import (
    CalibrationBin,
    CalibrationResult,
    mean_absolute_error,
    pinball_loss,
    quantile_calibration,
)


def test_mean_absolute_error_computes_average_absolute_difference() -> None:
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1.0 / 3.0)
    assert mean_absolute_error([0.6, 0.8, 0.7], [0.6, 0.7, 0.9]) == pytest.approx(
        (0.0 + 0.1 + 0.2) / 3.0
    )


def test_mean_absolute_error_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least one"):
        mean_absolute_error([], [])
    with pytest.raises(TypeError, match=r"actual\[0\]"):
        mean_absolute_error(["a"], [1.0])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="sequence of numbers"):
        mean_absolute_error("12", [1.0, 2.0])  # type: ignore[arg-type]


def test_pinball_loss_matches_absolute_error_at_median() -> None:
    actual = [10.0, 20.0, 30.0]
    predicted = 20.0
    # Median pinball loss is 0.5 * MAE = 0.5 * (10 + 0 + 10) / 3 = 10/3.
    assert pinball_loss(actual, predicted, 0.5) == pytest.approx(10.0 / 3.0)


def test_pinball_loss_is_asymmetric_for_non_median_quantiles() -> None:
    actual = [10.0]
    # q=0.75: under-prediction (actual >= pred) penalized by q * error.
    assert pinball_loss(actual, 8.0, 0.75) == pytest.approx(0.75 * 2.0)
    # Over-prediction penalized by (1-q) * error.
    assert pinball_loss(actual, 12.0, 0.75) == pytest.approx(0.25 * 2.0)


def test_pinball_loss_accepts_per_observation_predictions() -> None:
    actual = [10.0, 20.0]
    predicted = [12.0, 18.0]
    # errors = -2 (over) and +2 (under), q=0.25 -> (0.75*2 + 0.25*2)/2 = 1.0
    assert pinball_loss(actual, predicted, 0.25) == pytest.approx(1.0)


def test_pinball_loss_rejects_invalid_quantiles_and_shapes() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        pinball_loss([1.0], 1.0, 0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        pinball_loss([1.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="same length"):
        pinball_loss([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="at least one"):
        pinball_loss([], [], 0.5)


def test_quantile_calibration_reports_empirical_coverage() -> None:
    # Actuals [1,2,3,4]; q=0.5 predicted as 2.5 -> 2/4 covered (<=2.5).
    # q=0.75 predicted as 3.5 -> 3/4 covered.
    result = quantile_calibration([1.0, 2.0, 3.0, 4.0], {0.5: 2.5, 0.75: 3.5})

    assert isinstance(result, CalibrationResult)
    assert [b.nominal_quantile for b in result.bins] == [0.5, 0.75]
    assert result.bins[0].empirical_coverage == pytest.approx(0.5)
    assert result.bins[0].calibration_error == pytest.approx(0.0)
    assert result.bins[0].n_observations == 4
    assert result.bins[1].empirical_coverage == pytest.approx(0.75)
    assert result.bins[1].calibration_error == pytest.approx(0.0)
    assert result.mean_calibration_error == pytest.approx(0.0)


def test_quantile_calibration_measures_miscalibration() -> None:
    # q=0.5 predicted too low (1.5): only 1/4 <= 1.5 -> coverage 0.25, error 0.25.
    result = quantile_calibration([1.0, 2.0, 3.0, 4.0], {0.5: 1.5})
    assert result.bins[0].empirical_coverage == pytest.approx(0.25)
    assert result.bins[0].calibration_error == pytest.approx(0.25)
    assert result.mean_calibration_error == pytest.approx(0.25)


def test_quantile_calibration_accepts_per_observation_predictions() -> None:
    actual = [1.0, 2.0, 3.0]
    result = quantile_calibration(actual, {0.5: [1.0, 2.5, 2.5]})
    # actual <= pred: 1<=1, 2<=2.5, 3<=2.5 -> 2/3
    assert result.bins[0].empirical_coverage == pytest.approx(2.0 / 3.0)


def test_quantile_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        quantile_calibration([], {0.5: 1.0})
    with pytest.raises(ValueError, match="at least one quantile level"):
        quantile_calibration([1.0], {})
    with pytest.raises(TypeError, match="mapping of quantile"):
        quantile_calibration([1.0], [0.5])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly inside"):
        quantile_calibration([1.0], {1.5: 1.0})
    with pytest.raises(ValueError, match="same length"):
        quantile_calibration([1.0, 2.0], {0.5: [1.0]})


def test_calibration_types_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one quantile bin"):
        CalibrationResult(bins=(), mean_calibration_error=0.0)
    with pytest.raises(ValueError, match="n_observations must be >= 1"):
        CalibrationBin(
            nominal_quantile=0.5,
            empirical_coverage=0.5,
            n_observations=0,
            calibration_error=0.0,
        )
    with pytest.raises(ValueError, match=r"empirical_coverage must be in \[0, 1\]"):
        CalibrationBin(
            nominal_quantile=0.5,
            empirical_coverage=1.5,
            n_observations=1,
            calibration_error=0.0,
        )
