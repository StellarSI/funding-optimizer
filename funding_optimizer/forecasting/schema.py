"""Forecast dataset and time-window schema (issue #22).

The data contract every disbursement-forecasting model in this package
consumes and produces:

* :class:`ObservationWindow` — the historical window a model is fitted on.
* :class:`ForecastHorizon` — the future window a model predicts, at a
  ``day``, ``week`` or ``month`` frequency.
* :class:`ForecastTarget` — what is being predicted: ``cash_out_rate``,
  ``delivery_delay`` or ``funding_gap``.
* :class:`ForecastKey` / :class:`ForecastRow` — every observation carries
  the identifiers the schema requires: a program, a region, an asset and a
  time window, plus the observed value for the target.
* :class:`ForecastDataset` — rows scoped to a single target and a single
  observation window, built from raw records with
  :meth:`ForecastDataset.from_records`.
* :class:`ForecastPrediction` / :class:`ForecastEvaluation` — typed
  predictions that carry their horizon, method, and evaluation metadata.

Rows that do not identify a program, region, asset and time window are
rejected at the boundary with :class:`ValueError` naming the missing
identifier, so malformed data fails loudly instead of reaching a model.

This module only describes data. The optimizer recommends allocations and
never signs or submits payouts, and nothing here performs network calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite

from funding_optimizer.models import validate_asset_code

_IDENTIFIERS = ("program", "region", "asset", "window")
_FREQUENCIES = ("day", "week", "month")


class ForecastTarget(str, Enum):
    """What a forecast predicts: the target column of a dataset."""

    CASH_OUT_RATE = "cash_out_rate"
    """Fraction of a disbursement that is cashed out within a window."""

    DELIVERY_DELAY = "delivery_delay"
    """Days between sending and receipt."""

    FUNDING_GAP = "funding_gap"
    """Shortfall (positive) or surplus (negative) of funds for a window."""

    @classmethod
    def parse(cls, value: object) -> ForecastTarget:
        """Coerce ``value`` into a known target.

        Raises :class:`ValueError` for an unknown target name and
        :class:`TypeError` for a non-string value.
        """
        if isinstance(value, ForecastTarget):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip())
            except ValueError:
                allowed = ", ".join(member.value for member in cls)
                raise ValueError(
                    f"unknown forecast target {value!r}; expected one of: {allowed}"
                ) from None
        raise TypeError(f"forecast target must be a string, got {type(value).__name__}")


def _coerce_date(value: object, name: str) -> date:
    """Normalize a date, datetime, or ISO-8601 ``YYYY-MM-DD`` string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(f"{name} {value!r} is not an ISO-8601 date (YYYY-MM-DD)") from None
    raise TypeError(f"{name} must be a date or ISO-8601 string, got {type(value).__name__}")


def _require_text(value: object, name: str) -> str:
    """Trim ``value`` and require a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _coerce_frequency(value: object) -> str:
    """Normalize a step frequency to one of ``day``, ``week``, ``month``."""
    if not isinstance(value, str):
        raise TypeError(f"frequency must be a string, got {type(value).__name__}")
    frequency = value.strip().lower()
    if frequency not in _FREQUENCIES:
        allowed = ", ".join(_FREQUENCIES)
        raise ValueError(f"frequency {value!r} is not supported; expected one of: {allowed}")
    return frequency


def _validate_target_value(target: ForecastTarget, value: object) -> float:
    """Normalize an observed or predicted value and bound it by its target."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"value must be a real number, got {type(value).__name__}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"value must be a finite number, got {value!r}")
    if target == ForecastTarget.CASH_OUT_RATE and not 0.0 <= number <= 1.0:
        raise ValueError(f"cash-out rate {number} is not in [0, 1]")
    if target == ForecastTarget.DELIVERY_DELAY and number < 0.0:
        raise ValueError(f"delivery delay {number} must be non-negative")
    return number


def _validate_metric(value: object, name: str) -> float:
    """Normalize an evaluation metric: a finite, non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return number


@dataclass(frozen=True)
class TimeWindow:
    """A closed date interval ``[start, end]``; both bounds are inclusive.

    Dates may be :class:`datetime.date`, :class:`datetime.datetime`, or an
    ISO-8601 ``YYYY-MM-DD`` string; they are normalized to dates and
    rejected when ``start`` falls after ``end``.
    """

    start: date
    end: date

    def __post_init__(self) -> None:
        start = _coerce_date(self.start, "window start")
        end = _coerce_date(self.end, "window end")
        if start > end:
            raise ValueError(
                f"window start {start.isoformat()} is after window end {end.isoformat()}"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def days(self) -> int:
        """Number of days covered, counting both bounds."""
        return (self.end - self.start).days + 1

    def contains(self, moment: date) -> bool:
        """True when ``moment`` falls inside the window (bounds inclusive)."""
        instant = _coerce_date(moment, "moment")
        return self.start <= instant <= self.end

    def covers(self, other: TimeWindow) -> bool:
        """True when ``other`` lies entirely inside this window."""
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: TimeWindow) -> bool:
        """True when the two windows share at least one day."""
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class ObservationWindow(TimeWindow):
    """The historical window a forecast model is fitted on."""

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass(frozen=True)
class ForecastHorizon(TimeWindow):
    """The future window a forecast covers, at a fixed step ``frequency``.

    ``frequency`` is one of ``day``, ``week`` or ``month`` (case-insensitive)
    and is normalized; :attr:`periods` counts how many steps of that
    frequency the horizon spans, rounding partial steps up.
    """

    frequency: str = "day"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency", _coerce_frequency(self.frequency))
        super().__post_init__()

    @property
    def periods(self) -> int:
        """Steps of ``frequency`` covered by the horizon (rounded up)."""
        if self.frequency == "day":
            return self.days
        if self.frequency == "week":
            return (self.days + 6) // 7
        return (self.end.year - self.start.year) * 12 + (self.end.month - self.start.month) + 1


@dataclass(frozen=True)
class ForecastKey:
    """The identifiers every forecast row and prediction must carry.

    Asset codes are validated and normalized with
    :func:`funding_optimizer.models.validate_asset_code`; program and region
    are trimmed and must be non-empty.
    """

    program: str
    region: str
    asset: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "program", _require_text(self.program, "program"))
        object.__setattr__(self, "region", _require_text(self.region, "region"))
        object.__setattr__(self, "asset", validate_asset_code(self.asset))

    @property
    def label(self) -> str:
        """Stable ``program/region/asset`` label for logs and error messages."""
        return f"{self.program}/{self.region}/{self.asset}"


@dataclass(frozen=True)
class ForecastRow:
    """One observed target value for a key over a time window."""

    key: ForecastKey
    window: TimeWindow
    target: ForecastTarget
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, ForecastKey):
            raise TypeError(f"key must be a ForecastKey, got {type(self.key).__name__}")
        if not isinstance(self.window, TimeWindow):
            raise TypeError(f"window must be a TimeWindow, got {type(self.window).__name__}")
        target = ForecastTarget.parse(self.target)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "value", _validate_target_value(target, self.value))

    @property
    def program(self) -> str:
        """Program identifier of this row."""
        return self.key.program

    @property
    def region(self) -> str:
        """Region identifier of this row."""
        return self.key.region

    @property
    def asset(self) -> str:
        """Asset code of this row."""
        return self.key.asset


def missing_identifiers(record: Mapping[str, object]) -> tuple[str, ...]:
    """Required identifiers absent from ``record``.

    A forecast row must identify a program, a region, an asset and a time
    window; :class:`TypeError` is raised for a non-mapping record.
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"forecast record must be a mapping, got {type(record).__name__}")
    return tuple(name for name in _IDENTIFIERS if name not in record)


def _window_from_record(raw: object) -> TimeWindow:
    """Read a time window from a record: a window or a ``start``/``end`` mapping."""
    if isinstance(raw, TimeWindow):
        return raw
    if isinstance(raw, Mapping):
        if "start" not in raw or "end" not in raw:
            raise ValueError("forecast row window must provide both 'start' and 'end'")
        return TimeWindow(
            start=_coerce_date(raw["start"], "window start"),
            end=_coerce_date(raw["end"], "window end"),
        )
    raise ValueError(
        "forecast row window must be a TimeWindow or a mapping with 'start' and 'end', "
        f"got {type(raw).__name__}"
    )


def _row_from_record(record: Mapping[str, object], target: ForecastTarget) -> ForecastRow:
    """Build one row from a raw record, rejecting incomplete identifiers."""
    missing = missing_identifiers(record)
    if missing:
        raise ValueError(
            "forecast row must identify a program, region, asset and time window; "
            f"missing: {', '.join(missing)}"
        )
    if "value" not in record:
        raise ValueError("forecast row is missing the observed value for the target")
    raw_asset = record["asset"]
    if not isinstance(raw_asset, str):
        raise TypeError(f"asset must be a string, got {type(raw_asset).__name__}")
    return ForecastRow(
        key=ForecastKey(
            program=_require_text(record["program"], "program"),
            region=_require_text(record["region"], "region"),
            asset=validate_asset_code(raw_asset),
        ),
        window=_window_from_record(record["window"]),
        target=target,
        value=_validate_target_value(target, record["value"]),
    )


@dataclass(frozen=True)
class ForecastDataset:
    """A validated forecast dataset: one target, one observation window, N rows.

    Every row must target the dataset's target and sit inside the
    observation window, so a dataset is always a coherent slice of history
    for a single forecasting question.
    """

    target: ForecastTarget
    observation: ObservationWindow
    rows: Sequence[ForecastRow]

    def __post_init__(self) -> None:
        target = ForecastTarget.parse(self.target)
        if not isinstance(self.observation, ObservationWindow):
            raise TypeError(
                "observation must be an ObservationWindow, "
                f"got {type(self.observation).__name__}"
            )
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("a forecast dataset needs at least one row")
        for position, row in enumerate(rows):
            if not isinstance(row, ForecastRow):
                raise TypeError(f"rows[{position}] must be a ForecastRow, got {type(row).__name__}")
            if row.target != target:
                raise ValueError(
                    f"rows[{position}] targets {row.target.value}, but the dataset targets "
                    f"{target.value}"
                )
            if not self.observation.covers(row.window):
                raise ValueError(
                    f"rows[{position}] window {row.window.start.isoformat()}.."
                    f"{row.window.end.isoformat()} falls outside the observation window "
                    f"{self.observation.start.isoformat()}.."
                    f"{self.observation.end.isoformat()}"
                )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "rows", rows)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        target: ForecastTarget,
        observation: ObservationWindow,
    ) -> ForecastDataset:
        """Build a dataset from raw records (CSV/JSON rows).

        Each record needs ``program``, ``region``, ``asset``, ``window``
        (a :class:`TimeWindow` or a ``{"start", "end"}`` mapping) and
        ``value``. Records missing an identifier raise :class:`ValueError`
        naming the missing field instead of producing a partial row.
        """
        rows = [_row_from_record(record, target) for record in records]
        return cls(target=target, observation=observation, rows=rows)

    def __len__(self) -> int:
        """Number of rows in the dataset."""
        return len(self.rows)

    def group_by_key(self) -> dict[ForecastKey, tuple[ForecastRow, ...]]:
        """Rows grouped by identifier, in first-seen order."""
        grouped: dict[ForecastKey, list[ForecastRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.key, []).append(row)
        return {key: tuple(keyed_rows) for key, keyed_rows in grouped.items()}


@dataclass(frozen=True)
class ForecastEvaluation:
    """Evaluation metadata recorded alongside a prediction.

    Optional metrics are finite and non-negative; ``evaluated_at`` is the
    date the evaluation was produced, when known.
    """

    n_observations: int
    """How many observations the method was evaluated on."""

    mae: float | None = None
    rmse: float | None = None
    pinball_loss: float | None = None
    evaluated_at: date | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_observations, bool) or not isinstance(self.n_observations, int):
            raise TypeError(
                f"n_observations must be an integer, got {type(self.n_observations).__name__}"
            )
        if self.n_observations < 0:
            raise ValueError(f"n_observations must be non-negative, got {self.n_observations}")
        for name in ("mae", "rmse", "pinball_loss"):
            metric = getattr(self, name)
            if metric is not None:
                object.__setattr__(self, name, _validate_metric(metric, name))
        if self.evaluated_at is not None:
            object.__setattr__(
                self, "evaluated_at", _coerce_date(self.evaluated_at, "evaluated_at")
            )


@dataclass(frozen=True)
class ForecastPrediction:
    """A typed point prediction for one key over a forecast horizon.

    Carries the horizon it covers, the ``method`` that produced it, and
    optional :class:`ForecastEvaluation` metadata. Predictions describe
    expected outcomes only; they never move funds.
    """

    key: ForecastKey
    target: ForecastTarget
    horizon: ForecastHorizon
    method: str
    value: float
    evaluation: ForecastEvaluation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ForecastKey):
            raise TypeError(f"key must be a ForecastKey, got {type(self.key).__name__}")
        if not isinstance(self.horizon, ForecastHorizon):
            raise TypeError(f"horizon must be a ForecastHorizon, got {type(self.horizon).__name__}")
        if self.evaluation is not None and not isinstance(self.evaluation, ForecastEvaluation):
            raise TypeError(
                "evaluation must be a ForecastEvaluation or None, "
                f"got {type(self.evaluation).__name__}"
            )
        target = ForecastTarget.parse(self.target)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "method", _require_text(self.method, "method"))
        object.__setattr__(self, "value", _validate_target_value(target, self.value))
