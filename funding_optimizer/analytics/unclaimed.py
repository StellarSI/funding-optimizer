"""Unclaimed-funds balances and aging buckets (issue #37).

Computes, from normalized events (P34), disbursements that have no matching
receipt at the observation boundary. Units and conventions:

- ``amount`` values are in asset units (floats, matching ``NormalizedEvent``).
- ``age_days`` is ``(as_of - disbursed_at)`` in days (24h = 1 day).
- ``as_of`` and event timestamps are inclusive UTC instants.
- Aging buckets are ``[lower, upper]`` day ranges, evaluated in order; the
  first matching bucket wins.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from funding_optimizer.ingest.normalize import NormalizedEvent

SECONDS_PER_DAY = 86_400.0

AgingBucket = tuple[str, float, float | None]

DEFAULT_AGING_BUCKETS: tuple[AgingBucket, ...] = (
    ("0-7", 0.0, 7.0),
    ("8-30", 7.0, 30.0),
    ("31-90", 30.0, 90.0),
    ("90+", 90.0, None),
)


class UnclaimedFundsError(ValueError):
    """Raised when unclaimed-funds analytics inputs are invalid."""


@dataclass(frozen=True)
class UnclaimedBalance:
    """One disbursement with no matching receipt at the observation boundary."""

    disbursement_event_id: str
    account: str
    counterparty: str
    asset: str
    amount: float
    disbursed_at: datetime
    as_of: datetime
    age_days: float
    aging_bucket: str


@dataclass(frozen=True)
class AgingBucketSummary:
    """Total unclaimed amount and count for one aging bucket."""

    bucket: str
    min_age_days: float
    max_age_days: float | None
    total_amount: float
    disbursements: int


@dataclass(frozen=True)
class UnclaimedFundsReport:
    """Unclaimed balances at one observation boundary.

    Units: ``amount``/``total_amount`` are asset units; ages are days;
    timestamps are UTC ISO-8601 strings in ``as_dict()``.
    """

    as_of: datetime
    disbursement_account: str | None
    balances: Sequence[UnclaimedBalance]
    buckets: Sequence[AgingBucketSummary]
    by_asset: Mapping[str, float]

    @property
    def total_amount(self) -> float:
        return sum(balance.amount for balance in self.balances)

    @property
    def disbursements(self) -> int:
        return len(self.balances)

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat().replace("+00:00", "Z"),
            "units": {"amount": "asset units", "age": "days", "timestamp": "utc-iso8601"},
            "disbursement_account": self.disbursement_account,
            "total_amount": self.total_amount,
            "disbursements": self.disbursements,
            "by_asset": dict(self.by_asset),
            "buckets": [
                {
                    "bucket": b.bucket,
                    "min_age_days": b.min_age_days,
                    "max_age_days": b.max_age_days,
                    "total_amount": b.total_amount,
                    "disbursements": b.disbursements,
                }
                for b in self.buckets
            ],
            "balances": [
                {
                    "disbursement_event_id": b.disbursement_event_id,
                    "account": b.account,
                    "counterparty": b.counterparty,
                    "asset": b.asset,
                    "amount": b.amount,
                    "disbursed_at": b.disbursed_at.isoformat().replace("+00:00", "Z"),
                    "age_days": b.age_days,
                    "aging_bucket": b.aging_bucket,
                }
                for b in self.balances
            ],
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise UnclaimedFundsError(
                f"{name} must be an ISO-8601 datetime, got {value!r}"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise UnclaimedFundsError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise UnclaimedFundsError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise UnclaimedFundsError(
                f"events[{index}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


def _validate_buckets(buckets: Sequence[AgingBucket]) -> tuple[AgingBucket, ...]:
    if isinstance(buckets, (str, bytes)) or not isinstance(buckets, Sequence):
        raise UnclaimedFundsError("aging_buckets must be a sequence of (name, lower, upper)")
    result: list[AgingBucket] = []
    previous_upper = -1.0
    for index, entry in enumerate(buckets):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise UnclaimedFundsError(f"aging_buckets[{index}] must be (name, lower, upper)")
        name, lower, upper = entry
        if not isinstance(name, str) or not name.strip():
            raise UnclaimedFundsError(f"aging_buckets[{index}] name must be a non-empty string")
        if isinstance(lower, bool) or not isinstance(lower, (int, float)) or not isfinite(lower):
            raise UnclaimedFundsError(f"aging_buckets[{index}] lower must be a finite number")
        if lower < 0 or lower < previous_upper:
            raise UnclaimedFundsError(f"aging_buckets[{index}] must be ordered and non-negative")
        if upper is not None:
            if isinstance(upper, bool) or not isinstance(upper, (int, float)):
                raise UnclaimedFundsError(
                    f"aging_buckets[{index}] upper must be a finite number or None"
                )
            if not isfinite(upper) or upper <= lower:
                raise UnclaimedFundsError(
                    f"aging_buckets[{index}] upper must be finite and greater than lower"
                )
            previous_upper = float(upper)
        else:
            previous_upper = float(lower)
        result.append((name.strip(), float(lower), None if upper is None else float(upper)))
    if not result:
        raise UnclaimedFundsError("aging_buckets must contain at least one bucket")
    return tuple(result)


def _bucket_for(age_days: float, buckets: Sequence[AgingBucket]) -> str:
    for name, lower, upper in buckets:
        if age_days >= lower and (upper is None or age_days <= upper):
            return name
    return buckets[-1][0]


def compute_unclaimed_funds(
    events: Iterable[NormalizedEvent],
    *,
    disbursement_account: str | None = None,
    as_of: datetime | str | None = None,
    aging_buckets: Sequence[AgingBucket] | None = None,
) -> UnclaimedFundsReport:
    """Compute unclaimed disbursements and their aging buckets.

    Matching rule: a disbursement is an OUT event (from ``disbursement_account``
    when supplied) with a counterparty. It is considered claimed when an IN
    event for the same asset, received by that counterparty at or after the
    disbursement and no later than ``as_of``, exists in the event set. Each
    receipt is consumed at most once. The default ``as_of`` is the latest
    event time, making results deterministic without a wall clock.
    """
    normalized = _coerce_events(events)
    subject = disbursement_account.strip() if isinstance(disbursement_account, str) else None
    if disbursement_account is not None and not subject:
        raise UnclaimedFundsError("disbursement_account must be a non-empty string when provided")
    buckets = DEFAULT_AGING_BUCKETS if aging_buckets is None else _validate_buckets(aging_buckets)

    boundary = (
        max((_event_time(event) for event in normalized), default=datetime.now(UTC))
        if as_of is None
        else _as_utc(as_of, "as_of")
    )
    ordered = sorted(normalized, key=lambda event: (_event_time(event), event.event_id))

    disbursements = [
        event
        for event in ordered
        if event.direction.value == "out"
        and event.counterparty
        and (subject is None or event.account == subject)
        and _event_time(event) <= boundary
    ]

    used_receipts: set[str] = set()
    balances: list[UnclaimedBalance] = []
    for disbursement in disbursements:
        disbursed_at = _event_time(disbursement)
        receipt = None
        for candidate in ordered:
            if candidate.event_id in used_receipts:
                continue
            if candidate.direction.value != "in":
                continue
            if candidate.asset != disbursement.asset:
                continue
            if candidate.account != disbursement.counterparty:
                continue
            if candidate.counterparty not in (None, disbursement.account):
                continue
            received_at = _event_time(candidate)
            if received_at < disbursed_at or received_at > boundary:
                continue
            receipt = candidate
            break
        if receipt is not None:
            used_receipts.add(receipt.event_id)
            continue

        age_days = (boundary - disbursed_at).total_seconds() / SECONDS_PER_DAY
        if not isfinite(age_days) or age_days < 0:
            raise UnclaimedFundsError(
                f"invalid unclaimed age for disbursement {disbursement.event_id}"
            )
        balances.append(
            UnclaimedBalance(
                disbursement_event_id=disbursement.event_id,
                account=disbursement.account,
                counterparty=disbursement.counterparty or "",
                asset=disbursement.asset,
                amount=disbursement.amount,
                disbursed_at=disbursed_at,
                as_of=boundary,
                age_days=age_days,
                aging_bucket=_bucket_for(age_days, buckets),
            )
        )

    by_asset: dict[str, float] = {}
    for balance in balances:
        by_asset[balance.asset] = by_asset.get(balance.asset, 0.0) + balance.amount

    summaries = [
        AgingBucketSummary(
            bucket=name,
            min_age_days=lower,
            max_age_days=upper,
            total_amount=sum(b.amount for b in balances if b.aging_bucket == name),
            disbursements=sum(1 for b in balances if b.aging_bucket == name),
        )
        for name, lower, upper in buckets
    ]

    return UnclaimedFundsReport(
        as_of=boundary,
        disbursement_account=subject,
        balances=tuple(balances),
        buckets=tuple(summaries),
        by_asset=by_asset,
    )
