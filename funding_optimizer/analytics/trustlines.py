"""Trustline-readiness delivery analytics (issue #35).

Computes trustline-readiness rates by cohort from normalized P34 events.
The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from funding_optimizer.ingest.normalize import NormalizedEvent


class TrustlineReadinessError(ValueError):
    """Raised when trustline-readiness inputs are invalid."""


@dataclass(frozen=True)
class ObservationWindow:
    """Inclusive UTC observation window for readiness analytics."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _as_utc(self.start, "start")
        end = _as_utc(self.end, "end")
        if start > end:
            raise TrustlineReadinessError(
                f"observation window start {start.isoformat()} is after end {end.isoformat()}"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, moment: datetime) -> bool:
        ts = _as_utc(moment, "moment")
        return self.start <= ts <= self.end


@dataclass(frozen=True)
class CohortReadiness:
    """Trustline-readiness result for one cohort.

    Units:
    - readiness_rate is a fraction in [0, 1].
    - counts are accounts.
    """

    cohort: str
    asset: str
    total_accounts: int
    ready_accounts: int
    readiness_rate: float
    ready_account_ids: Sequence[str]


@dataclass(frozen=True)
class TrustlineReadinessReport:
    """Trustline-readiness report across cohorts."""

    observation_window: ObservationWindow | None
    asset: str
    cohorts: Sequence[CohortReadiness]

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_window": None
            if self.observation_window is None
            else {
                "start": self.observation_window.start.isoformat().replace("+00:00", "Z"),
                "end": self.observation_window.end.isoformat().replace("+00:00", "Z"),
                "unit": "utc-iso8601",
            },
            "units": {"readiness_rate": "fraction", "accounts": "count"},
            "asset": self.asset,
            "cohorts": [
                {
                    "cohort": c.cohort,
                    "asset": c.asset,
                    "total_accounts": c.total_accounts,
                    "ready_accounts": c.ready_accounts,
                    "readiness_rate": c.readiness_rate,
                    "ready_account_ids": list(c.ready_account_ids),
                }
                for c in self.cohorts
            ],
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TrustlineReadinessError(f"{name} must be an ISO-8601 datetime, got {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TrustlineReadinessError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise TrustlineReadinessError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for i, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise TrustlineReadinessError(
                f"events[{i}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


_TRUSTLINE_EVENT_TYPES = {"change_trust", "trustline_created", "trustline_ready", "create_trustline"}


def _is_trustline_ready(event: NormalizedEvent, account: str, asset: str) -> bool:
    if event.asset != asset:
        return False
    if event.account != account:
        return False
    et = (event.event_type or "").lower()
    if et in _TRUSTLINE_EVENT_TYPES or "trustline" in et or "change_trust" in et:
        return True
    # Receiving a non-native asset implies the trustline exists for that account.
    if event.direction.value == "in" and asset.upper() != "XLM":
        return True
    return False


def compute_trustline_readiness_rates(
    events: Iterable[NormalizedEvent],
    cohorts: Mapping[str, Sequence[str]],
    *,
    asset: str,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
) -> TrustlineReadinessReport:
    """Compute trustline-readiness rates by cohort from normalized P34 events.

    Readiness rule:
    - An account is considered ready for `asset` when a normalized event in the
      observation window shows trustline establishment (`change_trust` /
      trustline event types) or an inbound receipt of that non-native asset
      for the account.
    - XLM does not require a trustline; readiness is treated as true for XLM.

    Units:
    - readiness_rate is a fraction in [0, 1].
    - observation window is inclusive UTC.
    """
    normalized = _coerce_events(events)
    if not isinstance(asset, str) or not asset.strip():
        raise TrustlineReadinessError("asset must be a non-empty string")
    asset = asset.strip().upper()

    if not isinstance(cohorts, Mapping):
        raise TrustlineReadinessError("cohorts must be a mapping of cohort name to account ids")
    if isinstance(observation_window, Mapping):
        observation_window = ObservationWindow(
            start=_as_utc(observation_window["start"], "observation_window.start"),
            end=_as_utc(observation_window["end"], "observation_window.end"),
        )
    elif observation_window is not None and not isinstance(observation_window, ObservationWindow):
        raise TrustlineReadinessError(
            "observation_window must be an ObservationWindow, mapping, or None"
        )

    window_events = [
        e for e in normalized if observation_window is None or observation_window.contains(_event_time(e))
    ]

    cohort_results: list[CohortReadiness] = []
    for cohort_name, accounts in cohorts.items():
        if not isinstance(cohort_name, str) or not cohort_name.strip():
            raise TrustlineReadinessError("cohort names must be non-empty strings")
        if isinstance(accounts, (str, bytes)) or not isinstance(accounts, Sequence):
            raise TrustlineReadinessError(
                f"cohort {cohort_name!r} must be a sequence of account ids"
            )
        cleaned: list[str] = []
        seen: set[str] = set()
        for i, acct in enumerate(accounts):
            if not isinstance(acct, str) or not acct.strip():
                raise TrustlineReadinessError(
                    f"cohort {cohort_name!r} account[{i}] must be a non-empty string"
                )
            a = acct.strip()
            if a not in seen:
                seen.add(a)
                cleaned.append(a)

        ready_ids: list[str] = []
        for acct in cleaned:
            if asset == "XLM":
                ready_ids.append(acct)
                continue
            is_ready = any(_is_trustline_ready(event, acct, asset) for event in window_events)
            if is_ready:
                ready_ids.append(acct)
        total = len(cleaned)
        ready = len(ready_ids)
        cohort_results.append(
            CohortReadiness(
                cohort=cohort_name.strip(),
                asset=asset,
                total_accounts=total,
                ready_accounts=ready,
                readiness_rate=(ready / total) if total else 0.0,
                ready_account_ids=tuple(ready_ids),
            )
        )

    return TrustlineReadinessReport(
        observation_window=observation_window,
        asset=asset,
        cohorts=tuple(cohort_results),
    )
