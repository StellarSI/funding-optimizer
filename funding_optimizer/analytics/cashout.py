"""Realized cash-out rates from on-chain flows (issue #38).

Pairs disbursements (OUT events) with matching receipts (IN events) from the
same normalized event stream (P34) and reports how much of the disbursed
amount was actually received. Units and conventions:

- ``disbursed_amount`` / ``realized_amount`` are in asset units.
- ``realization_rate`` and ``recipient_cashout_rate`` are fractions in [0, 1].
- The observation window is inclusive UTC; both legs must fall inside it.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from funding_optimizer.analytics.delivery import ObservationWindow
from funding_optimizer.ingest.normalize import NormalizedEvent


class CashoutAnalyticsError(ValueError):
    """Raised when realized cash-out analytics inputs are invalid."""


@dataclass(frozen=True)
class AssetCashoutRate:
    """Realized cash-out metrics for one asset.

    Units: amounts in asset units; both rates are fractions in [0, 1].
    """

    asset: str
    disbursements: int
    matched_disbursements: int
    disbursed_amount: float
    realized_amount: float
    realization_rate: float
    recipients: int
    cashed_out_recipients: int
    recipient_cashout_rate: float


@dataclass(frozen=True)
class CashoutReport:
    """Realized cash-out rates across assets."""

    observation_window: ObservationWindow | None
    disbursement_account: str | None
    assets: Sequence[AssetCashoutRate]
    disbursements: int
    matched_disbursements: int
    disbursed_amount: float
    realized_amount: float
    realization_rate: float
    recipient_cashout_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_window": (
                None
                if self.observation_window is None
                else {
                    "start": self.observation_window.start.isoformat().replace("+00:00", "Z"),
                    "end": self.observation_window.end.isoformat().replace("+00:00", "Z"),
                    "unit": "utc-iso8601",
                }
            ),
            "units": {
                "amount": "asset units",
                "realization_rate": "fraction",
                "recipient_cashout_rate": "fraction",
            },
            "disbursement_account": self.disbursement_account,
            "disbursements": self.disbursements,
            "matched_disbursements": self.matched_disbursements,
            "disbursed_amount": self.disbursed_amount,
            "realized_amount": self.realized_amount,
            "realization_rate": self.realization_rate,
            "recipient_cashout_rate": self.recipient_cashout_rate,
            "assets": [
                {
                    "asset": a.asset,
                    "disbursements": a.disbursements,
                    "matched_disbursements": a.matched_disbursements,
                    "disbursed_amount": a.disbursed_amount,
                    "realized_amount": a.realized_amount,
                    "realization_rate": a.realization_rate,
                    "recipients": a.recipients,
                    "cashed_out_recipients": a.cashed_out_recipients,
                    "recipient_cashout_rate": a.recipient_cashout_rate,
                }
                for a in self.assets
            ],
        }


def _as_utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise CashoutAnalyticsError(
                f"{name} must be an ISO-8601 datetime, got {value!r}"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise CashoutAnalyticsError(
            f"{name} must be a datetime or ISO-8601 string, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_time(event: NormalizedEvent) -> datetime:
    return _as_utc(event.occurred_at, "occurred_at")


def _coerce_events(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        raise CashoutAnalyticsError("events must be an iterable of NormalizedEvent")
    result: list[NormalizedEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, NormalizedEvent):
            raise CashoutAnalyticsError(
                f"events[{index}] must be a NormalizedEvent, got {type(event).__name__}"
            )
        result.append(event)
    return result


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_realized_cashout_rates(
    events: Iterable[NormalizedEvent],
    *,
    disbursement_account: str | None = None,
    observation_window: ObservationWindow | Mapping[str, datetime | str] | None = None,
) -> CashoutReport:
    """Compute realized cash-out rates from normalized P34 events.

    A disbursement is an OUT event (from ``disbursement_account`` when given)
    with a counterparty. It is realized when an IN event for the same asset,
    received by that counterparty at or after the disbursement, exists inside
    the inclusive observation window. Receipts are consumed once so a single
    receipt cannot realize two disbursements.
    """
    normalized = _coerce_events(events)
    window: ObservationWindow | None
    if isinstance(observation_window, Mapping):
        window = ObservationWindow(
            start=_as_utc(observation_window["start"], "observation_window.start"),
            end=_as_utc(observation_window["end"], "observation_window.end"),
        )
    elif observation_window is not None and not isinstance(observation_window, ObservationWindow):
        raise CashoutAnalyticsError(
            "observation_window must be an ObservationWindow, mapping, or None"
        )
    else:
        window = observation_window

    subject = disbursement_account.strip() if isinstance(disbursement_account, str) else None
    if disbursement_account is not None and not subject:
        raise CashoutAnalyticsError("disbursement_account must be a non-empty string when provided")

    ordered = sorted(normalized, key=lambda event: (_event_time(event), event.event_id))
    in_window = [
        event
        for event in ordered
        if window is None or window.contains(_event_time(event))
    ]

    disbursements = [
        event
        for event in in_window
        if event.direction.value == "out"
        and event.counterparty
        and (subject is None or event.account == subject)
    ]

    used_receipts: set[str] = set()
    realized_by_asset: dict[str, float] = {}
    matched_by_asset: dict[str, int] = {}
    disbursed_by_asset: dict[str, float] = {}
    recipients_by_asset: dict[str, set[str]] = {}
    cashed_by_asset: dict[str, set[str]] = {}
    all_recipients: set[str] = set()
    cashed_recipients: set[str] = set()

    for disbursement in disbursements:
        asset = disbursement.asset
        recipient = disbursement.counterparty or ""
        disbursed_by_asset[asset] = disbursed_by_asset.get(asset, 0.0) + disbursement.amount
        recipients_by_asset.setdefault(asset, set()).add(recipient)
        all_recipients.add(recipient)

        disbursed_at = _event_time(disbursement)
        receipt = None
        for candidate in in_window:
            if candidate.event_id in used_receipts:
                continue
            if candidate.direction.value != "in":
                continue
            if candidate.asset != asset or candidate.account != recipient:
                continue
            if candidate.counterparty not in (None, disbursement.account):
                continue
            if _event_time(candidate) < disbursed_at:
                continue
            receipt = candidate
            break

        if receipt is None:
            continue
        used_receipts.add(receipt.event_id)
        matched_by_asset[asset] = matched_by_asset.get(asset, 0) + 1
        realized_by_asset[asset] = realized_by_asset.get(asset, 0.0) + disbursement.amount
        cashed_by_asset.setdefault(asset, set()).add(recipient)
        cashed_recipients.add(recipient)

    assets = [
        AssetCashoutRate(
            asset=asset,
            disbursements=sum(1 for d in disbursements if d.asset == asset),
            matched_disbursements=matched_by_asset.get(asset, 0),
            disbursed_amount=disbursed_by_asset[asset],
            realized_amount=realized_by_asset.get(asset, 0.0),
            realization_rate=_rate(realized_by_asset.get(asset, 0.0), disbursed_by_asset[asset]),
            recipients=len(recipients_by_asset.get(asset, set())),
            cashed_out_recipients=len(cashed_by_asset.get(asset, set())),
            recipient_cashout_rate=_rate(
                float(len(cashed_by_asset.get(asset, set()))),
                float(len(recipients_by_asset.get(asset, set()))),
            ),
        )
        for asset in disbursed_by_asset
    ]

    disbursed_total = sum(entry.disbursed_amount for entry in assets)
    realized_total = sum(entry.realized_amount for entry in assets)
    return CashoutReport(
        observation_window=window,
        disbursement_account=subject,
        assets=tuple(assets),
        disbursements=len(disbursements),
        matched_disbursements=sum(entry.matched_disbursements for entry in assets),
        disbursed_amount=disbursed_total,
        realized_amount=realized_total,
        realization_rate=_rate(realized_total, disbursed_total),
        recipient_cashout_rate=_rate(float(len(cashed_recipients)), float(len(all_recipients))),
    )
