"""Typed transaction-analytics-api client (issue #33).

Reads flow data from a StellarSI ``transaction-analytics-api`` endpoint over
HTTP. The client follows the same strict, deterministic contract as
:mod:`funding_optimizer.ingest.horizon`:

- **Explicit timeouts** — every request carries a finite ``timeout``.
- **Retries with backoff** — ``429`` and ``5xx`` responses and transport
  failures are retried up to ``max_retries`` times with exponential
  backoff (``backoff_factor * 2**attempt``). ``Retry-After`` (seconds or
  HTTP-date) is honored up to ``max_retry_after`` seconds; other ``4xx``
  responses are not retried.
- **Deterministic** — no randomized jitter; an optional bearer API key is
  sent only when configured. ``sleep`` and ``clock`` can be injected for
  tests.
- **Pagination** — ``iter_records`` follows ``next`` links across pages and
  ``max_records`` bounds how many records are consumed.
- **Typed parsing** — flow payloads are parsed into frozen dataclasses;
  malformed records raise :class:`AnalyticsRecordError` naming the record
  index, and malformed envelopes raise :class:`AnalyticsEnvelopeError`.

The base URL defaults to ``ANALYTICS_API_URL`` and the optional bearer token
to ``ANALYTICS_API_KEY``. The API root path defaults to ``/v1``. This module
only reads analytics data: it recommends allocations and never signs or
submits payouts. Tests inject a fake session and never hit a live endpoint.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Any, Protocol

from funding_optimizer.models import validate_asset_code

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_MAX_RETRY_AFTER = 60.0
DEFAULT_API_VERSION = "/v1"


class AnalyticsApiError(Exception):
    """Base error for transaction-analytics-api ingestion failures."""


class AnalyticsTransportError(AnalyticsApiError):
    """The endpoint could not be reached within the retry budget."""


class AnalyticsClientError(AnalyticsApiError):
    """A non-retryable ``4xx`` response (other than ``429``)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnalyticsRateLimitError(AnalyticsApiError):
    """``429`` responses persisted past the retry budget."""


class AnalyticsServerError(AnalyticsApiError):
    """``5xx`` responses persisted past the retry budget."""


class AnalyticsRecordError(AnalyticsApiError):
    """A record in an otherwise successful response failed validation."""


class AnalyticsEnvelopeError(AnalyticsApiError):
    """The response envelope failed validation (missing records, bad JSON)."""


@dataclass(frozen=True)
class FlowBucket:
    """One time bucket of asset flow analytics."""

    start: str
    volume: float
    payments: int
    net_flow: float


@dataclass(frozen=True)
class AssetFlow:
    """Volume, velocity and net flow for one asset."""

    asset: str
    interval: str
    buckets: tuple[FlowBucket, ...]


@dataclass(frozen=True)
class AccountFlow:
    """Inflow, outflow and counterparty count for one account."""

    account: str
    interval: str
    inflow: float
    outflow: float
    counterparties: int


@dataclass(frozen=True)
class AnalyticsPage:
    """One page of raw records from a collection endpoint."""

    records: tuple[dict[str, Any], ...]
    next_url: str | None


class Response(Protocol):
    """Minimal response contract the client relies on (test-friendly)."""

    @property
    def status_code(self) -> int:  # pragma: no cover - structural protocol
        ...

    @property
    def headers(self) -> Mapping[str, str]:  # pragma: no cover - structural protocol
        ...

    def json(self) -> Any:  # pragma: no cover - structural protocol
        ...


class Session(Protocol):
    """Minimal session contract satisfied by a requests session or fake."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Response:  # pragma: no cover - structural protocol
        ...


class _UrllibResponse:
    def __init__(self, status_code: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = dict(headers)
        self._body = body

    def json(self) -> Any:
        return json.loads(self._body.decode("utf-8"))


class _UrllibSession:
    """Default session built on :mod:`urllib.request` (no dependencies)."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Response:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as raw:
                return _UrllibResponse(
                    status_code=raw.status,
                    headers={k.lower(): v for k, v in raw.headers.items()},
                    body=raw.read(),
                )
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # noqa: BLE001 - best effort, never mask the HTTP status
                body = b""
            return _UrllibResponse(
                status_code=exc.code,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=body,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AnalyticsTransportError(
                f"transaction-analytics-api request to {url} failed: {exc}"
            ) from exc


def _validate_base_url(base_url: str) -> str:
    candidate = base_url.strip() if isinstance(base_url, str) else ""
    if not candidate:
        raise AnalyticsApiError(
            "a transaction-analytics-api base URL is required "
            "(pass base_url=... or set the ANALYTICS_API_URL env var)"
        )
    if not candidate.lower().startswith(("http://", "https://")):
        raise AnalyticsApiError(
            f"analytics API base URL must start with http:// or https://, got {candidate!r}"
        )
    return candidate.rstrip("/")


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            raw = value
            break
    if raw is None:
        return None
    raw = str(raw).strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, when.timestamp() - time.time())
    return seconds if isfinite(seconds) and seconds >= 0 else None


def _string_field(record: Mapping[str, Any], key: str, *, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AnalyticsRecordError(
            f"record {index}: {key!r} must be a non-empty string, got {value!r}"
        )
    return value.strip()


def _number_field(record: Mapping[str, Any], key: str, *, index: int) -> float:
    value = record.get(key)
    if isinstance(value, bool):
        raise AnalyticsRecordError(f"record {index}: {key!r} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            raise AnalyticsRecordError(
                f"record {index}: {key!r} must be a number, got {value!r}"
            ) from None
    else:
        raise AnalyticsRecordError(f"record {index}: {key!r} must be a number, got {value!r}")
    if not isfinite(number):
        raise AnalyticsRecordError(f"record {index}: {key!r} must be finite, got {value!r}")
    return number


def _count_field(record: Mapping[str, Any], key: str, *, index: int) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalyticsRecordError(
            f"record {index}: {key!r} must be a non-negative integer, got {value!r}"
        )
    return value


def parse_flow_bucket(record: Mapping[str, Any], *, index: int = 0) -> FlowBucket:
    """Parse and validate one asset-flow bucket, naming ``index`` on failure."""
    if not isinstance(record, Mapping):
        raise AnalyticsRecordError(f"record {index}: expected an object, got {record!r}")
    return FlowBucket(
        start=_string_field(record, "start", index=index),
        volume=_number_field(record, "volume", index=index),
        payments=_count_field(record, "payments", index=index),
        net_flow=_number_field(record, "net_flow", index=index),
    )


def parse_asset_flow(payload: Any) -> AssetFlow:
    """Parse a ``/v1/flows/assets/...`` response into an :class:`AssetFlow`."""
    if not isinstance(payload, Mapping):
        raise AnalyticsRecordError(f"asset flow response must be an object, got {payload!r}")
    raw_asset = _string_field(payload, "asset", index=0)
    try:
        asset = validate_asset_code(raw_asset)
    except ValueError as exc:
        raise AnalyticsRecordError(f"asset: {exc}") from None
    interval = _string_field(payload, "interval", index=0)
    buckets = payload.get("buckets")
    if not isinstance(buckets, list):
        raise AnalyticsRecordError("asset flow response is missing a 'buckets' list")
    parsed = tuple(parse_flow_bucket(bucket, index=index) for index, bucket in enumerate(buckets))
    return AssetFlow(asset=asset, interval=interval, buckets=parsed)


def parse_account_flow(payload: Any) -> AccountFlow:
    """Parse a ``/v1/flows/accounts/...`` response into an :class:`AccountFlow`."""
    if not isinstance(payload, Mapping):
        raise AnalyticsRecordError(f"account flow response must be an object, got {payload!r}")
    return AccountFlow(
        account=_string_field(payload, "account", index=0),
        interval=_string_field(payload, "interval", index=0),
        inflow=_number_field(payload, "inflow", index=0),
        outflow=_number_field(payload, "outflow", index=0),
        counterparties=_count_field(payload, "counterparties", index=0),
    )


class AnalyticsApiClient:
    """Synchronous transaction-analytics-api client with retries and paging.

    Parameters
    ----------
    base_url:
        API root; defaults to the ``ANALYTICS_API_URL`` environment variable.
        Must start with ``http://`` or ``https://``.
    api_key:
        Optional bearer token; defaults to ``ANALYTICS_API_KEY``. Sent only
        when non-empty, as ``Authorization: Bearer <key>``.
    timeout:
        Per-request timeout in seconds (must be positive and finite).
    max_retries:
        Retries after the initial attempt for ``429``/``5xx`` and transport
        failures. ``0`` disables retrying.
    backoff_factor:
        Base delay; attempt ``n`` waits ``backoff_factor * 2**n`` seconds.
    max_retry_after:
        Cap applied to ``Retry-After`` values, in seconds.
    session:
        Optional object with ``get(url, *, headers, timeout)``. Defaults to a
        built-in urllib adapter.
    sleep / clock:
        Injectable time primitives for deterministic tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        session: Session | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        resolved = base_url if base_url is not None else os.environ.get("ANALYTICS_API_URL", "")
        self.base_url = _validate_base_url(resolved)

        key = api_key if api_key is not None else os.environ.get("ANALYTICS_API_KEY", "")
        self.api_key = key.strip() if isinstance(key, str) else ""

        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise AnalyticsApiError(f"timeout must be a number, got {timeout!r}")
        if not isfinite(float(timeout)) or float(timeout) <= 0.0:
            raise AnalyticsApiError(f"timeout must be a positive, finite number, got {timeout!r}")
        self.timeout = float(timeout)

        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise AnalyticsApiError(f"max_retries must be an integer, got {max_retries!r}")
        if max_retries < 0:
            raise AnalyticsApiError(f"max_retries must be non-negative, got {max_retries}")
        self.max_retries = max_retries

        if not isinstance(backoff_factor, (int, float)) or isinstance(backoff_factor, bool):
            raise AnalyticsApiError(f"backoff_factor must be a number, got {backoff_factor!r}")
        if not isfinite(float(backoff_factor)) or float(backoff_factor) < 0.0:
            raise AnalyticsApiError(
                f"backoff_factor must be a non-negative, finite number, got {backoff_factor!r}"
            )
        self.backoff_factor = float(backoff_factor)

        if not isinstance(max_retry_after, (int, float)) or isinstance(max_retry_after, bool):
            raise AnalyticsApiError(f"max_retry_after must be a number, got {max_retry_after!r}")
        if not isfinite(float(max_retry_after)) or float(max_retry_after) < 0.0:
            raise AnalyticsApiError(
                f"max_retry_after must be a non-negative, finite number, got {max_retry_after!r}"
            )
        self.max_retry_after = float(max_retry_after)

        self._session: Session = session if session is not None else _UrllibSession()
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic

    # ------------------------------------------------------------------ #
    # Transport with retries and typed error mapping                      #
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, url: str) -> Any:
        attempt = 0
        while True:
            response: Response | None = None
            status: int | None = None
            error: Exception | None = None
            try:
                response = self._session.get(
                    url,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                status = response.status_code
            except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
                error = exc
                response = None
                status = None

            if response is not None:
                assert status is not None
                if 200 <= status < 300:
                    try:
                        return response.json()
                    except Exception as exc:
                        raise AnalyticsEnvelopeError(
                            f"transaction-analytics-api request to {url} returned "
                            f"a non-JSON body: {exc}"
                        ) from exc
                if 300 <= status < 400:
                    raise AnalyticsApiError(
                        f"transaction-analytics-api request to {url} returned an "
                        f"unexpected redirect (HTTP {status})"
                    )
                if status != 429 and status < 500:
                    raise AnalyticsClientError(
                        f"transaction-analytics-api request to {url} failed with HTTP {status}",
                        status_code=status,
                    )
                # 429 and 5xx are retryable; fall through to retry handling.

            attempt += 1
            if attempt > self.max_retries:
                if status == 429:
                    raise AnalyticsRateLimitError(
                        f"transaction-analytics-api request to {url} was rate limited "
                        f"past {self.max_retries} retries"
                    )
                if status is not None and status >= 500:
                    raise AnalyticsServerError(
                        f"transaction-analytics-api request to {url} failed with HTTP "
                        f"{status} past {self.max_retries} retries"
                    )
                raise AnalyticsTransportError(
                    f"transaction-analytics-api request to {url} failed past "
                    f"{self.max_retries} retries: {error}"
                )

            retry_after: float | None = None
            if response is not None:
                retry_after = _retry_after_seconds(
                    {str(k).lower(): str(v) for k, v in response.headers.items()}
                )
                if retry_after is not None:
                    retry_after = min(retry_after, self.max_retry_after)
            delay = (
                retry_after
                if retry_after is not None
                else self.backoff_factor * (2 ** (attempt - 1))
            )
            self._sleep(delay)

    # ------------------------------------------------------------------ #
    # URL handling and page envelopes                                     #
    # ------------------------------------------------------------------ #

    def _build_url(self, path: str, params: Mapping[str, Any] | None) -> str:
        if isinstance(path, str) and path.lower().startswith(("http://", "https://")):
            base = path
        else:
            raw = path if isinstance(path, str) else ""
            if not raw.strip():
                raise AnalyticsApiError("path must be a non-empty string")
            segment = raw.strip().lstrip("/")
            version = DEFAULT_API_VERSION.strip("/")
            if segment == version or segment.startswith(f"{version}/"):
                base = f"{self.base_url}/{segment}"
            else:
                base = f"{self.base_url}/{version}/{segment}"
        if not params:
            return base
        query = urllib.parse.urlencode(
            {key: str(value) for key, value in params.items()}, doseq=True
        )
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}{query}" if query else base

    @staticmethod
    def _records_from_envelope(
        payload: Any, *, url: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not isinstance(payload, dict):
            raise AnalyticsEnvelopeError(
                f"analytics API response from {url} must be a JSON object, got {payload!r}"
            )
        embedded = payload.get("_embedded")
        records: Any = None
        if isinstance(embedded, dict):
            records = embedded.get("records")
        for key in ("records", "data", "items"):
            if records is None and isinstance(payload.get(key), list):
                records = payload[key]
        if not isinstance(records, list):
            raise AnalyticsEnvelopeError(
                f"analytics API response from {url} is missing a records list"
            )
        validated: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise AnalyticsRecordError(f"record {index}: expected an object, got {record!r}")
            validated.append(record)

        next_url: str | None = None
        links = payload.get("_links")
        if isinstance(links, dict):
            candidate = links.get("next")
            if isinstance(candidate, dict):
                href = candidate.get("href")
                if isinstance(href, str) and href.strip():
                    next_url = href.strip()
        if next_url is None:
            for key in ("next", "next_url"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    next_url = candidate.strip()
                    break
        if next_url is None:
            meta = payload.get("meta")
            if isinstance(meta, dict):
                candidate = meta.get("next")
                if isinstance(candidate, str) and candidate.strip():
                    next_url = candidate.strip()
        return validated, next_url

    def get_page(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> AnalyticsPage:
        """Fetch a single page of records from a collection endpoint."""
        url = self._build_url(path, params)
        payload = self._request(url)
        records, next_url = self._records_from_envelope(payload, url=url)
        return AnalyticsPage(records=tuple(records), next_url=next_url)

    def iter_records(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_records: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate raw records across pages, following ``next`` links."""
        if max_records is not None:
            if not isinstance(max_records, int) or isinstance(max_records, bool):
                raise AnalyticsApiError(f"max_records must be an integer, got {max_records!r}")
            if max_records < 1:
                raise AnalyticsApiError(
                    f"max_records must be a positive integer, got {max_records}"
                )
        url: str | None = self._build_url(path, params)
        seen = 0
        while url is not None:
            page = self.get_page(url)
            for record in page.records:
                if max_records is not None and seen >= max_records:
                    return
                yield record
                seen += 1
            if max_records is not None and seen >= max_records:
                return
            url = page.next_url

    def get_asset_flow(
        self,
        asset: str,
        params: Mapping[str, Any] | None = None,
    ) -> AssetFlow:
        """Fetch and parse the flow summary for one asset."""
        if not isinstance(asset, str) or not asset.strip():
            raise AnalyticsApiError("asset must be a non-empty CODE or CODE:ISSUER string")
        payload = self._request(self._build_url(f"/flows/assets/{asset.strip()}", params))
        return parse_asset_flow(payload)

    def get_account_flow(
        self,
        account: str,
        params: Mapping[str, Any] | None = None,
    ) -> AccountFlow:
        """Fetch and parse the flow summary for one account."""
        if not isinstance(account, str) or not account.strip():
            raise AnalyticsApiError("account must be a non-empty Stellar address")
        payload = self._request(self._build_url(f"/flows/accounts/{account.strip()}", params))
        return parse_account_flow(payload)

    def iter_asset_flows(
        self,
        asset: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_records: int | None = None,
    ) -> Iterator[AssetFlow]:
        """Iterate typed asset-flow records across pages."""
        for index, record in enumerate(
            self.iter_records(f"/flows/assets/{asset}", params, max_records=max_records)
        ):
            yield parse_asset_flow(record)

    def get_health(self) -> dict[str, Any]:
        """Return the raw health payload as a validated JSON object."""
        payload = self._request(self._build_url("/health", None))
        if not isinstance(payload, dict):
            raise AnalyticsEnvelopeError("health response must be a JSON object")
        return payload
