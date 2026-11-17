"""Horizon client with pagination, retries, and rate limiting (issue #32).

Reads on-chain delivery data from a Stellar Horizon endpoint over HTTP.
The client is deliberately small and strict:

- **Explicit timeouts** — every request carries a finite ``timeout``.
- **Retries with backoff** — ``429`` and ``5xx`` responses and transport
  failures are retried up to ``max_retries`` times with exponential
  backoff (``backoff_factor * 2**attempt``). ``429`` responses may carry a
  ``Retry-After`` header (seconds or HTTP-date), honored up to
  ``max_retry_after`` seconds. Other ``4xx`` responses are not retried.
- **Deterministic** — no randomized jitter is applied; identical
  sequences of responses produce identical retry behavior. A custom
  ``sleep`` and ``clock`` can be injected for tests.
- **Rate limiting** — a minimum interval between requests is enforced
  from ``rate_limit`` (requests per second, default 5). The first request
  is not delayed.
- **Typed parsing** — payment-like records are parsed into
  :class:`HorizonPayment`; malformed records raise
  :class:`HorizonRecordError` naming the record index rather than being
  silently dropped. Envelope validation (``_embedded.records``) fails
  loudly with :class:`HorizonError`.

The on-chain data path recommends allocations for operators; this module
never signs or submits transactions. Tests inject a fake transport and
never hit a live endpoint.
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
DEFAULT_RATE_LIMIT = 5.0
DEFAULT_MAX_RETRY_AFTER = 60.0
NATIVE_ASSET = "XLM"


class HorizonError(Exception):
    """Base error for Horizon ingestion failures."""


class HorizonTransportError(HorizonError):
    """The endpoint could not be reached within the retry budget."""


class HorizonClientError(HorizonError):
    """A non-retryable ``4xx`` response (other than ``429``)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HorizonRateLimitError(HorizonError):
    """``429`` responses persisted past the retry budget."""


class HorizonServerError(HorizonError):
    """``5xx`` responses persisted past the retry budget."""


class HorizonRecordError(HorizonError):
    """A record in an otherwise successful response failed validation."""


class HorizonEnvelopeError(HorizonError):
    """The response envelope failed validation (missing records list, bad JSON)."""


@dataclass(frozen=True)
class HorizonPayment:
    """A typed, validated payment-like Horizon operation."""

    id: str
    paging_token: str
    transaction_hash: str
    created_at: str
    type: str
    source_account: str
    from_account: str
    to_account: str
    asset: str
    """``XLM`` for the native asset, otherwise ``CODE:ISSUER``."""
    amount: float


@dataclass(frozen=True)
class HorizonPage:
    """One page of records from a Horizon collection endpoint."""

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
    """Minimal session contract; ``requests.Session`` and the built-in
    urllib adapter both satisfy it."""

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
                pass
            return _UrllibResponse(
                status_code=exc.code,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=body,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HorizonTransportError(f"Horizon request to {url} failed: {exc}") from exc


def _validate_base_url(base_url: str) -> str:
    candidate = base_url.strip() if isinstance(base_url, str) else ""
    if not candidate:
        raise HorizonError(
            "a Horizon base URL is required (pass base_url=... or set the HORIZON_URL env var)"
        )
    if not candidate.lower().startswith(("http://", "https://")):
        raise HorizonError(
            f"Horizon base URL must start with http:// or https://, got {candidate!r}"
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
        raise HorizonRecordError(
            f"record {index}: {key!r} must be a non-empty string, got {value!r}"
        )
    return value.strip()


def _amount_field(record: Mapping[str, Any], key: str, *, index: int) -> float:
    value = record.get(key)
    if isinstance(value, bool):
        raise HorizonRecordError(f"record {index}: {key!r} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        amount = float(value)
    elif isinstance(value, str):
        try:
            amount = float(value)
        except ValueError:
            raise HorizonRecordError(
                f"record {index}: {key!r} must be a number, got {value!r}"
            ) from None
    else:
        raise HorizonRecordError(f"record {index}: {key!r} must be a number, got {value!r}")
    if not isfinite(amount):
        raise HorizonRecordError(f"record {index}: {key!r} must be finite, got {value!r}")
    return amount


def parse_payment_record(record: Mapping[str, Any], *, index: int = 0) -> HorizonPayment:
    """Parse and validate a raw Horizon payment-like operation record.

    Raises :class:`HorizonRecordError` naming ``index`` on the first
    malformed field.
    """
    if not isinstance(record, Mapping):
        raise HorizonRecordError(f"record {index}: expected an object, got {record!r}")

    op_type = _string_field(record, "type", index=index)
    asset_type = record.get("asset_type")
    asset_code = record.get("asset_code")
    if isinstance(asset_code, str) and asset_code.strip():
        code = asset_code.strip().upper()
        issuer = record.get("asset_issuer")
        if not isinstance(issuer, str) or not issuer.strip():
            raise HorizonRecordError(
                f"record {index}: issued asset {code!r} is missing 'asset_issuer'"
            )
        try:
            asset = validate_asset_code(f"{code}:{issuer.strip()}")
        except ValueError as exc:
            raise HorizonRecordError(f"record {index}: {exc}") from None
    else:
        if asset_type not in (None, "native"):
            raise HorizonRecordError(
                f"record {index}: non-native asset_type {asset_type!r} requires 'asset_code'"
            )
        asset = NATIVE_ASSET

    return HorizonPayment(
        id=_string_field(record, "id", index=index),
        paging_token=_string_field(record, "paging_token", index=index),
        transaction_hash=_string_field(record, "transaction_hash", index=index),
        created_at=_string_field(record, "created_at", index=index),
        type=op_type,
        source_account=_string_field(record, "source_account", index=index),
        from_account=_string_field(record, "from", index=index),
        to_account=_string_field(record, "to", index=index),
        asset=asset,
        amount=_amount_field(record, "amount", index=index),
    )


class HorizonClient:
    """Synchronous Horizon client with pagination, retries, and rate limits.

    Parameters
    ----------
    base_url:
        Horizon endpoint root; defaults to the ``HORIZON_URL`` environment
        variable. Must start with ``http://`` or ``https://``.
    timeout:
        Per-request timeout in seconds (must be positive and finite).
    max_retries:
        Number of retries after the initial attempt for ``429``/``5xx``
        responses and transport failures. ``0`` disables retrying.
    backoff_factor:
        Base delay in seconds; attempt ``n`` waits ``backoff_factor * 2**n``.
    rate_limit:
        Maximum requests per second; a minimum spacing of ``1/rate_limit``
        seconds is enforced between requests. Must be positive and finite.
    max_retry_after:
        Cap (in seconds) applied to ``Retry-After`` header values.
    session:
        Optional object with a ``get(url, *, headers, timeout)`` method
        returning a response with ``status_code``, ``headers``, and
        ``json()``. Defaults to a built-in urllib adapter.
    sleep / clock:
        Injectable time primitives for deterministic tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        session: Session | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        resolved = base_url if base_url is not None else os.environ.get("HORIZON_URL", "")
        self.base_url = _validate_base_url(resolved)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise HorizonError(f"timeout must be a number, got {timeout!r}")
        if not isfinite(float(timeout)) or float(timeout) <= 0.0:
            raise HorizonError(f"timeout must be a positive, finite number, got {timeout!r}")
        self.timeout = float(timeout)
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise HorizonError(f"max_retries must be an integer, got {max_retries!r}")
        if max_retries < 0:
            raise HorizonError(f"max_retries must be non-negative, got {max_retries}")
        self.max_retries = max_retries
        if not isinstance(backoff_factor, (int, float)) or isinstance(backoff_factor, bool):
            raise HorizonError(f"backoff_factor must be a number, got {backoff_factor!r}")
        if not isfinite(float(backoff_factor)) or float(backoff_factor) < 0.0:
            raise HorizonError(
                f"backoff_factor must be a non-negative, finite number, got {backoff_factor!r}"
            )
        self.backoff_factor = float(backoff_factor)
        if not isinstance(rate_limit, (int, float)) or isinstance(rate_limit, bool):
            raise HorizonError(f"rate_limit must be a number, got {rate_limit!r}")
        if not isfinite(float(rate_limit)) or float(rate_limit) <= 0.0:
            raise HorizonError(f"rate_limit must be a positive, finite number, got {rate_limit!r}")
        self.rate_limit = float(rate_limit)
        if not isinstance(max_retry_after, (int, float)) or isinstance(max_retry_after, bool):
            raise HorizonError(f"max_retry_after must be a number, got {max_retry_after!r}")
        if not isfinite(float(max_retry_after)) or float(max_retry_after) < 0.0:
            raise HorizonError(
                f"max_retry_after must be a non-negative, finite number, got {max_retry_after!r}"
            )
        self.max_retry_after = float(max_retry_after)
        self._session: Session = session if session is not None else _UrllibSession()
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self._min_interval = 1.0 / self.rate_limit
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------ #
    # Transport with throttling, retries, and typed error mapping         #
    # ------------------------------------------------------------------ #

    def _throttle(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = self._clock()
            return
        now = self._clock()
        wait = self._min_interval - (now - self._last_request_at)
        if wait > 0:
            self._sleep(wait)
        self._last_request_at = self._clock()

    @staticmethod
    def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
        return {str(key).lower(): str(value) for key, value in headers.items()}

    def _request(self, url: str) -> Any:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._session.get(
                    url,
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
                status: int | None = None
                error: Exception | None = exc
                response = None
            else:
                status = response.status_code
                error = None

            if response is not None:
                assert status is not None
                if 200 <= status < 300:
                    try:
                        return response.json()
                    except Exception as exc:
                        raise HorizonEnvelopeError(
                            f"Horizon request to {url} returned a non-JSON body: {exc}"
                        ) from exc
                if status == 429 or status >= 500:
                    pass  # retryable; handled below
                if 300 <= status < 400:
                    raise HorizonError(
                        f"Horizon request to {url} returned an unexpected redirect (HTTP {status})"
                    )
                raise HorizonClientError(
                    f"Horizon request to {url} failed with HTTP {status}",
                    status_code=status,
                )

            attempt += 1
            if attempt > self.max_retries:
                if status == 429:
                    raise HorizonRateLimitError(
                        f"Horizon request to {url} was rate limited past {self.max_retries} retries"
                    )
                if status is not None and status >= 500:
                    raise HorizonServerError(
                        f"Horizon request to {url} failed with HTTP {status} past "
                        f"{self.max_retries} retries"
                    )
                raise HorizonTransportError(
                    f"Horizon request to {url} failed past {self.max_retries} retries: {error}"
                )

            retry_after: float | None = None
            if response is not None:
                retry_after = _retry_after_seconds(self._lower_headers(response.headers))
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
                raise HorizonError("path must be a non-empty string")
            base = f"{self.base_url}/{raw.lstrip('/')}"
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
            raise HorizonEnvelopeError(
                f"Horizon response from {url} must be a JSON object, got {payload!r}"
            )
        embedded = payload.get("_embedded")
        records: Any = None
        if isinstance(embedded, dict):
            records = embedded.get("records")
        if records is None and isinstance(payload.get("records"), list):
            records = payload["records"]
        if not isinstance(records, list):
            raise HorizonEnvelopeError(
                f"Horizon response from {url} is missing '_embedded.records' list"
            )
        validated: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise HorizonRecordError(
                    f"record {index}: expected an object, got {record!r}"
                )
            validated.append(record)
        links = payload.get("_links")
        next_url: str | None = None
        if isinstance(links, dict):
            candidate = links.get("next")
            if isinstance(candidate, dict):
                href = candidate.get("href")
                if isinstance(href, str) and href.strip():
                    next_url = href.strip()
        if next_url is None:
            cursor = payload.get("next")
            if isinstance(cursor, str) and cursor.strip():
                next_url = cursor.strip()
        return validated, next_url

    def get_page(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> HorizonPage:
        """Fetch a single page of records from a collection endpoint."""
        url = self._build_url(path, params)
        payload = self._request(url)
        records, next_url = self._records_from_envelope(payload, url=url)
        return HorizonPage(records=tuple(records), next_url=next_url)

    def iter_records(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_records: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate records across pages, following ``next`` links.

        ``max_records`` bounds the total number of records yielded; the
        iterator stops early (without fetching further pages) when the
        bound is reached.
        """
        if max_records is not None:
            if not isinstance(max_records, int) or isinstance(max_records, bool):
                raise HorizonError(f"max_records must be an integer, got {max_records!r}")
            if max_records < 1:
                raise HorizonError(f"max_records must be a positive integer, got {max_records}")
        url = self._build_url(path, params)
        seen = 0
        while url is not None:
            page = self.get_page(url)
            for record in page.records:
                if max_records is not None and seen >= max_records:
                    return
                yield record
                seen += 1
            url = page.next_url

    def iter_payments(
        self,
        account: str,
        *,
        params: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> Iterator[HorizonPayment]:
        """Iterate typed payment operations for ``account``."""
        if not isinstance(account, str) or not account.strip():
            raise HorizonError("account must be a non-empty Stellar address")
        path = f"/accounts/{account.strip()}/payments"
        for index, record in enumerate(self.iter_records(path, params, max_records=max_records)):
            yield parse_payment_record(record, index=index)

    def get_payments(
        self,
        account: str,
        *,
        params: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[HorizonPayment]:
        """Collect :func:`iter_payments` into a list."""
        return list(self.iter_payments(account, params=params, max_records=max_records))
