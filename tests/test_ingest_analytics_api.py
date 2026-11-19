"""Tests for the typed transaction-analytics-api client (issue #33).

Sanitized fixtures only; every transport is a fake and no test performs a
network call. Retry, pagination, typed-parsing, and error-mapping behavior
are deterministic under the injected clock/sleep.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from funding_optimizer.ingest.analytics_api import (
    AccountFlow,
    AnalyticsApiClient,
    AnalyticsApiError,
    AnalyticsClientError,
    AnalyticsEnvelopeError,
    AnalyticsRateLimitError,
    AnalyticsRecordError,
    AnalyticsServerError,
    AnalyticsTransportError,
    AssetFlow,
    parse_account_flow,
    parse_asset_flow,
    parse_flow_bucket,
)

BASE_URL = "https://analytics-testnet.stellar.org"


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: Any = None,
        headers: dict[str, str] | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._body


class FakeSession:
    """Scripted session: pops one response (or exception) per GET."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.requests: list[tuple[str, dict[str, str], float]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.requests.append((url, dict(headers), timeout))
        if not self._script:
            raise AssertionError(f"unexpected request to {url}")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return cast(FakeResponse, step)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _asset_flow_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "asset": "usdc:G" + "A" * 55,
        "interval": "1h",
        "buckets": [
            {
                "start": "2026-10-01T00:00:00Z",
                "volume": "184220.55",
                "payments": 12,
                "net_flow": "-1200.00",
            },
        ],
    }
    body.update(overrides)
    return body


def _account_flow_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "account": "GRECIPIENT",
        "interval": "1h",
        "inflow": "1500.25",
        "outflow": "200.00",
        "counterparties": 7,
    }
    body.update(overrides)
    return body


def _page(records: list[dict[str, Any]], next_url: str | None = None) -> FakeResponse:
    body: dict[str, Any] = {"records": records}
    if next_url is not None:
        body["next"] = next_url
    return FakeResponse(200, body)


def _client(
    session: FakeSession,
    *,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    api_key: str | None = None,
    clock: FakeClock | None = None,
) -> AnalyticsApiClient:
    fake_clock = clock or FakeClock()
    return AnalyticsApiClient(
        BASE_URL,
        api_key=api_key,
        timeout=10.0,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        session=session,
        sleep=fake_clock.sleep,
        clock=fake_clock,
    )


# --------------------------------------------------------------------- #
# Typed parsing                                                           #
# --------------------------------------------------------------------- #


def test_parse_flow_bucket_parses_numeric_strings() -> None:
    bucket = parse_flow_bucket(
        {"start": "2026-10-01T00:00:00Z", "volume": "100.5", "payments": 3, "net_flow": "-2"},
        index=4,
    )
    assert bucket.start == "2026-10-01T00:00:00Z"
    assert bucket.volume == pytest.approx(100.5)
    assert bucket.payments == 3
    assert bucket.net_flow == pytest.approx(-2.0)


def test_parse_flow_bucket_rejects_malformed_volume() -> None:
    record = {"start": "t", "volume": "nope", "payments": 1, "net_flow": "0"}
    with pytest.raises(AnalyticsRecordError, match="record 2"):
        parse_flow_bucket(record, index=2)


def test_parse_flow_bucket_rejects_negative_count() -> None:
    record = {"start": "t", "volume": "1", "payments": -1, "net_flow": "0"}
    with pytest.raises(AnalyticsRecordError, match="payments"):
        parse_flow_bucket(record)


def test_parse_asset_flow_normalizes_asset_code() -> None:
    flow = parse_asset_flow(_asset_flow_body())
    assert isinstance(flow, AssetFlow)
    assert flow.asset == "USDC:" + "G" + "A" * 55
    assert flow.interval == "1h"
    assert len(flow.buckets) == 1
    assert flow.buckets[0].volume == pytest.approx(184220.55)


def test_parse_asset_flow_requires_buckets_list() -> None:
    with pytest.raises(AnalyticsRecordError, match="buckets"):
        parse_asset_flow({"asset": "XLM", "interval": "1h"})


def test_parse_asset_flow_rejects_invalid_asset() -> None:
    body = _asset_flow_body(asset="!!!")
    with pytest.raises(AnalyticsRecordError, match="asset"):
        parse_asset_flow(body)


def test_parse_account_flow_happy_path() -> None:
    flow = parse_account_flow(_account_flow_body())
    assert isinstance(flow, AccountFlow)
    assert flow.account == "GRECIPIENT"
    assert flow.inflow == pytest.approx(1500.25)
    assert flow.counterparties == 7


def test_parse_account_flow_rejects_non_object() -> None:
    with pytest.raises(AnalyticsRecordError, match="must be an object"):
        parse_account_flow(["not", "an", "object"])


# --------------------------------------------------------------------- #
# Transport: configuration, retries, error mapping                        #
# --------------------------------------------------------------------- #


def test_requires_http_base_url() -> None:
    with pytest.raises(AnalyticsApiError, match="base URL"):
        AnalyticsApiClient("ftp://example.org")
    with pytest.raises(AnalyticsApiError, match="base URL"):
        AnalyticsApiClient("")


def test_timeout_and_retry_configuration_are_validated() -> None:
    with pytest.raises(AnalyticsApiError, match="timeout"):
        AnalyticsApiClient(BASE_URL, timeout=0)
    with pytest.raises(AnalyticsApiError, match="max_retries"):
        AnalyticsApiClient(BASE_URL, max_retries=-1)
    with pytest.raises(AnalyticsApiError, match="backoff_factor"):
        AnalyticsApiClient(BASE_URL, backoff_factor=-1.0)


def test_bearer_key_is_only_sent_when_configured() -> None:
    session = FakeSession([FakeResponse(200, {"status": "ok"})])
    client = _client(session, api_key="test-only-key")
    client.get_health()
    headers = session.requests[0][1]
    assert headers["Authorization"] == "Bearer test-only-key"

    plain = FakeSession([FakeResponse(200, {"status": "ok"})])
    _client(plain).get_health()
    assert "Authorization" not in plain.requests[0][1]


def test_retries_rate_limits_then_succeeds() -> None:
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "2"}),
            FakeResponse(429),
            FakeResponse(200, {"status": "ok"}),
        ]
    )
    client = _client(session, clock=clock, backoff_factor=0.5)

    assert client.get_health() == {"status": "ok"}
    assert len(session.requests) == 3
    assert clock.sleeps == [2.0, 0.5]  # Retry-After, then 0.5 * 2**0


def test_retries_server_errors_with_exponential_backoff() -> None:
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(500),
            FakeResponse(503),
            FakeResponse(200, {"status": "ok"}),
        ]
    )
    client = _client(session, clock=clock, backoff_factor=0.5)
    client.get_health()
    assert clock.sleeps == [0.5, 1.0]


def test_retry_after_is_capped() -> None:
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(429, headers={"Retry-After": "999"}),
            FakeResponse(200, {"status": "ok"}),
        ]
    )
    client = AnalyticsApiClient(
        BASE_URL,
        timeout=10.0,
        max_retry_after=5.0,
        session=session,
        sleep=clock.sleep,
    )
    client.get_health()
    assert clock.sleeps == [5.0]


def test_transport_failure_is_retried_then_raises() -> None:
    session = FakeSession([OSError("offline"), OSError("offline"), OSError("offline")])
    client = _client(session, max_retries=2)
    with pytest.raises(AnalyticsTransportError, match="past 2 retries"):
        client.get_health()
    assert len(session.requests) == 3


def test_non_retryable_4xx_fails_immediately() -> None:
    session = FakeSession([FakeResponse(404)])
    client = _client(session)
    with pytest.raises(AnalyticsClientError, match="HTTP 404") as exc_info:
        client.get_health()
    assert exc_info.value.status_code == 404
    assert len(session.requests) == 1


def test_rate_limit_past_budget_raises_dedicated_error() -> None:
    session = FakeSession([FakeResponse(429), FakeResponse(429)])
    client = _client(session, max_retries=1)
    with pytest.raises(AnalyticsRateLimitError, match="rate limited"):
        client.get_health()


def test_server_error_past_budget_raises_dedicated_error() -> None:
    session = FakeSession([FakeResponse(500), FakeResponse(500)])
    client = _client(session, max_retries=1)
    with pytest.raises(AnalyticsServerError, match="HTTP 500"):
        client.get_health()


def test_non_json_body_raises_envelope_error() -> None:
    session = FakeSession([FakeResponse(200, json_error=ValueError("bad json"))])
    with pytest.raises(AnalyticsEnvelopeError, match="non-JSON"):
        _client(session).get_health()


# --------------------------------------------------------------------- #
# Endpoint helpers and pagination                                         #
# --------------------------------------------------------------------- #


def test_url_defaults_to_v1_and_keeps_absolute_next_links() -> None:
    session = FakeSession(
        [
            _page([_asset_flow_body()], next_url=f"{BASE_URL}/v1/flows/assets?page=2"),
            _page([]),
        ]
    )
    client = _client(session)
    records = list(client.iter_records("/flows/assets"))
    assert len(records) == 1
    assert session.requests[0][0] == f"{BASE_URL}/v1/flows/assets"
    assert session.requests[1][0] == f"{BASE_URL}/v1/flows/assets?page=2"


def test_max_records_stops_before_fetching_more_pages() -> None:
    session = FakeSession(
        [
            _page([_asset_flow_body()], next_url=f"{BASE_URL}/v1/flows/assets?page=2"),
        ]
    )
    client = _client(session)
    records = list(client.iter_records("/flows/assets", max_records=1))
    assert len(records) == 1
    assert len(session.requests) == 1


def test_max_records_must_be_positive_integer() -> None:
    client = _client(FakeSession([]))
    with pytest.raises(AnalyticsApiError, match="max_records"):
        list(client.iter_records("/flows/assets", max_records=0))


def test_get_asset_flow_and_account_flow() -> None:
    session = FakeSession(
        [
            FakeResponse(200, _asset_flow_body()),
            FakeResponse(200, _account_flow_body()),
            FakeResponse(200, {"status": "ok", "lag_seconds": 5}),
        ]
    )
    client = _client(session)
    flow = client.get_asset_flow("usdc:G" + "A" * 55, params={"interval": "1h"})
    assert flow.buckets[0].payments == 12
    assert "interval=1h" in session.requests[0][0]

    account = client.get_account_flow("GRECIPIENT")
    assert account.outflow == pytest.approx(200.0)
    assert client.get_health()["lag_seconds"] == 5


def test_iter_asset_flows_parses_each_page_record() -> None:
    session = FakeSession([_page([_asset_flow_body(), _asset_flow_body()])])
    flows = list(_client(session).iter_asset_flows("USDC"))
    assert len(flows) == 2
    assert all(isinstance(flow, AssetFlow) for flow in flows)


def test_malformed_record_in_envelope_is_rejected() -> None:
    session = FakeSession([_page([{"asset": "XLM"}]), _page([{"asset": "XLM"}])])
    client = _client(session)
    with pytest.raises(AnalyticsRecordError, match="buckets"):
        list(client.iter_asset_flows("XLM"))


def test_missing_records_list_raises_envelope_error() -> None:
    session = FakeSession([FakeResponse(200, {"unexpected": True})])
    with pytest.raises(AnalyticsEnvelopeError, match="records list"):
        _client(session).get_page("/flows/assets")
