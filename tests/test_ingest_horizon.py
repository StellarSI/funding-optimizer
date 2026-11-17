"""Tests for the Horizon ingestion client (issue #32).

Sanitized fixtures only; every transport is a fake and no test performs a
network call. Retry, pagination, rate-limit, malformed-record, and error
mapping behavior are all deterministic under the injected clock/sleep.
"""

from __future__ import annotations

from typing import Any

import pytest

from funding_optimizer.ingest.horizon import (
    HorizonClient,
    HorizonClientError,
    HorizonEnvelopeError,
    HorizonError,
    HorizonPayment,
    HorizonRateLimitError,
    HorizonRecordError,
    HorizonServerError,
    HorizonTransportError,
    parse_payment_record,
)

BASE_URL = "https://horizon-testnet.stellar.org"


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
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.requests.append((url, dict(headers), timeout))
        if not self._script:
            raise AssertionError(f"unexpected request to {url}")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _payment_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "123",
        "paging_token": "123",
        "transaction_hash": "deadbeef",
        "created_at": "2026-01-01T00:00:00Z",
        "type": "payment",
        "source_account": "GSENDER",
        "from": "GSENDER",
        "to": "GRECIPIENT",
        "asset_type": "native",
        "amount": "12.5000000",
    }
    record.update(overrides)
    return record


def _page(records: list[dict[str, Any]], next_url: str | None = None) -> FakeResponse:
    body: dict[str, Any] = {"_embedded": {"records": records}, "_links": {}}
    if next_url is not None:
        body["_links"]["next"] = {"href": next_url}
    return FakeResponse(200, body)


def _client(
    session: FakeSession,
    *,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    rate_limit: float = 1000.0,
    clock: FakeClock | None = None,
) -> HorizonClient:
    fake_clock = clock or FakeClock()
    return HorizonClient(
        BASE_URL,
        timeout=10.0,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        rate_limit=rate_limit,
        session=session,
        sleep=fake_clock.sleep,
        clock=fake_clock,
    )


# --------------------------------------------------------------------- #
# parse_payment_record                                                   #
# --------------------------------------------------------------------- #


def test_parse_payment_record_native_asset() -> None:
    payment = parse_payment_record(_payment_record(), index=0)

    assert payment == HorizonPayment(
        id="123",
        paging_token="123",
        transaction_hash="deadbeef",
        created_at="2026-01-01T00:00:00Z",
        type="payment",
        source_account="GSENDER",
        from_account="GSENDER",
        to_account="GRECIPIENT",
        asset="XLM",
        amount=pytest.approx(12.5),
    )


def test_parse_payment_record_issued_asset() -> None:
    record = _payment_record(
        asset_type="credit_alphanum4",
        asset_code="USDC",
        asset_issuer="G" + "A" * 55,
    )
    payment = parse_payment_record(record, index=3)

    assert payment.asset == "USDC:" + "A" * 55
    assert payment.amount == pytest.approx(12.5)


def test_parse_payment_record_malformed_raises_with_index() -> None:
    record = _payment_record(amount="not-a-number")
    with pytest.raises(HorizonRecordError, match="record 7"):
        parse_payment_record(record, index=7)


def test_parse_payment_record_issued_asset_requires_issuer() -> None:
    record = _payment_record(asset_type="credit_alphanum4", asset_code="USDC")
    with pytest.raises(HorizonRecordError, match="asset_issuer"):
        parse_payment_record(record, index=0)


def test_parse_payment_record_missing_field_rejected() -> None:
    record = _payment_record()
    del record["transaction_hash"]
    with pytest.raises(HorizonRecordError, match="transaction_hash"):
        parse_payment_record(record, index=0)


# --------------------------------------------------------------------- #
# Rate limiting                                                          #
# --------------------------------------------------------------------- #


def test_rate_limit_enforces_minimum_interval() -> None:
    session = FakeSession([_page([]), _page([]), _page([])])
    clock = FakeClock()
    client = _client(session, rate_limit=2.0, clock=clock)  # 0.5s spacing

    list(client.iter_records("/accounts/GABC/payments"))
    list(client.iter_records("/accounts/GABC/payments"))
    list(client.iter_records("/accounts/GABC/payments"))

    assert clock.sleeps == [pytest.approx(0.5), pytest.approx(0.5)]


def test_first_request_is_not_delayed() -> None:
    session = FakeSession([_page([])])
    clock = FakeClock()
    client = _client(session, rate_limit=1.0, clock=clock)

    list(client.iter_records("/accounts/GABC/payments"))

    assert clock.sleeps == []


# --------------------------------------------------------------------- #
# Retries and error mapping                                              #
# --------------------------------------------------------------------- #


def test_transient_5xx_is_retried_then_succeeds() -> None:
    session = FakeSession([FakeResponse(502), FakeResponse(500), _page([_payment_record()])])
    clock = FakeClock()
    client = _client(session, max_retries=3, backoff_factor=0.5, clock=clock)

    page = client.get_page("/accounts/GABC/payments")

    assert len(page.records) == 1
    assert clock.sleeps == [pytest.approx(0.5), pytest.approx(1.0)]
    assert len(session.requests) == 3


def test_rate_limited_response_is_retried_and_honors_retry_after() -> None:
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}),
        _page([]),
    ])
    clock = FakeClock()
    client = _client(session, max_retries=2, clock=clock)

    list(client.iter_records("/accounts/GABC/payments"))

    assert clock.sleeps == [pytest.approx(7.0)]


def test_retry_after_is_capped_by_max_retry_after() -> None:
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "3600"}),
        _page([]),
    ])
    clock = FakeClock()
    client = HorizonClient(
        BASE_URL,
        max_retries=2,
        max_retry_after=30.0,
        session=session,
        sleep=clock.sleep,
        clock=clock,
    )

    list(client.iter_records("/accounts/GABC/payments"))

    assert clock.sleeps == [pytest.approx(30.0)]


def test_rate_limit_persists_past_retries_raises_specific_error() -> None:
    session = FakeSession([FakeResponse(429), FakeResponse(429), FakeResponse(429)])
    client = _client(session, max_retries=1, backoff_factor=0.25)

    with pytest.raises(HorizonRateLimitError):
        client.get_page("/accounts/GABC/payments")
    assert len(session.requests) == 2  # initial attempt + 1 retry


def test_server_error_persists_past_retries_raises_specific_error() -> None:
    session = FakeSession([FakeResponse(503), FakeResponse(503)])
    client = _client(session, max_retries=0, backoff_factor=0)

    with pytest.raises(HorizonServerError):
        client.get_page("/accounts/GABC/payments")


def test_client_error_4xx_is_not_retried() -> None:
    session = FakeSession([FakeResponse(400, {"title": "bad request"})])
    client = _client(session, max_retries=3)

    with pytest.raises(HorizonClientError) as excinfo:
        client.get_page("/accounts/GABC/payments")
    assert excinfo.value.status_code == 400
    assert len(session.requests) == 1


def test_transport_failure_is_retried_then_raises() -> None:
    session = FakeSession([
        HorizonTransportError("connection reset"),
        HorizonTransportError("connection reset"),
    ])
    client = _client(session, max_retries=1, backoff_factor=0.5)

    with pytest.raises(HorizonTransportError):
        client.get_page("/accounts/GABC/payments")
    assert len(session.requests) == 2


def test_transport_failure_recovers_on_retry() -> None:
    session = FakeSession([
        HorizonTransportError("connection reset"),
        _page([_payment_record()]),
    ])
    client = _client(session, max_retries=2, backoff_factor=0.5)

    page = client.get_page("/accounts/GABC/payments")
    assert len(page.records) == 1


def test_non_json_body_on_success_raises_envelope_error() -> None:
    session = FakeSession([
        FakeResponse(200, json_error=ValueError("no json")),
    ])
    client = _client(session, max_retries=2)

    with pytest.raises(HorizonEnvelopeError):
        client.get_page("/accounts/GABC/payments")
    assert len(session.requests) == 1  # not retried


# --------------------------------------------------------------------- #
# Pagination                                                             #
# --------------------------------------------------------------------- #


def test_pagination_follows_next_links() -> None:
    page2_url = f"{BASE_URL}/accounts/GABC/payments?cursor=abc"
    session = FakeSession([
        _page([_payment_record(id="1"), _payment_record(id="2")], next_url=page2_url),
        _page([_payment_record(id="3")]),
    ])
    client = _client(session)

    records = list(client.iter_records("/accounts/GABC/payments"))

    assert [record["id"] for record in records] == ["1", "2", "3"]
    assert session.requests[1][0] == page2_url


def test_pagination_stops_at_max_records() -> None:
    page2_url = f"{BASE_URL}/accounts/GABC/payments?cursor=abc"
    session = FakeSession([
        _page([_payment_record(id="1"), _payment_record(id="2")], next_url=page2_url),
    ])
    client = _client(session)

    records = list(client.iter_records("/accounts/GABC/payments", max_records=1))

    assert [record["id"] for record in records] == ["1"]
    assert len(session.requests) == 1  # second page never fetched


def test_iter_payments_parses_typed_records() -> None:
    session = FakeSession([
        _page([
            _payment_record(id="p1", asset_type="credit_alphanum4",
                            asset_code="USDC", asset_issuer="G" + "B" * 55,
                            amount="100"),
            _payment_record(id="p2", amount="7"),
        ]),
    ])
    client = _client(session)

    payments = client.get_payments("GABC")

    assert [p.id for p in payments] == ["p1", "p2"]
    assert payments[0].asset == "USDC:" + "B" * 55
    assert payments[0].amount == pytest.approx(100.0)
    assert payments[1].asset == "XLM"


def test_iter_payments_rejects_invalid_account() -> None:
    client = _client(FakeSession([]))

    with pytest.raises(HorizonError):
        list(client.iter_payments("   "))


def test_iter_payments_surfaces_malformed_record_with_index() -> None:
    session = FakeSession([
        _page([_payment_record(), _payment_record(amount="oops")]),
    ])
    client = _client(session)

    with pytest.raises(HorizonRecordError, match="record 1"):
        list(client.iter_payments("GABC"))


def test_envelope_missing_records_list_rejected() -> None:
    session = FakeSession([FakeResponse(200, {"_embedded": {}})])
    client = _client(session)

    with pytest.raises(HorizonEnvelopeError):
        client.get_page("/accounts/GABC/payments")


def test_envelope_malformed_record_entry_rejected() -> None:
    session = FakeSession([
        FakeResponse(200, {"_embedded": {"records": ["not-an-object"]}}),
    ])
    client = _client(session)

    with pytest.raises(HorizonRecordError):
        client.get_page("/accounts/GABC/payments")


# --------------------------------------------------------------------- #
# Options validation                                                     #
# --------------------------------------------------------------------- #


def test_client_rejects_invalid_configuration() -> None:
    session = FakeSession([])

    with pytest.raises(HorizonError):
        HorizonClient("not-a-url", session=session)
    with pytest.raises(HorizonError):
        HorizonClient("", session=session)
    with pytest.raises(HorizonError):
        HorizonClient(BASE_URL, timeout=0, session=session)
    with pytest.raises(HorizonError):
        HorizonClient(BASE_URL, max_retries=-1, session=session)
    with pytest.raises(HorizonError):
        HorizonClient(BASE_URL, backoff_factor=-0.5, session=session)
    with pytest.raises(HorizonError):
        HorizonClient(BASE_URL, rate_limit=0, session=session)
    with pytest.raises(HorizonError):
        HorizonClient(BASE_URL, max_retry_after=-1, session=session)


def test_iter_records_rejects_invalid_max_records() -> None:
    client = _client(FakeSession([]))

    with pytest.raises(HorizonError):
        list(client.iter_records("/x", max_records=0))
    with pytest.raises(HorizonError):
        list(client.iter_records("/x", max_records=1.5))  # type: ignore[arg-type]


def test_build_url_appends_query_params() -> None:
    session = FakeSession([_page([])])
    client = _client(session)

    client.get_page("/accounts/GABC/payments", params={"limit": 200, "order": "desc"})

    url = session.requests[0][0]
    assert url.startswith(f"{BASE_URL}/accounts/GABC/payments?")
    assert "limit=200" in url
    assert "order=desc" in url


def test_get_payments_uses_absolute_next_href() -> None:
    page2_url = f"{BASE_URL}/accounts/GABC/payments?cursor=xyz"
    session = FakeSession([
        _page([_payment_record(id="1")], next_url=page2_url),
        _page([_payment_record(id="2")]),
    ])
    client = _client(session)

    payments = client.get_payments("GABC")

    assert [p.id for p in payments] == ["1", "2"]
    assert session.requests[1][0] == page2_url


def test_client_sends_accept_json_header() -> None:
    session = FakeSession([_page([])])
    client = _client(session)

    client.get_page("/accounts/GABC/payments")

    assert session.requests[0][1]["Accept"] == "application/json"
