"""Unit tests for SDP payload export, idempotency key generation, and duplicate detection."""

import json
from decimal import Decimal

import pytest

from funding_optimizer.sdp import (
    DisbursementItem,
    DuplicateDisbursementError,
    SDPExportValidationError,
    SDPIdempotencyEngine,
)

VALID_ADDR_1 = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
VALID_ADDR_2 = "GC3C4ACWQLCFFNZMDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLB2"
VALID_ISSUER = "GDQERFIB4LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLM7"


def test_disbursement_item_valid_native_asset() -> None:
    item = DisbursementItem(
        recipient_id="rec-001",
        stellar_address=VALID_ADDR_1,
        amount=Decimal("150.5000000"),
        asset_code="XLM",
        memo="grant-q4",
    )
    assert item.recipient_id == "rec-001"
    assert item.amount == Decimal("150.5000000")
    canon = item.to_canonical_dict()
    assert canon["amount"] == "150.5000000"
    assert canon["asset_code"] == "XLM"
    assert canon["memo"] == "grant-q4"
    assert "asset_issuer" not in canon


def test_disbursement_item_valid_credit_asset() -> None:
    item = DisbursementItem(
        recipient_id="rec-002",
        stellar_address=VALID_ADDR_2,
        amount=Decimal("2500.00"),
        asset_code="USDC",
        asset_issuer=VALID_ISSUER,
    )
    canon = item.to_canonical_dict()
    assert canon["asset_code"] == "USDC"
    assert canon["asset_issuer"] == VALID_ISSUER
    assert canon["amount"] == "2500.0000000"


def test_disbursement_item_invalid_stellar_address() -> None:
    with pytest.raises(SDPExportValidationError, match="Invalid Stellar public key format"):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address="INVALID_ADDRESS_123",
            amount=Decimal("10.0"),
            asset_code="XLM",
        )


def test_disbursement_item_non_positive_amount() -> None:
    with pytest.raises(SDPExportValidationError, match="must be positive"):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address=VALID_ADDR_1,
            amount=Decimal("-5.00"),
            asset_code="XLM",
        )

    with pytest.raises(SDPExportValidationError, match="must be positive"):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address=VALID_ADDR_1,
            amount=Decimal("0.00"),
            asset_code="XLM",
        )


def test_disbursement_item_excessive_precision() -> None:
    with pytest.raises(SDPExportValidationError, match="maximum allowed SDP precision"):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address=VALID_ADDR_1,
            amount=Decimal("10.123456789"),  # 9 decimal places
            asset_code="XLM",
        )


def test_disbursement_item_issuer_rules() -> None:
    # Native asset with issuer must fail
    with pytest.raises(
        SDPExportValidationError, match="Native asset 'XLM' cannot have an asset_issuer"
    ):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address=VALID_ADDR_1,
            amount=Decimal(100),
            asset_code="XLM",
            asset_issuer=VALID_ISSUER,
        )

    # Non-native asset without issuer must fail
    with pytest.raises(SDPExportValidationError, match="requires an asset_issuer"):
        DisbursementItem(
            recipient_id="rec-bad",
            stellar_address=VALID_ADDR_1,
            amount=Decimal(100),
            asset_code="EURC",
        )


def test_idempotency_engine_deterministic_key() -> None:
    engine = SDPIdempotencyEngine()
    item1 = DisbursementItem("r1", VALID_ADDR_1, Decimal(100), "XLM")
    item2 = DisbursementItem("r2", VALID_ADDR_2, Decimal(200), "XLM")

    key1 = engine.generate_idempotency_key("batch-2026-q4", [item1, item2])
    # Order independence test: swap items
    key2 = engine.generate_idempotency_key("batch-2026-q4", [item2, item1])

    assert key1 == key2
    assert key1.startswith("sdp:alloc:")


def test_idempotency_engine_different_batches_different_keys() -> None:
    engine = SDPIdempotencyEngine()
    item1 = DisbursementItem("r1", VALID_ADDR_1, Decimal(100), "XLM")

    key1 = engine.generate_idempotency_key("batch-A", [item1])
    key2 = engine.generate_idempotency_key("batch-B", [item1])
    assert key1 != key2


def test_build_and_validate_batch_happy_path() -> None:
    engine = SDPIdempotencyEngine()
    items = [
        DisbursementItem("r1", VALID_ADDR_1, Decimal("500.25"), "XLM"),
        DisbursementItem("r2", VALID_ADDR_2, Decimal("250.75"), "XLM"),
    ]

    batch = engine.build_and_validate_batch("cohort-alpha", items)
    assert batch.batch_id == "cohort-alpha"
    assert batch.item_count == 2
    assert batch.total_amount == Decimal("751.00")
    assert engine.is_processed(batch.idempotency_key)

    # Test serialization to valid JSON
    json_str = batch.to_json()
    parsed = json.loads(json_str)
    assert parsed["batch_id"] == "cohort-alpha"
    assert parsed["item_count"] == 2
    assert parsed["total_amount"] == "751.0000000"
    assert len(parsed["disbursements"]) == 2


def test_duplicate_recipient_address_in_single_batch() -> None:
    engine = SDPIdempotencyEngine()
    items = [
        DisbursementItem("r1", VALID_ADDR_1, Decimal(100), "XLM"),
        DisbursementItem("r2", VALID_ADDR_1, Decimal(200), "XLM"),  # Duplicate address
    ]

    with pytest.raises(DuplicateDisbursementError, match="Duplicate recipient address"):
        engine.build_and_validate_batch("batch-dup", items)


def test_duplicate_batch_rejection() -> None:
    engine = SDPIdempotencyEngine()
    items = [DisbursementItem("r1", VALID_ADDR_1, Decimal(100), "XLM")]

    # First attempt succeeds
    batch = engine.build_and_validate_batch("batch-repeat", items)
    assert batch is not None

    # Immediate second attempt with same ID and items must raise DuplicateDisbursementError
    with pytest.raises(DuplicateDisbursementError, match="Duplicate batch detected"):
        engine.build_and_validate_batch("batch-repeat", items)


def test_empty_batch_rejection() -> None:
    engine = SDPIdempotencyEngine()
    with pytest.raises(SDPExportValidationError, match="zero disbursement items"):
        engine.build_and_validate_batch("empty-batch", [])


def test_disbursement_item_non_finite_and_scientific_notation() -> None:
    for bad in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(SDPExportValidationError, match="must be finite"):
            DisbursementItem("rec-nan", VALID_ADDR_1, bad, "XLM")

    # Large whole integer in scientific notation (exponent +8) is valid (0 decimal places)
    large_item = DisbursementItem("rec-big", VALID_ADDR_1, Decimal("1E+8"), "XLM")
    assert large_item.to_canonical_dict()["amount"] == "100000000.0000000"


def test_mixed_asset_batch_rejected() -> None:
    engine = SDPIdempotencyEngine()
    items = [
        DisbursementItem("r1", VALID_ADDR_1, Decimal(100), "XLM"),
        DisbursementItem("r2", VALID_ADDR_2, Decimal(100), "USDC", asset_issuer=VALID_ISSUER),
    ]
    with pytest.raises(SDPExportValidationError, match="same asset_code and asset_issuer"):
        engine.build_and_validate_batch("mixed-batch", items)
