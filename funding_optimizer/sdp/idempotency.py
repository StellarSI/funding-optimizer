"""SDP (Stellar Disbursement Platform) payout payload export and idempotency module.

Produces valid, validated disbursement batch payloads ready for SDP handoff,
accompanied by deterministic idempotency keys and strict duplicate detection.

Boundary rule:
This module only prepares, hashes, and validates recommendation export payloads.
It NEVER signs transactions, manages private keys, or initiates live network payouts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

# Standard Stellar Public Key format: 'G' followed by 55 base32 uppercase characters
STELLAR_ACCOUNT_REGEX = re.compile(r"^G[A-Z2-7]{55}$")

# Maximum permitted decimal precision for standard SDP asset amounts
MAX_SDP_DECIMAL_PLACES = 7


class SDPExportValidationError(ValueError):
    """Raised when an allocation record violates SDP schema or precision requirements."""


class DuplicateDisbursementError(ValueError):
    """Raised when duplicate recipients or idempotency collisions are detected within a batch."""


@dataclass(frozen=True)
class DisbursementItem:
    """Individual recipient allocation item matching SDP bulk disbursement format."""

    recipient_id: str
    stellar_address: str
    amount: Decimal
    asset_code: str
    asset_issuer: str | None = None
    memo: str | None = None

    def __post_init__(self) -> None:
        if not self.recipient_id or not self.recipient_id.strip():
            raise SDPExportValidationError("recipient_id must be a non-empty string.")

        if not STELLAR_ACCOUNT_REGEX.match(self.stellar_address):
            raise SDPExportValidationError(
                f"Invalid Stellar public key format: '{self.stellar_address}'. "
                "Must be a 56-character string starting with 'G'."
            )

        if not isinstance(self.amount, Decimal):
            try:
                object.__setattr__(self, "amount", Decimal(str(self.amount)))
            except (InvalidOperation, TypeError) as exc:
                raise SDPExportValidationError(f"Invalid decimal amount: {self.amount}") from exc

        if not self.amount.is_finite():
            raise SDPExportValidationError(
                f"Disbursement amount must be finite. Received: {self.amount}"
            )

        if self.amount <= Decimal(0):
            raise SDPExportValidationError(
                f"Disbursement amount must be positive. Received: {self.amount}"
            )

        # Check fractional precision (Stellar Horizon/SDP max 7 decimal places)
        exponent = self.amount.normalize().as_tuple().exponent
        if isinstance(exponent, int) and -exponent > MAX_SDP_DECIMAL_PLACES:
            raise SDPExportValidationError(
                f"Amount {self.amount} exceeds maximum allowed SDP precision of 7 decimal places."
            )

        if not self.asset_code or not self.asset_code.strip():
            raise SDPExportValidationError("asset_code must be specified (e.g. 'XLM' or 'USDC').")

        # Native asset (XLM) must not specify an issuer
        if self.asset_code.upper() == "XLM" and self.asset_issuer:
            raise SDPExportValidationError("Native asset 'XLM' cannot have an asset_issuer.")

        # Non-native asset must specify an issuer
        if self.asset_code.upper() != "XLM" and not self.asset_issuer:
            raise SDPExportValidationError(
                f"Non-native asset '{self.asset_code}' requires an asset_issuer."
            )

        if self.asset_issuer and not STELLAR_ACCOUNT_REGEX.match(self.asset_issuer):
            raise SDPExportValidationError(f"Invalid asset_issuer format: '{self.asset_issuer}'")

    def to_canonical_dict(self) -> dict[str, Any]:
        """Returns a canonical dictionary representation for stable hashing and JSON serialization."""
        d: dict[str, Any] = {
            "amount": f"{self.amount:.7f}",
            "asset_code": self.asset_code.upper(),
            "recipient_id": self.recipient_id.strip(),
            "stellar_address": self.stellar_address,
        }
        if self.asset_issuer:
            d["asset_issuer"] = self.asset_issuer
        if self.memo:
            d["memo"] = self.memo.strip()
        return d


@dataclass(frozen=True)
class SDPBatchPayload:
    """Validated export payload ready for SDP ingestion with cryptographic idempotency key."""

    batch_id: str
    idempotency_key: str
    created_at_iso: str
    items: Sequence[DisbursementItem]
    total_amount: Decimal
    item_count: int

    def to_json(self, indent: int | None = None) -> str:
        """Serializes the batch to deterministic JSON string."""
        data = {
            "batch_id": self.batch_id,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at_iso,
            "item_count": self.item_count,
            "total_amount": f"{self.total_amount:.7f}",
            "disbursements": [item.to_canonical_dict() for item in self.items],
        }
        return json.dumps(data, indent=indent, sort_keys=True)


class SDPIdempotencyEngine:
    """Manages deterministic batch key generation and tracks processed batches to prevent duplicates."""

    def __init__(self, key_prefix: str = "sdp:alloc") -> None:
        self.key_prefix = key_prefix
        self._seen_idempotency_keys: set[str] = set()

    def generate_idempotency_key(self, batch_id: str, items: Sequence[DisbursementItem]) -> str:
        """Calculates a deterministic SHA-256 idempotency key based on batch items and attributes.

        Two batches with identical recipients, amounts, and asset parameters will yield
        the exact same key, enabling upstream SDP de-duplication.
        """
        canonical_items = [item.to_canonical_dict() for item in items]
        # Sort canonically by recipient address and ID to guarantee order independence
        sorted_items = sorted(
            canonical_items, key=lambda x: (x["stellar_address"], x["recipient_id"])
        )

        hasher = hashlib.sha256()
        hasher.update(batch_id.encode("utf-8"))
        hasher.update(json.dumps(sorted_items, sort_keys=True).encode("utf-8"))
        digest = hasher.hexdigest()
        return f"{self.key_prefix}:{digest[:32]}"

    def build_and_validate_batch(
        self,
        batch_id: str,
        items: Sequence[DisbursementItem],
        *,
        allow_duplicate_addresses: bool = False,
    ) -> SDPBatchPayload:
        """Validates all allocation items, checks for duplicate recipients, and generates the batch payload.

        Parameters
        ----------
        batch_id:
            Unique identifier or allocation recommendation cycle tag.
        items:
            Sequence of DisbursementItem records.
        allow_duplicate_addresses:
            If False, prevents multiple disbursements to the same Stellar address within a single batch.

        Raises
        ------
        SDPExportValidationError:
            If the batch is empty or items fail validation.
        DuplicateDisbursementError:
            If duplicate recipients are detected or if the exact idempotency key has already been issued.
        """
        if not batch_id or not batch_id.strip():
            raise SDPExportValidationError("batch_id cannot be empty.")

        if not items:
            raise SDPExportValidationError(
                "Cannot create an SDP batch with zero disbursement items."
            )

        # Ensure all items in a single SDP batch share the same asset (code + issuer)
        first_asset = (items[0].asset_code.upper(), items[0].asset_issuer)
        for item in items[1:]:
            if (item.asset_code.upper(), item.asset_issuer) != first_asset:
                raise SDPExportValidationError(
                    "All disbursement items in a single SDP batch must share the same asset_code and asset_issuer."
                )

        # Duplicate address check within the batch
        if not allow_duplicate_addresses:
            seen_addrs: set[str] = set()
            for item in items:
                if item.stellar_address in seen_addrs:
                    raise DuplicateDisbursementError(
                        f"Duplicate recipient address '{item.stellar_address}' detected in batch '{batch_id}'."
                    )
                seen_addrs.add(item.stellar_address)

        idempotency_key = self.generate_idempotency_key(batch_id, items)

        # Check against previously processed batches in this session/engine
        if idempotency_key in self._seen_idempotency_keys:
            raise DuplicateDisbursementError(
                f"Duplicate batch detected! Idempotency key '{idempotency_key}' has already been processed."
            )

        total_amount = sum((item.amount for item in items), Decimal(0))
        now_iso = datetime.now(UTC).isoformat()

        self._seen_idempotency_keys.add(idempotency_key)

        return SDPBatchPayload(
            batch_id=batch_id.strip(),
            idempotency_key=idempotency_key,
            created_at_iso=now_iso,
            items=tuple(items),
            total_amount=total_amount,
            item_count=len(items),
        )

    def is_processed(self, idempotency_key: str) -> bool:
        """Checks if a given idempotency key has already been issued by this engine."""
        return idempotency_key in self._seen_idempotency_keys

    def reset_registry(self) -> None:
        """Clears registered idempotency keys (useful for test isolation)."""
        self._seen_idempotency_keys.clear()
