"""Tests for security audit logging module (funding_optimizer/audit.py).

Verifies Issue #81 requirements:
- Structured event creation for runs, exports, and reports
- Sensitive secret and PII redaction (Stellar keys, tokens, emails) without clobbering benign keys
- Oversized payload rejection
- Append-only sink recording and serialization
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from funding_optimizer.audit import (
    AuditEvent,
    AuditEventStatus,
    AuditEventType,
    AuditLogger,
    AuditPayloadTooLargeError,
    FileAuditSink,
    InMemoryAuditSink,
    redact_sensitive_string,
    sanitize_payload,
)


def test_audit_event_immutability() -> None:
    event = AuditEvent(
        event_type=AuditEventType.OPTIMIZATION_RUN,
        status=AuditEventStatus.SUCCESS,
        action="Run completed",
        actor="admin",
        metadata={"run_id": "test-123"},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        # AuditEvent is frozen
        event.action = "Modified"  # type: ignore


def test_redact_sensitive_string_stellar_secret() -> None:
    # Stellar Secret key (56 chars base32 starting with S)
    test_secret = "SBK4QY2Y75R7Z7MQLJFLXZX2E32OOD34LGBJ7G2KFLR3UEX7U33OXYX2"
    raw_text = f"User authenticated with key {test_secret} in session."
    redacted = redact_sensitive_string(raw_text)

    assert test_secret not in redacted
    assert "[REDACTED_STELLAR_SECRET]" in redacted


def test_redact_sensitive_string_email_and_token() -> None:
    email = "contributor@example.com"
    token = "Bearer ghp_Abc123XyZ98765432109876"
    raw = f"Contact {email} authorized via {token}"
    redacted = redact_sensitive_string(raw)

    assert email not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_TOKEN]" in redacted


def test_sanitize_payload_nested_dict_and_lists() -> None:
    payload = {
        "user_email": "admin@organization.org",
        "api_token": "secret_token_123456789",
        "author": "Alice Researcher",
        "idempotency_key": "sdp:alloc:987654321",
        "credentials": {
            "password": "SuperSecretPassword123!",
            "stellar_key": "SAAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLLMMMMNNN",
        },
        "tags": ["normal_tag", "contact: alert@domain.com"],
        "budget": 50000.0,
    }

    clean = sanitize_payload(payload)

    assert clean["budget"] == 50000.0
    assert clean["author"] == "Alice Researcher"
    assert clean["idempotency_key"] == "sdp:alloc:987654321"
    assert clean["user_email"] == "[REDACTED_EMAIL]"
    assert clean["api_token"] == "[REDACTED_SECRET]"
    assert clean["credentials"]["password"] == "[REDACTED_SECRET]"
    assert clean["credentials"]["stellar_key"] == "[REDACTED_SECRET]"
    assert clean["tags"][0] == "normal_tag"
    assert "[REDACTED_EMAIL]" in clean["tags"][1]


def test_audit_logger_in_memory_sink() -> None:
    sink = InMemoryAuditSink()
    logger = AuditLogger(sink)

    # 1. Audit optimization run
    logger.audit_optimization_run(
        run_id="run-999",
        num_candidates=45,
        total_budget=100000.0,
        actor="operator_1",
        extra={"notes": "Token: bearer_999888777666555"},
    )

    # 2. Audit export generation
    logger.audit_export_generation(
        export_id="exp-101",
        export_format="csv",
        actor="analyst@corp.com",
    )

    # 3. Audit report access
    logger.audit_report_access(
        report_id="rep-555",
        actor="viewer@external.org",
        status=AuditEventStatus.DENIED,
    )

    assert len(sink.events) == 3

    # Check first event
    ev1 = sink.events[0]
    assert ev1.event_type == AuditEventType.OPTIMIZATION_RUN
    assert ev1.status == AuditEventStatus.SUCCESS
    assert ev1.metadata["run_id"] == "run-999"
    assert "[REDACTED_TOKEN]" in ev1.metadata["notes"]

    # Check second event actor redacted
    ev2 = sink.events[1]
    assert ev2.event_type == AuditEventType.EXPORT_GENERATION
    assert ev2.actor == "[REDACTED_EMAIL]"

    # Check third event denied status
    ev3 = sink.events[2]
    assert ev3.event_type == AuditEventType.REPORT_ACCESS
    assert ev3.status == AuditEventStatus.DENIED


def test_file_audit_sink_append_only(tmp_path: Path) -> None:
    log_file = tmp_path / "test_audit.jsonl"
    sink = FileAuditSink(log_file)
    logger = AuditLogger(sink)

    logger.audit_optimization_run("run-001", 10, 5000.0)
    logger.audit_export_generation("exp-001", "json")

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    record_1 = json.loads(lines[0])
    assert record_1["event_type"] == "OPTIMIZATION_RUN"
    assert record_1["metadata"]["run_id"] == "run-001"

    record_2 = json.loads(lines[1])
    assert record_2["event_type"] == "EXPORT_GENERATION"
    assert record_2["metadata"]["format"] == "json"


def test_sanitize_payload_handles_non_string_keys() -> None:
    """Verifies that sanitize_payload safely handles integer and non-string dictionary keys."""
    payload = {
        101: "safe_value",
        "nested": {
            202: "another_safe",
            "api_key_override": "secret_token_12345",
        },
    }
    sanitized = sanitize_payload(payload)
    assert sanitized[101] == "safe_value"
    assert sanitized["nested"][202] == "another_safe"
    assert sanitized["nested"]["api_key_override"] == "[REDACTED_SECRET]"


def test_oversized_and_unsafe_audit_payload_rejected() -> None:
    """Verifies that oversized metadata payloads and empty actions are rejected."""
    sink = InMemoryAuditSink()
    logger = AuditLogger(sink, max_metadata_bytes=512)

    with pytest.raises(AuditPayloadTooLargeError, match="exceeds maximum limit"):
        logger.log(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="Oversized run",
            metadata={"huge_blob": "X" * 1024},
        )

    with pytest.raises(ValueError, match="must be non-empty"):
        logger.log(event_type=AuditEventType.OPTIMIZATION_RUN, action="   ")
