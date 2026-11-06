"""Security audit logging module for funding-optimizer.

Addresses Issue #81:
Audit-log optimization runs, exports, and report access with PII & secret redaction
and append-only sink recording.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Sensitive patterns to redact
# 1. Stellar Secret Keys (starts with S, 56 base32 chars)
STELLAR_SECRET_PATTERN = re.compile(r"\bS[A-Z2-7]{55}\b")
# 2. Bearer / API Tokens
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\b(bearer\s+|bearer_|token\s+|token_|key\s*[:=_\-]\s*)([A-Za-z0-9_\-\.]{10,})\b")
# 3. Email Addresses
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
# 4. Standard private keys / secrets
GENERIC_SECRET_PATTERN = re.compile(r"(?i)(secret|password|private_key|api_key|auth_token)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?")


class AuditEventType(str, Enum):
    """Categorized security audit event types."""
    OPTIMIZATION_RUN = "OPTIMIZATION_RUN"
    EXPORT_GENERATION = "EXPORT_GENERATION"
    REPORT_ACCESS = "REPORT_ACCESS"
    AUTH_CHECK = "AUTH_CHECK"
    CONFIG_CHANGE = "CONFIG_CHANGE"


class AuditEventStatus(str, Enum):
    """Execution status for the audited event."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"


@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event payload."""
    event_type: AuditEventType
    status: AuditEventStatus
    action: str
    actor: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_json(self) -> str:
        """Serializes the audit record to JSON string."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["status"] = self.status.value
        return json.dumps(d, ensure_ascii=False)


def redact_sensitive_string(text: str) -> str:
    """Redacts Stellar secret keys, bearer tokens, emails, and credentials."""
    if not isinstance(text, str):
        return text

    # Redact Stellar private keys
    redacted = STELLAR_SECRET_PATTERN.sub("[REDACTED_STELLAR_SECRET]", text)
    # Redact Bearer / API tokens
    redacted = BEARER_TOKEN_PATTERN.sub(r"\1[REDACTED_TOKEN]", redacted)
    # Redact Emails
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    # Redact generic secrets
    redacted = GENERIC_SECRET_PATTERN.sub(r"\1: [REDACTED_SECRET]", redacted)

    return redacted


def sanitize_payload(obj: Any) -> Any:
    """Recursively traverses dictionaries, lists, and strings to sanitize PII and secrets."""
    if isinstance(obj, str):
        return redact_sensitive_string(obj)
    elif isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for k, v in obj.items():
            # Check key name for sensitive keywords
            if any(s in k.lower() for s in ("secret", "password", "private", "token", "auth", "key")):
                sanitized[k] = "[REDACTED_SECRET]"
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    elif isinstance(obj, list):
        return [sanitize_payload(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_payload(elem) for elem in obj)
    return obj


class AuditSink:
    """Base abstract sink for append-only audit persistence."""
    def record(self, event: AuditEvent) -> None:
        raise NotImplementedError


class FileAuditSink(AuditSink):
    """Append-only thread-safe file sink that writes line-delimited JSON (JSONL) audit logs."""
    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        import threading
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock, open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")


class InMemoryAuditSink(AuditSink):
    """In-memory sink useful for testing and transient environments."""
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()


class AuditLogger:
    """Central audit manager enforcing data redaction and sink delegation."""
    def __init__(self, sink: AuditSink | None = None) -> None:
        self.sink = sink or InMemoryAuditSink()

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Sanitizes metadata, constructs the immutable event, and commits to the sink."""
        clean_metadata = sanitize_payload(metadata or {})
        clean_actor = redact_sensitive_string(actor)

        event = AuditEvent(
            event_type=event_type,
            status=status,
            action=action,
            actor=clean_actor,
            metadata=clean_metadata,
        )
        self.sink.record(event)
        return event

    def audit_optimization_run(
        self,
        run_id: str,
        num_candidates: int,
        total_budget: float,
        actor: str = "system",
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        extra: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Audit log helper for optimization runs."""
        meta = {
            "run_id": run_id,
            "num_candidates": num_candidates,
            "total_budget": total_budget,
            **(extra or {}),
        }
        return self.log(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action=f"Optimization run executed for ID {run_id}",
            status=status,
            actor=actor,
            metadata=meta,
        )

    def audit_export_generation(
        self,
        export_id: str,
        export_format: str,
        actor: str = "system",
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        extra: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Audit log helper for bulk export generation (CSV, JSON, SDP payload)."""
        meta = {
            "export_id": export_id,
            "format": export_format,
            **(extra or {}),
        }
        return self.log(
            event_type=AuditEventType.EXPORT_GENERATION,
            action=f"Export generated format {export_format}",
            status=status,
            actor=actor,
            metadata=meta,
        )

    def audit_report_access(
        self,
        report_id: str,
        actor: str,
        status: AuditEventStatus = AuditEventStatus.SUCCESS,
        extra: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Audit log helper for accessing reports and dashboard metrics."""
        meta = {
            "report_id": report_id,
            **(extra or {}),
        }
        return self.log(
            event_type=AuditEventType.REPORT_ACCESS,
            action=f"Report accessed: {report_id}",
            status=status,
            actor=actor,
            metadata=meta,
        )


# Global default logger instance for convenient integration
_default_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Returns or creates the global default audit logger."""
    global _default_audit_logger
    if _default_audit_logger is None:
        log_dir = os.environ.get("OPTIMIZER_AUDIT_LOG_DIR", "./logs")
        log_file = Path(log_dir) / "audit.jsonl"
        _default_audit_logger = AuditLogger(FileAuditSink(log_file))
    return _default_audit_logger
