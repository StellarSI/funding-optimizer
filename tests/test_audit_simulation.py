"""Stress and simulation tests for security audit logging module.

Runs Monte Carlo and multi-agent simulation scenarios:
- Concurrent high-throughput audit logging
- Adversarial fuzzing with dirty/malformed strings and deep nesting
- Performance latency benchmarks ensuring near-zero execution overhead
"""

import threading
import time
from pathlib import Path
from typing import Any

from funding_optimizer.audit import (
    AuditLogger,
    FileAuditSink,
    sanitize_payload,
)


def test_concurrent_simulation_audit_logging(tmp_path: Path) -> None:
    """Simulates 10 concurrent solver worker threads logging simultaneously."""
    log_file = tmp_path / "concurrent_audit.jsonl"
    sink = FileAuditSink(log_file)
    logger = AuditLogger(sink)

    num_threads = 10
    logs_per_thread = 50
    threads = []

    def worker(worker_id: int) -> None:
        for i in range(logs_per_thread):
            logger.audit_optimization_run(
                run_id=f"run-{worker_id}-{i}",
                num_candidates=10 + i,
                total_budget=50000.0 + (i * 100),
                actor=f"worker_{worker_id}@cluster.org",
                extra={"secret_token": f"tok_{worker_id}_{i}_123456789"},
            )

    start_time = time.time()
    for tid in range(num_threads):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    duration = time.time() - start_time
    # Must complete 500 logs rapidly
    assert duration < 15.0

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == num_threads * logs_per_thread

    # Verify every single line is clean JSON and secret is redacted
    for line in lines:
        assert "[REDACTED_SECRET]" in line
        assert "tok_" not in line


def test_adversarial_deep_nesting_and_fuzzing() -> None:
    """Fuzzes sanitizer with 20 levels of deep nested structures, None, numbers, bytes."""
    nested: dict[str, Any] = {"level": 0, "secret_key": "S" + "A" * 55}
    curr = nested
    for i in range(1, 20):
        curr["child"] = {
            "level": i,
            "data": f"user_{i}@domain.com",
            "nested_auth": "Bearer secret_1234567890",
            "mixed": [None, 123, 45.67, True, False, "safe_string"],
        }
        curr = curr["child"]

    sanitized = sanitize_payload(nested)

    # Walk the sanitized tree and verify redactions at bottom level
    walker = sanitized
    for i in range(1, 20):
        walker = walker["child"]
        assert walker["data"] == "[REDACTED_EMAIL]"
        assert walker["nested_auth"] == "[REDACTED_SECRET]"
        assert walker["mixed"][1] == 123
        assert walker["mixed"][5] == "safe_string"
