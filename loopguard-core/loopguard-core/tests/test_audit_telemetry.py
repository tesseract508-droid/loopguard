import os
import json
import time
import pytest
from loopguard.core.audit import CryptographicAuditTrail
from loopguard.core.telemetry import TelemetryLogger

def test_audit_trail_chaining():
    audit_path = ".test_audit.jsonl"
    if os.path.exists(audit_path):
        os.remove(audit_path)

    audit = CryptographicAuditTrail(path=audit_path)
    audit.log_interception("agent1", {"turn": 5}, "interception")
    audit.log_interception("agent1", {"turn": 10}, "interception")
    audit.close()

    with open(audit_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        event0_str = lines[0].strip()
        event1 = json.loads(lines[1])

        import hashlib
        expected_prev_hash = hashlib.sha256(event0_str.encode("utf-8")).hexdigest()
        assert event1["previous_hash"] == expected_prev_hash

    os.remove(audit_path)

def test_telemetry_async_behavior():
    # Since it uses a Queue, it should be non-blocking
    logger = TelemetryLogger()
    start = time.perf_counter()
    for i in range(1000):
        logger.log_turn(i, 1.0, 1.0, 0.5)
    end = time.perf_counter()

    # 1000 logs should be very fast
    duration = end - start
    assert duration < 0.1 # Should be very quick
