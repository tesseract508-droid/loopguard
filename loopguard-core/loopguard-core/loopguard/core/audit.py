from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


class CryptographicAuditTrail:
    """
    Append-only cryptographic audit trail stored as JSON Lines.

    Each line is chained with SHA-256 of the immediately preceding log line.
    If any historical line is modified, the chain breaks.
    """

    __slots__ = ("_path", "_prev_hash", "_last_line", "_buffer")

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            # Keep file co-located with the package runtime, but stable across working dirs.
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, ".loopguard_audit.jsonl")

        self._path: str = path
        self._prev_hash: str = "0" * 64
        self._last_line: Optional[str] = None
        self._buffer: str = ""

        self._initialize_from_existing()

    def _initialize_from_existing(self) -> None:
        try:
            if not os.path.exists(self._path):
                return

            last_line: Optional[str] = None
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    # Keep raw line text for hashing parity.
                    stripped = line.rstrip("\n")
                    if stripped:
                        last_line = stripped

            if last_line is None:
                return

            self._last_line = last_line
            self._prev_hash = hashlib.sha256(last_line.encode("utf-8")).hexdigest()
        except OSError:
            # Fail-safe: if the log cannot be read, keep chain anchored.
            self._prev_hash = "0" * 64
            self._last_line = None

    def log_interception(self, agent_id: str, context: Dict[str, Any], directive: str) -> None:
        """
        Serialize and append a chained hash event.

        Event fields:
          - ts: event time (unix seconds, float)
          - agent_id: the agent that intercepted
          - context: serialized context payload
          - directive: deterministic directive string
          - previous_hash: sha256 hash of immediately preceding log line
          - event_hash: sha256(previous_hash + serialized payload for integrity)
        """
        event: Dict[str, Any] = {
            "ts": time.time(),
            "agent_id": agent_id,
            "context": context,
            "directive": directive,
            "previous_hash": self._prev_hash,
        }

        # Deterministic serialization for hashing: JSON canonical-ish via sorted keys.
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        event_hash = hashlib.sha256((self._prev_hash + payload).encode("utf-8")).hexdigest()
        event["event_hash"] = event_hash

        line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        # Append-only write. Use an in-memory buffer to reduce syscall frequency.
        # Buffer is flushed on size threshold or at process exit (best-effort).
        if self._buffer:
            self._buffer += "\n"
        self._buffer += line

        self._prev_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        self._last_line = line

        # Flush threshold tuned for latency: small, dependency-free.
        if len(self._buffer) >= 64 * 1024:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        # Ensure append-only by using 'a' mode.
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(self._buffer)
            f.write("\n")
        self._buffer = ""

    def close(self) -> None:
        self._flush()

    def __del__(self) -> None:
        # Best-effort flush; do not raise from destructor.
        try:
            self._flush()
        except Exception:
            pass
