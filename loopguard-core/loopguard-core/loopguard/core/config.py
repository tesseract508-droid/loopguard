from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LoopGuardConfig:
    welford_threshold: float = -3.0
    cold_start_turns: int = 15
    max_tracked_sessions: int = 10000
