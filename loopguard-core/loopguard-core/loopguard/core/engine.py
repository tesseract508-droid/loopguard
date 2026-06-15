from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from loopguard.core.audit import CryptographicAuditTrail
from loopguard.core.config import LoopGuardConfig
from loopguard.core.telemetry import telemetry_logger
from loopguard.core.vector import StateVector
from loopguard.core.exceptions import InterceptionContext
from loopguard.detection.anomaly import AnomalyDetector
from loopguard.detection.exact_match import ExactStateDetector


@dataclass(slots=True)
class AgentSession:
    turn: int = 0
    exact_detector: ExactStateDetector = field(default_factory=ExactStateDetector)
    anomaly_detector: AnomalyDetector = field(default_factory=AnomalyDetector)


class LoopGuardEngine:
    __slots__ = ("_sessions", "config", "_audit")

    def __init__(self, config: LoopGuardConfig | None = None) -> None:
        self.config = config if config is not None else LoopGuardConfig()
        self._sessions: Dict[str, AgentSession] = {}
        self._audit = CryptographicAuditTrail()

    def terminate_session(self, agent_id: str) -> None:
        self._sessions.pop(agent_id, None)

    def _get_or_create_session(self, agent_id: str) -> AgentSession:
        session = self._sessions.get(agent_id)
        if session is not None:
            return session
        if len(self._sessions) >= self.config.max_tracked_sessions:
            raise RuntimeError("LoopGuardEngine session limit reached; refusing to create new session.")
        session = AgentSession()
        self._sessions[agent_id] = session
        return session

    def step(
        self,
        s_hash: float,
        a_hash: float,
        t_entropy: float,
        agent_id: str = "default",
    ) -> None:
        session = self._get_or_create_session(agent_id)

        state_vector: StateVector = StateVector(
            s_hash=s_hash,
            a_hash=a_hash,
            t_entropy=t_entropy,
            t=session.turn,
        )
        vector_array: np.ndarray = state_vector.to_numpy()

        if session.turn <= self.config.cold_start_turns:
            session.exact_detector.check_and_add(vector_array)
        else:
            try:
                session.anomaly_detector.update_and_check(vector_array)
            except Exception as exc:
                # Log right before propagating LoopInterruptException to preserve chain semantics.
                if hasattr(exc, "context"):
                    context_obj = getattr(exc, "context")
                    if isinstance(context_obj, InterceptionContext):
                        directive = f"interception:severity={context_obj.z_score}|loop_start_turn={context_obj.loop_start_turn}|viscosity_d={context_obj.viscosity_d}"
                        self._audit.log_interception(
                            agent_id=agent_id,
                            context={
                                "loop_start_turn": context_obj.loop_start_turn,
                                "current_turn": context_obj.current_turn,
                                "viscosity_d": context_obj.viscosity_d,
                                "z_score": context_obj.z_score,
                            },
                            directive=directive,
                        )
                raise

        session.turn += 1
        telemetry_logger.log_turn(
            turn=session.turn,
            s_hash=s_hash,
            a_hash=a_hash,
            t_entropy=t_entropy,
            agent_id=agent_id,
        )
