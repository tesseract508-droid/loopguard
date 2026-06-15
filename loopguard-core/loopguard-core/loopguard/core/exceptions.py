from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InterceptionContext:
    loop_start_turn: int
    current_turn: int
    viscosity_d: float
    z_score: float


class LoopInterruptException(Exception):
    """Raised to interrupt execution when a LoopGuard termination condition is met."""

    __slots__ = ("_context",)

    def __init__(self, message: str, context: InterceptionContext) -> None:
        super().__init__(message)
        self._context = context

    @property
    def context(self) -> InterceptionContext:
        return self._context
