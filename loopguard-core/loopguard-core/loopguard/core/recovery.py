from __future__ import annotations

from loopguard.core.exceptions import InterceptionContext


class StatePruner:
    @staticmethod
    def calculate_safe_slice(history_length: int, context: InterceptionContext) -> slice:
        if context.loop_start_turn < history_length:
            safe_turn_index = max(0, int(context.loop_start_turn))
        else:
            safe_turn_index = max(0, history_length - 2)

        return slice(0, safe_turn_index)

    @staticmethod
    def generate_correction_directive(context: InterceptionContext) -> str:
        return (
            "System Intercept: Degenerate loop detected at turn "
            f"{context.loop_start_turn}. Viscosity dropped to {context.viscosity_d:.2f}. "
            "Previous strategy failed. Discard prior sequential tool calls and formulate "
            "an entirely new approach."
        )
