from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener
import os
import queue
import threading
from typing import Final


_LOG_QUEUE_MAXSIZE: Final[int] = 16384
_LOG_FILE_NAME: Final[str] = ".loopguard.log"


_queue: queue.Queue[str] = queue.Queue(maxsize=_LOG_QUEUE_MAXSIZE)
_listener: QueueListener[str] | None = None
_listener_lock: Final[threading.Lock] = threading.Lock()


def _get_log_path() -> str:
    return os.path.join(os.getcwd(), _LOG_FILE_NAME)


def _ensure_listener() -> QueueListener[str]:
    global _listener
    if _listener is not None:
        return _listener

    with _listener_lock:
        if _listener is not None:
            return _listener

        file_handler = logging.FileHandler(_get_log_path(), mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        queue_handler = QueueHandler(_queue)
        queue_handler.setLevel(logging.INFO)

        # The listener consumes records from the queue and writes via the file handler.
        _listener = QueueListener(_queue, file_handler, respect_handler_level=True)
        _listener.start()
        return _listener


class TelemetryLogger:
    __slots__ = ("_queue_handler",)

    def __init__(self) -> None:
        self._queue_handler = QueueHandler(_queue)
        self._queue_handler.setLevel(logging.INFO)
        _ensure_listener()

    def log_turn(
        self,
        turn: int,
        s_hash: float,
        a_hash: float,
        t_entropy: float,
        viscosity: float | None = None,
        z_score: float | None = None,
    ) -> None:
        # Preformat message; hot-path work kept minimal.
        if viscosity is None or z_score is None:
            msg = (
                f"turn={turn} s_hash={s_hash} a_hash={a_hash} "
                f"t_entropy={t_entropy}"
            )
        else:
            msg = (
                f"turn={turn} s_hash={s_hash} a_hash={a_hash} "
                f"t_entropy={t_entropy} viscosity={viscosity} z_score={z_score}"
            )

        # Queue insertion is O(1) amortized; handler formatting is bypassed.
        record = logging.LogRecord(
            name="loopguard.telemetry",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        self._queue_handler.emit(record)


telemetry_logger: Final[TelemetryLogger] = TelemetryLogger()
