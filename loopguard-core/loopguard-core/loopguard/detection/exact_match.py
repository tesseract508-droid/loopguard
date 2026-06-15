from __future__ import annotations

import numpy as np

from loopguard.core.exceptions import InterceptionContext, LoopInterruptException


class ExactStateDetector:
    """
    Detects exact (S_hash, A_hash) repetition during the Cold-Start phase (t in [0..15]).

    Stores only:
      - S_hash (column 0)
      - A_hash (column 1)

    No Python-level loops are used for comparison.
    """

    __slots__ = ("_buf", "_row")

    def __init__(self) -> None:
        self._buf: np.ndarray = np.full((16, 2), np.nan, dtype=np.float32)
        self._row: int = 0

    def check_and_add(self, vector_array: np.ndarray) -> None:
        """
        Args:
            vector_array: shape (4,) NumPy array from StateVector.to_numpy().

        Raises:
            LoopInterruptException: when an exact match is found within populated rows.
        """
        # Extract S_hash and A_hash (first two elements)
        s_hash = float(vector_array[0])
        a_hash = float(vector_array[1])

        # Extract t for Cold-Start gating (vector_array[3] is stored as float32 in StateVector)
        t_int = int(vector_array[3])

        if self._row > 0:
            existing = self._buf[: self._row]  # shape (k, 2)
            target = np.array((s_hash, a_hash), dtype=np.float32)  # shape (2,)
            match_mask = np.all(np.isclose(existing, target, equal_nan=True), axis=1)  # shape (k,)
            if bool(np.any(match_mask)):
                origin_t = int(np.argmax(match_mask))  # first matching row index
                raise LoopInterruptException(
                    "Infinite loop detected: Exact state-action match within Cold-Start Phase.",
                    context=InterceptionContext(
                        loop_start_turn=origin_t,
                        current_turn=t_int,
                        viscosity_d=0.0,
                        z_score=0.0,
                    ),
                )

        if t_int <= 15:
            self._buf[self._row, 0] = np.float32(s_hash)
            self._buf[self._row, 1] = np.float32(a_hash)
            self._row += 1
