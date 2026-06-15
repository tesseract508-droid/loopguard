from __future__ import annotations

from typing import Final

import numpy as np


class StateVector:
    """
    Core 4D feature vector for LoopGuard engine:

    V = [S_hash, A_hash, T_entropy, t]
    """

    __slots__ = ("_S_hash", "_A_hash", "_T_entropy", "_t", "_buf")

    _DTYPE: Final[np.dtype] = np.dtype(np.float32)

    def __init__(self, s_hash: float, a_hash: float, t_entropy: float, t: int) -> None:
        self._S_hash = float(s_hash)
        self._A_hash = float(a_hash)
        self._T_entropy = float(t_entropy)
        self._t = int(t)
        # Minimal overhead: keep a single fixed-size float32 buffer.
        self._buf = np.array((self._S_hash, self._A_hash, self._T_entropy, float(self._t)), dtype=np.float32)

    def to_numpy(self) -> np.ndarray:
        """Return the 4D feature vector as np.ndarray with shape (4,) and dtype float32."""
        return self._buf
