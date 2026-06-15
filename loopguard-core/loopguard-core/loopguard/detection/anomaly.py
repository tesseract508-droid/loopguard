from __future__ import annotations

import numpy as np

from loopguard.core.exceptions import InterceptionContext, LoopInterruptException


class AnomalyDetector:
    """Streaming anomaly detector for Turns 16+ using Welford's Online Algorithm."""

    __slots__ = ("_n", "_mean", "_M2", "_prev_vector")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0
        self._prev_vector: np.ndarray = np.full((4,), np.nan, dtype=np.float32)

    def update_and_check(self, vector_array: np.ndarray) -> tuple[float, float, int]:
        """
        Update online viscosity statistics and raise on severe structural viscosity anomaly.

        Args:
            vector_array: shape (4,) NumPy array from StateVector.to_numpy().
        """
        s_t = float(vector_array[0])
        a_t = float(vector_array[1])
        t_int = int(vector_array[3])

        # On first step (t=0 in typical usage), initialize prev vector and streaming stats.
        if self._n == 0:
            self._prev_vector[...] = vector_array
            self._n = 1
            self._mean = 0.0
            self._M2 = 0.0
            return 0.0, 0.0, 0

        s_prev = float(self._prev_vector[0])
        a_prev = float(self._prev_vector[1])

        # If previous is NaN, treat as cold-start re-initialization.
        if np.isnan(s_prev) or np.isnan(a_prev):
            self._prev_vector[...] = vector_array
            self._n = 1
            self._mean = 0.0
            self._M2 = 0.0
            return 0.0, 0.0, 0

        # Viscosity D = sqrt((S_t - S_{t-1})^2 + (A_t - A_{t-1})^2)
        d = np.sqrt(
            (s_t - s_prev) * (s_t - s_prev) + (a_t - a_prev) * (a_t - a_prev)
        )
        D = float(d)

        # Welford update on D
        n_new = self._n + 1
        self._n = n_new

        delta = D - self._mean
        self._mean = self._mean + (delta / n_new)
        self._M2 = self._M2 + delta * (D - self._mean)

        loop_start_turn = 0
        # Anomaly threshold only applies for t >= 16
        if t_int >= 16:
            # var = M2 / n ; sigma = sqrt(var)
            var = self._M2 / self._n if self._n > 0 else 0.0

            # Zero-variance guard: if sigma==0 => Z=0
            if var <= 0.0:
                z = 0.0
            else:
                sigma = float(np.sqrt(var))
                z = (D - self._mean) / sigma

            if z < -3.0:
                # Estimate loop start as variance began dropping.
                loop_start_turn = t_int - 2
                raise LoopInterruptException(
                    "Infinite loop detected: Severe structural viscosity (Z < -3.0).",
                    context=InterceptionContext(
                        loop_start_turn=loop_start_turn,
                        current_turn=t_int,
                        viscosity_d=D,
                        z_score=z,
                    ),
                )
        else:
            z = 0.0
            loop_start_turn = 0

        # Persist current vector for next-step viscosity computation
        self._prev_vector[...] = vector_array
        return D, z, loop_start_turn
