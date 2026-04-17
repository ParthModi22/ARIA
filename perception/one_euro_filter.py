"""One Euro Filter implementation.

Reference: http://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math


class _LowPassFilter:
    """Simple exponential smoothing filter."""

    def __init__(self) -> None:
        self._initialized = False
        self._value = 0.0

    def __call__(self, alpha: float, value: float) -> float:
        if not self._initialized:
            self._initialized = True
            self._value = value
            return value

        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value


class OneEuroFilter:
    """One Euro Filter for time-series smoothing with adaptive cutoff."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be > 0")
        if d_cutoff <= 0.0:
            raise ValueError("d_cutoff must be > 0")

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._last_time: float | None = None
        self._last_raw_value: float | None = None
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        """Filter value ``x`` observed at time ``t``."""
        if self._last_time is None:
            self._last_time = float(t)
            self._last_raw_value = float(x)
            return self._x_filter(1.0, float(x))

        dt = float(t) - self._last_time
        if dt <= 0.0:
            return self._x_filter(1.0, float(x))

        if self._last_raw_value is None:
            dx = 0.0
        else:
            dx = (float(x) - self._last_raw_value) / dt

        dx_hat = self._dx_filter(self._alpha(dt, self.d_cutoff), dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_filter(self._alpha(dt, cutoff), float(x))

        self._last_time = float(t)
        self._last_raw_value = float(x)
        return x_hat
