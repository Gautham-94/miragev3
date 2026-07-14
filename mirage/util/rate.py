"""Rolling-window events-per-second counter.

Used for camera_fps / skipped_fps / detection_fps style metrics throughout the pipeline
(spec section 1.7, "reconnects-per-hour and stalls-per-hour" style bounded windows).
"""

from __future__ import annotations

import time
from collections import deque
from multiprocessing import Value


class EventsPerSecond:
    def __init__(self, window_seconds: float = 10.0) -> None:
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    def update(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def eps(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        self._prune(now)
        return len(self._timestamps) / self._window if self._timestamps else 0.0

    def count(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        self._prune(now)
        return len(self._timestamps)


class BoundedWindowCounter:
    """Counts timestamped events within a rolling window (e.g. reconnects/stalls per hour)."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    def record(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def count(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        self._prune(now)
        return len(self._timestamps)


def shared_double(initial: float = 0.0) -> "Value":
    return Value("d", initial)
