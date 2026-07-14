"""ServiceWatchdog: restarts crashed service processes (detector, recording, review,
output) with a bounded restart-rate circuit breaker.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.7 ("A separate,
higher-level watchdog in the main process supervises the service processes").
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

MAX_RESTARTS = 5
RESTART_WINDOW_SECONDS = 60.0


@dataclass
class MonitoredProcess:
    name: str
    process: object  # multiprocessing.Process
    factory: Callable[[], object]
    on_restart: Callable[[object], None]
    restart_timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_RESTARTS))


class ServiceWatchdog(threading.Thread):
    def __init__(self, tick_seconds: float = 10.0) -> None:
        super().__init__(daemon=True, name="watchdog")
        self.tick_seconds = tick_seconds
        self._entries: dict[str, MonitoredProcess] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register(self, name: str, process, factory: Callable[[], object], on_restart: Callable[[object], None]) -> None:
        with self._lock:
            self._entries[name] = MonitoredProcess(name=name, process=process, factory=factory, on_restart=on_restart)

    def run(self) -> None:
        while not self._stop_event.wait(self.tick_seconds):
            self._check_all()

    def stop(self) -> None:
        """Signals the watchdog thread to stop and blocks until it has actually exited.

        Callers rely on this to guarantee no further restarts can happen once stop()
        returns -- without joining, the thread's own wait() loop can sleep for up to
        tick_seconds before noticing the stop event, leaving a window where it can race
        a caller's own process teardown (e.g. "helpfully" restarting a detector process
        the caller is in the middle of terminating).
        """
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=self.tick_seconds + 5)

    def _check_all(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            self._check_one(entry)

    def _check_one(self, entry: MonitoredProcess) -> None:
        if entry.process.is_alive():
            return

        exitcode = entry.process.exitcode
        if exitcode == 0:
            logger.info("%s exited cleanly, not restarting", entry.name)
            return

        now = time.time()
        if self._is_restarting_too_fast(entry, now):
            logger.error("%s restarted too many times (%d) within %ds, giving up", entry.name, MAX_RESTARTS, RESTART_WINDOW_SECONDS)
            return

        logger.warning("%s died (exitcode=%s), restarting", entry.name, exitcode)
        try:
            entry.process.close()
        except (ValueError, AttributeError):
            pass  # process object may already be in a state that disallows close()

        new_process = entry.factory()
        new_process.start()
        entry.restart_timestamps.append(now)
        entry.process = new_process
        entry.on_restart(new_process)

    def _is_restarting_too_fast(self, entry: MonitoredProcess, now: float) -> bool:
        if len(entry.restart_timestamps) < MAX_RESTARTS:
            return False
        oldest = entry.restart_timestamps[0]
        return (now - oldest) < RESTART_WINDOW_SECONDS
