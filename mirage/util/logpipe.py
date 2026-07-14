"""Buffered stderr pipe for ffmpeg subprocesses.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.6.

Wraps ffmpeg's stderr in a thread that reads an os.pipe() and buffers the last N lines in a
bounded deque WITHOUT forwarding them to the logger during normal operation -- only emits the
buffered lines when explicitly dumped (call .dump() when ffmpeg crashes/needs a restart).
This preserves forensic detail exactly when needed while keeping logs quiet in the healthy
case.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque


class LogPipe(threading.Thread):
    def __init__(self, logger_name: str, max_lines: int = 100) -> None:
        super().__init__(daemon=True)
        self.logger = logging.getLogger(logger_name)
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._fd_read, self._fd_write = os.pipe()
        self._pipe_reader = os.fdopen(self._fd_read)
        self._stop = threading.Event()

    def fileno(self) -> int:
        """So this can be passed directly as `stderr=logpipe` to subprocess.Popen."""
        return self._fd_write

    def run(self) -> None:
        for line in iter(self._pipe_reader.readline, ""):
            if self._stop.is_set():
                break
            self._lines.append(line.rstrip("\n"))

    def dump(self, level: int = logging.ERROR) -> None:
        """Emit all buffered lines to the logger. Call this when the ffmpeg process this
        pipe belongs to crashes or is being restarted, so the forensic detail leading up
        to the failure is preserved.
        """
        for line in self._lines:
            self.logger.log(level, line)
        self._lines.clear()

    def close(self) -> None:
        self._stop.set()
        try:
            os.close(self._fd_write)
        except OSError:
            pass
        try:
            self._pipe_reader.close()
        except OSError:
            pass
