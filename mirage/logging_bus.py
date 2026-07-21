"""Structured activity log, shared between the `python -m mirage` pipeline process (the
writer) and `mirage.api` (the reader) -- powers the frontend's Logs page.

Same file-based IPC pattern already used for supervisor status/restart-request (see
mirage/supervisor.py's own docstring): the pipeline and API are separate OS processes
that share nothing but disk and the SQLite DB, so a live in-memory queue/deque can't be
read across that boundary. Instead the pipeline process appends one JSON object per line
to a plain file under cache_dir, and the API just reads the file fresh on every request
(mirage.config.schema.MirageConfig.from_db's own "always read fresh, no stale in-memory
cache" reasoning applies here too).

Within the pipeline process itself, mirage.app.MirageApp is still heavily multi-process
(DetectorProcess, CameraTracker, SpeciesProcess, ...) -- rather than have every one of
those processes open/append to the log file directly (risking interleaved partial
writes), each process puts LogEvent instances onto one shared mp.Queue
(MirageApp.activity_log_queue), and only the main process's result-consumer thread
drains that queue and appends to the file, exactly the same "one writer" shape
detected_frames_queue/species result_queue already use.

Bounded via simple truncation-on-write: rewriting the whole file past MAX_LOG_LINES
would be wasteful at this size (a few thousand short JSON lines), so instead the file is
periodically trimmed from the front by _maybe_trim (called from the same append path) --
cheap enough that doing it every ~200 appends is unnoticeable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ACTIVITY_LOG_FILENAME = "activity_log.jsonl"

MAX_LOG_LINES = 5000
TRIM_CHECK_INTERVAL = 200  # only bother checking/trimming file length every N appends


@dataclass
class LogEvent:
    # Category is a closed, small set of pipeline-stage names (not free text) so the
    # frontend can offer a filter dropdown without scraping distinct values from
    # historical data: motion, detect, track, species, system.
    category: str
    message: str
    camera: str | None = None
    timestamp: float = field(default_factory=time.time)


def _log_path(cache_dir: str) -> Path:
    return Path(cache_dir) / ACTIVITY_LOG_FILENAME


class ActivityLogWriter:
    """Lives in the main process's result-consumer thread (see mirage.app.MirageApp) --
    the only thing that ever appends to the log file, draining LogEvents put onto
    MirageApp.activity_log_queue by any pipeline process/thread.
    """

    def __init__(self, cache_dir: str) -> None:
        self.path = _log_path(cache_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._appends_since_trim = 0

    def append(self, event: LogEvent) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)))
            f.write("\n")
        self._appends_since_trim += 1
        if self._appends_since_trim >= TRIM_CHECK_INTERVAL:
            self._appends_since_trim = 0
            self._maybe_trim()

    def _maybe_trim(self) -> None:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return
        if len(lines) <= MAX_LOG_LINES:
            return
        trimmed = lines[-MAX_LOG_LINES:]
        self.path.write_text("\n".join(trimmed) + "\n")


def read_recent_logs(cache_dir: str, limit: int = 500, since: float | None = None) -> list[dict]:
    """Reads the most recent `limit` log entries, newest last (chronological order) --
    mirage.api's GET /api/system/logs, called fresh on every request (no caching; see
    this module's own docstring for why). `since`, if given, only returns entries with
    timestamp strictly greater than it, for the frontend's incremental-poll case.
    """
    path = _log_path(cache_dir)
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []

    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue  # a read racing a partial/interrupted write is possible; skip it

    if since is not None:
        entries = [e for e in entries if e.get("timestamp", 0) > since]

    return entries[-limit:]
