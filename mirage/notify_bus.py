"""Structured "a row was created/updated" bus, shared between the `python -m mirage`
pipeline process (the writer) and `mirage.api` (the reader) -- powers the frontend's
live SSE stream (Events/Review/Queries pages).

Same file-based IPC pattern as mirage/logging_bus.py (read that module's docstring for
the full rationale): the pipeline and API are separate OS processes sharing nothing but
disk and the SQLite DB, so the pipeline appends one small JSON line per row change to a
plain file under cache_dir, and mirage.api's background tailer polls that file for new
lines and re-fetches the actual row fresh from SQLite before broadcasting over SSE --
never the row data itself, just `{table, id, op}`, so a stale/point-in-time snapshot
can never reach the frontend (mirrors the "always read fresh" philosophy already used
for config/logs).

Kept as a dedicated file (not folded into activity_log.jsonl) since the two are
consumed differently: the activity log is read on-demand per HTTP request for the Logs
page, this file is continuously tailed by a background asyncio task. Different
consumer, different cadence, different schema -- same reasoning that already keeps
activity_log.jsonl separate from other pipeline state files.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

NOTIFY_LOG_FILENAME = "notify_log.jsonl"

MAX_LOG_LINES = 5000
TRIM_CHECK_INTERVAL = 200  # only bother checking/trimming file length every N appends


@dataclass
class NotifyEvent:
    table: str  # "event" | "review_segment" | "query_match"
    id: str
    op: str  # "create" | "update"
    timestamp: float = field(default_factory=time.time)


def _log_path(cache_dir: str) -> Path:
    return Path(cache_dir) / NOTIFY_LOG_FILENAME


class NotifyLogWriter:
    """Lives in the main process's result-consumer thread (see mirage.app.MirageApp) --
    the only thing that ever appends to the notify log file, draining NotifyEvents put
    onto MirageApp.notify_queue by EventProcessor/ReviewSegmentMaintainer.
    """

    def __init__(self, cache_dir: str) -> None:
        self.path = _log_path(cache_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._appends_since_trim = 0

    def append(self, event: NotifyEvent) -> None:
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


def read_notify_events_since(cache_dir: str, since_offset: int) -> tuple[list[dict], int]:
    """Reads every notify entry at or past line `since_offset` in the file's CURRENT
    state. Returns (new_entries, new_offset) -- pass `new_offset` back in as
    `since_offset` on the next call. If the file was trimmed (or doesn't exist yet)
    such that `since_offset` is now beyond the file's length, this resets to reading
    from the top of what remains rather than erroring -- a rare, harmless case (only
    means a tailer that was stalled for MAX_LOG_LINES's worth of writes might replay a
    few already-seen entries, which the frontend's prependLive-by-id already tolerates).
    """
    path = _log_path(cache_dir)
    if not path.exists():
        return [], since_offset
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return [], since_offset

    if since_offset > len(lines):
        since_offset = 0

    new_lines = lines[since_offset:]
    entries = []
    for line in new_lines:
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue  # a read racing a partial/interrupted write is possible; skip it

    return entries, len(lines)
