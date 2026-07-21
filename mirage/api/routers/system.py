"""Pipeline restart control: lets the frontend's "Apply changes" button request a clean
restart of the `python -m mirage` pipeline process (run under `mirage.supervisor`, see
that module's docstring for the full design/why) and poll its status while restarting.

mirage.api never starts, stops, or owns the pipeline process itself -- it only reads/
writes two small files under the shared cache_dir that mirage.supervisor also reads/
writes, exactly the same file-based IPC pattern already used elsewhere in this codebase
(see mirage/const.py's ipc_addr for the analogous ZMQ-socket-path convention).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from mirage.logging_bus import read_recent_logs
from mirage.supervisor import RESTART_REQUEST_FILENAME, STATUS_FILENAME

router = APIRouter(prefix="/api/system", tags=["system"])


class SystemStatus(BaseModel):
    # "unknown" if the supervisor has never written a status file at all -- e.g.
    # mirage.api was started before mirage.supervisor ever ran once, or the pipeline is
    # being run directly via `python -m mirage` (no supervisor) rather than through
    # `python -m mirage.supervisor`. The frontend treats this the same as "running" for
    # display purposes (no banner), since there's nothing wrong, just nothing to report.
    state: str
    pid: int | None
    updated_at: float | None


def _status_path(request: Request) -> Path:
    return Path(request.app.state.cache_dir) / STATUS_FILENAME


def _restart_request_path(request: Request) -> Path:
    return Path(request.app.state.cache_dir) / RESTART_REQUEST_FILENAME


@router.get("/status", response_model=SystemStatus)
def get_status(request: Request) -> SystemStatus:
    path = _status_path(request)
    if not path.exists():
        return SystemStatus(state="unknown", pid=None, updated_at=None)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A read racing the supervisor's atomic rename is possible in principle but
        # exceptionally unlikely in practice (the write is a rename, not an in-place
        # edit) -- treat a transient read failure as "unknown" rather than erroring the
        # whole status endpoint, since the next poll a second later will succeed.
        return SystemStatus(state="unknown", pid=None, updated_at=None)
    return SystemStatus(state=data.get("state", "unknown"), pid=data.get("pid"), updated_at=data.get("updated_at"))


@router.post("/restart", status_code=202)
def request_restart(request: Request) -> dict:
    path = _restart_request_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()))
    return {"ok": True}


class LogEntryOut(BaseModel):
    timestamp: float
    category: str
    message: str
    camera: str | None


@router.get("/logs", response_model=list[LogEntryOut])
def get_logs(request: Request, limit: int = 500, since: float | None = None) -> list[LogEntryOut]:
    """Powers the Logs page -- reads mirage.logging_bus's shared activity_log.jsonl file
    fresh on every call (see that module's docstring for why mirage.api can't just hold
    a live queue/deque: it's a separate OS process from the pipeline that writes this).
    `since` lets the frontend poll incrementally (only fetch entries newer than the last
    one it already has) instead of re-fetching the whole tail every time.
    """
    entries = read_recent_logs(str(request.app.state.cache_dir), limit=limit, since=since)
    return [LogEntryOut(**e) for e in entries]


class SystemCapabilitiesOut(BaseModel):
    cpu_count: int
    # Physical memory, in MB, on the machine mirage.api is running on -- reported so
    # the Config page can size its num_workers guidance against this specific host's
    # real headroom rather than a generic rule of thumb (a detector worker's memory
    # cost is roughly fixed regardless of CPU count, so this matters just as much as
    # cpu_count for the "should I add another worker" decision).
    total_memory_mb: int


@router.get("/capabilities", response_model=SystemCapabilitiesOut)
def get_capabilities() -> SystemCapabilitiesOut:
    """Real host capability numbers -- powers the Config page's num_workers guidance
    ("you have N cores, M GB RAM; each extra detector worker costs roughly its model's
    own memory footprint again and competes for the same CPU/accelerator") rather than
    a hardcoded generic recommendation baked into the frontend.
    """
    cpu_count = os.cpu_count() or 1

    total_memory_mb = 0
    try:
        import psutil

        total_memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass  # psutil is a pinned dependency, but degrade gracefully rather than 500 if it's ever missing

    return SystemCapabilitiesOut(cpu_count=cpu_count, total_memory_mb=total_memory_mb)
