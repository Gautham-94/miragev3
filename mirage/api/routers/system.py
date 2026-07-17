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
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

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
