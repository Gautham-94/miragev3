"""SSE endpoint for live push updates -- lets the frontend learn about new
Event/ReviewSegment rows as soon as the pipeline process creates them, instead of
relying purely on polling (see the Events/Review pages' `visibleInterval`
reconciliation polls, which now just self-heal any gap from a dropped connection).

Bridges the pipeline/API process boundary via mirage.notify_bus (same file-based IPC
pattern as mirage.logging_bus, see that module's docstring for the full rationale): a
single background asyncio task (`_tail_notify_log`, started in the app's lifespan)
polls the notify log file every NOTIFY_POLL_SECONDS, re-fetches each new row fresh from
SQLite (never trusting a payload serialized back when the pipeline process wrote the
notification -- a ReviewSegment in particular can be notified multiple times as it
evolves, so the row must always be read at broadcast time, not cached from creation),
and fans it out to every currently-connected SSE client's own asyncio.Queue.

One shared asyncio.Task/subscriber-set is safe here specifically because
mirage.api.__main__ runs uvicorn with a single worker process (no `workers=N`) -- see
that module's docstring. A multi-worker deployment would need each worker's own tailer
plus a cross-worker fan-out (e.g. Redis pub/sub), which this app doesn't need today.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mirage.api.schemas import EventOut, ReviewSegmentOut
from mirage.db.models import Event, ReviewSegment
from mirage.notify_bus import read_notify_events_since

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["notifications"])

NOTIFY_POLL_SECONDS = 0.25
KEEPALIVE_SECONDS = 15.0
CLIENT_QUEUE_MAXSIZE = 200

_TABLE_LOOKUP = {
    "event": (Event, EventOut),
    "review_segment": (ReviewSegment, ReviewSegmentOut),
}

# Every currently-connected SSE client's own queue -- fan-out target for
# _tail_notify_log. Module-level (not app.state) since it's only ever touched from
# this module's own background task + its own route handler, both in the same process.
_subscribers: set["asyncio.Queue[dict]"] = set()


def _fetch_and_serialize(table: str, row_id: str) -> dict | None:
    lookup = _TABLE_LOOKUP.get(table)
    if lookup is None:
        return None
    model, schema = lookup
    row = model.get_or_none(model.id == row_id)
    if row is None:
        return None  # row was deleted, or this read raced a not-yet-committed write
    return {"type": table, "data": schema.from_model(row).model_dump()}


async def notify_tailer_loop(cache_dir: str) -> None:
    """Runs for the lifetime of the API process (started/cancelled from the app's
    lifespan, mirroring how app.state.database is set up/torn down there).
    """
    offset = 0
    while True:
        try:
            entries, offset = await asyncio.to_thread(read_notify_events_since, cache_dir, offset)
            for entry in entries:
                table = entry.get("table")
                row_id = entry.get("id")
                if not table or not row_id:
                    continue
                message = await asyncio.to_thread(_fetch_and_serialize, table, row_id)
                if message is None:
                    continue
                for queue in list(_subscribers):
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        pass  # a slow client misses a message rather than blocking everyone else
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("error tailing notify log")
        await asyncio.sleep(NOTIFY_POLL_SECONDS)


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    async def event_generator():
        queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=CLIENT_QUEUE_MAXSIZE)
        _subscribers.add(queue)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
