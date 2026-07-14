from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from mirage.api.schemas import EventOut
from mirage.db.models import Event
from mirage.util.time import utc_from_timestamp

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    camera: str | None = None,
    label: str | None = None,
    after: float | None = Query(None, description="epoch seconds, inclusive lower bound on start_time"),
    before: float | None = Query(None, description="epoch seconds, exclusive upper bound on start_time"),
    include_false_positive: bool = Query(False, description="include events still flagged false_positive"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[EventOut]:
    query = Event.select()
    if camera is not None:
        query = query.where(Event.camera == camera)
    if label is not None:
        query = query.where(Event.label == label)
    if after is not None:
        query = query.where(Event.start_time >= utc_from_timestamp(after))
    if before is not None:
        query = query.where(Event.start_time < utc_from_timestamp(before))
    if not include_false_positive:
        query = query.where(Event.false_positive == False)  # noqa: E712 -- peewee expression, not a Python bool check

    query = query.order_by(Event.start_time.desc()).limit(limit).offset(offset)
    return [EventOut.from_model(e) for e in query]


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: str) -> EventOut:
    event = Event.get_or_none(Event.id == event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")
    return EventOut.from_model(event)


@router.get("/{event_id}/snapshot")
def get_event_snapshot(event_id: str) -> FileResponse:
    event = Event.get_or_none(Event.id == event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")
    if not event.snapshot_path:
        raise HTTPException(status_code=404, detail="this event has no snapshot")
    path = Path(event.snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"snapshot file no longer exists on disk: {event.snapshot_path}")
    return FileResponse(path, media_type="image/jpeg")
