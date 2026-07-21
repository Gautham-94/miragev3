from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from mirage.api.schemas import EventOut
from mirage.db.models import Event
from mirage.util.thumbnail import draw_boxes_on_jpeg_bytes
from mirage.util.time import utc_from_timestamp

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[EventOut])
def list_events(
    camera: str | None = None,
    label: str | None = None,
    after: float | None = Query(None, description="epoch seconds, inclusive lower bound on start_time"),
    before: float | None = Query(None, description="epoch seconds, exclusive upper bound on start_time"),
    include_false_positive: bool = Query(False, description="include events still flagged false_positive"),
    species_status: str | None = Query(
        None, description="filter by species classification status: pending/complete/failed/skipped/not_applicable",
    ),
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
    if species_status is not None:
        # Event.data is a JSONField (playhouse.sqlite_ext) -- this expression compiles to
        # a SQLite json_extract() comparison, no schema migration needed since
        # species_status lives inside the JSON blob, not a real column (see
        # mirage.events.processor.EventProcessor._on_start / mirage.api.schemas.EventOut).
        query = query.where(Event.data["species_status"] == species_status)

    query = query.order_by(Event.start_time.desc()).limit(limit).offset(offset)
    return [EventOut.from_model(e) for e in query]


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: str) -> EventOut:
    event = Event.get_or_none(Event.id == event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")
    return EventOut.from_model(event)


@router.get("/{event_id}/snapshot")
def get_event_snapshot(
    event_id: str,
    bbox: bool = Query(True, description="draw bounding boxes for every confirmed object in the frame"),
) -> Response:
    """The stored file (event.snapshot_path) is always the CLEAN, unannotated frame --
    see mirage.util.thumbnail's own module docstring for why (mirrors Frigate's own
    confirmed design). Boxes are drawn fresh on every request from
    event.data["snapshot_boxes"] when bbox=True (the default) -- pass bbox=false to get
    the raw clean file back, e.g. for downloading an unannotated copy.
    """
    event = Event.get_or_none(Event.id == event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")
    if not event.snapshot_path:
        raise HTTPException(status_code=404, detail="this event has no snapshot")
    path = Path(event.snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"snapshot file no longer exists on disk: {event.snapshot_path}")

    if not bbox:
        return FileResponse(path, media_type="image/jpeg")

    snapshot_boxes = (event.data or {}).get("snapshot_boxes") or []
    if not snapshot_boxes:
        return FileResponse(path, media_type="image/jpeg")

    boxes = [(entry["label"], tuple(entry["box"])) for entry in snapshot_boxes]
    boxed_bytes = draw_boxes_on_jpeg_bytes(path.read_bytes(), boxes)
    if boxed_bytes is None:
        # Drawing failed (corrupt file, decode error) -- serve the clean original
        # rather than a 500, matching draw_boxes_on_jpeg_bytes's own best-effort contract.
        return FileResponse(path, media_type="image/jpeg")

    return Response(boxed_bytes, media_type="image/jpeg")
