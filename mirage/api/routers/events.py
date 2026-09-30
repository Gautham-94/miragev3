from __future__ import annotations

import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from mirage.api.schemas import EventOut, ReviewSegmentOut
from mirage.db.models import Event, ReviewSegment
from mirage.recording.stitch import recordings_overlapping, stitch_recordings
from mirage.util.thumbnail import draw_boxes_on_jpeg_bytes
from mirage.util.time import utc_from_timestamp

router = APIRouter(prefix="/api/events", tags=["events"])

# An Event with no end_time yet (object still being tracked) or a very short-lived one
# has too narrow a window to reliably overlap a recording segment boundary -- pad both
# sides, same margin the frontend's old Events->Recordings deep-link already used
# (see EventsPage.viewVideo's RECORDING_LOOKUP_PADDING_SECONDS).
CLIP_WINDOW_PADDING_SECONDS = 30


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


@router.get("/scenes", response_model=dict[str, ReviewSegmentOut | None])
def get_scenes_for_events(
    ids: str = Query(..., description="comma-separated event ids"),
) -> dict[str, ReviewSegmentOut | None]:
    """Reverse of mirage.api.routers.review's GET /{segment_id}/events -- given a batch
    of event ids (one page of the Detections page's flat sighting list), resolves each
    one's containing ReviewSegment (or null if it never qualified for review at all,
    e.g. its label isn't in review.alerts.labels/review.detections.labels -- see
    mirage.events.review.classify_severity), in ONE extra query rather than one per row.

    Registered ABOVE /{event_id} deliberately -- FastAPI/Starlette matches routes in
    registration order, so /scenes must come first or it'd be swallowed as
    event_id="scenes".

    Each event resolves to at most one segment in practice: ReviewSegmentMaintainer
    only ever has one OPEN segment per camera at a time (mirage/events/review.py's
    PendingReviewSegment._pending, keyed by camera name only), so segments never
    overlap in time for the same camera. `next(...)` below just takes the first
    (only) match rather than assuming that invariant blindly.
    """
    event_ids = [i for i in ids.split(",") if i]
    if not event_ids:
        return {}
    events = list(Event.select().where(Event.id.in_(event_ids)))
    if not events:
        return {}

    cameras = {e.camera for e in events}
    min_start = min(e.start_time for e in events)
    max_end = max((e.end_time or e.start_time) for e in events)

    segments = list(
        ReviewSegment.select()
        .where(ReviewSegment.camera.in_(cameras))
        .where(ReviewSegment.start_time <= max_end)
        .where((ReviewSegment.end_time.is_null()) | (ReviewSegment.end_time >= min_start))
    )

    result: dict[str, ReviewSegmentOut | None] = {}
    for event in events:
        window_end = event.end_time or event.start_time
        match = next(
            (
                seg
                for seg in segments
                if seg.camera == event.camera
                and seg.start_time <= window_end
                and (seg.end_time is None or seg.end_time >= event.start_time)
            ),
            None,
        )
        result[event.id] = ReviewSegmentOut.from_model(match) if match else None
    return result


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


@router.get("/{event_id}/clip")
def get_event_clip(event_id: str, request: Request) -> FileResponse:
    """Stitches every permanent recording clip overlapping this event's time window
    (padded by CLIP_WINDOW_PADDING_SECONDS on both sides) into one continuous video --
    same mirage.recording.stitch primitive mirage.api.routers.review's review-segment
    clip endpoint uses, since there's no single recording file that already equals an
    event's own (usually sub-recording-length) time window, and no direct Event<->
    Recordings foreign key to look up instead. Cached under
    request.app.state.export_dir keyed by event id, same as the review clip endpoint.

    No filename=... passed to FileResponse -- see mirage.api.routers.review's own clip
    endpoint docstring for why that would break inline <video> playback.
    """
    event = Event.get_or_none(Event.id == event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"unknown event {event_id!r}")

    dest_path = Path(request.app.state.export_dir) / "event_clips" / f"{event_id}.mp4"
    if dest_path.exists():
        return FileResponse(dest_path, media_type="video/mp4")

    padding = datetime.timedelta(seconds=CLIP_WINDOW_PADDING_SECONDS)
    window_start = event.start_time - padding
    window_end = (event.end_time or event.start_time) + padding

    recordings = recordings_overlapping(event.camera, window_start, window_end)
    if not recordings:
        raise HTTPException(
            status_code=404,
            detail="no recordings cover this event's time window (may have been deleted by retention)",
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ok = stitch_recordings(recordings, dest_path)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to stitch recordings for this event")

    return FileResponse(dest_path, media_type="video/mp4")
