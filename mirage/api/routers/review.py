from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from mirage.api.schemas import EventOut, ReviewSegmentOut
from mirage.db.models import Event, ReviewSegment
from mirage.recording.stitch import recordings_overlapping, stitch_recordings
from mirage.util.time import utc_from_timestamp, utcnow

router = APIRouter(prefix="/api/review", tags=["review"])


@router.get("", response_model=list[ReviewSegmentOut])
def list_review_segments(
    camera: str | None = None,
    severity: str | None = Query(None, description="'alert' or 'detection'"),
    after: float | None = Query(None, description="epoch seconds, inclusive lower bound on start_time"),
    before: float | None = Query(None, description="epoch seconds, exclusive upper bound on start_time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[ReviewSegmentOut]:
    query = ReviewSegment.select()
    if camera is not None:
        query = query.where(ReviewSegment.camera == camera)
    if severity is not None:
        query = query.where(ReviewSegment.severity == severity)
    if after is not None:
        query = query.where(ReviewSegment.start_time >= utc_from_timestamp(after))
    if before is not None:
        query = query.where(ReviewSegment.start_time < utc_from_timestamp(before))

    query = query.order_by(ReviewSegment.start_time.desc()).limit(limit).offset(offset)
    return [ReviewSegmentOut.from_model(s) for s in query]


@router.get("/{segment_id}", response_model=ReviewSegmentOut)
def get_review_segment(segment_id: str) -> ReviewSegmentOut:
    seg = ReviewSegment.get_or_none(ReviewSegment.id == segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown review segment {segment_id!r}")
    return ReviewSegmentOut.from_model(seg)


@router.get("/{segment_id}/events", response_model=list[EventOut])
def get_review_segment_events(segment_id: str) -> list[EventOut]:
    """Every distinct Event (one per tracked object, see mirage.events.processor -- no
    cross-object suppression there) whose own [start_time, end_time] overlaps this
    segment's window on the same camera. Derived purely from existing data -- Events
    and ReviewSegments are independently built from the same per-camera tracker obj_id
    stream, so a time-range join is enough to recover "which distinct sightings made up
    this scene" without needing any new capture/persistence work or an explicit
    obj_id<->event_id link (ReviewSegment.data["detections"] keys are tracker obj_ids,
    which aren't the same value as Event.id -- see mirage.events.review's own
    PendingReviewSegment.detections vs mirage.events.processor's freshly-random
    _event_id).

    This is what backs the Detections page's per-scene sighting list -- ReviewSegment
    itself only ever stores one frozen thumbnail from the moment the segment opened, so
    an animal that entered the scene later has no box representation there at all; each
    Event does have its own snapshot with its own boxes (see events.py's
    /{event_id}/snapshot?bbox=true), which is what makes this join useful instead of
    just reading review segment's own data.
    """
    seg = ReviewSegment.get_or_none(ReviewSegment.id == segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown review segment {segment_id!r}")

    window_end = seg.end_time or utcnow()
    query = (
        Event.select()
        .where(Event.camera == seg.camera)
        .where(Event.start_time <= window_end)
        .where((Event.end_time.is_null()) | (Event.end_time >= seg.start_time))
        .order_by(Event.start_time.asc())
    )
    return [EventOut.from_model(e) for e in query]


@router.get("/{segment_id}/thumbnail")
def get_review_thumbnail(segment_id: str) -> FileResponse:
    seg = ReviewSegment.get_or_none(ReviewSegment.id == segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown review segment {segment_id!r}")
    if not seg.thumb_path:
        raise HTTPException(status_code=404, detail="this review segment has no thumbnail")
    path = Path(seg.thumb_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"thumbnail file no longer exists on disk: {seg.thumb_path}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{segment_id}/clip")
def get_review_segment_clip(segment_id: str, request: Request) -> FileResponse:
    """Stitches together every permanent recording clip overlapping this review
    segment's [start_time, end_time] window into one continuous video ("the footage
    that led to this alert") -- a review segment's own duration almost always spans
    multiple separate fixed-length recording files, never exactly one (see
    mirage/recording/stitch.py's module docstring). Cached under
    request.app.state.export_dir (defaults to the real EXPORT_DIR, injectable per-app
    for tests -- see create_app) keyed by segment id -- a segment's time window and its
    underlying recordings are immutable once the segment has ended, so a second request
    for the same segment reuses the already-stitched file rather than re-running ffmpeg.

    Deliberately does NOT pass filename=... to FileResponse -- that makes Starlette set
    Content-Disposition: attachment, which tells the browser to treat the response as a
    download rather than inline media. This single endpoint backs BOTH the review page's
    inline <video src=...> playback AND its separate download link (see
    frontend's video-lightbox.html: [videoUrl] and [downloadUrl] both point at this same
    URL) -- attachment broke inline playback (confirmed live: the <video> element never
    left its loading spinner despite the underlying HTTP request succeeding with a real
    200/206 and a valid, ffprobe-clean H.264/AAC file) while leaving the download link
    working, which is exactly what attachment is designed to do and why the bug was easy
    to miss from the download side alone. The download link's own HTML `download`
    attribute already fully handles the "save as" filename/behavior without needing the
    server to set Content-Disposition at all.
    """
    seg = ReviewSegment.get_or_none(ReviewSegment.id == segment_id)
    if seg is None:
        raise HTTPException(status_code=404, detail=f"unknown review segment {segment_id!r}")
    if seg.end_time is None:
        raise HTTPException(status_code=409, detail="this review segment hasn't ended yet")

    dest_path = Path(request.app.state.export_dir) / "review_clips" / f"{segment_id}.mp4"
    if dest_path.exists():
        return FileResponse(dest_path, media_type="video/mp4")

    recordings = recordings_overlapping(seg.camera, seg.start_time, seg.end_time)
    if not recordings:
        raise HTTPException(
            status_code=404,
            detail="no recordings cover this review segment's time window (may have been deleted by retention)",
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ok = stitch_recordings(recordings, dest_path)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to stitch recordings for this review segment")

    return FileResponse(dest_path, media_type="video/mp4")
