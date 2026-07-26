from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from mirage.api.schemas import RecordingOut
from mirage.db.models import Recordings
from mirage.util.time import utc_from_timestamp

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


@router.get("", response_model=list[RecordingOut])
def list_recordings(
    camera: str | None = None,
    after: float | None = Query(None, description="epoch seconds, inclusive lower bound on start_time"),
    before: float | None = Query(None, description="epoch seconds, exclusive upper bound on start_time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[RecordingOut]:
    query = Recordings.select()
    if camera is not None:
        query = query.where(Recordings.camera == camera)
    if after is not None:
        query = query.where(Recordings.start_time >= utc_from_timestamp(after))
    if before is not None:
        query = query.where(Recordings.start_time < utc_from_timestamp(before))

    query = query.order_by(Recordings.start_time.desc()).limit(limit).offset(offset)
    return [RecordingOut.from_model(r) for r in query]


@router.get("/{recording_id}", response_model=RecordingOut)
def get_recording(recording_id: str) -> RecordingOut:
    rec = Recordings.get_or_none(Recordings.id == recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown recording {recording_id!r}")
    return RecordingOut.from_model(rec)


@router.get("/{recording_id}/clip")
def get_recording_clip(recording_id: str) -> FileResponse:
    # No `filename=` here -- that sets Content-Disposition: attachment, which blocks
    # inline <video> playback (browser treats the response as a forced download only).
    # The frontend's download link/button sets its own [download] attribute instead.
    rec = Recordings.get_or_none(Recordings.id == recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown recording {recording_id!r}")
    path = Path(rec.path)
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"recording file no longer exists on disk: {rec.path}")
    return FileResponse(path, media_type="video/mp4")
