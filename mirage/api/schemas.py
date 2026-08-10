"""Pydantic response models for the API -- separate from the peewee ORM models in
mirage.db.models so the wire format is decoupled from storage details (e.g. converting
naive-UTC DateTimeField values to epoch seconds for the frontend, and hiding db-internal
columns we don't want to commit to as a public contract).
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel


def _epoch(dt: datetime.datetime | None) -> float | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()


class CameraOut(BaseModel):
    name: str
    enabled: bool
    width: int | None
    height: int | None
    fps: int
    detector: str
    record_enabled: bool
    track_objects: list[str]
    track_all: bool = False


class EventOut(BaseModel):
    id: str
    camera: str
    label: str
    sub_label: str | None
    start_time: float
    end_time: float | None
    score: float
    top_score: float
    false_positive: bool
    zones: list[str]
    has_clip: bool
    has_snapshot: bool
    # Mirage V3 async species classification (mirage/species/, mirage/events/processor.py)
    # -- all read out of Event.data, not real columns (see mirage.db.models.Event.data's
    # own comment: "box, region, attributes, path history, max_severity, etc." are
    # expected to live there). species_status is always present ("not_applicable" for
    # Person/Vehicle/etc., "pending"/"complete"/"failed"/"skipped" for Animal/Bird).
    species: str | None = None
    species_status: str = "not_applicable"
    species_confidence: float | None = None
    species_taxonomy: dict | None = None

    @classmethod
    def from_model(cls, event) -> "EventOut":
        data = event.data or {}
        return cls(
            id=event.id,
            camera=event.camera,
            label=event.label,
            sub_label=event.sub_label,
            start_time=_epoch(event.start_time),
            end_time=_epoch(event.end_time),
            score=event.score,
            top_score=event.top_score,
            false_positive=event.false_positive,
            zones=event.zones or [],
            has_clip=event.has_clip,
            has_snapshot=event.has_snapshot,
            species=data.get("species"),
            species_status=data.get("species_status", "not_applicable"),
            species_confidence=data.get("species_confidence"),
            species_taxonomy=data.get("species_taxonomy"),
        )


class RecordingOut(BaseModel):
    id: str
    camera: str
    path: str
    start_time: float
    end_time: float
    duration: float
    motion: int | None
    objects: int | None
    dBFS: int | None
    segment_size_mb: float

    @classmethod
    def from_model(cls, rec) -> "RecordingOut":
        return cls(
            id=rec.id,
            camera=rec.camera,
            path=rec.path,
            start_time=_epoch(rec.start_time),
            end_time=_epoch(rec.end_time),
            duration=rec.duration,
            motion=rec.motion,
            objects=rec.objects,
            dBFS=rec.dBFS,
            segment_size_mb=rec.segment_size_mb,
        )


class ReviewSegmentOut(BaseModel):
    id: str
    camera: str
    start_time: float
    end_time: float | None
    severity: str
    thumb_path: str | None
    data: dict

    @classmethod
    def from_model(cls, seg) -> "ReviewSegmentOut":
        return cls(
            id=seg.id,
            camera=seg.camera,
            start_time=_epoch(seg.start_time),
            end_time=_epoch(seg.end_time),
            severity=seg.severity,
            thumb_path=seg.thumb_path,
            data=seg.data or {},
        )
