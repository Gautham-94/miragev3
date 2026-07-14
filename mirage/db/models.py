"""Database schema.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 10.1.

SQLite via peewee, WAL mode. Uses a plain peewee SqliteDatabase wrapped with the pragmas
the spec calls for; the spec's `SqliteQueueDatabase` async-write-queue recommendation is a
performance/concurrency refinement for multi-process write contention -- the schema and
correctness properties below are independent of that choice and this module can be
upgraded to it later without changing any model definition.
"""

from __future__ import annotations

from peewee import (
    SQL,
    BooleanField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.sqlite_ext import JSONField

from mirage.util.time import utcnow

db_proxy = DatabaseProxy()


class BaseModel(Model):
    class Meta:
        database = db_proxy


class Event(BaseModel):
    """Spec section 10.1: one tracked object's detection lifecycle."""

    id = CharField(primary_key=True, max_length=64)
    label = CharField(index=True)
    sub_label = CharField(null=True)
    camera = CharField(index=True)
    start_time = DateTimeField(index=True)
    end_time = DateTimeField(null=True, index=True)
    score = FloatField(default=0.0)
    top_score = FloatField(default=0.0)
    false_positive = BooleanField(default=True)
    zones = JSONField(default=list)
    has_clip = BooleanField(default=False)
    has_snapshot = BooleanField(default=False)
    snapshot_path = CharField(null=True)
    data = JSONField(default=dict)  # box, region, attributes, path history, max_severity, etc.


class Recordings(BaseModel):
    """Spec section 8.1/10.1: one permanently-stored recording segment."""

    id = CharField(primary_key=True, max_length=64)
    camera = CharField(index=True)
    path = CharField(unique=True)
    start_time = DateTimeField(index=True)
    end_time = DateTimeField(index=True)
    duration = FloatField()
    motion = IntegerField(null=True)
    objects = IntegerField(null=True)
    regions = IntegerField(null=True)
    dBFS = IntegerField(null=True)
    segment_size_mb = FloatField(default=0.0)


class ReviewSegment(BaseModel):
    """Spec section 9/10.1: human-facing activity-aggregation window."""

    id = CharField(primary_key=True, max_length=64)
    camera = CharField(index=True)
    start_time = DateTimeField(index=True)
    end_time = DateTimeField(null=True, index=True)
    severity = CharField()  # "alert" | "detection"
    thumb_path = CharField(unique=True, null=True)
    data = JSONField(default=dict)  # detections, objects, zones, audio, thumb_time, metadata


class Timeline(BaseModel):
    """Spec section 10.1: optional scrubbable activity timeline."""

    timestamp = DateTimeField(index=True)
    camera = CharField(index=True)
    source = CharField(index=True)  # e.g. "tracked_object"
    source_id = CharField(index=True)
    class_type = CharField()  # e.g. "entered_zone"
    data = JSONField(default=dict)


class Regions(BaseModel):
    """Spec section 4/10.1: learned per-camera object-size grid."""

    camera = CharField(primary_key=True)
    grid = JSONField(default=dict)
    last_update = DateTimeField(default=utcnow)


class AppConfig(BaseModel):
    """Singleton row holding the entire MirageConfig tree as JSON. Cameras/detectors are
    no longer authored by hand-editing a YAML file -- see mirage.config.schema.MirageConfig
    .from_db()/.save_to_db(), which read/write this row. The `id` CHECK constraint keeps
    this table to exactly one row (id=1) by construction, not just by convention.
    """

    id = IntegerField(primary_key=True, constraints=[SQL("CHECK (id = 1)")])
    data = JSONField(default=dict)
    updated_at = DateTimeField(default=utcnow)


class QueryMatch(BaseModel):
    """One saved open-vocabulary query (mirage.config.schema.OpenVocabQuery) matching a
    tracked object on a camera, per TODO_FIX_LIST.md items 4/6's design: the fast
    YOLOv8n detector + tracker confirms an object, a cheap perceptual-hash gate decides
    whether it's changed enough since last checked, and only then is its crop sent to
    the OWLv2 open-vocab process (mirage/openvocab/process.py) to check against every
    enabled query scoped to that camera. A dedicated table (rather than folding this
    into Event.data) so matches are independently queryable/listable/filterable by
    query/camera/time without JSON-scanning every event row -- this is the "show match
    found" list the Queries page is built around, not per-event metadata.
    """

    id = CharField(primary_key=True, max_length=64)
    query_id = CharField(index=True)
    query_text = CharField()  # denormalized snapshot -- survives the query being edited/deleted later
    camera = CharField(index=True)
    object_id = CharField(index=True)  # TrackedObjectState.id this match is for
    matched_at = DateTimeField(index=True, default=utcnow)
    score = FloatField(default=0.0)  # OWLv2's own confidence for this box
    box = JSONField(default=list)  # [y1, x1, y2, x2], full-frame pixel coords
    thumb_path = CharField(null=True)


ALL_MODELS = [Event, Recordings, ReviewSegment, Timeline, Regions, AppConfig, QueryMatch]
