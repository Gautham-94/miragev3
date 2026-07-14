from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pytest

from mirage.db.database import close_database, init_database
from mirage.db.models import Event, Recordings, Regions, ReviewSegment, Timeline
from mirage.util.time import utcnow


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        database = init_database(db_path)
        yield database
        close_database(database)


def test_wal_mode_is_active(db):
    (mode,) = db.execute_sql("PRAGMA journal_mode;").fetchone()
    assert mode.lower() == "wal"


def test_tables_created(db):
    tables = {row[0] for row in db.execute_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    for expected in ("event", "recordings", "reviewsegment", "timeline", "regions"):
        assert expected in tables


def test_event_crud_with_json_data_field(db):
    now = datetime.datetime.now(datetime.timezone.utc)
    Event.create(
        id="1234567890-abcdef",
        label="person",
        camera="front_door",
        start_time=now,
        score=0.91,
        top_score=0.95,
        false_positive=False,
        zones=["driveway"],
        data={"box": [10, 20, 30, 40], "attributes": [{"label": "face", "score": 0.8}]},
    )
    fetched = Event.get(Event.id == "1234567890-abcdef")
    assert fetched.label == "person"
    assert fetched.zones == ["driveway"]
    assert fetched.data["box"] == [10, 20, 30, 40]
    assert fetched.data["attributes"][0]["label"] == "face"
    assert fetched.false_positive is False


def test_event_end_time_nullable(db):
    Event.create(id="in-progress-1", label="car", camera="cam1", start_time=datetime.datetime.now(datetime.timezone.utc))
    fetched = Event.get(Event.id == "in-progress-1")
    assert fetched.end_time is None


def test_recordings_unique_path_constraint(db):
    now = datetime.datetime.now(datetime.timezone.utc)
    Recordings.create(
        id="rec1", camera="cam1", path="/media/nvr/recordings/2026-01-01/00/cam1/00.00.mp4",
        start_time=now, end_time=now + datetime.timedelta(seconds=10), duration=10.0,
    )
    with pytest.raises(Exception):
        Recordings.create(
            id="rec2", camera="cam1", path="/media/nvr/recordings/2026-01-01/00/cam1/00.00.mp4",
            start_time=now, end_time=now + datetime.timedelta(seconds=10), duration=10.0,
        )


def test_review_segment_severity_and_data(db):
    now = datetime.datetime.now(datetime.timezone.utc)
    ReviewSegment.create(
        id="rev1", camera="cam1", start_time=now, severity="alert",
        thumb_path="/media/nvr/clips/review/thumb-cam1-rev1.webp",
        data={"detections": ["evt1"], "objects": ["person"], "zones": ["yard"]},
    )
    fetched = ReviewSegment.get(ReviewSegment.id == "rev1")
    assert fetched.severity == "alert"
    assert fetched.data["objects"] == ["person"]


def test_timeline_row(db):
    now = datetime.datetime.now(datetime.timezone.utc)
    Timeline.create(
        timestamp=now, camera="cam1", source="tracked_object", source_id="evt1",
        class_type="entered_zone", data={"zone": "yard"},
    )
    rows = list(Timeline.select().where(Timeline.camera == "cam1"))
    assert len(rows) == 1
    assert rows[0].data["zone"] == "yard"


def test_regions_grid_upsert(db):
    Regions.create(camera="cam1", grid={"0,0": {"w": 100, "h": 80}})
    fetched = Regions.get(Regions.camera == "cam1")
    assert fetched.grid["0,0"]["w"] == 100

    # Update in place (camera is the primary key).
    fetched.grid = {"0,0": {"w": 120, "h": 90}}
    fetched.save()
    refetched = Regions.get(Regions.camera == "cam1")
    assert refetched.grid["0,0"]["w"] == 120


def test_reopening_existing_db_file_preserves_data():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "persist.db")
        db1 = init_database(db_path)
        Event.create(id="persisted-1", label="person", camera="cam1", start_time=datetime.datetime.now(datetime.timezone.utc))
        close_database(db1)

        db2 = init_database(db_path)
        fetched = Event.get(Event.id == "persisted-1")
        assert fetched.label == "person"
        close_database(db2)


def test_naive_utc_datetime_survives_roundtrip_as_a_real_datetime(db):
    """Regression test for a real bug: peewee's DateTimeField.formats has no
    timezone-offset pattern, so a timezone-AWARE datetime (e.g.
    datetime.now(datetime.timezone.utc), NOT stripped of tzinfo) silently comes back as
    an unparsed STR after a round trip through SQLite, rather than a datetime object --
    breaking any arithmetic on it. mirage.util.time.utcnow() (a naive datetime) is the
    fix used everywhere in this codebase; this test pins down the underlying mechanism
    so a future change can't silently reintroduce tz-aware datetimes as a write path.
    """
    naive = utcnow()
    aware = datetime.datetime.now(datetime.timezone.utc)

    Event.create(id="naive-dt", label="person", camera="cam1", start_time=naive)
    Event.create(id="aware-dt", label="person", camera="cam1", start_time=aware)

    fetched_naive = Event.get(Event.id == "naive-dt")
    fetched_aware = Event.get(Event.id == "aware-dt")

    assert isinstance(fetched_naive.start_time, datetime.datetime), (
        "naive UTC datetime must round-trip as a real datetime object"
    )
    # This assertion documents the actual (buggy, if ever reintroduced) peewee behavior:
    # a tz-aware datetime does NOT round-trip as a datetime -- it comes back as a string.
    assert isinstance(fetched_aware.start_time, str), (
        "if this now fails because peewee/playhouse started parsing tz-aware datetimes "
        "correctly, the workaround in mirage.util.time may no longer be necessary -- but "
        "do not remove it without verifying arithmetic on every DateTimeField call site"
    )

    # Confirm arithmetic actually works on the naive (correctly-typed) datetime.
    _ = fetched_naive.start_time + datetime.timedelta(seconds=30)
