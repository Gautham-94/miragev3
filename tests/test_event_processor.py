from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mirage.config.schema import CameraConfig, CameraInputConfig, FfmpegConfig
from mirage.db.database import close_database, init_database
from mirage.db.models import Event
from mirage.events.processor import EventProcessor
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def _camera(name: str = "cam1") -> CameraConfig:
    return CameraConfig(name=name, ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]))


def _state(
    obj_id: str, label: str, score: float, box=(0, 0, 100, 100), frame_time: float = 0.0, is_false_positive: bool = True
) -> TrackedObjectState:
    return TrackedObjectState(
        id=obj_id, label=label, box=box, score=score,
        stationary=StationaryClassifier(threshold_frames=50), frame_time=frame_time,
        is_false_positive=is_false_positive,
    )


def test_new_object_creates_event_row(db):
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state})

    rows = list(Event.select())
    assert len(rows) == 1
    assert rows[0].label == "person"
    assert rows[0].camera == "cam1"
    assert rows[0].end_time is None
    assert rows[0].false_positive is True  # sticky-until-proven-true, per spec section 6.1


def test_new_object_captures_snapshot_via_injected_fetcher(db):
    with tempfile.TemporaryDirectory() as thumb_dir:
        fetched_cameras = []

        def fake_fetcher(camera_name: str) -> bytes:
            fetched_cameras.append(camera_name)
            return b"\xff\xd8fakejpegbytes"

        processor = EventProcessor(thumbnail_fetcher=fake_fetcher, thumb_dir=thumb_dir)
        camera = _camera()
        state = _state("obj1", "person", 0.9, frame_time=100.0)

        processor.process(camera, 100.0, {"obj1": state})

        assert fetched_cameras == ["cam1"]
        row = Event.select().get()
        assert row.has_snapshot is True
        assert row.snapshot_path is not None
        assert Path(row.snapshot_path).read_bytes() == b"\xff\xd8fakejpegbytes"


def test_new_object_with_no_fetcher_leaves_snapshot_unset(db):
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state})

    row = Event.select().get()
    assert row.has_snapshot is False
    assert row.snapshot_path is None


def test_object_removed_sets_end_time(db):
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)
    processor.process(camera, 100.0, {"obj1": state})

    processor.process(camera, 105.0, {})

    row = Event.get(Event.camera == "cam1")
    assert row.end_time is not None


def test_update_throttled_within_window(db):
    processor = EventProcessor()
    camera = _camera()
    state1 = _state("obj1", "person", 0.9, frame_time=100.0)
    processor.process(camera, 100.0, {"obj1": state1})

    row_before = Event.get(Event.camera == "cam1")
    original_score = row_before.score

    # Update 1 second later with a LOWER score (not a new top_score) -- should be
    # throttled (spec section 6.3: at most once per 5s unless top_score changed).
    state2 = _state("obj1", "person", 0.5, frame_time=101.0)
    processor.process(camera, 101.0, {"obj1": state2})

    row_after = Event.get(Event.camera == "cam1")
    assert row_after.score == original_score  # not updated, still throttled


def test_update_forced_immediately_when_top_score_increases(db):
    processor = EventProcessor()
    camera = _camera()
    state1 = _state("obj1", "person", 0.5, frame_time=100.0)
    processor.process(camera, 100.0, {"obj1": state1})

    state2 = _state("obj1", "person", 0.95, frame_time=100.5)  # higher score, same second
    processor.process(camera, 100.5, {"obj1": state2})

    row = Event.get(Event.camera == "cam1")
    assert row.top_score == 0.95
    assert row.score == 0.95


def test_update_allowed_after_throttle_window_elapses(db):
    processor = EventProcessor()
    camera = _camera()
    state1 = _state("obj1", "person", 0.7, frame_time=100.0)
    processor.process(camera, 100.0, {"obj1": state1})

    # 6 seconds later (past the 5s throttle window), even with same/lower score.
    state2 = _state("obj1", "person", 0.6, frame_time=106.0)
    processor.process(camera, 106.0, {"obj1": state2})

    row = Event.get(Event.camera == "cam1")
    assert row.score == 0.6


def test_false_positive_flag_reflects_state(db):
    processor = EventProcessor()
    camera = _camera()
    state1 = _state("obj1", "person", 0.9, frame_time=100.0, is_false_positive=True)
    processor.process(camera, 100.0, {"obj1": state1})

    state2 = _state("obj1", "person", 0.95, frame_time=100.5, is_false_positive=False)
    processor.process(camera, 100.5, {"obj1": state2})

    row = Event.get(Event.camera == "cam1")
    assert row.false_positive is False


def test_false_positive_flip_on_a_throttled_frame_still_persists_when_object_ends(db):
    """Regression test for a real bug found live against a real camera: an object's
    is_false_positive flip (true_positive -> after crossing the score threshold) could
    happen on a frame that _should_update_db throttles out of an immediate DB write --
    if the object then disappears (tracker loses it, e.g. person walks out of frame)
    BEFORE the next scheduled/heartbeat write fires, the flip was never persisted at
    all, since _on_end only ever wrote end_time, not false_positive. Confirmed live: 84
    real tracked people (some with top_score 0.85-0.92) all stayed false_positive=True
    in the database forever, despite the in-process lifecycle correctly computing
    is_false_positive=False well before they disappeared.
    """
    processor = EventProcessor()
    camera = _camera()

    # Start still false-positive.
    state1 = _state("obj1", "person", 0.9, frame_time=100.0, is_false_positive=True)
    processor.process(camera, 100.0, {"obj1": state1})

    # One second later it flips to a real (non-false-positive) detection -- but this is
    # WELL inside the 5s throttle window and top_score isn't increasing, so
    # _should_update_db returns False and no Event.update() call happens for this frame.
    state2 = _state("obj1", "person", 0.85, frame_time=101.0, is_false_positive=False)
    processor.process(camera, 101.0, {"obj1": state2})

    row_still_throttled = Event.get(Event.camera == "cam1")
    assert row_still_throttled.false_positive is True  # not yet written -- still throttled

    # The object disappears (tracker no longer reports it) before the throttle window
    # or heartbeat elapses -- this must NOT lose the false_positive=False flip.
    processor.process(camera, 101.5, {})

    row_final = Event.get(Event.camera == "cam1")
    assert row_final.false_positive is False
    assert row_final.end_time is not None


def test_multiple_cameras_tracked_independently(db):
    processor = EventProcessor()
    cam1 = _camera("cam1")
    cam2 = _camera("cam2")

    processor.process(cam1, 100.0, {"obj1": _state("obj1", "person", 0.9, frame_time=100.0)})
    processor.process(cam2, 100.0, {"obj2": _state("obj2", "car", 0.8, frame_time=100.0)})

    assert Event.select().where(Event.camera == "cam1").count() == 1
    assert Event.select().where(Event.camera == "cam2").count() == 1


def test_close_dangling_events_on_startup(db):
    import datetime

    # NOTE: store a naive UTC datetime, matching what EventProcessor itself writes (see
    # _to_datetime's docstring for why -- peewee's DateTimeField can't parse a
    # timezone-aware "+00:00"-suffixed string back out of SQLite).
    naive_utc_5min_ago = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(minutes=5)
    Event.create(
        id="dangling1", label="person", camera="cam1",
        start_time=naive_utc_5min_ago,
        end_time=None,
    )
    processor = EventProcessor()
    processor.close_dangling_events()

    row = Event.get(Event.id == "dangling1")
    assert row.end_time is not None
