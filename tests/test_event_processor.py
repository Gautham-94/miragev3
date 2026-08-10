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


def _real_jpeg(width: int = 100, height: int = 80) -> bytes:
    import cv2
    import numpy as np

    image = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_new_object_with_frame_jpeg_writes_clean_frame_and_never_calls_the_live_fetcher(db):
    """Regression test for a real bug: the old snapshot path called a LIVE go2rtc
    fetch independently of when the tracker actually computed the box, so for a
    moving object the box could land nowhere near its real position by the time that
    later frame was captured (confirmed against real footage -- see
    TODO_FIX_LIST.md). When frame_jpeg is available (the actual frame the box was
    computed from, passed synchronously from CameraTracker's own process), it must be
    used INSTEAD of the live fetcher -- proven here by never calling the fetcher at
    all when frame_jpeg is provided. The saved file is the CLEAN frame (unmodified,
    byte-for-byte identical to frame_jpeg) -- boxes are rendered later, on demand, by
    the API layer (mirage/api/routers/events.py), not burned in here -- see
    mirage/util/thumbnail.py's own module docstring for why (mirrors Frigate's design).
    """
    with tempfile.TemporaryDirectory() as thumb_dir:
        fetcher_called = []

        def fetcher_that_must_not_be_called(camera_name: str) -> bytes:
            fetcher_called.append(camera_name)
            return b"\xff\xd8livefetchbytes"

        processor = EventProcessor(thumbnail_fetcher=fetcher_that_must_not_be_called, thumb_dir=thumb_dir)
        camera = _camera()
        state = _state("obj1", "person", 0.9, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0)
        frame_jpeg = _real_jpeg()

        processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=frame_jpeg)

        assert fetcher_called == []
        row = Event.select().get()
        assert row.has_snapshot is True
        assert Path(row.snapshot_path).exists()
        # the CLEAN frame, byte-for-byte -- not the live-fetch's bytes, and not boxed
        assert Path(row.snapshot_path).read_bytes() == frame_jpeg


def test_new_object_without_frame_jpeg_falls_back_to_live_fetcher(db):
    with tempfile.TemporaryDirectory() as thumb_dir:
        fetched = []

        def fake_fetcher(camera_name: str) -> bytes:
            fetched.append(camera_name)
            return b"\xff\xd8fakejpegbytes"

        processor = EventProcessor(thumbnail_fetcher=fake_fetcher, thumb_dir=thumb_dir)
        camera = _camera()
        state = _state("obj1", "person", 0.9, frame_time=100.0)

        processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=None)

        assert fetched == ["cam1"]


def test_new_event_stores_snapshot_boxes_for_every_confirmed_object_in_frame(db):
    """Real bug the user found: a snapshot only ever showed a box for the ONE object
    that triggered this particular Event, even when multiple other people were
    clearly visible and already confirmed in the same frame. The fix stores every
    Gate-1-confirmed object's box in Event.data["snapshot_boxes"] (rendered on demand
    by mirage/api/routers/events.py, not burned into the file here -- see
    mirage/util/thumbnail.py's module docstring), not just the triggering one.
    """
    with tempfile.TemporaryDirectory() as thumb_dir:
        processor = EventProcessor(thumb_dir=thumb_dir)
        camera = _camera()

        # obj1 is the NEW object that triggers this Event's creation; obj2 is a
        # DIFFERENT, already-confirmed object visible in the same frame.
        new_state = _state("obj1", "person", 0.9, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0, is_false_positive=False)
        other_state = _state("obj2", "person", 0.85, box=(100.0, 80.0, 180.0, 140.0), frame_time=100.0, is_false_positive=False)

        processor.process(camera, 100.0, {"obj1": new_state, "obj2": other_state}, frame_jpeg=_real_jpeg())

        row = Event.select().where(Event.camera == "cam1").get()
        boxes = row.data["snapshot_boxes"]
        assert len(boxes) == 2
        assert {"label": "person", "box": [10.0, 10.0, 50.0, 60.0]} in boxes
        assert {"label": "person", "box": [100.0, 80.0, 180.0, 140.0]} in boxes


def test_new_event_snapshot_boxes_excludes_unconfirmed_objects(db):
    """An object still mid initialization_delay (is_false_positive still True) must
    NOT appear in snapshot_boxes -- it might still turn out to be noise.
    """
    with tempfile.TemporaryDirectory() as thumb_dir:
        processor = EventProcessor(thumb_dir=thumb_dir)
        camera = _camera()

        confirmed = _state("obj1", "person", 0.9, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0, is_false_positive=False)
        unconfirmed = _state("obj2", "person", 0.3, box=(100.0, 80.0, 180.0, 140.0), frame_time=100.0, is_false_positive=True)

        processor.process(camera, 100.0, {"obj1": confirmed, "obj2": unconfirmed}, frame_jpeg=_real_jpeg())

        row = Event.select().where(Event.camera == "cam1").get()
        boxes = row.data["snapshot_boxes"]
        assert len(boxes) == 1
        assert boxes[0]["box"] == [10.0, 10.0, 50.0, 60.0]


def test_snapshot_boxes_survive_a_later_throttled_update(db):
    """Real bug caught live: Event.update(data=...) REPLACES the whole JSON field, so
    an _on_update call writing {"box": ...} alone silently wiped out snapshot_boxes
    that _on_start had just set -- a fresh Event's snapshot had a correct box
    immediately after creation, then no box at all by the time its track ended. Must
    survive every subsequent update, unchanged (see _on_update's own comment for why
    it must stay UNCHANGED, not recomputed from the later frame's box position).
    """
    with tempfile.TemporaryDirectory() as thumb_dir:
        processor = EventProcessor(thumb_dir=thumb_dir)
        camera = _camera()

        state1 = _state("obj1", "person", 0.5, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0, is_false_positive=False)
        processor.process(camera, 100.0, {"obj1": state1}, frame_jpeg=_real_jpeg())

        row_after_start = Event.select().where(Event.camera == "cam1").get()
        assert row_after_start.data["snapshot_boxes"] == [{"label": "person", "box": [10.0, 10.0, 50.0, 60.0]}]

        # Force an immediate DB write on the next frame (top_score increases, so
        # _should_update_db returns True even though we're well inside the 5s
        # throttle window) -- this is the exact call that used to wipe snapshot_boxes.
        state2 = _state("obj1", "person", 0.95, box=(400.0, 300.0, 450.0, 350.0), frame_time=100.5, is_false_positive=False)
        processor.process(camera, 100.5, {"obj1": state2})

        row_after_update = Event.select().where(Event.camera == "cam1").get()
        # snapshot_boxes preserved EXACTLY as it was at creation -- NOT recomputed
        # from state2's new (400, 300, 450, 350) box, since the saved snapshot IMAGE
        # is still the original frame from creation time, never re-captured.
        assert row_after_update.data["snapshot_boxes"] == [{"label": "person", "box": [10.0, 10.0, 50.0, 60.0]}]
        assert row_after_update.top_score == 0.95  # the actual update DID apply


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


# --------------------------------------------------------------------------------------
# Mirage V3 async species classification -- Event.data species_* defaults, dispatch
# gating (animal/bird only), and the _on_update read-modify-write race fix.
# --------------------------------------------------------------------------------------


def test_person_event_species_status_is_not_applicable(db):
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state})

    row = Event.select().get()
    assert row.data["species_status"] == "not_applicable"
    assert row.data["species"] is None


def test_animal_event_with_no_dispatcher_is_skipped_not_pending(db):
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "animal", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=_real_jpeg())

    row = Event.select().get()
    assert row.data["species_status"] == "skipped"


def test_animal_event_without_frame_jpeg_is_skipped(db):
    """No synchronously-available frame to crop (e.g. the capture_thumbnail live-fetch
    fallback path) -- can't dispatch, so species_status must be "skipped", not stuck
    claiming "pending" forever with nothing ever going to complete it.
    """
    processor = EventProcessor()
    camera = _camera()
    state = _state("obj1", "bird", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=None)

    row = Event.select().get()
    assert row.data["species_status"] == "skipped"


def test_animal_event_with_dispatcher_dispatches_a_crop_and_is_pending(db):
    dispatched = []

    class FakeDispatcher:
        def dispatch(self, event_id, camera_name, label, crop_jpeg):
            dispatched.append((event_id, camera_name, label, crop_jpeg))

    processor = EventProcessor()
    processor.species_dispatcher = FakeDispatcher()
    camera = _camera()
    state = _state("obj1", "animal", 0.9, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=_real_jpeg())

    row = Event.select().get()
    assert row.data["species_status"] == "pending"
    assert len(dispatched) == 1
    event_id, camera_name, label, crop_jpeg = dispatched[0]
    assert event_id == row.id
    assert camera_name == "cam1"
    assert label == "animal"
    assert crop_jpeg  # non-empty encoded jpeg bytes


def test_person_event_never_dispatches_even_with_a_dispatcher_configured(db):
    dispatched = []

    class FakeDispatcher:
        def dispatch(self, event_id, camera_name, label, crop_jpeg):
            dispatched.append(event_id)

    processor = EventProcessor()
    processor.species_dispatcher = FakeDispatcher()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state}, frame_jpeg=_real_jpeg())

    assert dispatched == []
    row = Event.select().get()
    assert row.data["species_status"] == "not_applicable"


def test_species_fields_survive_a_later_throttled_update(db):
    """The core race this fix closes: Event.update(data=...) in _on_update must be a
    read-modify-write off the CURRENT row, not a blind literal -- otherwise a
    throttled/heartbeat update landing AFTER the species worker has already stamped a
    real classification onto this row would silently wipe it back to pending/None.
    """
    processor = EventProcessor()
    camera = _camera()

    state1 = _state("obj1", "animal", 0.5, box=(10.0, 10.0, 50.0, 60.0), frame_time=100.0, is_false_positive=False)
    processor.process(camera, 100.0, {"obj1": state1}, frame_jpeg=_real_jpeg())

    row = Event.select().where(Event.camera == "cam1").get()
    event_id = row.id

    # Simulate the species worker completing asynchronously, via its own read-modify-
    # write Event.update() (mirrors mirage.species.dispatcher's real implementation).
    completed_row = Event.get(Event.id == event_id)
    merged = {**completed_row.data, "species": "Odocoileus virginianus", "species_status": "complete", "species_confidence": 0.93}
    Event.update(data=merged).where(Event.id == event_id).execute()

    # Now a throttled/heartbeat EventProcessor update fires for the SAME event (higher
    # score forces an immediate write past the throttle window) -- this must NOT
    # clobber the species fields just written above.
    state2 = _state("obj1", "animal", 0.95, box=(400.0, 300.0, 450.0, 350.0), frame_time=100.5, is_false_positive=False)
    processor.process(camera, 100.5, {"obj1": state2})

    row_after = Event.get(Event.id == event_id)
    assert row_after.data["species"] == "Odocoileus virginianus"
    assert row_after.data["species_status"] == "complete"
    assert row_after.data["species_confidence"] == 0.93
    assert row_after.top_score == 0.95  # the actual box/score update DID apply
    assert row_after.data["snapshot_boxes"] == row.data["snapshot_boxes"]  # unchanged, per existing behavior


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


def test_new_object_notifies_when_notify_queue_is_wired(db):
    import queue as queue_module

    processor = EventProcessor()
    processor.notify_queue = queue_module.Queue()
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state})

    event = Event.get()
    notified = processor.notify_queue.get_nowait()
    assert notified.table == "event"
    assert notified.id == event.id
    assert notified.op == "create"


def test_new_object_with_no_notify_queue_does_not_raise(db):
    processor = EventProcessor()  # notify_queue stays None
    camera = _camera()
    state = _state("obj1", "person", 0.9, frame_time=100.0)

    processor.process(camera, 100.0, {"obj1": state})  # should not raise

    assert Event.select().count() == 1
