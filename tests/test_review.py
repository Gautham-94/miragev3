from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    FfmpegConfig,
    ReviewConfig,
    ReviewLabelConfig,
)
from mirage.db.database import close_database, init_database
from mirage.db.models import ReviewSegment
from mirage.events.review import ReviewSegmentMaintainer, Severity, classify_severity, qualifies_for_review
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def _camera(name: str = "cam1") -> CameraConfig:
    return CameraConfig(
        name=name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        review=ReviewConfig(
            alerts=ReviewLabelConfig(labels=["person"]),
            detections=ReviewLabelConfig(labels=["car", "cat"]),
            cutoff_seconds=10,
        ),
    )


def _active_state(obj_id: str, label: str, frame_time: float = 0.0, is_false_positive: bool = False) -> TrackedObjectState:
    stationary = StationaryClassifier(threshold_frames=50)
    stationary.is_active = True
    return TrackedObjectState(
        id=obj_id, label=label, box=(0, 0, 10, 10), score=0.9, stationary=stationary, frame_time=frame_time,
        is_false_positive=is_false_positive,
    )


def _stationary_state(obj_id: str, label: str, frame_time: float = 0.0) -> TrackedObjectState:
    state = _active_state(obj_id, label, frame_time)
    state.stationary.motionless_count = 1000  # force is_stationary() True
    return state


def test_classify_severity_alert():
    camera = _camera()
    assert classify_severity("person", camera) == Severity.alert


def test_classify_severity_detection():
    camera = _camera()
    assert classify_severity("car", camera) == Severity.detection


def test_classify_severity_none_for_untracked_label():
    camera = _camera()
    assert classify_severity("dog", camera) is None


def test_qualifies_for_review_false_when_false_positive():
    state = _active_state("obj1", "person", is_false_positive=True)
    assert qualifies_for_review(state) is False


def test_qualifies_for_review_false_when_stationary():
    state = _stationary_state("obj1", "person")
    assert qualifies_for_review(state) is False


def test_classify_severity_rule_for_a_rule_prefixed_object_id():
    """A rule-triggered synthetic object (mirage.tracking.rules.RulesEngine) is always
    Severity.rule, regardless of its dynamic label text and regardless of whether that
    label happens to also be in the camera's alert/detection lists.
    """
    camera = _camera()
    assert classify_severity("crowd (12)", camera, obj_id="rule-crowd-cam1") == Severity.rule
    # Even a label that WOULD otherwise classify as "alert" -- the id prefix wins.
    assert classify_severity("person", camera, obj_id="rule-dwell-obj1") == Severity.rule


def test_qualifies_for_review_true_for_rule_object_even_if_stationary():
    """The one deliberate exception to the stationary-exclusion rule: dwell-time/
    loitering is specifically about objects that may not be moving much, so a
    rule-triggered synthetic object must NOT be excluded just because its underlying
    real object happens to be classified as stationary.
    """
    state = _stationary_state("rule-dwell-obj1", "loitering (65s)")
    assert qualifies_for_review(state, obj_id="rule-dwell-obj1") is True


def test_qualifies_for_review_true_when_active_and_not_false_positive():
    state = _active_state("obj1", "person", is_false_positive=False)
    assert qualifies_for_review(state) is True


def test_new_alert_object_creates_review_segment(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state = _active_state("obj1", "person", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": state})

    rows = list(ReviewSegment.select().where(ReviewSegment.camera == "cam1"))
    assert len(rows) == 1
    assert rows[0].severity == "alert"
    assert rows[0].end_time is None


def test_segment_start_captures_thumbnail_via_injected_fetcher(db):
    with tempfile.TemporaryDirectory() as thumb_dir:
        fetched_cameras = []

        def fake_fetcher(camera_name: str) -> bytes:
            fetched_cameras.append(camera_name)
            return b"\xff\xd8fakejpegbytes"

        maintainer = ReviewSegmentMaintainer(cutoff_seconds=10, thumbnail_fetcher=fake_fetcher, thumb_dir=thumb_dir)
        camera = _camera()
        state = _active_state("obj1", "person", frame_time=100.0)

        maintainer.process(camera, 100.0, {"obj1": state})

        assert fetched_cameras == ["cam1"]
        row = ReviewSegment.select().where(ReviewSegment.camera == "cam1").get()
        assert row.thumb_path is not None
        assert Path(row.thumb_path).exists()
        assert Path(row.thumb_path).read_bytes() == b"\xff\xd8fakejpegbytes"


def test_segment_start_with_no_fetcher_leaves_thumb_path_none(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state = _active_state("obj1", "person", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": state})

    row = ReviewSegment.select().where(ReviewSegment.camera == "cam1").get()
    assert row.thumb_path is None


def test_segment_start_thumbnail_fetch_failure_does_not_break_segment_creation(db):
    def failing_fetcher(camera_name: str) -> bytes:
        raise ConnectionError("go2rtc unreachable")

    with tempfile.TemporaryDirectory() as thumb_dir:
        maintainer = ReviewSegmentMaintainer(cutoff_seconds=10, thumbnail_fetcher=failing_fetcher, thumb_dir=thumb_dir)
        camera = _camera()
        state = _active_state("obj1", "person", frame_time=100.0)

        maintainer.process(camera, 100.0, {"obj1": state})

        row = ReviewSegment.select().where(ReviewSegment.camera == "cam1").get()
        assert row.thumb_path is None
        assert row.severity == "alert"


def test_detection_only_object_creates_detection_severity_segment(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state = _active_state("obj1", "car", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": state})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "detection"


def test_detection_segment_upgrades_to_alert_when_alert_object_appears(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(camera, 100.0, {"car1": _active_state("car1", "car", 100.0)})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "detection"

    maintainer.process(
        camera, 101.0,
        {"car1": _active_state("car1", "car", 101.0), "person1": _active_state("person1", "person", 101.0)},
    )
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "alert"


def test_rule_triggered_object_creates_a_rule_severity_segment(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state = _active_state("rule-crowd-cam1", "crowd (12)", frame_time=100.0)

    maintainer.process(camera, 100.0, {"rule-crowd-cam1": state})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "rule"
    assert "crowd (12)" in row.data["objects"]


def test_rule_label_updates_in_place_instead_of_accumulating_every_second(db):
    """Regression test for a REAL bug the user caught live via a screenshot: a
    loitering alert's card showed "Loitering (120s), Loitering (121s), Loitering
    (122s), ..." -- dozens of near-identical entries, one per second, instead of a
    single entry whose count keeps updating. Caused by objects being a plain set of
    label STRINGS with no concept of "this is the same underlying rule trigger, just a
    later reading" -- fixed by keying rule-triggered labels by obj_id
    (PendingReviewSegment.rule_labels) so the SAME entry updates in place.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    for elapsed in range(120, 178):
        state = _active_state("rule-dwell-person1", f"loitering ({elapsed}s)", frame_time=100.0 + elapsed)
        maintainer.process(camera, 100.0 + elapsed, {"rule-dwell-person1": state})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    loitering_entries = [o for o in row.data["objects"] if "loitering" in o]
    # Exactly ONE entry for this rule trigger, holding the MOST RECENT count -- not 58
    # (one per second processed above).
    assert loitering_entries == ["loitering (177s)"]


def test_rule_label_and_a_real_object_label_coexist_correctly(db):
    """A real object's label (e.g. "person") still accumulates in the historical
    `objects` set as before -- only rule-triggered labels get the update-in-place
    treatment, so this must not regress the pre-existing behavior for real labels.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(
        camera, 100.0,
        {
            "person1": _active_state("person1", "person", 100.0),
            "rule-dwell-person1": _active_state("rule-dwell-person1", "loitering (120s)", 100.0),
        },
    )
    maintainer.process(
        camera, 101.0,
        {
            "person1": _active_state("person1", "person", 101.0),
            "rule-dwell-person1": _active_state("rule-dwell-person1", "loitering (121s)", 101.0),
        },
    )

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert "person" in row.data["objects"]
    assert "loitering (121s)" in row.data["objects"]
    assert "loitering (120s)" not in row.data["objects"]


def test_alert_segment_upgrades_to_rule_when_a_rule_object_appears(db):
    """rule outranks alert (see mirage.events.review._SEVERITY_RANK) -- an
    already-open alert segment must upgrade to rule severity, never the reverse.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(camera, 100.0, {"person1": _active_state("person1", "person", 100.0)})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "alert"

    maintainer.process(
        camera, 101.0,
        {
            "person1": _active_state("person1", "person", 101.0),
            "rule-dwell-person1": _active_state("rule-dwell-person1", "loitering (65s)", 101.0),
        },
    )
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "rule"


def test_rule_segment_never_downgrades_back_to_alert(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(
        camera, 100.0,
        {
            "person1": _active_state("person1", "person", 100.0),
            "rule-crowd-cam1": _active_state("rule-crowd-cam1", "crowd (5)", 100.0),
        },
    )
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "rule"

    # Rule condition clears (crowd drops below threshold), but the person alert is
    # still active -- severity must NOT drop back down to "alert".
    maintainer.process(camera, 101.0, {"person1": _active_state("person1", "person", 101.0)})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.severity == "rule"


def test_segment_closes_after_cutoff_with_no_activity(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(camera, 100.0, {"obj1": _active_state("obj1", "person", 100.0)})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.end_time is None

    # No qualifying activity, but within cutoff -- stays open.
    maintainer.process(camera, 105.0, {})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.end_time is None

    # Past cutoff (10s) with no activity -- should close.
    maintainer.process(camera, 111.0, {})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.end_time is not None


def test_new_segment_starts_after_previous_one_closes(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()

    maintainer.process(camera, 100.0, {"obj1": _active_state("obj1", "person", 100.0)})
    maintainer.process(camera, 111.0, {})  # closes segment 1

    maintainer.process(camera, 120.0, {"obj2": _active_state("obj2", "person", 120.0)})

    rows = list(ReviewSegment.select().where(ReviewSegment.camera == "cam1"))
    assert len(rows) == 2
    assert rows[0].id != rows[1].id


def test_non_qualifying_objects_do_not_start_a_segment(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    stationary_person = _stationary_state("obj1", "person", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": stationary_person})

    assert ReviewSegment.select().where(ReviewSegment.camera == "cam1").count() == 0


def test_false_positive_objects_do_not_start_a_segment(db):
    """Regression test: qualifies_for_review's false-positive check was completely dead
    code in production before this fix -- ReviewSegmentMaintainer.process's own
    get_lifecycle callback always returned None (see mirage.app's old
    _get_lifecycle_stub), so `lifecycle is not None and lifecycle.is_false_positive` was
    always False, meaning review segments included every tracked object regardless of
    its false-positive status. This checks the fixed path (state.is_false_positive read
    directly, no cross-process lifecycle lookup needed) actually excludes them.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    false_positive_person = _active_state("obj1", "person", frame_time=100.0, is_false_positive=True)

    maintainer.process(camera, 100.0, {"obj1": false_positive_person})

    assert ReviewSegment.select().where(ReviewSegment.camera == "cam1").count() == 0


def test_data_payload_includes_contributing_objects_and_labels(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state = _active_state("obj1", "person", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": state})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert "obj1" in row.data["detections"]
    assert "person" in row.data["objects"]


# --------------------------------------------------------------------------------------
# thumb_boxes -- captured ONCE, at the exact instant the segment's thumbnail itself is
# taken (in _start_segment), so a box drawn on the Review page's thumbnail lines up with
# where the object actually was in that frozen frame -- not wherever it later moved to.
# --------------------------------------------------------------------------------------


def test_thumb_boxes_captured_at_segment_start_normalized_to_frame_shape(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()  # default DetectConfig: 640x480
    state = _active_state("obj1", "person", frame_time=100.0)
    state.box = (64.0, 36.0, 320.0, 180.0)  # (x1, y1, x2, y2) pixel coords

    maintainer.process(camera, 100.0, {"obj1": state})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert "obj1" in row.data["thumb_boxes"]
    entry = row.data["thumb_boxes"]["obj1"]
    assert entry["label"] == "person"
    # normalized against width=640, height=480
    assert entry["box"] == pytest.approx([64.0 / 640, 36.0 / 480, 320.0 / 640, 180.0 / 480])


def test_thumb_boxes_excludes_stationary_and_false_positive_objects(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    qualifying = _active_state("obj1", "person", frame_time=100.0)
    non_qualifying = _stationary_state("obj2", "person", frame_time=100.0)

    maintainer.process(camera, 100.0, {"obj1": qualifying, "obj2": non_qualifying})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert "obj1" in row.data["thumb_boxes"]
    assert "obj2" not in row.data["thumb_boxes"]


def test_thumb_boxes_excludes_non_review_labels(db):
    """A tracked object whose label isn't in either alert_labels or detection_labels
    doesn't qualify for review at all (classify_severity returns None) -- it must not
    get a thumb_box either, even if some OTHER object in the same frame does qualify.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    person = _active_state("person1", "person", frame_time=100.0)
    dog = _active_state("dog1", "dog", frame_time=100.0)  # "dog" is in neither label list

    maintainer.process(camera, 100.0, {"person1": person, "dog1": dog})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert "person1" in row.data["thumb_boxes"]
    assert "dog1" not in row.data["thumb_boxes"]


def test_thumb_boxes_stay_frozen_from_segment_start_even_as_object_moves(db):
    """The whole point of thumb_boxes: it must NOT be re-captured/updated on later
    process() calls, since the thumbnail image itself is only ever taken once (at
    segment start) -- a box that kept tracking the object's LATEST position would show
    it somewhere it had already moved away from by the time that frozen frame was shot.
    """
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    state_at_start = _active_state("obj1", "person", frame_time=100.0)
    state_at_start.box = (0.0, 0.0, 50.0, 50.0)

    maintainer.process(camera, 100.0, {"obj1": state_at_start})
    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    original_box = row.data["thumb_boxes"]["obj1"]["box"]

    state_later = _active_state("obj1", "person", frame_time=105.0)
    state_later.box = (400.0, 200.0, 500.0, 300.0)  # object has moved a lot
    maintainer.process(camera, 105.0, {"obj1": state_later})

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.data["thumb_boxes"]["obj1"]["box"] == original_box


def test_close_all_pending_force_closes_open_segments(db):
    maintainer = ReviewSegmentMaintainer(cutoff_seconds=10)
    camera = _camera()
    maintainer.process(camera, 100.0, {"obj1": _active_state("obj1", "person", 100.0)})

    maintainer.close_all_pending(frame_time=105.0)

    row = ReviewSegment.get(ReviewSegment.camera == "cam1")
    assert row.end_time is not None
