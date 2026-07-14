from __future__ import annotations

import numpy as np

from mirage.tracking.distance import box_to_points, points_to_box, tracking_distance
from mirage.tracking.stationary import StationaryClassifier, iou
from mirage.tracking.tracker import ObjectTracker


def test_box_to_points_and_back_roundtrip():
    box = (10.0, 20.0, 110.0, 220.0)
    points = box_to_points(box)
    assert points.shape == (2, 2)
    assert points_to_box(points) == box


def test_tracking_distance_zero_for_identical_boxes():
    box = box_to_points((10, 10, 50, 50))
    assert tracking_distance(box, box) == 0.0


def test_tracking_distance_increases_with_position_delta():
    est = box_to_points((10, 10, 50, 50))
    near = box_to_points((15, 10, 55, 50))
    far = box_to_points((100, 10, 140, 50))
    assert tracking_distance(near, est) < tracking_distance(far, est)


def test_tracking_distance_increases_with_size_ratio_change():
    est = box_to_points((10, 10, 50, 50))  # 40x40
    same_size = box_to_points((10, 10, 50, 50))
    much_bigger = box_to_points((10, 10, 200, 200))  # same position, very different size
    assert tracking_distance(same_size, est) < tracking_distance(much_bigger, est)


def test_iou_identical_boxes():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    result = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < result < 1.0


def test_stationary_classifier_flags_stationary_after_threshold_frames():
    clf = StationaryClassifier(threshold_frames=5)
    box = (100, 100, 150, 150)
    for _ in range(10):
        clf.update(box)  # identical box every frame -> IoU=1.0 -> stationary
    assert clf.is_stationary() is True


def test_stationary_classifier_stays_active_with_movement():
    clf = StationaryClassifier(threshold_frames=5)
    x = 100
    for _ in range(10):
        clf.update((x, 100, x + 50, 150))
        x += 30  # moves enough each frame to keep IoU low
    assert clf.is_stationary() is False


def test_stationary_classifier_expires_after_max_frames():
    clf = StationaryClassifier(threshold_frames=2, max_frames=3)
    box = (100, 100, 150, 150)
    for _ in range(3):
        clf.update(box)
    assert clf.is_expired() is False
    for _ in range(5):
        clf.update(box)
    assert clf.is_expired() is True


def test_stationary_classifier_never_expires_without_max_frames():
    clf = StationaryClassifier(threshold_frames=2, max_frames=None)
    box = (100, 100, 150, 150)
    for _ in range(100):
        clf.update(box)
    assert clf.is_expired() is False


def _sequence_of_boxes(start_x: int, count: int, step: int = 5) -> list[tuple[int, int, int, int]]:
    return [(start_x + i * step, 100, start_x + i * step + 50, 150) for i in range(count)]


def test_object_tracker_assigns_persistent_id_across_frames():
    tracker = ObjectTracker(fps=5)
    boxes = _sequence_of_boxes(start_x=100, count=6)

    seen_ids = set()
    for i, box in enumerate(boxes):
        states = tracker.update(frame_time=float(i), detections=[("person", 0.9, box)])
        seen_ids.update(states.keys())

    # min_initialized for fps=5 is max(5//2,2)=2, so the object should be confirmed by
    # frame 2 at the latest and keep the SAME id thereafter.
    assert len(seen_ids) == 1


def test_object_tracker_different_labels_tracked_independently():
    tracker = ObjectTracker(fps=5)
    for i in range(6):
        states = tracker.update(
            frame_time=float(i),
            detections=[("person", 0.9, (100 + i * 2, 100, 150 + i * 2, 150)), ("car", 0.8, (300 + i * 2, 300, 400 + i * 2, 400))],
        )
    labels = {s.label for s in states.values()}
    assert labels == {"person", "car"}
    assert len(states) == 2


def test_object_tracker_object_disappears_and_is_removed_after_hit_counter_expires():
    tracker = ObjectTracker(fps=5)
    boxes = _sequence_of_boxes(start_x=100, count=6)
    for i, box in enumerate(boxes):
        tracker.update(frame_time=float(i), detections=[("person", 0.9, box)])

    # max_disappeared for fps=5 is 5*5=25 frames of no detections before norfair drops
    # the track. Feed enough empty frames to exceed that.
    last_states = {}
    for i in range(6, 6 + 30):
        last_states = tracker.update(frame_time=float(i), detections=[])

    assert last_states == {}


def test_object_tracker_stationary_object_flagged_after_holding_still():
    tracker = ObjectTracker(fps=5)
    box = (100, 100, 150, 150)
    states = {}
    for i in range(60):  # stationary_threshold_frames(5) = 50
        states = tracker.update(frame_time=float(i), detections=[("person", 0.9, box)])

    assert len(states) == 1
    state = next(iter(states.values()))
    assert state.stationary.is_stationary() is True


def test_object_tracker_moving_object_never_flagged_stationary():
    tracker = ObjectTracker(fps=5)
    boxes = _sequence_of_boxes(start_x=100, count=60, step=10)
    states = {}
    for i, box in enumerate(boxes):
        states = tracker.update(frame_time=float(i), detections=[("person", 0.9, box)])

    assert len(states) == 1
    state = next(iter(states.values()))
    assert state.stationary.is_stationary() is False


def test_object_tracker_expiry_removes_permanently_stationary_object():
    """spec section 5.7: a label configured with a max-stationary-frame limit gets
    force-deregistered once motionless_count exceeds threshold+max_frames, even though
    the object is technically still "there" (still being detected every frame). Since
    detections keep arriving at the same position, a brand-new track will legitimately
    re-form a few frames later (indistinguishable, from the tracker's perspective, from a
    genuinely new object appearing in the same spot) -- this test asserts the original
    track's id is dropped at least once, not that the tracker stays permanently empty.
    """
    tracker = ObjectTracker(fps=5, stationary_max_frames={"person": 5})
    box = (100, 100, 150, 150)
    seen_ids: list[str] = []
    saw_empty_state = False
    for i in range(70):
        states = tracker.update(frame_time=float(i), detections=[("person", 0.9, box)])
        if not states:
            saw_empty_state = True
        else:
            seen_ids.append(next(iter(states.values())).id)

    assert saw_empty_state, "expected the permanently-stationary track to be force-expired at least once"
    # Confirm it's actually a NEW id after expiry, not the same one somehow surviving.
    assert len(set(seen_ids)) >= 2, f"expected at least 2 distinct track ids (original + re-formed), got {set(seen_ids)}"
