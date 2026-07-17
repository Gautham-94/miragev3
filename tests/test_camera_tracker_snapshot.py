"""Tests for mirage.tracking.camera_tracker's frame-snapshot-encoding helpers --
_has_qualifying_object (cheap gate) and _maybe_encode_frame (the actual JPEG encode).

This is the fix for a real bug: the old snapshot path re-fetched a frame LIVE from
go2rtc, independently of when the tracker actually computed a tracked object's box --
for a moving object, the box could land nowhere near its real position in that later,
unrelated frame. The fix encodes the box's own SAME frame synchronously, inside
CameraTracker's own process, before its SHM slot can be recycled -- these tests cover
the gating logic (only encode when there's real activity, to avoid the real CPU cost of
encoding every single frame at 5-10fps/camera) and the encode itself.
"""

from __future__ import annotations

import cv2
import numpy as np

from mirage.tracking.camera_tracker import _has_qualifying_object, _maybe_encode_frame
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState


def _state(obj_id: str, is_false_positive: bool) -> TrackedObjectState:
    return TrackedObjectState(
        id=obj_id, label="person", box=(0, 0, 10, 10), score=0.9,
        stationary=StationaryClassifier(threshold_frames=50), is_false_positive=is_false_positive,
    )


def _yuv_frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height * 3 // 2, width), 128, dtype=np.uint8)


def test_has_qualifying_object_true_when_any_object_passed_gate_1():
    tracked = {"obj1": _state("obj1", is_false_positive=False)}
    assert _has_qualifying_object(tracked) is True


def test_has_qualifying_object_false_when_all_objects_are_false_positive():
    tracked = {"obj1": _state("obj1", is_false_positive=True)}
    assert _has_qualifying_object(tracked) is False


def test_has_qualifying_object_false_when_no_objects_at_all():
    assert _has_qualifying_object({}) is False


def test_has_qualifying_object_true_if_at_least_one_of_several_qualifies():
    tracked = {
        "obj1": _state("obj1", is_false_positive=True),
        "obj2": _state("obj2", is_false_positive=False),
    }
    assert _has_qualifying_object(tracked) is True


def test_maybe_encode_frame_returns_none_when_nothing_qualifies():
    tracked = {"obj1": _state("obj1", is_false_positive=True)}
    result = _maybe_encode_frame(_yuv_frame(), (48, 64), tracked)
    assert result is None


def test_maybe_encode_frame_returns_real_jpeg_bytes_when_something_qualifies():
    tracked = {"obj1": _state("obj1", is_false_positive=False)}
    result = _maybe_encode_frame(_yuv_frame(64, 48), (48, 64), tracked)

    assert result is not None
    assert isinstance(result, bytes)
    # A real, valid JPEG -- decodable back to the same frame dimensions.
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (48, 64)


def test_maybe_encode_frame_returns_none_for_empty_tracked_objects():
    result = _maybe_encode_frame(_yuv_frame(), (48, 64), {})
    assert result is None
