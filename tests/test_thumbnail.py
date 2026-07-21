"""Unit tests for the shared thumbnail-capture helper (mirage/util/thumbnail.py), used by
both EventProcessor and ReviewSegmentMaintainer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from mirage.util.thumbnail import capture_thumbnail, crop_jpeg_to_box, draw_boxes_on_jpeg_bytes, write_clean_snapshot


def test_capture_thumbnail_writes_file_and_returns_path():
    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(lambda cam: b"\xff\xd8jpeg", thumb_dir, "seg1", "cam1")

        assert result == str(Path(thumb_dir) / "seg1.jpg")
        assert Path(result).read_bytes() == b"\xff\xd8jpeg"


def test_capture_thumbnail_no_fetcher_returns_none():
    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(None, thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_fetcher_returns_empty_bytes_returns_none():
    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(lambda cam: b"", thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_fetcher_raises_returns_none_not_propagated():
    def failing(cam):
        raise ConnectionError("go2rtc unreachable")

    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(failing, thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_retries_once_on_transient_empty_response():
    # Regression test: go2rtc's frame.jpeg endpoint can return an empty body during a
    # brief transient window (observed for real during end-to-end validation -- see
    # IMPLEMENTATION_NOTES.md). A single retry should recover a thumbnail that only
    # fails on the very first attempt.
    calls = []

    def flaky(cam):
        calls.append(cam)
        if len(calls) == 1:
            return b""  # transient empty response, matching go2rtc's real behavior
        return b"\xff\xd8jpeg"

    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(flaky, thumb_dir, "seg1", "cam1")

        assert len(calls) == 2
        assert result == str(Path(thumb_dir) / "seg1.jpg")
        assert Path(result).read_bytes() == b"\xff\xd8jpeg"


def test_capture_thumbnail_gives_up_after_two_failed_attempts():
    calls = []

    def always_empty(cam):
        calls.append(cam)
        return b""

    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(always_empty, thumb_dir, "seg1", "cam1")

        assert len(calls) == 2
        assert result is None


# --------------------------------------------------------------------------------------
# write_clean_snapshot -- writes a JPEG to disk completely unmodified. The saved file is
# ALWAYS the clean, unannotated frame (mirrors Frigate's own confirmed design, see
# mirage/util/thumbnail.py's module docstring) -- boxes are never burned in here.
# --------------------------------------------------------------------------------------


def _real_jpeg(width: int = 100, height: int = 80) -> bytes:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_write_clean_snapshot_writes_the_exact_same_bytes(tmp_path):
    frame_jpeg = _real_jpeg()
    result = write_clean_snapshot(frame_jpeg, str(tmp_path), "ev1")

    assert result == str(tmp_path / "ev1.jpg")
    assert Path(result).read_bytes() == frame_jpeg  # byte-for-byte unmodified, no drawing


def test_write_clean_snapshot_creates_missing_directory(tmp_path):
    thumb_dir = tmp_path / "nested" / "dir"
    result = write_clean_snapshot(_real_jpeg(), str(thumb_dir), "ev1")

    assert result is not None
    assert Path(result).exists()


# --------------------------------------------------------------------------------------
# draw_boxes_on_jpeg_bytes -- renders boxes on demand from already-in-memory bytes,
# returning NEW bytes (never touches disk) -- called at HTTP-request time by
# mirage/api/routers/events.py's snapshot endpoint, fixing a real bug: burning a box in
# once at Event-creation time couldn't handle an open-vocab match, which is produced by
# a completely separate process with no shared frame to burn into.
# --------------------------------------------------------------------------------------


def test_draw_boxes_on_jpeg_bytes_returns_a_real_decodable_image():
    frame_jpeg = _real_jpeg()
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, [("person", (10.0, 10.0, 50.0, 60.0))])

    assert result is not None
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (80, 100)  # same dimensions as the source frame


def test_draw_boxes_on_jpeg_bytes_does_not_mutate_the_input_bytes():
    frame_jpeg = _real_jpeg()
    original = bytes(frame_jpeg)
    draw_boxes_on_jpeg_bytes(frame_jpeg, [("person", (10.0, 10.0, 50.0, 60.0))])

    assert frame_jpeg == original  # operates on a decoded COPY, input bytes untouched


def test_draw_boxes_on_jpeg_bytes_actually_draws_a_visible_box():
    """Confirms the box is REALLY drawn into the pixels, not just re-encoded unchanged
    -- picks a pixel exactly on the box's border (where cv2.rectangle draws) and checks
    it differs from the plain gray background.
    """
    frame_jpeg = _real_jpeg(width=100, height=80)
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, [("person", (10.0, 10.0, 50.0, 60.0))])

    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    border_pixel = decoded[10, 30]  # top edge of the box, well inside its width
    background_pixel = decoded[70, 90]  # far corner, untouched by any box
    assert not np.array_equal(border_pixel, background_pixel)


def test_draw_boxes_on_jpeg_bytes_draws_multiple_boxes():
    frame_jpeg = _real_jpeg(width=200, height=150)
    boxes = [("person", (10.0, 10.0, 50.0, 60.0)), ("car", (100.0, 80.0, 180.0, 140.0))]
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, boxes)

    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    box1_border = decoded[10, 30]
    box2_border = decoded[80, 140]
    background = decoded[0, 0]
    assert not np.array_equal(box1_border, background)
    assert not np.array_equal(box2_border, background)


def test_draw_boxes_on_jpeg_bytes_clamps_out_of_bounds_box():
    """A box partially or fully outside the frame (e.g. a norfair estimate that briefly
    overshoots frame bounds) must not crash -- clamped to the frame.
    """
    frame_jpeg = _real_jpeg(width=100, height=80)
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, [("person", (-50.0, -50.0, 200.0, 200.0))])

    assert result is not None
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (80, 100)


def test_draw_boxes_on_jpeg_bytes_skips_degenerate_box_without_crashing():
    frame_jpeg = _real_jpeg()
    # x1 > x2 -- inverted/degenerate (a real symptom of the original box-timing bug --
    # see TODO_FIX_LIST.md).
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, [("person", (200.0, 10.0, 50.0, 60.0))])

    assert result is not None  # the frame itself is still returned, just without that box


def test_draw_boxes_on_jpeg_bytes_with_no_boxes_returns_the_same_image():
    frame_jpeg = _real_jpeg()
    result = draw_boxes_on_jpeg_bytes(frame_jpeg, [])

    assert result is not None
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (80, 100)


def test_draw_boxes_on_jpeg_bytes_returns_none_for_invalid_jpeg():
    result = draw_boxes_on_jpeg_bytes(b"not a real jpeg", [("person", (0.0, 0.0, 10.0, 10.0))])
    assert result is None


def test_crop_jpeg_to_box_returns_a_smaller_valid_jpeg():
    frame_jpeg = _real_jpeg(width=100, height=80)
    result = crop_jpeg_to_box(frame_jpeg, (10.0, 10.0, 50.0, 60.0))

    assert result is not None
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (50, 40)  # (y2-y1, x2-x1)


def test_crop_jpeg_to_box_clamps_out_of_bounds_box():
    frame_jpeg = _real_jpeg(width=100, height=80)
    result = crop_jpeg_to_box(frame_jpeg, (-20.0, -20.0, 200.0, 200.0))

    assert result is not None
    arr = np.frombuffer(result, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (80, 100)  # clamped to the full frame


def test_crop_jpeg_to_box_returns_none_for_degenerate_box():
    frame_jpeg = _real_jpeg(width=100, height=80)
    assert crop_jpeg_to_box(frame_jpeg, (500.0, 500.0, 600.0, 600.0)) is None  # entirely out of bounds
    assert crop_jpeg_to_box(frame_jpeg, (10.0, 10.0, 10.0, 60.0)) is None  # zero width


def test_crop_jpeg_to_box_returns_none_for_invalid_jpeg():
    result = crop_jpeg_to_box(b"not a real jpeg", (0.0, 0.0, 10.0, 10.0))
    assert result is None
