from __future__ import annotations

import numpy as np

from mirage.config.schema import MotionConfig
from mirage.motion.detector import MotionDetector, rasterize_mask_polygons


def _static_frame(height: int, width: int, value: int = 100) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _frame_with_block(height: int, width: int, base_value: int, block_value: int,
                       box: tuple[int, int, int, int]) -> np.ndarray:
    frame = np.full((height, width), base_value, dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = block_value
    return frame


def test_no_motion_on_identical_static_frames():
    height, width = 200, 300
    cfg = MotionConfig(frame_height=100)
    detector = MotionDetector((height, width), cfg)

    frame = _static_frame(height, width, 100)
    # Feed the same static frame repeatedly to let the background model settle.
    for _ in range(40):
        boxes = detector.detect(frame)

    assert boxes == []


def test_motion_detected_when_a_block_appears():
    height, width = 200, 300
    cfg = MotionConfig(frame_height=100, threshold=30, contour_area=5)
    detector = MotionDetector((height, width), cfg)

    static_frame = _static_frame(height, width, 100)
    # Settle the background on the static scene first.
    for _ in range(40):
        detector.detect(static_frame)
    assert not detector.is_calibrating()

    moving_frame = _frame_with_block(height, width, 100, 220, (50, 50, 150, 150))
    boxes = detector.detect(moving_frame)

    assert len(boxes) >= 1
    # The detected box should roughly overlap the inserted block (allow for downscale
    # rounding at frame_height=100 -> resize_factor=2).
    x1, y1, x2, y2 = boxes[0]
    assert x1 < 150 and x2 > 50
    assert y1 < 150 and y2 > 50


def test_masked_region_never_triggers_motion():
    height, width = 200, 300
    # Mask covers the exact region where we'll insert the moving block (normalized coords).
    cfg = MotionConfig(
        frame_height=100, threshold=30, contour_area=5,
        mask=[[(0.1, 0.2), (0.6, 0.2), (0.6, 0.8), (0.1, 0.8)]],
    )
    detector = MotionDetector((height, width), cfg)

    static_frame = _static_frame(height, width, 100)
    for _ in range(40):
        detector.detect(static_frame)

    # Block placed inside the masked region (x:30-90 of 300 width, y:40-160 of 200 height
    # roughly maps inside the 0.1-0.6 x / 0.2-0.8 y normalized polygon above).
    moving_frame = _frame_with_block(height, width, 100, 220, (30, 40, 90, 160))
    boxes = detector.detect(moving_frame)

    assert boxes == [], f"expected masked region to suppress motion, got {boxes}"


def test_calibrating_true_initially_and_false_after_settling():
    height, width = 200, 300
    cfg = MotionConfig(frame_height=100)
    detector = MotionDetector((height, width), cfg)
    assert detector.is_calibrating() is True

    static_frame = _static_frame(height, width, 100)
    for _ in range(40):
        detector.detect(static_frame)
    assert detector.is_calibrating() is False


def test_lightning_threshold_forces_recalibration_without_dropping_boxes():
    height, width = 200, 300
    cfg = MotionConfig(frame_height=100, threshold=10, contour_area=1, lightning_threshold=0.3)
    detector = MotionDetector((height, width), cfg)

    static_frame = _static_frame(height, width, 100)
    for _ in range(40):
        detector.detect(static_frame)
    assert not detector.is_calibrating()

    # A near-full-frame brightness flip should exceed the 0.3 lightning threshold.
    flash_frame = _static_frame(height, width, 250)
    boxes = detector.detect(flash_frame)

    assert detector.is_calibrating() is True
    # lightning_threshold does NOT drop boxes (unlike skip_motion_threshold) -- the huge
    # brightness delta should still have produced a large motion contour.
    assert len(boxes) >= 1


def test_skip_motion_threshold_drops_all_boxes_and_forces_calibration():
    height, width = 200, 300
    cfg = MotionConfig(
        frame_height=100, threshold=10, contour_area=1,
        skip_motion_threshold=0.3, lightning_threshold=0.9,
    )
    detector = MotionDetector((height, width), cfg)

    static_frame = _static_frame(height, width, 100)
    for _ in range(40):
        detector.detect(static_frame)
    assert not detector.is_calibrating()

    flash_frame = _static_frame(height, width, 250)
    boxes = detector.detect(flash_frame)

    assert boxes == [], "skip_motion_threshold should drop ALL boxes for this frame"
    assert detector.is_calibrating() is True


def test_disabled_motion_returns_no_boxes():
    height, width = 200, 300
    cfg = MotionConfig(enabled=False)
    detector = MotionDetector((height, width), cfg)
    moving_frame = _frame_with_block(height, width, 100, 220, (50, 50, 150, 150))
    assert detector.detect(moving_frame) == []


def test_update_mask_resets_background_and_calibration():
    height, width = 200, 300
    cfg = MotionConfig(frame_height=100)
    detector = MotionDetector((height, width), cfg)
    static_frame = _static_frame(height, width, 100)
    for _ in range(40):
        detector.detect(static_frame)
    assert not detector.is_calibrating()

    new_mask = rasterize_mask_polygons([[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]], width, height)
    detector.update_mask(new_mask)

    assert detector.is_calibrating() is True
    assert detector.motion_frame_count == 0
    assert detector.avg_frame.sum() == 0


def test_rasterize_mask_polygons_none_when_empty():
    assert rasterize_mask_polygons([], 100, 100) is None


def test_rasterize_mask_polygons_produces_expected_shape_and_values():
    mask = rasterize_mask_polygons([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]], 50, 40)
    assert mask is not None
    assert mask.shape == (40, 50)
    assert mask.max() == 255
