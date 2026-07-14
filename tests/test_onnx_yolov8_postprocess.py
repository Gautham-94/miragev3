"""Tests the YOLOv8 ONNX decode/postprocess math directly against synthetic raw-output
arrays shaped like a real model's output (no actual .onnx model file needed) -- this
validates the cx,cy,w,h -> normalized-xyxy conversion, per-class-argmax scoring, score
thresholding, and NMS wiring independent of whether a real exported model is available.
"""

from __future__ import annotations

import numpy as np

from mirage.config.schema import ModelConfig
from mirage.detection.plugins.onnx_yolov8 import OnnxYolov8Detector


def _make_detector_without_loading_model(model_config: ModelConfig) -> OnnxYolov8Detector:
    """Bypasses __init__'s onnxruntime.InferenceSession(...) call (no real model file
    needed) by constructing the object via __new__ and setting the attributes _postprocess
    actually uses.
    """
    detector = OnnxYolov8Detector.__new__(OnnxYolov8Detector)
    detector.model_config = model_config
    detector.score_threshold = 0.4
    detector.nms_threshold = 0.4
    return detector


def _raw_output_for_single_box(model_size: int, num_classes: int, cx: float, cy: float, w: float, h: float,
                                class_id: int, score: float) -> np.ndarray:
    """Builds a [1, 4+num_classes, N] raw output array with exactly one "real" anchor
    prediction (the rest zeroed/low-score), matching YOLOv8's ONNX export layout.
    """
    n_anchors = 100
    raw = np.zeros((1, 4 + num_classes, n_anchors), dtype=np.float32)
    raw[0, 0, 0] = cx
    raw[0, 1, 0] = cy
    raw[0, 2, 0] = w
    raw[0, 3, 0] = h
    raw[0, 4 + class_id, 0] = score
    return raw


def test_single_confident_box_decodes_to_correct_normalized_coords():
    model_size = 320
    model_config = ModelConfig(width=model_size, height=model_size)
    detector = _make_detector_without_loading_model(model_config)

    # A box centered at (160,160) with width/height 64 -> spans x:[128,192], y:[128,192]
    # in pixel space of a 320x320 model input.
    raw = _raw_output_for_single_box(model_size, num_classes=3, cx=160, cy=160, w=64, h=64, class_id=1, score=0.9)

    output = detector._postprocess(raw)

    assert output.shape == (20, 6)
    row = output[0]
    class_id, score, y1, x1, y2, x2 = row
    assert int(class_id) == 1
    assert abs(score - 0.9) < 1e-4
    assert abs(x1 - 128 / model_size) < 0.02
    assert abs(y1 - 128 / model_size) < 0.02
    assert abs(x2 - 192 / model_size) < 0.02
    assert abs(y2 - 192 / model_size) < 0.02
    # All rows after the first real detection should be zero-padded.
    assert np.all(output[1:] == 0)


def test_low_score_box_is_filtered_out():
    model_config = ModelConfig(width=320, height=320)
    detector = _make_detector_without_loading_model(model_config)
    raw = _raw_output_for_single_box(320, num_classes=3, cx=160, cy=160, w=64, h=64, class_id=0, score=0.1)

    output = detector._postprocess(raw)
    assert np.all(output == 0), "a box scoring below score_threshold must be fully filtered"


def test_boxes_are_clipped_to_frame_bounds():
    model_size = 320
    model_config = ModelConfig(width=model_size, height=model_size)
    detector = _make_detector_without_loading_model(model_config)

    # Box centered near the top-left corner with a size that pushes it out of bounds.
    raw = _raw_output_for_single_box(model_size, num_classes=2, cx=5, cy=5, w=40, h=40, class_id=0, score=0.8)
    output = detector._postprocess(raw)

    row = output[0]
    _, _, y1, x1, y2, x2 = row
    assert x1 >= 0.0
    assert y1 >= 0.0
    assert x2 <= 1.0
    assert y2 <= 1.0


def test_nms_suppresses_heavily_overlapping_duplicate_boxes():
    model_size = 320
    num_classes = 2
    n_anchors = 10
    raw = np.zeros((1, 4 + num_classes, n_anchors), dtype=np.float32)

    # Two nearly-identical boxes for the same class -- NMS should collapse them to one.
    for i in (0, 1):
        raw[0, 0, i] = 160
        raw[0, 1, i] = 160
        raw[0, 2, i] = 64
        raw[0, 3, i] = 64
        raw[0, 4, i] = 0.9 - i * 0.01  # slightly different scores

    model_config = ModelConfig(width=model_size, height=model_size)
    detector = _make_detector_without_loading_model(model_config)
    output = detector._postprocess(raw)

    non_zero_rows = output[np.any(output != 0, axis=1)]
    assert len(non_zero_rows) == 1, f"expected NMS to collapse duplicate boxes to 1, got {len(non_zero_rows)}"


def test_no_detections_returns_all_zero_output():
    model_config = ModelConfig(width=320, height=320)
    detector = _make_detector_without_loading_model(model_config)
    raw = np.zeros((1, 4 + 5, 50), dtype=np.float32)  # all scores 0

    output = detector._postprocess(raw)
    assert output.shape == (20, 6)
    assert np.all(output == 0)


def test_output_rows_sorted_descending_by_score():
    model_size = 320
    num_classes = 2
    n_anchors = 20
    raw = np.zeros((1, 4 + num_classes, n_anchors), dtype=np.float32)

    # Three well-separated boxes (different positions so NMS doesn't merge them) with
    # deliberately out-of-order scores.
    boxes = [(50, 50, 0.5), (200, 200, 0.9), (260, 60, 0.7)]
    for i, (cx, cy, score) in enumerate(boxes):
        raw[0, 0, i] = cx
        raw[0, 1, i] = cy
        raw[0, 2, i] = 30
        raw[0, 3, i] = 30
        raw[0, 4, i] = score

    model_config = ModelConfig(width=model_size, height=model_size)
    detector = _make_detector_without_loading_model(model_config)
    output = detector._postprocess(raw)

    scores = output[:3, 1]
    assert list(scores) == sorted(scores, reverse=True)
    assert abs(scores[0] - 0.9) < 1e-4
