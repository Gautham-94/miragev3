"""Real end-to-end test: a real photo -> YUV420 conversion (matching how frames actually
arrive from ffmpeg in this pipeline) -> create_tensor_input -> the real exported
yolov8n.onnx model -> decoded detections, verifying the whole chain actually finds the
right real-world objects, not just that the code runs without error.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mirage.config.schema import InputDType, ModelConfig, PixelFormat
from mirage.detection.labelmap import load_labels
from mirage.detection.plugins.onnx_yolov8 import OnnxYolov8Detector
from mirage.detection.tensor import create_tensor_input

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
BUS_IMAGE = Path(__file__).resolve().parent.parent / "media" / "bus.jpg"
ZIDANE_IMAGE = Path(__file__).resolve().parent.parent / "media" / "zidane.jpg"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and BUS_IMAGE.exists()),
    reason="yolov8n.onnx model or test image fixture not available",
)


def _bgr_image_to_yuv420(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Converts a real BGR photo into the same YUV420 planar buffer layout the capture
    pipeline produces from ffmpeg, so this test exercises the actual crop_yuv_region /
    create_tensor_input code path rather than feeding a pre-made RGB array directly.
    """
    height, width = bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # shape (height*3//2, width)
    return yuv_i420, (height, width)


@pytest.fixture(scope="module")
def detector():
    model_config = ModelConfig(
        width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
        model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
    )
    return OnnxYolov8Detector(model_config)


@pytest.fixture(scope="module")
def labels():
    return load_labels(str(LABELMAP_PATH))


def test_detects_bus_and_person_in_real_photo(detector, labels):
    bgr = cv2.imread(str(BUS_IMAGE))
    assert bgr is not None
    yuv, frame_shape = _bgr_image_to_yuv420(bgr)

    tensor = create_tensor_input(yuv, frame_shape, detector.model_config, (0, 0, frame_shape[1], frame_shape[0]))
    raw = detector.detect_raw(tensor)

    assert raw.shape == (20, 6)
    detected_labels = set()
    for row in raw:
        class_id, score, y1, x1, y2, x2 = row
        if score <= 0:
            continue
        detected_labels.add(labels.get(int(class_id), "unknown"))

    assert "bus" in detected_labels, f"expected 'bus' in detections, got {detected_labels}"
    assert "person" in detected_labels, f"expected 'person' in detections, got {detected_labels}"


def test_detects_person_in_zidane_photo(detector, labels):
    if not ZIDANE_IMAGE.exists():
        pytest.skip("zidane.jpg fixture not available")
    bgr = cv2.imread(str(ZIDANE_IMAGE))
    assert bgr is not None
    yuv, frame_shape = _bgr_image_to_yuv420(bgr)

    tensor = create_tensor_input(yuv, frame_shape, detector.model_config, (0, 0, frame_shape[1], frame_shape[0]))
    raw = detector.detect_raw(tensor)

    detected_labels = {labels.get(int(row[0]), "unknown") for row in raw if row[1] > 0}
    assert "person" in detected_labels


def test_detection_boxes_are_within_normalized_bounds(detector):
    bgr = cv2.imread(str(BUS_IMAGE))
    yuv, frame_shape = _bgr_image_to_yuv420(bgr)
    tensor = create_tensor_input(yuv, frame_shape, detector.model_config, (0, 0, frame_shape[1], frame_shape[0]))
    raw = detector.detect_raw(tensor)

    for row in raw:
        _, score, y1, x1, y2, x2 = row
        if score <= 0:
            continue
        assert 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0
        assert 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0
        assert x2 > x1 and y2 > y1


def test_no_high_confidence_detections_on_blank_frame(detector):
    blank_bgr = np.full((320, 320, 3), 128, dtype=np.uint8)
    yuv, frame_shape = _bgr_image_to_yuv420(blank_bgr)
    tensor = create_tensor_input(yuv, frame_shape, detector.model_config, (0, 0, frame_shape[1], frame_shape[0]))
    raw = detector.detect_raw(tensor)
    # A flat gray frame should produce no confident detections -- assert nothing is
    # reported with high confidence (some low-score noise passing score_threshold=0.4 is
    # tolerable and not the property under test here).
    assert not any(row[1] > 0.8 for row in raw)
