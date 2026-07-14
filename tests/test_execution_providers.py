"""Tests for mirage/detection/execution_providers.py -- the ModelConfig.execution_provider
-> onnxruntime `providers=[...]` list resolution, plus a real end-to-end check that a
detector actually configured for CoreML still produces correct detections on this machine
(not just that construction doesn't crash).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import onnxruntime as ort
import pytest

from mirage.config.schema import ExecutionProvider, InputDType, ModelConfig, PixelFormat
from mirage.detection.execution_providers import available_execution_providers, resolve_providers
from mirage.detection.labelmap import load_labels
from mirage.detection.plugins.onnx_yolov8 import OnnxYolov8Detector
from mirage.detection.tensor import create_tensor_input

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
BUS_IMAGE = Path(__file__).resolve().parent.parent / "media" / "bus.jpg"


def test_cpu_resolves_to_cpu_provider_only():
    assert resolve_providers(ExecutionProvider.cpu) == ["CPUExecutionProvider"]


def test_coreml_resolves_with_cpu_fallback_appended():
    providers = resolve_providers(ExecutionProvider.coreml)
    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_cuda_resolves_with_cpu_fallback_appended():
    providers = resolve_providers(ExecutionProvider.cuda)
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_auto_always_ends_with_cpu_fallback():
    providers = resolve_providers(ExecutionProvider.auto)
    assert providers[-1] == "CPUExecutionProvider"


def test_available_execution_providers_always_includes_cpu_and_auto():
    providers = available_execution_providers()
    assert "cpu" in providers
    assert "auto" in providers


def test_available_execution_providers_matches_real_onnxruntime_installation():
    # Ground-truth check against the actual onnxruntime build installed in this
    # environment, not a mocked/assumed provider list.
    installed = set(ort.get_available_providers())
    providers = available_execution_providers()
    if "CoreMLExecutionProvider" in installed:
        assert "coreml" in providers
    if "CUDAExecutionProvider" in installed:
        assert "cuda" in providers


@pytest.mark.skipif(
    "CoreMLExecutionProvider" not in ort.get_available_providers(),
    reason="CoreML execution provider not available on this machine",
)
def test_coreml_configured_detector_still_detects_correctly():
    # Real end-to-end proof this isn't just "constructs without error" -- a detector
    # explicitly configured for CoreML must still find the same real objects in the same
    # real photo as the CPU-configured detector (test_onnx_yolov8_e2e.py's CPU baseline),
    # confirming the accelerator path is functionally correct, not just wired up.
    if not (MODEL_PATH.exists() and BUS_IMAGE.exists()):
        pytest.skip("yolov8n.onnx model or test image fixture not available")

    model_config = ModelConfig(
        width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
        model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        execution_provider=ExecutionProvider.coreml,
    )
    detector = OnnxYolov8Detector(model_config)
    assert detector.session.get_providers()[0] == "CoreMLExecutionProvider"

    labels = load_labels(str(LABELMAP_PATH))
    bgr = cv2.imread(str(BUS_IMAGE))
    height, width = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)

    tensor = create_tensor_input(yuv, (height, width), detector.model_config, (0, 0, width, height))
    raw = detector.detect_raw(tensor)

    detected_labels = {labels.get(int(row[0]), "unknown") for row in raw if row[1] > 0}
    assert "bus" in detected_labels
    assert "person" in detected_labels
