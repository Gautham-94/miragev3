"""Real end-to-end test: a real photo -> a real OWLv2 model -> decoded query matches,
verifying the whole mirage.openvocab.process._process_request chain actually finds the
right real-world object for a real text query, not just that the code runs without
error. Mirrors tests/test_onnx_yolov8_e2e.py's philosophy for the closed-vocab detector.

Skipped entirely if `transformers` isn't installed -- it's not yet a hard dependency of
this repo (see requirements.txt), only needed once the open-vocab feature is actually
exercised. Slow (loads a real ~600MB model) -- this is deliberately the one open-vocab
test that pays that cost, everything else (gating, dispatcher) uses fakes/small fixtures.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

BUS_IMAGE = Path(__file__).resolve().parent.parent / "media" / "bus.jpg"

transformers = pytest.importorskip("transformers", reason="transformers not installed -- open-vocab feature not yet exercised")

pytestmark = pytest.mark.skipif(not BUS_IMAGE.exists(), reason="bus.jpg test fixture not available")


@pytest.fixture(scope="module")
def loaded_model():
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    return processor, model, device


def test_process_request_finds_bus_and_person_in_real_photo(loaded_model):
    from mirage.openvocab.process import OpenVocabRequest, _process_request

    processor, model, device = loaded_model

    bgr = cv2.imread(str(BUS_IMAGE))
    assert bgr is not None
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    request = OpenVocabRequest(
        request_id="test-req",
        camera_name="test_cam",
        object_id="obj1",
        crop_jpeg=encoded.tobytes(),
        frame_box=(0.0, 0.0, float(bgr.shape[0]), float(bgr.shape[1])),
        queries=[("q_bus", "a bus"), ("q_person", "a person")],
    )

    result = _process_request(request, processor, model, device, score_threshold=0.1)

    assert result.request_id == "test-req"
    assert result.camera_name == "test_cam"
    assert result.object_id == "obj1"

    matched_query_ids = {m.query_id for m in result.matches}
    assert "q_bus" in matched_query_ids, f"expected a bus match, got queries: {matched_query_ids}"
    assert "q_person" in matched_query_ids, f"expected a person match, got queries: {matched_query_ids}"

    # every match's box must be within the real image's full-frame pixel bounds
    height, width = bgr.shape[:2]
    for match in result.matches:
        y1, x1, y2, x2 = match.box
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert 0.0 <= match.score <= 1.0


def test_process_request_finds_nothing_for_an_irrelevant_query(loaded_model):
    from mirage.openvocab.process import OpenVocabRequest, _process_request

    processor, model, device = loaded_model

    bgr = cv2.imread(str(BUS_IMAGE))
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    request = OpenVocabRequest(
        request_id="test-req-2",
        camera_name="test_cam",
        object_id="obj2",
        crop_jpeg=encoded.tobytes(),
        frame_box=(0.0, 0.0, float(bgr.shape[0]), float(bgr.shape[1])),
        queries=[("q_spaceship", "a spaceship")],
    )

    # A high score threshold plus a genuinely absent object should yield no matches --
    # a real negative-case check, not just "the code runs."
    result = _process_request(request, processor, model, device, score_threshold=0.5)
    assert result.matches == []
