"""ONNX Runtime backend for YOLO-family models exported WITH NMS baked in (ultralytics'
own `model.export(format="onnx", nms=True)` "end2end" export path produces exactly this
shape) -- distinct from onnx_yolov8.py (raw, pre-NMS [1, 4+num_classes, num_anchors]
output, needs NMS applied here) and onnx_megadetector.py (MegaDetector v5's own raw
YOLOv5-style [1, N, 4+1+num_classes] export with a separate objectness score, ALSO
pre-NMS). A model already answering to the name "MegaDetector" is not proof it matches
onnx_megadetector.py's expected shape -- this was added for a model literally named
mega.onnx whose real output, confirmed by direct inspection (dummy-input run against the
actual file, not assumed from a shape number alone), turned out to be this end2end
format instead, not MegaDetector v5's.

Output shape confirmed live: [1, 300, 6], each row already-NMS'd
[x1, y1, x2, y2, confidence, class_id] in PIXEL space of the model's (fixed) input size
-- no further NMS needed here, just a score-threshold filter and normalization.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from mirage.config.schema import ModelConfig
from mirage.const import resolve_model_path
from mirage.detection.api import DetectionApi, empty_detection_output
from mirage.detection.execution_providers import resolve_providers
from mirage.detection.postprocess import pack_detections

DEFAULT_SCORE_THRESHOLD = 0.2


class OnnxYoloNmsDetector(DetectionApi):
    type_key = "onnx_yolo_nms"

    def __init__(self, model_config: ModelConfig, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> None:
        self.model_config = model_config
        self.score_threshold = score_threshold
        providers = resolve_providers(model_config.execution_provider)
        self.session = ort.InferenceSession(resolve_model_path(model_config.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # tensor_input: [1, H, W, 3] uint8 (NHWC) -- this export expects NCHW float32 in
        # [0,1], same convention as onnx_megadetector.py's own MDv5 preprocessing.
        nchw = np.transpose(tensor_input, (0, 3, 1, 2)).astype(np.float32) / 255.0

        raw_output = self.session.run(None, {self.input_name: nchw})[0]  # [1, 300, 6]
        return self._postprocess(raw_output)

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]  # [N, 6]: x1, y1, x2, y2, confidence, class_id
        scores = predictions[:, 4]

        keep = scores >= self.score_threshold
        if not np.any(keep):
            return empty_detection_output()

        boxes_xyxy = predictions[keep, :4]
        scores = scores[keep]
        class_ids = predictions[keep, 5].astype(np.int64)

        model_w, model_h = self.model_config.width, self.model_config.height
        x1 = np.clip(boxes_xyxy[:, 0] / model_w, 0, 1)
        y1 = np.clip(boxes_xyxy[:, 1] / model_h, 0, 1)
        x2 = np.clip(boxes_xyxy[:, 2] / model_w, 0, 1)
        y2 = np.clip(boxes_xyxy[:, 3] / model_h, 0, 1)
        boxes_xyxy_norm = np.stack([x1, y1, x2, y2], axis=1)

        return pack_detections(class_ids, scores, boxes_xyxy_norm)
