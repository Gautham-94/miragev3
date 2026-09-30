"""ONNX Runtime backend for RT-DETR-family detectors exported with built-in top-k
postprocessing (MegaDetector v6's RT-DETR variants -- MDV6-apa-rtdetr-c/e -- use this
export). Distinct from onnx_yolo_nms.py despite the same [1, 300, 6] output shape: the
box columns here are [cx, cy, w, h] (center + size), normalized to [0,1] already -- NOT
[x1, y1, x2, y2] in pixel space like the YOLO NMS-export format. Confirmed by direct
inspection against a real image: treating column 0-3 as x1y1x2y2 produces an impossible
box (x2 < x1) for some rows, while cx/cy/w/h decodes cleanly and matches the model's own
high-confidence prediction to a real animal in frame.

RT-DETR predicts one query per detection slot, scored against every class -- so 300 rows
here is ~100 spatial queries x 3 classes, with 2-3 nearly-identical-box rows per real
detection (one per class). No extra NMS/dedup needed: the true class's score dominates
the other classes' scores for the same box by a wide margin in practice (confirmed:
0.975 vs 0.061/0.043 for the same box), so a plain per-row score threshold naturally
keeps just the correct row.
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


class OnnxRtdetrDetector(DetectionApi):
    type_key = "onnx_rtdetr"

    def __init__(self, model_config: ModelConfig, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> None:
        self.model_config = model_config
        self.score_threshold = score_threshold
        providers = resolve_providers(model_config.execution_provider)
        self.session = ort.InferenceSession(resolve_model_path(model_config.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # tensor_input: [1, H, W, 3] uint8 (NHWC) -- same NCHW float32 [0,1] convention
        # as the other ONNX plugins here.
        nchw = np.transpose(tensor_input, (0, 3, 1, 2)).astype(np.float32) / 255.0

        raw_output = self.session.run(None, {self.input_name: nchw})[0]  # [1, 300, 6]
        return self._postprocess(raw_output)

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]  # [N, 6]: cx, cy, w, h, confidence, class_id (all normalized)
        scores = predictions[:, 4]

        keep = scores >= self.score_threshold
        if not np.any(keep):
            return empty_detection_output()

        cx, cy, w, h = predictions[keep, 0], predictions[keep, 1], predictions[keep, 2], predictions[keep, 3]
        scores = scores[keep]
        class_ids = predictions[keep, 5].astype(np.int64)

        x1 = np.clip(cx - w / 2, 0, 1)
        y1 = np.clip(cy - h / 2, 0, 1)
        x2 = np.clip(cx + w / 2, 0, 1)
        y2 = np.clip(cy + h / 2, 0, 1)
        boxes_xyxy_norm = np.stack([x1, y1, x2, y2], axis=1)

        return pack_detections(class_ids, scores, boxes_xyxy_norm)
