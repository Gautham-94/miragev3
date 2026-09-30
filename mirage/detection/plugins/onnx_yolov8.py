"""ONNX Runtime backend for YOLOv8-family models. Runs on whichever execution provider
ModelConfig.execution_provider resolves to (CPU by default; CoreML/CUDA if configured and
available -- see mirage/detection/execution_providers.py).

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.1, 3.3.2.

YOLOv8's ONNX export (`model.export(format="onnx")`) produces a single output tensor of
shape [1, 4+num_classes, num_anchors] (e.g. [1, 84, 8400] for 80 COCO classes at 640x640):
4 box regression values (cx, cy, w, h in pixel space of the model's input size) followed by
one row per class's confidence score, with the anchor dimension last (a "channels-first,
transposed" layout relative to older YOLO ONNX exports). This is NOT already NMS'd -- NMS
must be applied here.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from mirage.config.schema import ModelConfig
from mirage.const import resolve_model_path
from mirage.detection.api import DetectionApi, empty_detection_output
from mirage.detection.execution_providers import resolve_providers
from mirage.detection.labelmap import load_labels
from mirage.detection.postprocess import nms_xywh, pack_detections

logger = logging.getLogger(__name__)

DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_NMS_THRESHOLD = 0.4

# TEMP diagnostic (see mirage.tracking.orchestration.DIAGNOSTIC_SCORE_LOGGING_LABEL --
# same investigation, one layer deeper): score_threshold (0.4 by default) is a HARD,
# currently-uncustomizable floor applied here, before mirage's own per-camera
# min_score/threshold config ever sees a detection -- a candidate scoring below this
# never reaches Gate 0/Gate 1 logging in orchestration.py, so a genuine model
# false-negative is otherwise indistinguishable from "never ran inference near there at
# all" after the fact. Logs the highest-scoring candidates in this near-miss band
# (deliberately excludes near-zero background-clutter noise, which would otherwise
# flood every single frame) so a real "the model saw SOMETHING there but scored it too
# low" case is visible. No camera name available at this layer (this plugin is shared
# across every camera routed to one detector) -- correlate by timestamp with the
# adjacent "detector processing camera X" line detector_process_main now logs
# immediately before each detect_raw() call, since that loop is strictly synchronous
# (one call completes before the next begins). Remove once the investigation is done.
DIAGNOSTIC_NEAR_MISS_MIN_SCORE = 0.15
DIAGNOSTIC_NEAR_MISS_MAX_CANDIDATES = 5


class OnnxYolov8Detector(DetectionApi):
    type_key = "onnx_yolov8"

    def __init__(self, model_config: ModelConfig, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> None:
        self.model_config = model_config
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        providers = resolve_providers(model_config.execution_provider)
        self.session = ort.InferenceSession(resolve_model_path(model_config.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # TEMP diagnostic only -- see DIAGNOSTIC_NEAR_MISS_MIN_SCORE's own comment.
        self._diagnostic_labels = load_labels(resolve_model_path(model_config.labelmap_path))

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # tensor_input: [1, H, W, 3] uint8 (NHWC) -- YOLOv8 ONNX expects NCHW float32 in [0,1].
        nchw = np.transpose(tensor_input, (0, 3, 1, 2)).astype(np.float32) / 255.0

        raw_output = self.session.run(None, {self.input_name: nchw})[0]  # [1, 4+C, N]
        return self._postprocess(raw_output)

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]  # [4+C, N]
        num_classes = predictions.shape[0] - 4
        boxes_cxcywh = predictions[:4, :].T  # [N, 4]
        class_scores = predictions[4:, :].T  # [N, C]

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        if logger.isEnabledFor(logging.DEBUG):
            self._log_near_miss_candidates(class_ids, scores)

        keep = scores >= self.score_threshold
        if not np.any(keep):
            return empty_detection_output()

        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        # cx,cy,w,h (pixel space of model input) -> x,y,w,h (top-left) for cv2.dnn.NMSBoxes.
        x = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]
        boxes_xywh = np.stack([x, y, w, h], axis=1)

        keep_indices = nms_xywh(boxes_xywh, scores, self.score_threshold, self.nms_threshold)
        if not keep_indices:
            return empty_detection_output()

        boxes_xywh = boxes_xywh[keep_indices]
        class_ids = class_ids[keep_indices]
        scores = scores[keep_indices]

        model_w, model_h = self.model_config.width, self.model_config.height
        x1 = np.clip(boxes_xywh[:, 0] / model_w, 0, 1)
        y1 = np.clip(boxes_xywh[:, 1] / model_h, 0, 1)
        x2 = np.clip((boxes_xywh[:, 0] + boxes_xywh[:, 2]) / model_w, 0, 1)
        y2 = np.clip((boxes_xywh[:, 1] + boxes_xywh[:, 3]) / model_h, 0, 1)
        boxes_xyxy_norm = np.stack([x1, y1, x2, y2], axis=1)

        return pack_detections(class_ids, scores, boxes_xyxy_norm)

    def _log_near_miss_candidates(self, class_ids: np.ndarray, scores: np.ndarray) -> None:
        """TEMP diagnostic -- see DIAGNOSTIC_NEAR_MISS_MIN_SCORE's own module-level
        comment. Only ever called when DEBUG is actually enabled (see the
        isEnabledFor guard at the call site), so this costs nothing in normal
        operation.
        """
        band = (scores >= DIAGNOSTIC_NEAR_MISS_MIN_SCORE) & (scores < self.score_threshold)
        if not np.any(band):
            return
        band_scores = scores[band]
        band_labels = class_ids[band]
        order = np.argsort(-band_scores)[:DIAGNOSTIC_NEAR_MISS_MAX_CANDIDATES]
        candidates = ", ".join(
            f"{self._diagnostic_labels.get(int(band_labels[i]), 'unknown')}={band_scores[i]:.3f}"
            for i in order
        )
        logger.debug(
            "near-miss sub-threshold candidates (threshold=%.3f): %s",
            self.score_threshold, candidates,
        )
