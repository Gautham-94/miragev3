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

import numpy as np
import onnxruntime as ort

from mirage.config.schema import ModelConfig
from mirage.detection.api import DetectionApi, empty_detection_output
from mirage.detection.execution_providers import resolve_providers
from mirage.detection.postprocess import nms_xywh, pack_detections

DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_NMS_THRESHOLD = 0.4


class OnnxYolov8Detector(DetectionApi):
    type_key = "onnx_yolov8"

    def __init__(self, model_config: ModelConfig, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> None:
        self.model_config = model_config
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        providers = resolve_providers(model_config.execution_provider)
        self.session = ort.InferenceSession(model_config.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

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
