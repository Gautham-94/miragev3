"""ONNX Runtime backend for MegaDetector v5 (YOLOv5x6-based, exported via the official
ultralytics/yolov5 repo's export.py -- MDv5 predates the unified `ultralytics` package
and can't be loaded/exported with the same onnx_yolov8.py code path).

Class map (MegaDetector's own convention, 1-indexed in the original checkpoint,
0-indexed here to match mirage's labelmap convention): 0=animal, 1=person, 2=vehicle.
No "bird" class -- MegaDetector buckets birds under "animal" (see
mirage/detection/registry.py's plugin scan; camera configs using this backend should
set objects.track accordingly, and note SPECIES_ENRICHABLE_LABELS in
mirage/events/processor.py only special-cases "bird" for cameras using a detector that
actually emits it).

Output shape is [1, N, 4+1+num_classes] = [1, N, 8] for 3 classes -- YOLOv5's own
export convention, DIFFERENT from YOLOv8's [1, 4+num_classes, N] (see
onnx_yolov8.py's docstring for that shape): the anchor dimension is second, not last,
and there's a separate objectness score (index 4) multiplied into each class score,
rather than YOLOv8 folding objectness into the class scores directly. Confirmed by
direct inspection of a real export (dummy-input run against md_v5a.0.0.onnx) -- not
assumed from documentation alone, since this detail is easy to get wrong and would
silently corrupt every detection if mismatched.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from mirage.config.schema import ModelConfig
from mirage.const import resolve_model_path
from mirage.detection.api import DetectionApi, empty_detection_output
from mirage.detection.execution_providers import resolve_providers
from mirage.detection.postprocess import nms_xywh, pack_detections

DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_NMS_THRESHOLD = 0.45


class OnnxMegadetectorDetector(DetectionApi):
    type_key = "onnx_megadetector"

    def __init__(self, model_config: ModelConfig, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> None:
        self.model_config = model_config
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        providers = resolve_providers(model_config.execution_provider)
        self.session = ort.InferenceSession(resolve_model_path(model_config.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # tensor_input: [1, H, W, 3] uint8 (NHWC) -- MDv5 ONNX expects NCHW float32 in [0,1].
        nchw = np.transpose(tensor_input, (0, 3, 1, 2)).astype(np.float32) / 255.0

        raw_output = self.session.run(None, {self.input_name: nchw})[0]  # [1, N, 4+1+C]
        return self._postprocess(raw_output)

    def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        predictions = raw_output[0]  # [N, 4+1+C]
        boxes_cxcywh = predictions[:, :4]  # [N, 4]
        objectness = predictions[:, 4]  # [N]
        class_scores_raw = predictions[:, 5:]  # [N, C]

        # Final per-class confidence is objectness * class_score (YOLOv5's own
        # convention, distinct from YOLOv8 folding this into the class score already).
        class_scores = class_scores_raw * objectness[:, np.newaxis]
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
