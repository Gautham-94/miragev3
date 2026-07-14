"""Post-processing helpers shared across detector backends: NMS wrapping and the
universal (20,6) output-contract packer.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.3.2.
"""

from __future__ import annotations

import cv2
import numpy as np

from mirage.const import DETECTIONS_PER_ROW, MAX_DETECTIONS


def pack_detections(
    class_ids: np.ndarray, scores: np.ndarray, boxes_xyxy_norm: np.ndarray
) -> np.ndarray:
    """Packs arbitrary-length (already NMS'd, already score-thresholded) detections into
    the fixed (20,6) contract: rows sorted descending by score, zero-padded, each row
    [class_id, score, y_min, x_min, y_max, x_max] normalized to [0,1].

    boxes_xyxy_norm: shape (N, 4), columns [x1, y1, x2, y2], already normalized to [0,1].
    """
    output = np.zeros((MAX_DETECTIONS, DETECTIONS_PER_ROW), dtype=np.float32)
    if len(scores) == 0:
        return output

    order = np.argsort(-scores)[:MAX_DETECTIONS]
    for i, idx in enumerate(order):
        x1, y1, x2, y2 = boxes_xyxy_norm[idx]
        output[i] = [class_ids[idx], scores[idx], y1, x1, y2, x2]
    return output


def nms_xywh(
    boxes_xywh: np.ndarray, scores: np.ndarray, score_threshold: float, nms_threshold: float
) -> list[int]:
    """Thin wrapper around cv2.dnn.NMSBoxes -- boxes_xywh: (N,4) columns [x, y, w, h] in
    pixel (not normalized) coordinates, as cv2.dnn.NMSBoxes expects.
    """
    if len(boxes_xywh) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), scores.tolist(), score_threshold=score_threshold, nms_threshold=nms_threshold
    )
    if len(indices) == 0:
        return []
    return list(np.array(indices).flatten())


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
