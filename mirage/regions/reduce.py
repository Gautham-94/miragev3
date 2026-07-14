"""Reduce/consolidate detections across all regions scanned in a frame.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 4, "After detection" paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

EDGE_CLIP_MARGIN_PX = 5
EDGE_CLIP_CONFIDENCE_FLOOR = 0.6
NMS_SCORE_THRESHOLD = 0.5
DEFAULT_NMS_IOU_THRESHOLD = 0.4
CONTAINMENT_FRACTION_THRESHOLD = 0.9  # smaller box >=90% contained in a larger one -> drop
CONTAINMENT_MIN_AREA_RATIO = 0.05  # unless smaller box is <5% of the larger's area


@dataclass
class RawDetection:
    label: str
    score: float
    box: Box  # full detect-frame pixel coordinates
    region: Box  # the region this detection came from


def is_clipped_at_region_edge(box: Box, region: Box, frame_shape: tuple[int, int], margin: int = EDGE_CLIP_MARGIN_PX) -> bool:
    """True if `box` touches a region boundary that is NOT also a frame edge (i.e. the
    crop actually cut the object off, rather than the object simply being near the edge
    of the full frame). Spec section 4: such detections get a confidence floor before NMS
    so a partially-cropped real object isn't unfairly suppressed for a low score.
    """
    height, width = frame_shape
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = region

    if bx1 <= rx1 + margin and rx1 > 0:
        return True
    if by1 <= ry1 + margin and ry1 > 0:
        return True
    if bx2 >= rx2 - margin and rx2 < width:
        return True
    if by2 >= ry2 - margin and ry2 < height:
        return True
    return False


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def reduce_overlapping_detections(
    detections: list[RawDetection], frame_shape: tuple[int, int], nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD
) -> list[RawDetection]:
    """Per-label NMS, with an edge-clip confidence floor applied before NMS runs (spec
    section 4).
    """
    result: list[RawDetection] = []
    by_label: dict[str, list[RawDetection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    for label, label_detections in by_label.items():
        boxes_xywh = []
        scores = []
        for d in label_detections:
            x1, y1, x2, y2 = d.box
            score = d.score
            if is_clipped_at_region_edge(d.box, d.region, frame_shape):
                score = max(score, EDGE_CLIP_CONFIDENCE_FLOOR)
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)

        if not boxes_xywh:
            continue

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=NMS_SCORE_THRESHOLD, nms_threshold=nms_iou_threshold)
        keep_indices = list(np.array(indices).flatten()) if len(indices) > 0 else []
        for i in keep_indices:
            result.append(label_detections[i])

    return result


def get_consolidated_object_detections(detections: list[RawDetection]) -> list[RawDetection]:
    """Drops smaller same-label detections that are mostly contained inside a larger one
    (protects against double-counting one physical object detected at two scales/regions),
    except when the smaller box is a tiny fraction of the larger's area (spec section 4).
    """
    by_label: dict[str, list[RawDetection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    result: list[RawDetection] = []
    for label, label_detections in by_label.items():
        sorted_by_area = sorted(label_detections, key=lambda d: box_area(d.box), reverse=True)
        kept: list[RawDetection] = []
        for candidate in sorted_by_area:
            candidate_area = box_area(candidate.box)
            suppressed = False
            for larger in kept:
                larger_area = box_area(larger.box)
                if larger_area <= 0:
                    continue
                if candidate_area / larger_area < CONTAINMENT_MIN_AREA_RATIO:
                    continue  # too small relative to the larger box -- likely a genuinely different object
                overlap = intersection_area(candidate.box, larger.box)
                if candidate_area > 0 and overlap / candidate_area >= CONTAINMENT_FRACTION_THRESHOLD:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)
        result.extend(kept)

    return result


def reduce_detections(detections: list[RawDetection], frame_shape: tuple[int, int]) -> list[RawDetection]:
    """The full spec section 4 reduce/consolidate pipeline: per-label NMS with edge-clip
    confidence floor, then containment suppression.
    """
    nms_result = reduce_overlapping_detections(detections, frame_shape)
    return get_consolidated_object_detections(nms_result)
