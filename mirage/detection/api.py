"""Detector abstraction -- one interface, multiple ML backends.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.1, 3.3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mirage.config.schema import ModelConfig
from mirage.const import DETECTIONS_PER_ROW, MAX_DETECTIONS


class DetectionApi(ABC):
    """One subclass per inference backend (CPU/ONNX/OpenVINO/EdgeTPU/etc).

    Every implementation must load its model in __init__ and, in detect_raw(), return a
    fixed-shape np.zeros((20, 6), dtype=np.float32) array: rows sorted descending by
    score, each row = [class_id, score, y_min, x_min, y_max, x_max] normalized to [0,1]
    relative to the model's input size. Unused trailing rows stay zeroed. Implementations
    are responsible for applying their own internal score cutoff and NMS before returning,
    if the model's raw output isn't already NMS'd (native SSD heads typically are;
    anchor-based YOLO-family outputs typically are not).
    """

    type_key: str = ""

    @abstractmethod
    def __init__(self, model_config: ModelConfig) -> None: ...

    @abstractmethod
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray: ...


def empty_detection_output() -> np.ndarray:
    return np.zeros((MAX_DETECTIONS, DETECTIONS_PER_ROW), dtype=np.float32)
