"""Tensor construction: crop a region out of a YUV420 frame and build the model input
tensor.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.3.1.
"""

from __future__ import annotations

import cv2
import numpy as np

from mirage.config.schema import InputDType, ModelConfig, PixelFormat


def yuv420_to_bgr(yuv_frame: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """yuv_frame: full YUV420 planar buffer, shape (height*3//2, width).
    frame_shape: (height, width) of the actual image (luma plane dimensions).
    """
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)


def yuv420_to_rgb(yuv_frame: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)


def crop_yuv_region(
    yuv_frame: np.ndarray, frame_shape: tuple[int, int], region: tuple[int, int, int, int], pixel_format: PixelFormat
) -> np.ndarray:
    """Converts the full frame to RGB/BGR once, then crops the region out of it.

    region: (x1, y1, x2, y2) in full detect-frame pixel coordinates, already clamped to
    frame bounds by the caller (see mirage.regions).
    """
    if pixel_format == PixelFormat.bgr:
        full = yuv420_to_bgr(yuv_frame, frame_shape)
    else:
        full = yuv420_to_rgb(yuv_frame, frame_shape)

    x1, y1, x2, y2 = region
    return full[y1:y2, x1:x2]


def create_tensor_input(
    yuv_frame: np.ndarray, frame_shape: tuple[int, int], model_config: ModelConfig, region: tuple[int, int, int, int]
) -> np.ndarray:
    """Crops `region` out of the full-resolution YUV420 frame, resizes to the model's
    input size if needed, and returns a [1, H, W, 3] batch tensor in the model's expected
    dtype. Layout permutation (NCHW etc.) is applied by the caller/backend if needed --
    this function always returns canonical NHWC.
    """
    cropped = crop_yuv_region(yuv_frame, frame_shape, region, model_config.pixel_format)

    target_shape = (model_config.height, model_config.width, 3)
    if cropped.shape != target_shape:
        cropped = cv2.resize(cropped, (model_config.width, model_config.height), interpolation=cv2.INTER_LINEAR)

    tensor = np.expand_dims(cropped, axis=0)

    if model_config.input_dtype == InputDType.float_:
        tensor = tensor.astype(np.float32) / 255.0
    elif model_config.input_dtype == InputDType.float_denorm:
        tensor = tensor.astype(np.float32)
    # InputDType.int_ (default): leave as uint8 passthrough.

    return tensor


def clamp_region_to_frame(region: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape
    x1, y1, x2, y2 = region
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)
