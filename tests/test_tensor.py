from __future__ import annotations

import numpy as np

from mirage.config.schema import InputDType, ModelConfig, PixelFormat
from mirage.detection.tensor import clamp_region_to_frame, create_tensor_input, crop_yuv_region


def _synthetic_yuv_frame(height: int, width: int, y_value: int = 128) -> np.ndarray:
    """A flat YUV420 planar buffer: Y plane at y_value, U/V planes at neutral 128 (gray)."""
    yuv = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    yuv[:height, :] = y_value
    return yuv


def test_crop_yuv_region_rgb_shape():
    height, width = 120, 160
    yuv = _synthetic_yuv_frame(height, width)
    cropped = crop_yuv_region(yuv, (height, width), (10, 10, 60, 70), PixelFormat.rgb)
    assert cropped.shape == (60, 50, 3)


def test_crop_yuv_region_bgr_vs_rgb_channel_order_differs_for_non_gray():
    height, width = 64, 64
    yuv = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    yuv[:height, :] = 200  # bright luma
    yuv[height : height + height // 4, :] = 90  # skew U plane so R/B channels differ
    yuv[height + height // 4 :, :] = 160  # skew V plane

    rgb = crop_yuv_region(yuv, (height, width), (0, 0, width, height), PixelFormat.rgb)
    bgr = crop_yuv_region(yuv, (height, width), (0, 0, width, height), PixelFormat.bgr)

    # For a genuinely colored (non-gray) image, RGB and BGR must NOT be identical --
    # channel order actually differs.
    assert not np.array_equal(rgb, bgr)
    # But BGR reversed on the channel axis should equal RGB (same underlying pixels).
    assert np.array_equal(bgr[..., ::-1], rgb)


def test_create_tensor_input_shape_and_batch_dim():
    height, width = 120, 160
    yuv = _synthetic_yuv_frame(height, width)
    model_config = ModelConfig(width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb)

    tensor = create_tensor_input(yuv, (height, width), model_config, (0, 0, width, height))

    assert tensor.shape == (1, 320, 320, 3)
    assert tensor.dtype == np.uint8


def test_create_tensor_input_no_resize_when_region_already_model_size():
    model_config = ModelConfig(width=64, height=48, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb)
    yuv = _synthetic_yuv_frame(48, 64)

    tensor = create_tensor_input(yuv, (48, 64), model_config, (0, 0, 64, 48))
    assert tensor.shape == (1, 48, 64, 3)


def test_create_tensor_input_float_normalization():
    model_config = ModelConfig(width=32, height=32, input_dtype=InputDType.float_, pixel_format=PixelFormat.rgb)
    yuv = _synthetic_yuv_frame(32, 32, y_value=255)

    tensor = create_tensor_input(yuv, (32, 32), model_config, (0, 0, 32, 32))
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0


def test_create_tensor_input_float_denorm_no_scaling():
    model_config = ModelConfig(width=32, height=32, input_dtype=InputDType.float_denorm, pixel_format=PixelFormat.rgb)
    yuv = _synthetic_yuv_frame(32, 32, y_value=200)

    tensor = create_tensor_input(yuv, (32, 32), model_config, (0, 0, 32, 32))
    assert tensor.dtype == np.float32
    # Not normalized -- values should still be in roughly the original 0-255ish range
    # (allowing for YUV->RGB conversion drift), definitely not clipped to [0,1].
    assert tensor.max() > 1.0


def test_clamp_region_to_frame():
    frame_shape = (100, 200)  # height, width
    assert clamp_region_to_frame((-10, -5, 50, 60), frame_shape) == (0, 0, 50, 60)
    assert clamp_region_to_frame((150, 90, 250, 150), frame_shape) == (150, 90, 200, 100)
    assert clamp_region_to_frame((10, 10, 50, 50), frame_shape) == (10, 10, 50, 50)
