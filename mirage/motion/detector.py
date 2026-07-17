"""Adaptive background-average motion detector.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 2 and Appendix B.3.

Algorithm: hand-rolled exponential-moving-average background model + contour extraction
(deliberately not OpenCV's built-in BackgroundSubtractorMOG2/KNN). Motion output does not
gate whether a frame is processed at all -- it determines which regions of the frame get
cropped and sent to the object detector (see mirage.regions).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from mirage.config.schema import MotionConfig


def rasterize_mask_polygons(
    polygons: list[list[tuple[float, float]]], width: int, height: int
) -> Optional[np.ndarray]:
    """Converts normalized [0,1] polygon coordinates into a uint8 mask image of shape
    (height, width), where 255 = inside a masked polygon, 0 = not masked. Returns None if
    no polygons are configured (i.e. no masking).
    """
    if not polygons:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array([(x * width, y * height) for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


class MotionDetector:
    def __init__(
        self,
        frame_shape: tuple[int, int],
        config: MotionConfig,
        blur_radius: int = 1,
        interpolation: int = cv2.INTER_NEAREST,
        contrast_frame_history: int = 50,
    ) -> None:
        self.config = config
        self.frame_shape = frame_shape  # (full_height, full_width) of the YUV luma plane
        frame_height = config.frame_height or frame_shape[0]
        self.resize_factor = frame_shape[0] / frame_height
        self.motion_frame_size = (frame_height, frame_height * frame_shape[1] // frame_shape[0])
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.motion_frame_count = 0
        self.blur_radius = blur_radius
        self.interpolation = interpolation
        self.calibrating = True
        self.contrast_values = np.zeros((contrast_frame_history, 2), np.uint8)
        self.contrast_values[:, 1:2] = 255
        self.contrast_values_index = 0

        rasterized = rasterize_mask_polygons(config.mask, frame_shape[1], frame_shape[0])
        self.update_mask(rasterized)

    def is_calibrating(self) -> bool:
        return self.calibrating

    def update_mask(self, rasterized_mask: Optional[np.ndarray]) -> None:
        if rasterized_mask is not None:
            resized_mask = cv2.resize(
                rasterized_mask,
                dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            self.mask = np.where(resized_mask != 0)
        else:
            self.mask = (np.array([], dtype=int), np.array([], dtype=int))
        # Reset so the background model is relearned quickly against the new mask.
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.calibrating = True
        self.motion_frame_count = 0

    def detect(self, luma_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """luma_frame: the Y (luma) plane only, shape (full_height, full_width), uint8."""
        motion_boxes: list[tuple[int, int, int, int]] = []
        if not self.config.enabled:
            return motion_boxes

        gray = luma_frame[0 : self.frame_shape[0], 0 : self.frame_shape[1]]
        resized_frame = cv2.resize(
            gray,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=self.interpolation,
        )

        if self.config.improve_contrast:
            min_value = np.percentile(resized_frame, 4).astype(np.uint8)
            max_value = np.percentile(resized_frame, 96).astype(np.uint8)
            if min_value < max_value:  # skip if the image is a single flat color
                self.contrast_values[self.contrast_values_index] = [min_value, max_value]
                self.contrast_values_index = (self.contrast_values_index + 1) % len(self.contrast_values)
                avg_min, avg_max = np.mean(self.contrast_values, axis=0)
                resized_frame = np.clip(resized_frame, avg_min, avg_max)
                resized_frame = (((resized_frame - avg_min) / (avg_max - avg_min)) * 255).astype(np.uint8)

        # Mask AFTER contrast stretch (so masked pixels don't skew the percentile calc),
        # but BEFORE blur/diff (so they never contribute to the background model).
        resized_frame[self.mask] = 0
        # cv2.GaussianBlur instead of scipy.ndimage.gaussian_filter (this function's
        # only non-cv2 call, in an otherwise all-cv2 hot path run unconditionally on
        # every frame of every camera) -- measured ~7x faster for this frame size
        # (OPTIMIZATION_OPPORTUNITIES.md item 7). ksize = 2*radius+1 matches scipy's
        # own kernel-size convention for an explicit `radius` argument.
        # BORDER_REFLECT (not cv2's default BORDER_REFLECT_101) is required to match
        # scipy's default 'reflect' edge handling -- confirmed by direct comparison:
        # BORDER_REFLECT_101 diverges by up to 61/255 at the edges, BORDER_REFLECT by
        # at most 2/255 everywhere (ordinary uint8 rounding, not a behavior change).
        blur_ksize = 2 * self.blur_radius + 1
        resized_frame = cv2.GaussianBlur(
            resized_frame, (blur_ksize, blur_ksize), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT,
        )

        frame_delta = cv2.absdiff(resized_frame, cv2.convertScaleAbs(self.avg_frame))
        thresh = cv2.threshold(frame_delta, self.config.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh_dilated = cv2.dilate(thresh, None, iterations=1)
        contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_contour_area = 0.0
        for c in contours:
            contour_area = cv2.contourArea(c)
            total_contour_area += contour_area
            if contour_area > self.config.contour_area:
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append(
                    (
                        int(x * self.resize_factor),
                        int(y * self.resize_factor),
                        int((x + w) * self.resize_factor),
                        int((y + h) * self.resize_factor),
                    )
                )

        pct_motion = total_contour_area / (self.motion_frame_size[0] * self.motion_frame_size[1])

        # skip_motion_threshold: drop ALL boxes for this frame and force recalibration.
        # Distinct from lightning_threshold below, which recalibrates WITHOUT dropping boxes.
        if self.config.skip_motion_threshold is not None and pct_motion > self.config.skip_motion_threshold:
            self.calibrating = True
            self._update_background(resized_frame, motion_boxes)
            return []

        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False
        if self.calibrating or pct_motion > self.config.lightning_threshold:
            self.calibrating = True

        self._update_background(resized_frame, motion_boxes)
        return motion_boxes

    def _update_background(self, resized_frame: np.ndarray, motion_boxes: list) -> None:
        """Debounced background accumulation: motion must persist >=10 consecutive frames
        before it's allowed to bleed into the background average (favors "this is
        actually a lighting change" over "this is a passing object").
        """
        if len(motion_boxes) > 0:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                cv2.accumulateWeighted(
                    resized_frame, self.avg_frame, 0.2 if self.calibrating else self.config.frame_alpha
                )
        else:
            cv2.accumulateWeighted(
                resized_frame, self.avg_frame, 0.2 if self.calibrating else self.config.frame_alpha
            )
            self.motion_frame_count = 0
