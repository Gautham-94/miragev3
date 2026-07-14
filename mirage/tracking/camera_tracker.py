"""CameraTracker: the per-camera OS process that runs motion detection, region
selection, object detection, and tracking.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.1 (CameraTracker), section 7
(per-frame orchestration loop).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_module

from mirage.config.schema import CameraConfig, DetectorInstanceConfig
from mirage.const import PROCESS_PRIORITY_HIGH
from mirage.detection.labelmap import load_labels
from mirage.detection.remote import RemoteObjectDetector
from mirage.motion.detector import MotionDetector
from mirage.tracking.orchestration import CameraOrchestrator
from mirage.tracking.tracker import ObjectTracker
from mirage.util.shm import SharedMemoryFrameManager

logger = logging.getLogger(__name__)

FRAME_QUEUE_POLL_TIMEOUT_SECONDS = 1.0


def camera_tracker_main(
    camera: CameraConfig,
    detector_config: DetectorInstanceConfig,
    detection_queue,
    detector_sub_addr: str,
    frame_queue,
    detected_frames_queue,
    stop_event,
) -> None:
    """Runs as the per-camera tracker OS process. Pulls (frame_name, frame_time) off
    frame_queue (populated by the CameraCapture process, see mirage.capture.capture),
    reads the actual pixel data from the shared frame ring, runs the full section 7
    per-frame pipeline, and publishes results to detected_frames_queue for the main
    process's result-consumer thread.
    """
    try:
        os.nice(PROCESS_PRIORITY_HIGH)
    except (AttributeError, PermissionError, OSError):
        pass

    logging.basicConfig(level=logging.INFO)
    frame_manager = SharedMemoryFrameManager()

    labels = load_labels(detector_config.model.labelmap_path)
    remote_detector = RemoteObjectDetector(
        camera_name=camera.name,
        labelmap=labels,
        detection_queue=detection_queue,
        model_config=detector_config.model,
        detector_sub_addr=detector_sub_addr,
        frame_manager=frame_manager,
    )
    motion_detector = MotionDetector(frame_shape=camera.frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)
    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, remote_detector)

    logger.info("%s: tracker ready", camera.name)

    while not stop_event.is_set():
        try:
            frame_name, frame_time = frame_queue.get(True, FRAME_QUEUE_POLL_TIMEOUT_SECONDS)
        except queue_module.Empty:
            continue
        except (OSError, EOFError):
            break

        yuv_frame = frame_manager.get(frame_name, camera.frame_shape_yuv)
        if yuv_frame is None:
            continue

        try:
            result = orchestrator.process_frame(yuv_frame, frame_time)
        except Exception:
            logger.exception("%s: error processing frame %s", camera.name, frame_name)
            frame_manager.close(frame_name)
            continue

        frame_manager.close(frame_name)

        try:
            detected_frames_queue.put(
                (camera.name, frame_name, frame_time, result.tracked_objects, result.motion_boxes, result.regions),
                block=False,
            )
        except queue_module.Full:
            pass  # backpressure: main process fell behind; drop this frame's publish

    remote_detector.close()
    logger.info("%s: tracker stopped", camera.name)


class CameraTracker(mp.Process):
    def __init__(
        self,
        camera: CameraConfig,
        detector_config: DetectorInstanceConfig,
        detection_queue,
        detector_sub_addr: str,
        frame_queue,
        detected_frames_queue,
        stop_event,
    ) -> None:
        super().__init__(name=f"tracker:{camera.name}")
        self.camera = camera
        self.detector_config = detector_config
        self.detection_queue = detection_queue
        self.detector_sub_addr = detector_sub_addr
        self.frame_queue = frame_queue
        self.detected_frames_queue = detected_frames_queue
        self.stop_event = stop_event

    def run(self) -> None:
        camera_tracker_main(
            self.camera,
            self.detector_config,
            self.detection_queue,
            self.detector_sub_addr,
            self.frame_queue,
            self.detected_frames_queue,
            self.stop_event,
        )
