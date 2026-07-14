"""DetectorProcess: the single dedicated OS process that loads a model once and serves
detection requests for every camera routed to it.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.2, 3.2.1.
NVR_EXTENSION_MULTI_MODEL_ROUTING.md section 2 (per-camera detector routing).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_module
import time

from mirage.config.schema import DetectorInstanceConfig
from mirage.const import DETECTIONS_PER_ROW, MAX_DETECTIONS, OUTPUT_SHM_SIZE, PROCESS_PRIORITY_HIGH
from mirage.detection.registry import create_detector
from mirage.ipc.zmq_pubsub import Publisher
from mirage.util.shm import SharedMemoryFrameManager, detector_input_shm_name, detector_output_shm_name

logger = logging.getLogger(__name__)


def detector_process_main(
    detector_config: DetectorInstanceConfig,
    detection_queue,
    camera_names: list[str],
    detector_pub_addr: str,
    detection_start,  # multiprocessing.Value("d", 0.0)
    avg_inference_speed,  # multiprocessing.Value("d", 0.0)
    stop_event,
) -> None:
    """Runs as the single dedicated OS process for one configured detector/accelerator.
    The model is loaded exactly ONCE here, regardless of how many cameras exist -- camera
    processes never load a model themselves; they only hold a RemoteObjectDetector client
    that talks to this process via shared memory + a tiny pub/sub "done" signal.
    """
    try:
        os.nice(PROCESS_PRIORITY_HIGH)
    except (AttributeError, PermissionError, OSError):
        pass

    logging.basicConfig(level=logging.INFO)
    frame_manager = SharedMemoryFrameManager()
    detector = create_detector(detector_config)
    publisher = Publisher(detector_pub_addr)

    import numpy as np

    for name in camera_names:
        frame_manager.create(detector_output_shm_name(name), OUTPUT_SHM_SIZE)

    logger.info("detector %s ready, serving cameras: %s", detector_config.name, camera_names)

    while not stop_event.is_set():
        try:
            camera_name = detection_queue.get(timeout=1)
        except queue_module.Empty:
            continue
        except (OSError, EOFError):
            break

        input_shm_name = detector_input_shm_name(camera_name)
        input_tensor = frame_manager.get(
            input_shm_name, (1, detector_config.model.height, detector_config.model.width, 3)
        )
        if input_tensor is None:
            logger.warning("detector %s: no input SHM for camera %s", detector_config.name, camera_name)
            continue

        detection_start.value = time.time()
        try:
            detections = detector.detect_raw(input_tensor)
        except Exception:
            logger.exception("detector %s: inference failed for camera %s", detector_config.name, camera_name)
            detection_start.value = 0.0
            continue
        duration = time.time() - detection_start.value
        frame_manager.close(input_shm_name)

        output_buf = frame_manager.write(detector_output_shm_name(camera_name), OUTPUT_SHM_SIZE)
        output_np = np.ndarray((MAX_DETECTIONS, DETECTIONS_PER_ROW), dtype=np.float32, buffer=output_buf)
        output_np[:] = detections[:]

        publisher.publish(f"object_detector/{camera_name}", "")

        detection_start.value = 0.0
        avg_inference_speed.value = (avg_inference_speed.value * 9 + duration) / 10

    publisher.close()


class DetectorProcess(mp.Process):
    def __init__(
        self,
        detector_config: DetectorInstanceConfig,
        detection_queue,
        camera_names: list[str],
        detector_pub_addr: str,
        stop_event,
    ) -> None:
        super().__init__(name=f"detector:{detector_config.name}")
        self.detector_config = detector_config
        self.detection_queue = detection_queue
        self.camera_names = camera_names
        self.detector_pub_addr = detector_pub_addr
        self.stop_event = stop_event
        ctx = mp.get_context()
        self.detection_start = ctx.Value("d", 0.0)
        self.avg_inference_speed = ctx.Value("d", 0.01)

    def run(self) -> None:
        detector_process_main(
            self.detector_config,
            self.detection_queue,
            self.camera_names,
            self.detector_pub_addr,
            self.detection_start,
            self.avg_inference_speed,
            self.stop_event,
        )
