"""RemoteObjectDetector: the per-camera client that talks to the shared detector process.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.2.2.

Holds no model weights and no inference runtime -- only shared-memory handles and a queue
reference shared with every other camera's RemoteObjectDetector instance.
"""

from __future__ import annotations

import numpy as np

from mirage.config.schema import ModelConfig
from mirage.const import DETECTIONS_PER_ROW, MAX_DETECTIONS, OUTPUT_SHM_SIZE
from mirage.ipc.zmq_pubsub import Subscriber
from mirage.util.shm import SharedMemoryFrameManager, detector_input_shm_name, detector_output_shm_name

DEFAULT_WAIT_TIMEOUT_SECONDS = 5.0


class RemoteObjectDetector:
    def __init__(
        self,
        camera_name: str,
        labelmap: dict[int, str],
        detection_queue,
        model_config: ModelConfig,
        detector_sub_addr: str,
        frame_manager: SharedMemoryFrameManager | None = None,
        zmq_context=None,
    ) -> None:
        self.name = camera_name
        self.labels = labelmap
        self.detection_queue = detection_queue
        self.model_config = model_config

        self.frame_manager = frame_manager or SharedMemoryFrameManager()
        self.input_shm_name = detector_input_shm_name(camera_name)
        self.output_shm_name = detector_output_shm_name(camera_name)

        input_size = model_config.height * model_config.width * 3
        self.frame_manager.create(self.input_shm_name, input_size)  # idempotent if already exists
        self.frame_manager.create(self.output_shm_name, OUTPUT_SHM_SIZE)

        self.subscriber = Subscriber(
            detector_sub_addr, topic_filter=f"object_detector/{camera_name}", context=zmq_context
        )

    def detect(self, tensor_input: np.ndarray, threshold: float = 0.4,
               timeout: float = DEFAULT_WAIT_TIMEOUT_SECONDS) -> list[tuple[str, float, tuple]]:
        self.subscriber.drain_stale()  # defensive: clear any leftover messages

        input_buf = self.frame_manager.write(self.input_shm_name, tensor_input.nbytes)
        input_np = np.ndarray(tensor_input.shape, dtype=tensor_input.dtype, buffer=input_buf)
        input_np[:] = tensor_input[:]  # zero-copy handoff, no serialization

        self.detection_queue.put(self.name)

        got_signal = self.subscriber.wait_for_message(timeout=timeout)
        if not got_signal:
            return []  # detector overloaded -- give up for this call only; caller retries next frame

        output_np = self.frame_manager.get(self.output_shm_name, (MAX_DETECTIONS, DETECTIONS_PER_ROW), dtype=np.float32)
        results = []
        for row in output_np:
            class_id, score, y1, x1, y2, x2 = row
            if score < threshold:
                break  # rows are sorted descending, zero-padded
            results.append((self.labels.get(int(class_id), "unknown"), float(score), (float(y1), float(x1), float(y2), float(x2))))
        return results

    def close(self) -> None:
        """Detach only -- does NOT unlink the SHM segments. In the real system these
        segments are pre-created by the main-process maintainer (spec section 3.2) and
        shared across the camera's tracker process and the detector process, so ownership
        of the unlink lifecycle belongs to whichever component created them, not to every
        individual client that merely attached to them.
        """
        self.frame_manager.close(self.input_shm_name)
        self.frame_manager.close(self.output_shm_name)
        self.subscriber.close()

    def unlink(self) -> None:
        """Fully destroys the SHM segments. Only call this if THIS instance is also the
        owner that created them (e.g. in a test, or a standalone script) -- in the real
        multi-process system this is the main-process maintainer's responsibility (see
        mirage.util.shm.teardown_ring for the equivalent frame-ring lifecycle), not
        something every RemoteObjectDetector should do on its own close().
        """
        self.frame_manager.delete(self.input_shm_name)
        self.frame_manager.delete(self.output_shm_name)
        self.subscriber.close()
