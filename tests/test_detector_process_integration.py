"""Real end-to-end test of the shared detector-process architecture: a real DetectorProcess
(separate OS process) loading the real yolov8n.onnx model, and a real RemoteObjectDetector
client feeding it a real photo's tensor via shared memory + the mp.Queue request path +
the ZMQ detector-ready signal -- validating the exact section 3.2 architecture end-to-end,
not just each piece in isolation.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path

import cv2
import pytest

from mirage.config.schema import DetectorInstanceConfig, InputDType, ModelConfig, PixelFormat
from mirage.detection.labelmap import load_labels
from mirage.detection.process import DetectorProcess
from mirage.detection.remote import RemoteObjectDetector
from mirage.detection.tensor import create_tensor_input
from mirage.ipc.zmq_pubsub import ZmqProxy
from mirage.util.shm import SharedMemoryFrameManager

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
BUS_IMAGE = Path(__file__).resolve().parent.parent / "media" / "bus.jpg"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and BUS_IMAGE.exists()),
    reason="yolov8n.onnx model or test image fixture not available",
)


@pytest.fixture
def short_ipc_dir():
    d = tempfile.mkdtemp(prefix="mrgdet", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_detector_process_serves_a_real_camera_client(short_ipc_dir):
    ctx = mp.get_context("spawn")

    detector_config = DetectorInstanceConfig(
        name="general",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )

    camera_name = "detector_test_cam"
    pub_addr = f"ipc://{short_ipc_dir}/detpub"
    sub_addr = f"ipc://{short_ipc_dir}/detsub"

    proxy = ZmqProxy(pub_addr, sub_addr)

    frame_manager = SharedMemoryFrameManager()
    detection_queue = ctx.Queue()
    stop_event = ctx.Event()

    detector_process = DetectorProcess(
        detector_config=detector_config,
        detection_queue=detection_queue,
        camera_names=[camera_name],
        detector_pub_addr=pub_addr,
        stop_event=stop_event,
    )

    labels = load_labels(str(LABELMAP_PATH))
    remote_detector = RemoteObjectDetector(
        camera_name=camera_name,
        labelmap=labels,
        detection_queue=detection_queue,
        model_config=detector_config.model,
        detector_sub_addr=sub_addr,
        frame_manager=frame_manager,
    )

    try:
        detector_process.start()

        bgr = cv2.imread(str(BUS_IMAGE))
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame_shape = bgr.shape[:2]
        tensor = create_tensor_input(yuv, frame_shape, detector_config.model, (0, 0, frame_shape[1], frame_shape[0]))

        detections = remote_detector.detect(tensor, threshold=0.4, timeout=30.0)

        assert len(detections) > 0, "expected at least one real detection from the shared detector process"
        detected_labels = {label for label, score, box in detections}
        assert "bus" in detected_labels
        assert "person" in detected_labels

        for label, score, box in detections:
            assert 0.0 < score <= 1.0
            y1, x1, y2, x2 = box
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0

    finally:
        stop_event.set()
        detector_process.join(timeout=15)
        if detector_process.is_alive():
            detector_process.terminate()
            detector_process.join(timeout=5)
        remote_detector.unlink()
        proxy.close()


def test_detector_process_serves_a_camera_with_a_long_real_world_name(short_ipc_dir):
    """Regression test: a real camera name longer than this codebase's usual short test
    fixtures (e.g. "hikvision_ds_2cd1023g0e_i", added for real through the add-camera
    wizard) broke end-to-end for real -- DetectorProcess logged "no input SHM for
    camera ..." and detection silently never ran for that camera. Root cause: SHM names
    are now a short derived key (mirage.util.shm._shm_key, needed since macOS caps POSIX
    shm names at 30 chars and long real camera names overflow that), and
    RemoteObjectDetector (the writer) already used detector_input_shm_name() correctly,
    but DetectorProcess's own read/close calls used the raw camera_name directly instead
    -- invisible before _shm_key existed, since detector_input_shm_name() used to just
    return the raw name unchanged, so both sides "agreed" by coincidence. This test uses
    a name specifically long enough that a bug in either the write or the read path
    would produce mismatched SHM segment names and silently drop every detection.
    """
    ctx = mp.get_context("spawn")

    detector_config = DetectorInstanceConfig(
        name="general3",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )

    camera_name = "hikvision_ds_2cd1023g0e_i_long_name_test"
    pub_addr = f"ipc://{short_ipc_dir}/detpub3"
    sub_addr = f"ipc://{short_ipc_dir}/detsub3"
    proxy = ZmqProxy(pub_addr, sub_addr)

    frame_manager = SharedMemoryFrameManager()
    detection_queue = ctx.Queue()
    stop_event = ctx.Event()

    detector_process = DetectorProcess(
        detector_config=detector_config, detection_queue=detection_queue,
        camera_names=[camera_name], detector_pub_addr=pub_addr, stop_event=stop_event,
    )
    labels = load_labels(str(LABELMAP_PATH))
    remote_detector = RemoteObjectDetector(
        camera_name=camera_name, labelmap=labels, detection_queue=detection_queue,
        model_config=detector_config.model, detector_sub_addr=sub_addr, frame_manager=frame_manager,
    )

    try:
        detector_process.start()
        bgr = cv2.imread(str(BUS_IMAGE))
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame_shape = bgr.shape[:2]
        tensor = create_tensor_input(yuv, frame_shape, detector_config.model, (0, 0, frame_shape[1], frame_shape[0]))

        detections = remote_detector.detect(tensor, threshold=0.4, timeout=30.0)

        assert len(detections) > 0, (
            "expected real detections for a long-camera-name client -- if this is empty, "
            "the write (RemoteObjectDetector) and read (DetectorProcess) sides are using "
            "mismatched SHM names again"
        )
        detected_labels = {label for label, score, box in detections}
        assert "bus" in detected_labels or "person" in detected_labels
    finally:
        stop_event.set()
        detector_process.join(timeout=15)
        if detector_process.is_alive():
            detector_process.terminate()
            detector_process.join(timeout=5)
        remote_detector.unlink()
        proxy.close()


def test_detector_process_handles_multiple_sequential_requests(short_ipc_dir):
    """Confirms the shared detector process correctly serves more than one request in a
    row without needing to be restarted -- i.e. it's a genuine long-running server loop,
    not a one-shot.
    """
    ctx = mp.get_context("spawn")
    detector_config = DetectorInstanceConfig(
        name="general2",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )
    camera_name = "detector_test_cam2"
    pub_addr = f"ipc://{short_ipc_dir}/detpub2"
    sub_addr = f"ipc://{short_ipc_dir}/detsub2"
    proxy = ZmqProxy(pub_addr, sub_addr)

    frame_manager = SharedMemoryFrameManager()
    detection_queue = ctx.Queue()
    stop_event = ctx.Event()

    detector_process = DetectorProcess(
        detector_config=detector_config, detection_queue=detection_queue,
        camera_names=[camera_name], detector_pub_addr=pub_addr, stop_event=stop_event,
    )
    labels = load_labels(str(LABELMAP_PATH))
    remote_detector = RemoteObjectDetector(
        camera_name=camera_name, labelmap=labels, detection_queue=detection_queue,
        model_config=detector_config.model, detector_sub_addr=sub_addr, frame_manager=frame_manager,
    )

    try:
        detector_process.start()
        bgr = cv2.imread(str(BUS_IMAGE))
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame_shape = bgr.shape[:2]
        tensor = create_tensor_input(yuv, frame_shape, detector_config.model, (0, 0, frame_shape[1], frame_shape[0]))

        for i in range(3):
            detections = remote_detector.detect(tensor, threshold=0.4, timeout=30.0)
            assert len(detections) > 0, f"request {i}: expected detections"
            assert {label for label, _, _ in detections} & {"bus", "person"}
    finally:
        stop_event.set()
        detector_process.join(timeout=15)
        if detector_process.is_alive():
            detector_process.terminate()
            detector_process.join(timeout=5)
        remote_detector.unlink()
        proxy.close()
