"""OpenVocabDispatcher: lives in the main process's result-consumer loop (see
mirage.app.MirageApp._result_consumer_loop), sitting alongside EventProcessor and
ReviewSegmentMaintainer as a third consumer of each frame's tracked_objects.

Two independent dispatch modes, selected per-camera by CameraConfig.openvocab_direct_frame:

1. **Confirmed-object mode** (default, openvocab_direct_frame=False): for every
   CONFIRMED tracked object (state.is_false_positive is False -- Gate 1) that has at
   least one enabled OpenVocabQuery scoped to this camera, crop its box out of the frame
   RIGHT NOW (synchronously, in this call) and check the perceptual-hash gate (Gate 2,
   mirage.openvocab.gating.OpenVocabGate, keyed by object id). This is TODO_FIX_LIST.md
   item 4's original design -- OWLv2 only ever sees crops of things the closed-vocab
   detector already found, so it can enrich/refine a detection but can't discover
   something outside that detector's label map.
2. **Direct-frame mode** (openvocab_direct_frame=True): skips the confirmed-object
   requirement entirely -- gated on motion presence instead (motion_boxes non-empty),
   and the WHOLE frame (not an object crop) is sent to OWLv2. This is for queries about
   things the closed-vocab detector was never trained to recognize at all (so it would
   never produce a confirmed object for OWLv2 to enrich in mode 1). The perceptual-hash
   gate (Gate 2) still applies, keyed by camera name instead of object id, since there's
   no tracked object to key it by.

In both modes: if the gates pass, JPEG-encode the crop/frame and enqueue an
OpenVocabRequest onto the OpenVocabProcess's request_queue; separately, drain any
OpenVocabResults that have come back from a previous request and persist real matches as
QueryMatch rows.

The crop/frame MUST be read out of the shared-memory frame ring and encoded to a plain
bytes JPEG synchronously here, in this call -- NOT deferred -- because the ring buffer is
finite-depth (mirage/util/shm.py's calculate_shm_ring_depth, typically 20-50 slots) and
gets overwritten by newer frames continuously; by the time OWLv2 finishes a single
request (~1-1.7s measured, TODO_FIX_LIST.md item 6), the same frame_name slot could
easily have wrapped around and been overwritten multiple times over. Once JPEG-encoded
into the request, the crop/frame is a plain, self-contained bytes payload with no further
dependency on the ring buffer's lifetime.
"""

from __future__ import annotations

import logging
import queue as queue_module
import time
import uuid

import cv2
import numpy as np

from mirage.config.schema import CameraConfig, MirageConfig, PixelFormat
from mirage.const import QUERY_MATCH_THUMB_DIR
from mirage.db.models import QueryMatch
from mirage.detection.tensor import crop_yuv_region, yuv420_to_rgb
from mirage.openvocab.gating import OpenVocabGate
from mirage.openvocab.process import OpenVocabRequest, OpenVocabResult
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.shm import SharedMemoryFrameManager
from mirage.util.time import utc_from_timestamp

logger = logging.getLogger(__name__)

# Placeholder object_id for direct-frame-mode matches, which have no real tracked object
# behind them -- QueryMatch.object_id is a required column, and this still uniquely
# identifies which specific frame produced the match without needing a schema change.
_DIRECT_FRAME_OBJECT_ID_PREFIX = "frame-"


class OpenVocabDispatcher:
    def __init__(
        self,
        request_queue,
        result_queue,
        frame_manager: SharedMemoryFrameManager,
        thumb_dir: str = QUERY_MATCH_THUMB_DIR,
    ) -> None:
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.frame_manager = frame_manager
        self.thumb_dir = thumb_dir
        self.gate = OpenVocabGate()
        # request_id -> (camera_name, object_id, frame_time) -- so a returned result can
        # be matched back to which frame/object it was for, since OpenVocabResult only
        # carries what OpenVocabProcess itself was given (it never sees frame_time).
        self._pending: dict[str, tuple[str, str, float]] = {}

    def process_frame(
        self,
        config: MirageConfig,
        camera: CameraConfig,
        frame_name: str,
        frame_time: float,
        tracked_objects: dict[str, TrackedObjectState],
        motion_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._drain_results()

        active_queries = [
            (q.id, q.text) for q in config.queries if q.enabled and (not q.cameras or camera.name in q.cameras)
        ]
        if not active_queries:
            return  # no queries apply to this camera at all -- skip the SHM read entirely

        if camera.openvocab_direct_frame:
            self._process_direct_frame(camera, frame_name, frame_time, motion_boxes or [], active_queries)
        else:
            self._process_confirmed_objects(camera, frame_name, frame_time, tracked_objects, active_queries)

    def _process_confirmed_objects(
        self,
        camera: CameraConfig,
        frame_name: str,
        frame_time: float,
        tracked_objects: dict[str, TrackedObjectState],
        active_queries: list[tuple[str, str]],
    ) -> None:
        confirmed = {oid: state for oid, state in tracked_objects.items() if not state.is_false_positive}
        if not confirmed:
            return

        yuv_frame = self.frame_manager.get(frame_name, camera.frame_shape_yuv)
        if yuv_frame is None:
            return  # ring slot already gone -- nothing to do this frame

        for object_id, state in confirmed.items():
            box = tuple(int(v) for v in state.box)  # (x1, y1, x2, y2) full-frame pixel coords
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = crop_yuv_region(yuv_frame, camera.frame_shape, box, PixelFormat.rgb)
            if crop_rgb.size == 0:
                continue

            if not self.gate.should_dispatch(object_id, crop_rgb, now=frame_time):
                continue

            self._encode_and_dispatch(
                camera, object_id, crop_rgb, frame_box=(float(y1), float(x1), float(y2), float(x2)),
                frame_time=frame_time, active_queries=active_queries,
            )

    def _process_direct_frame(
        self,
        camera: CameraConfig,
        frame_name: str,
        frame_time: float,
        motion_boxes: list[tuple[int, int, int, int]],
        active_queries: list[tuple[str, str]],
    ) -> None:
        if not motion_boxes:
            return  # nothing moving -- don't burn OWLv2 on a static scene

        gate_key = f"camera:{camera.name}"

        yuv_frame = self.frame_manager.get(frame_name, camera.frame_shape_yuv)
        if yuv_frame is None:
            return

        height, width = camera.frame_shape
        full_frame_rgb = yuv420_to_rgb(yuv_frame, camera.frame_shape)[:height, :width]

        if not self.gate.should_dispatch(gate_key, full_frame_rgb, now=frame_time):
            return

        object_id = f"{_DIRECT_FRAME_OBJECT_ID_PREFIX}{frame_time}"
        self._encode_and_dispatch(
            camera, object_id, full_frame_rgb, frame_box=(0.0, 0.0, float(height), float(width)),
            frame_time=frame_time, active_queries=active_queries, gate_key=gate_key,
        )

    def _encode_and_dispatch(
        self,
        camera: CameraConfig,
        object_id: str,
        image_rgb: np.ndarray,
        frame_box: tuple[float, float, float, float],
        frame_time: float,
        active_queries: list[tuple[str, str]],
        gate_key: str | None = None,
    ) -> None:
        gate_key = gate_key or object_id
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            self.gate.mark_result_received(gate_key)  # don't leave it permanently "in flight"
            return

        request_id = uuid.uuid4().hex
        request = OpenVocabRequest(
            request_id=request_id,
            camera_name=camera.name,
            object_id=object_id,
            crop_jpeg=encoded.tobytes(),
            frame_box=frame_box,
            queries=active_queries,
        )
        try:
            self.request_queue.put(request, block=False)
        except queue_module.Full:
            logger.warning("openvocab: request_queue full, skipping check for %s/%s", camera.name, object_id)
            self.gate.mark_result_received(gate_key)
            return

        self._pending[request_id] = (camera.name, object_id, frame_time)

    def forget_object(self, object_id: str) -> None:
        self.gate.forget(object_id)

    def _drain_results(self) -> None:
        while True:
            try:
                result: OpenVocabResult = self.result_queue.get(block=False)
            except queue_module.Empty:
                return
            except (OSError, EOFError):
                return

            self.gate.mark_result_received(result.object_id)
            pending = self._pending.pop(result.request_id, None)

            for match in result.matches:
                match_id = f"{time.time()}-{uuid.uuid4().hex[:6]}"
                thumb_path = self._save_thumb(result, match_id)
                QueryMatch.create(
                    id=match_id,
                    query_id=match.query_id,
                    query_text=match.query_text,
                    camera=result.camera_name,
                    object_id=result.object_id,
                    matched_at=utc_from_timestamp(pending[2]) if pending else utc_from_timestamp(time.time()),
                    score=match.score,
                    box=list(match.box),
                    thumb_path=thumb_path,
                )
                logger.info(
                    "openvocab: match found -- camera=%s object=%s query=%r score=%.3f",
                    result.camera_name, result.object_id, match.query_text, match.score,
                )

    def _save_thumb(self, result: OpenVocabResult, match_id: str) -> str | None:
        """Writes the exact crop OWLv2 was actually checking (echoed back on
        OpenVocabResult.crop_jpeg) to disk as this match's thumbnail. Best-effort: any
        failure returns None rather than raising, matching capture_thumbnail's
        established pattern elsewhere in this codebase (mirage/util/thumbnail.py) --
        a missing thumbnail should never break match persistence itself.
        """
        if not result.crop_jpeg:
            return None
        from pathlib import Path

        path = Path(self.thumb_dir) / f"{match_id}.jpg"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(result.crop_jpeg)
        except OSError:
            logger.exception("openvocab: failed to write match thumbnail for %s", match_id)
            return None
        return str(path)
