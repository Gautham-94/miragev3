"""OpenVocabDispatcher: lives in the main process's result-consumer loop (see
mirage.app.MirageApp._result_consumer_loop), sitting alongside EventProcessor and
ReviewSegmentMaintainer as a third consumer of each frame's tracked_objects -- AND,
since the track_objects/open-vocab bridge was built (see synthetic_tracked_objects()
below), as a PRODUCER that feeds synthetic entries back INTO tracked_objects before
EventProcessor/ReviewSegmentMaintainer run, so a real OWLv2 match for an
auto-provisioned query (e.g. "animals", typed into a camera's Track objects field --
see mirage/api/routers/config.py's _sync_track_object_queries) drives a real Event and
ReviewSegment through the EXACT SAME code path a closed-vocab detection does, rather
than being a second, disconnected notification system. See synthetic_tracked_objects()'s
own docstring for why this has to happen in _drain_results, not process_frame.

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

import dataclasses
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
from mirage.notify_bus import NotifyEvent
from mirage.openvocab.gating import OpenVocabGate
from mirage.openvocab.process import OpenVocabMatch, OpenVocabRequest, OpenVocabResult
from mirage.tracking.stationary import StationaryClassifier, iou
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.shm import SharedMemoryFrameManager
from mirage.util.time import utc_from_timestamp

logger = logging.getLogger(__name__)

# Placeholder object_id for direct-frame-mode matches, which have no real tracked object
# behind them -- QueryMatch.object_id is a required column, and this still uniquely
# identifies which specific frame produced the match without needing a schema change.
_DIRECT_FRAME_OBJECT_ID_PREFIX = "frame-"

# How long a synthetic tracked object (see SyntheticTrack below) stays "alive" in
# tracked_objects after its most recent real OWLv2 match, before EventProcessor/
# ReviewSegmentMaintainer see it disappear and close out the Event/ReviewSegment it
# drove. Deliberately generous relative to OpenVocabGate's own MIN_RECHECK_INTERVAL_
# SECONDS (5s, mirage/openvocab/gating.py) -- a real animal standing in frame for 20s
# might only get re-matched every 5-10s (Gate 2's hash-dedup can skip a recheck if the
# crop looks unchanged), and this needs to comfortably outlive that gap so a still-
# present animal doesn't flicker its Event/ReviewSegment closed and reopened between
# individual OWLv2 calls.
SYNTHETIC_TRACK_TTL_SECONDS = 20.0

# Minimum IoU (see mirage.tracking.stationary.iou) between an incoming match's box and
# an existing live SyntheticTrack for the SAME (camera, query_text) to be treated as
# "the same ongoing sighting, box moved a bit" rather than "a genuinely different,
# simultaneous instance" (e.g. a second animal that entered frame while the first one
# was still there). No real tuning basis yet -- chosen as a permissive-but-not-trivial
# starting point, same caveat as OpenVocabGate.DEFAULT_HAMMING_THRESHOLD.
SYNTHETIC_TRACK_MATCH_IOU_THRESHOLD = 0.3


@dataclasses.dataclass
class SyntheticTrack:
    """One DISTINCT open-vocab sighting's "currently active" state for one camera and
    query, bridging a real OWLv2 match into the SAME dict[str, TrackedObjectState]
    shape EventProcessor and ReviewSegmentMaintainer already consume from the
    closed-vocab pipeline -- see this module's top docstring for why.

    OpenVocabDispatcher._synthetic_tracks is keyed by (camera_name, query_text) ->
    list[SyntheticTrack], NOT a single track per key -- a single OWLv2 call can
    genuinely return multiple simultaneous matches for the same query (e.g. two
    separate animals both in frame at once; confirmed directly against real garage
    footage, where one direct-frame dispatch returned several distinct boxes for
    "animals" in a single response). Collapsing them all into one track per
    (camera, query) would silently undercount multi-instance activity -- "how many
    animals" would always read as "an animal, at most one" regardless of how many were
    actually present. Each incoming match is instead matched by IoU (see
    SYNTHETIC_TRACK_MATCH_IOU_THRESHOLD) against this query's OTHER currently-live
    tracks for this camera: a box that clearly overlaps an existing track updates it in
    place (same object_id, so EventProcessor sees one continuous Event); a box that
    doesn't overlap any existing live track well enough becomes a NEW track (new
    object_id, a genuinely separate Event) -- the same greedy nearest-neighbor
    association idea norfair itself uses for the closed-vocab tracker, just without a
    Kalman filter, since OWLv2 calls are too infrequent (seconds apart) for
    frame-to-frame motion prediction to be meaningful here anyway.

    Unlike a real norfair track, a match has no continuous identity across frames
    (direct-frame mode's object_id is a fresh "frame-<timestamp>" every single call,
    and even confirmed-object mode's object_id is the CLOSED-vocab tracker's id, not
    something OWLv2 itself maintains) -- IoU-based association within (camera, query)
    is the only stable identity a repeated match for "the same ongoing sighting" can
    have here.
    """

    object_id: str  # synthetic TrackedObjectState.id, stable for this sighting's lifetime
    label: str  # the query text itself, e.g. "animals" -- becomes Event.label / ReviewSegment's objects entry
    box: tuple[float, float, float, float]  # last match's box, full-frame pixel coords (x1, y1, x2, y2)
    score: float
    last_seen: float  # frame_time of the most recent real match, for TTL expiry
    # The exact frame OWLv2 was checking for this match, ONLY for direct-frame-mode
    # matches (None for confirmed-object mode) -- see synthetic_frame_jpegs()'s
    # docstring for why confirmed-object mode's crop can't be used the same way.
    # Updated on every subsequent match too (like `box`/`score` above), so an Event
    # snapshot always reflects the MOST RECENT sighting's frame, not the first one.
    frame_jpeg: bytes | None = None


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
        # Set by MirageApp AFTER construction -- an mp.Queue feeding the frontend's live
        # SSE stream (see mirage.notify_bus). None means notification is silently
        # skipped (e.g. direct-construction tests).
        self.notify_queue = None
        # request_id -> (camera_name, object_id, frame_time) -- so a returned result can
        # be matched back to which frame/object it was for, since OpenVocabResult only
        # carries what OpenVocabProcess itself was given (it never sees frame_time).
        self._pending: dict[str, tuple[str, str, float]] = {}
        # camera_name -> query_text -> list[SyntheticTrack] -- see SyntheticTrack's
        # docstring for why this is a LIST (multiple simultaneous matches for the same
        # query, e.g. two animals at once, must become two independent tracks, not one
        # that silently overwrites the other). Populated in _drain_results (a real
        # match arriving), read by synthetic_tracked_objects() (called once per camera
        # per frame from mirage.app._result_consumer_loop, BEFORE EventProcessor/
        # ReviewSegmentMaintainer run for that frame) and expired there too.
        self._synthetic_tracks: dict[str, dict[str, list[SyntheticTrack]]] = {}

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

    def _notify(self, row_id: str, op: str) -> None:
        if self.notify_queue is None:
            return
        try:
            self.notify_queue.put_nowait(NotifyEvent(table="query_match", id=row_id, op=op))
        except Exception:
            pass  # notify queue backpressure/full -- never let SSE affect dispatch

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
            matched_frame_time = pending[2] if pending else time.time()

            for match in result.matches:
                match_id = f"{time.time()}-{uuid.uuid4().hex[:6]}"
                thumb_path = self._save_thumb(result, match_id)
                QueryMatch.create(
                    id=match_id,
                    query_id=match.query_id,
                    query_text=match.query_text,
                    camera=result.camera_name,
                    object_id=result.object_id,
                    matched_at=utc_from_timestamp(matched_frame_time),
                    score=match.score,
                    box=list(match.box),
                    thumb_path=thumb_path,
                )
                self._notify(match_id, "create")
                logger.info(
                    "openvocab: match found -- camera=%s object=%s query=%r score=%.3f",
                    result.camera_name, result.object_id, match.query_text, match.score,
                )
                # Direct-frame mode's crop_jpeg IS the whole motion-triggered frame
                # (not an object crop), already in the exact coordinate space
                # match.box is expressed in -- confirmed-object mode's crop_jpeg is a
                # small crop of a single object, not usable as a whole-frame snapshot
                # background, so it's deliberately excluded here (frame_jpeg stays
                # None, and EventProcessor falls back to its existing no-frame path).
                is_direct_frame_match = result.object_id.startswith(_DIRECT_FRAME_OBJECT_ID_PREFIX)
                match_frame_jpeg = result.crop_jpeg if is_direct_frame_match else None
                self._upsert_synthetic_track(result.camera_name, match, matched_frame_time, match_frame_jpeg)

    def _upsert_synthetic_track(
        self, camera_name: str, match: OpenVocabMatch, frame_time: float, frame_jpeg: bytes | None = None,
    ) -> None:
        """Feeds a real match into the SAME dict[str, TrackedObjectState] shape
        EventProcessor/ReviewSegmentMaintainer already consume -- see this module's top
        docstring and SyntheticTrack's docstring for why this bridge exists, and why
        _synthetic_tracks holds a LIST per (camera, query_text) rather than a single
        track. Associates this match against this query's OTHER currently-live tracks
        for this camera by IoU (greedy nearest-neighbor, same idea norfair's own
        association step uses): a box that clearly overlaps an existing live track
        updates it IN PLACE (same object_id, so EventProcessor sees one continuous
        Event rather than a new one per OWLv2 call); a box that doesn't overlap any
        existing live track well enough becomes a brand-new track (new object_id, a
        genuinely separate Event) -- this is what makes two simultaneous matches for
        the same query (e.g. two animals at once) surface as two independent alerts
        instead of silently collapsing into one.
        """
        y1, x1, y2, x2 = match.box  # OpenVocabMatch.box is (y1, x1, y2, x2) -- see process.py
        box = (x1, y1, x2, y2)  # TrackedObjectState.box convention -- see crop_yuv_region's docstring

        by_query = self._synthetic_tracks.setdefault(camera_name, {})
        existing_tracks = by_query.setdefault(match.query_text, [])

        # Only associate against tracks that are actually still "live" right now (not
        # expired) -- an old track that hasn't been touched in a while shouldn't have a
        # brand-new, unrelated sighting silently glommed onto its stale identity just
        # because it happens to be the best IoU match among a list that's mostly dead.
        live_candidates = [t for t in existing_tracks if frame_time - t.last_seen <= SYNTHETIC_TRACK_TTL_SECONDS]

        best_match: SyntheticTrack | None = None
        best_iou = 0.0
        for track in live_candidates:
            score = iou(box, track.box)
            if score > best_iou:
                best_iou = score
                best_match = track

        if best_match is not None and best_iou >= SYNTHETIC_TRACK_MATCH_IOU_THRESHOLD:
            best_match.box = box
            best_match.score = match.score
            best_match.last_seen = frame_time
            if frame_jpeg is not None:
                best_match.frame_jpeg = frame_jpeg
        else:
            existing_tracks.append(
                SyntheticTrack(
                    object_id=f"openvocab-{uuid.uuid4().hex}", label=match.query_text,
                    box=box, score=match.score, last_seen=frame_time, frame_jpeg=frame_jpeg,
                )
            )

    def synthetic_tracked_objects(self, camera_name: str, now: float) -> dict[str, TrackedObjectState]:
        """Called once per camera per frame from mirage.app._result_consumer_loop,
        BEFORE EventProcessor.process()/ReviewSegmentMaintainer.process() run for that
        frame -- returns every still-live (matched within SYNTHETIC_TRACK_TTL_SECONDS of
        `now`) synthetic track for this camera as real TrackedObjectState instances --
        POSSIBLY MULTIPLE per query_text, see SyntheticTrack's docstring -- so the
        caller can merge them into that frame's tracked_objects dict and let
        EventProcessor/ReviewSegmentMaintainer drive a real Event/ReviewSegment for
        each distinct open-vocab sighting exactly as they would for a closed-vocab
        detection. Expired entries are dropped from internal state here too (not just
        filtered from the return value), so a synthetic track's id disappearing from
        tracked_objects next call is what makes EventProcessor's start/update/end diff
        correctly close out its Event -- same mechanism a real object's track ending
        already uses.
        """
        by_query = self._synthetic_tracks.get(camera_name)
        if not by_query:
            return {}

        live: dict[str, TrackedObjectState] = {}
        empty_queries = []
        for query_text, tracks in by_query.items():
            still_live = [t for t in tracks if now - t.last_seen <= SYNTHETIC_TRACK_TTL_SECONDS]
            by_query[query_text] = still_live
            if not still_live:
                empty_queries.append(query_text)
                continue
            for track in still_live:
                live[track.object_id] = self._to_tracked_object_state(track)

        for query_text in empty_queries:
            del by_query[query_text]
        if not by_query:
            self._synthetic_tracks.pop(camera_name, None)

        return live

    def synthetic_frame_jpegs(self, camera_name: str) -> dict[str, bytes]:
        """Companion to synthetic_tracked_objects(): the per-object_id clean-frame
        override EventProcessor._on_start needs to give a direct-frame-mode open-vocab
        Event a real snapshot with a real box (TODO_FIX_LIST.md item 9/11's "no shared
        frame" gap) -- confirmed-object-mode matches and matches that haven't landed
        yet simply aren't present as keys here (their SyntheticTrack.frame_jpeg is
        None), which is the correct "fall back to the existing no-frame path" signal
        for the caller. MUST be called AFTER synthetic_tracked_objects() for the same
        (camera_name, now) in mirage.app._result_consumer_loop -- that call is what
        prunes expired tracks from internal state; this one only reads, so an expired
        track's stale frame never leaks into a later call by accident.
        """
        by_query = self._synthetic_tracks.get(camera_name)
        if not by_query:
            return {}
        return {
            track.object_id: track.frame_jpeg
            for tracks in by_query.values()
            for track in tracks
            if track.frame_jpeg is not None
        }

    def _to_tracked_object_state(self, track: SyntheticTrack) -> TrackedObjectState:
        return TrackedObjectState(
            id=track.object_id,
            label=track.label,
            box=track.box,
            score=track.score,
            # Never stationary -- an open-vocab match has no continuous box history to
            # judge motionlessness from (each match is an independent OWLv2 call, not a
            # frame-by-frame position update), and qualifies_for_review
            # (mirage/events/review.py) would otherwise exclude it entirely once
            # StationaryClassifier's default threshold_frames was reached.
            stationary=StationaryClassifier(threshold_frames=10**9),
            frame_time=track.last_seen,
            is_false_positive=False,  # already a REAL OWLv2 match -- Gate 1 doesn't apply here
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
