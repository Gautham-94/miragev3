"""Main application bootstrap.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 0 (High-Level Architecture),
section 1.1 (CameraMaintainer).

Startup sequence: config -> ensure directories -> init database -> pre-create per-camera
SHM rings -> spawn DetectorProcess(es) -> spawn RecordProcess -> spawn per-camera
CameraCapture+CameraTracker pairs -> start the main-process result-consumer thread
(EventProcessor + ReviewSegmentMaintainer) -> start the watchdog -> block until shutdown.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_module
import threading
import time
from pathlib import Path

from mirage.config.schema import MirageConfig
from mirage.const import CACHE_DIR as DEFAULT_CACHE_DIR
from mirage.const import DB_PATH as DEFAULT_DB_PATH
from mirage.const import RECORD_DIR as DEFAULT_RECORD_DIR
from mirage.const import ensure_dirs, ipc_addr
from mirage.db.database import close_database, init_database
from mirage.detection.process import DetectorProcess
from mirage.events.processor import EventProcessor
from mirage.events.review import ReviewSegmentMaintainer
from mirage.go2rtc.process import Go2rtcProcess
from mirage.ipc.zmq_pubsub import ZmqProxy
from mirage.logging_bus import ActivityLogWriter, LogEvent
from mirage.notify_bus import NotifyEvent, NotifyLogWriter
from mirage.openvocab.device import resolve_device
from mirage.openvocab.dispatcher import OpenVocabDispatcher
from mirage.openvocab.process import OpenVocabProcess
from mirage.recording.maintainer import RecordingMaintainer, run_recording_maintainer_loop
from mirage.species.dispatcher import SpeciesDispatcher
from mirage.species.process import SpeciesProcess
from mirage.tracking.camera_tracker import CameraTracker
from mirage.tracking.rules import RulesEngine
from mirage.util.shm import (
    SharedMemoryFrameManager,
    calculate_shm_ring_depth,
    detector_input_shm_name,
    detector_output_shm_name,
    preallocate_ring,
    teardown_ring,
)
from mirage.watchdog import ServiceWatchdog

logger = logging.getLogger(__name__)

DETECTED_FRAMES_QUEUE_TIMEOUT = 1.0
WATCHDOG_TICK_SECONDS = 10.0


class MirageApp:
    def __init__(
        self,
        config: MirageConfig,
        cache_dir: str = DEFAULT_CACHE_DIR,
        record_dir: str = DEFAULT_RECORD_DIR,
        db_path: str = DEFAULT_DB_PATH,
        enable_go2rtc: bool = True,
        go2rtc_stream_overrides: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.record_dir = record_dir
        self.db_path = db_path
        self.enable_go2rtc = enable_go2rtc
        self.go2rtc_stream_overrides = go2rtc_stream_overrides
        self.stop_event = mp.Event()
        self.frame_manager = SharedMemoryFrameManager()

        self.database = None
        self.zmq_proxy = None
        self.go2rtc_process: Go2rtcProcess | None = None
        self.detector_pub_addr = ipc_addr("detector_pub", cache_dir=self.cache_dir)
        self.detector_sub_addr = ipc_addr("detector_sub", cache_dir=self.cache_dir)

        # detector name -> the ONE queue every camera routed to that detector puts onto,
        # and every one of that detector's num_workers processes consumes from (see
        # _start_detectors). A single shared multi-consumer mp.Queue gives free
        # least-busy-worker routing: mp.Queue's own internal lock/pipe already hands
        # each queued item to whichever consumer's blocking .get() happens to unblock
        # first, so a worker that just finished a fast inference naturally picks up the
        # next request before a still-busy worker does -- no separate queue-depth
        # tracking or broker needed. This is why detection_queues is keyed by detector
        # name (every camera on that detector shares the identical queue reference),
        # NOT by camera name or worker index.
        self.detection_queues: dict[str, "mp.Queue"] = {}
        # "detector_name" (num_workers==1) or "detector_name#0", "detector_name#1", ...
        # (num_workers>1) -> that worker's process -- see _start_detectors. Every
        # worker for a given detector is otherwise identical (same model, same shared
        # queue) -- the #N suffix exists only to give each worker's OS process/watchdog
        # entry a distinct key, not to signal a different camera assignment.
        self.detector_processes: dict[str, DetectorProcess] = {}
        self.record_process = None
        self.capture_processes: dict[str, object] = {}
        self.tracker_processes: dict[str, CameraTracker] = {}
        self.capture_frame_queues: dict[str, object] = {}
        self.camera_ring_depths: dict[str, int] = {}

        self.manager = mp.Manager()
        self.detected_frames_queue = self.manager.Queue(maxsize=(len(config.cameras) + 2) * 2)

        # Structured activity log (Logs page) -- every pipeline process/thread puts
        # LogEvents here; only the main-process result-consumer loop drains it and
        # appends to the shared activity_log.jsonl file mirage.api tails (see
        # mirage.logging_bus's module docstring for the full one-writer reasoning).
        # Unbounded-ish maxsize: log events are small and infrequent (one per
        # motion/detect/species milestone, not per-frame) compared to detection_queues,
        # so this is sized generously rather than tightly like those queues.
        self.activity_log_queue: "mp.Queue" = self.manager.Queue(maxsize=2000)
        self.activity_log_writer: ActivityLogWriter | None = None

        # Live-notification bus (frontend SSE stream) -- same one-writer shape as
        # activity_log_queue above, see mirage.notify_bus's module docstring.
        self.notify_queue: "mp.Queue" = self.manager.Queue(maxsize=2000)
        self.notify_writer: NotifyLogWriter | None = None

        thumbnail_fetcher = self._fetch_go2rtc_thumbnail if self.enable_go2rtc else None
        self.event_processor = EventProcessor(thumbnail_fetcher=thumbnail_fetcher)
        self.event_processor.activity_log_queue = self.activity_log_queue
        self.event_processor.notify_queue = self.notify_queue
        self.review_maintainer = ReviewSegmentMaintainer(thumbnail_fetcher=thumbnail_fetcher)
        self.review_maintainer.notify_queue = self.notify_queue
        self.rules_engine = RulesEngine()
        self.result_consumer_thread: threading.Thread | None = None
        self.record_thread: threading.Thread | None = None
        self.record_stop_event = threading.Event()
        self.watchdog: ServiceWatchdog | None = None

        # OpenVocabProcess is only spawned if at least one enabled OpenVocabQuery exists
        # (see _start_openvocab) -- no point loading a ~600MB OWLv2 model and paying its
        # startup time if the user hasn't configured any saved query, which is the
        # common case today (this is a new, opt-in feature -- TODO_FIX_LIST.md items
        # 4/6). self.openvocab_dispatcher stays None in that case, and
        # _result_consumer_loop simply skips the open-vocab check step entirely.
        self.openvocab_process: OpenVocabProcess | None = None
        self.openvocab_dispatcher: OpenVocabDispatcher | None = None
        self.openvocab_request_queue = None
        self.openvocab_result_queue = None

        # SpeciesProcess is only spawned if config.species_classifier.enabled -- mirrors
        # OpenVocabProcess's own lazy-start reasoning above (no point loading a species
        # classification model if the user hasn't opted in, which is the default). See
        # _start_species_worker. self.species_dispatcher stays None in that case, and
        # EventProcessor._on_start/​_result_consumer_loop simply skip the species step
        # entirely (see mirage.events.processor.EventProcessor.species_dispatcher).
        self.species_process: SpeciesProcess | None = None
        self.species_dispatcher: SpeciesDispatcher | None = None
        self.species_request_queue = None
        self.species_result_queue = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> None:
        ensure_dirs(self.cache_dir, self.record_dir)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.database = init_database(self.db_path)
        self.event_processor.close_dangling_events()

        self.activity_log_writer = ActivityLogWriter(self.cache_dir)
        self.notify_writer = NotifyLogWriter(self.cache_dir)

        self.zmq_proxy = ZmqProxy(self.detector_pub_addr, self.detector_sub_addr)

        if self.enable_go2rtc:
            self.go2rtc_process = Go2rtcProcess(
                self.config, cache_dir=self.cache_dir, stream_overrides=self.go2rtc_stream_overrides,
            )
            self.go2rtc_process.start()

        self._preallocate_camera_shm()
        self._start_detectors()
        self._start_openvocab()
        self._start_species_worker()
        self._start_record_process()
        self._start_cameras()

        self.result_consumer_thread = threading.Thread(target=self._result_consumer_loop, daemon=True, name="result-consumer")
        self.result_consumer_thread.start()

        self.watchdog = ServiceWatchdog(tick_seconds=WATCHDOG_TICK_SECONDS)
        self._register_watchdog_targets()
        self.watchdog.start()

        logger.info("mirage app started: %d camera(s), %d detector(s)", len(self.config.cameras), len(self.config.detectors))

    def _preallocate_camera_shm(self) -> None:
        camera_dims = [(cam.detect.width, cam.detect.height) for cam in self.config.cameras.values() if cam.enabled]
        import shutil

        available_shm_mb = shutil.disk_usage(self.cache_dir).free / (1024 * 1024)
        ring_depth, recommended_min_mb = calculate_shm_ring_depth(camera_dims, available_shm_mb)
        if ring_depth < 20:
            logger.warning("computed SHM ring depth %d is below the recommended minimum of 20", ring_depth)

        for camera in self.config.cameras.values():
            if not camera.enabled:
                continue
            detector_config = self.config.detectors.get(camera.detector)
            if detector_config is None or not detector_config.enabled:
                # Logged once already in _start_detectors (which runs first) -- skip
                # this camera's SHM entirely rather than pre-creating input/output
                # rings for a detector process that will never exist.
                continue

            self.camera_ring_depths[camera.name] = ring_depth
            preallocate_ring(self.frame_manager, camera.name, ring_depth, camera.detect.width, camera.detect.height)

            input_size = detector_config.model.height * detector_config.model.width * 3
            self.frame_manager.create(detector_input_shm_name(camera.name), input_size)
            from mirage.const import OUTPUT_SHM_SIZE

            self.frame_manager.create(detector_output_shm_name(camera.name), OUTPUT_SHM_SIZE)

    def _start_detectors(self) -> None:
        cameras_by_detector: dict[str, list[str]] = {}
        for camera in self.config.cameras.values():
            if camera.enabled:
                cameras_by_detector.setdefault(camera.detector, []).append(camera.name)

        for detector_name, detector_config in self.config.detectors.items():
            if not detector_config.enabled:
                routed_cameras = cameras_by_detector.get(detector_name, [])
                if routed_cameras:
                    logger.warning(
                        "detector %r is disabled; camera(s) %s will not detect until reassigned or "
                        "this detector is re-enabled", detector_name, routed_cameras,
                    )
                else:
                    logger.info("detector %r is disabled, skipping process startup", detector_name)
                continue

            camera_names = cameras_by_detector.get(detector_name, [])
            num_workers = detector_config.num_workers

            # ONE shared queue for every worker of this detector -- every camera routed
            # to this detector puts onto the SAME queue regardless of num_workers, and
            # every worker process blocks on .get() against that SAME queue. This is
            # what gives free least-busy-worker routing (see self.detection_queues's
            # own docstring): whichever worker is idle and calls .get() first wins the
            # next request, so a worker that just finished naturally picks up more work
            # before a still-busy sibling does -- no queue-depth polling needed.
            queue = mp.Queue()
            self.detection_queues[detector_name] = queue

            for worker_index in range(num_workers):
                worker_key = detector_name if num_workers == 1 else f"{detector_name}#{worker_index}"
                # Every worker gets the FULL camera_names list -- not a subset -- purely
                # so each worker pre-creates every routed camera's output SHM segment on
                # startup (detector_process_main's own one-time setup loop). Which
                # worker actually SERVES a given request is decided per-request by the
                # shared queue above, not by this list.
                process = DetectorProcess(
                    detector_config=detector_config, detection_queue=queue, camera_names=camera_names,
                    detector_pub_addr=self.detector_pub_addr, stop_event=self.stop_event,
                )
                process.start()
                self.detector_processes[worker_key] = process

            if num_workers > 1:
                logger.info(
                    "detector %r: %d worker process(es) sharing one queue, serving cameras %s",
                    detector_name, num_workers, camera_names,
                )

    def _start_openvocab(self) -> None:
        """Only spawns OpenVocabProcess if config.openvocab_enabled AND at least one
        enabled query exists -- see __init__'s docstring on self.openvocab_dispatcher
        for why this is conditional. openvocab_enabled is checked FIRST and is a hard
        override: it lets queries stay configured/saved (not disabled/deleted one by
        one) while still keeping the ~600MB OWLv2 model out of memory entirely, e.g.
        via the Config page's toggle.
        """
        if not self.config.openvocab_enabled:
            logger.info("openvocab: disabled via config, skipping OWLv2 process startup")
            return

        active_queries = [q for q in self.config.queries if q.enabled]
        if not active_queries:
            logger.info("openvocab: no enabled queries configured, skipping OWLv2 process startup")
            return

        device = resolve_device("auto")
        self.openvocab_request_queue = mp.Queue()
        self.openvocab_result_queue = mp.Queue()
        self.openvocab_process = OpenVocabProcess(
            request_queue=self.openvocab_request_queue, result_queue=self.openvocab_result_queue,
            stop_event=self.stop_event, device=device,
        )
        self.openvocab_process.start()
        self.openvocab_dispatcher = OpenVocabDispatcher(
            request_queue=self.openvocab_request_queue, result_queue=self.openvocab_result_queue,
            frame_manager=self.frame_manager,
        )
        self.openvocab_dispatcher.notify_queue = self.notify_queue
        logger.info("openvocab: started (device=%s), %d enabled quer%s", device, len(active_queries), "y" if len(active_queries) == 1 else "ies")

    def _start_species_worker(self) -> None:
        """Only spawns SpeciesProcess if config.species_classifier.enabled -- see
        __init__'s docstring on self.species_dispatcher for why this is conditional.
        Must run AFTER self.event_processor is constructed (in __init__) but assigns
        into it here, post-construction -- EventProcessor itself has no species-related
        constructor param, since whether species classification is available isn't
        known until config is loaded in start(), well after __init__ already ran.
        """
        species_config = self.config.species_classifier
        if not species_config.enabled:
            logger.info("species: classifier not enabled, skipping species worker startup")
            return

        self.species_request_queue = mp.Queue()
        self.species_result_queue = mp.Queue()
        self.species_process = SpeciesProcess(
            request_queue=self.species_request_queue, result_queue=self.species_result_queue,
            stop_event=self.stop_event, classifier_config=species_config,
        )
        self.species_process.start()
        self.species_dispatcher = SpeciesDispatcher(
            request_queue=self.species_request_queue, result_queue=self.species_result_queue,
            activity_log_queue=self.activity_log_queue,
        )
        self.event_processor.species_dispatcher = self.species_dispatcher
        logger.info("species: started (device=%s)", species_config.device)

    def _start_record_process(self) -> None:
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True, name="record-maintainer")
        self.record_thread.start()

    def _record_loop(self) -> None:
        maintainer = RecordingMaintainer(cameras=self.config.cameras, cache_dir=self.cache_dir, record_dir=self.record_dir)
        run_recording_maintainer_loop(maintainer, self.record_stop_event)

    def _start_cameras(self) -> None:
        for camera in self.config.cameras.values():
            if not camera.enabled:
                continue
            detector_config = self.config.detectors.get(camera.detector)
            if detector_config is None or not detector_config.enabled:
                # No SHM ring, no detection queue, no DetectorProcess exists for this
                # camera (see _preallocate_camera_shm/_start_detectors) -- starting a
                # CameraTracker here would crash looking any of those up. Already
                # logged once in _start_detectors.
                continue
            self._start_camera(camera)

    def _start_camera(self, camera) -> None:
        from mirage.capture.camera_capture import CameraCapture

        frame_queue = self.manager.Queue(maxsize=2)
        current_frame_ts = self.manager.Value("d", 0.0)
        camera_fps_value = self.manager.Value("d", 0.0)
        skipped_fps_value = self.manager.Value("d", 0.0)
        self.capture_frame_queues[camera.name] = frame_queue

        capture = CameraCapture(
            camera=camera, cache_dir=self.cache_dir, ring_depth=self.camera_ring_depths[camera.name],
            frame_queue=frame_queue, current_frame_ts=current_frame_ts,
            camera_fps_value=camera_fps_value, skipped_fps_value=skipped_fps_value,
            stop_event=self.stop_event,
        )
        capture.start()
        self.capture_processes[camera.name] = capture

        detector_config = self.config.detectors[camera.detector]
        tracker = CameraTracker(
            camera=camera, detector_config=detector_config,
            detection_queue=self.detection_queues[camera.detector],
            detector_sub_addr=self.detector_sub_addr, frame_queue=frame_queue,
            detected_frames_queue=self.detected_frames_queue, stop_event=self.stop_event,
            activity_log_queue=self.activity_log_queue,
        )
        tracker.start()
        self.tracker_processes[camera.name] = tracker

    def _fetch_go2rtc_thumbnail(self, camera_name: str) -> bytes | None:
        """Real thumbnail source for ReviewSegmentMaintainer: a JPEG snapshot from
        go2rtc's own frame endpoint, the same mechanism the live-view snapshot proxy
        (mirage/api/routers/live.py) uses. Runs in the main-process result-consumer
        thread, so this is a short synchronous HTTP call, not fire-and-forget -- kept
        deliberately simple (one segment start is a rare event, not per-frame) rather
        than adding async plumbing for a call that happens a handful of times a minute
        at most.
        """
        if self.go2rtc_process is None:
            return None
        import httpx

        url = f"http://127.0.0.1:{self.go2rtc_process.api_port}/api/frame.jpeg"
        try:
            resp = httpx.get(url, params={"src": camera_name}, timeout=5.0)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200 or not resp.content:
            return None
        return resp.content

    def _register_watchdog_targets(self) -> None:
        for worker_key, process in self.detector_processes.items():
            # worker_key is "detector_name" (num_workers==1) or "detector_name#N"
            # (num_workers>1) -- strip any "#N" suffix to recover the real detector
            # name this worker belongs to (see _start_detectors's own worker_key
            # construction). Every worker of a detector shares that SAME detector's
            # config and queue (see self.detection_queues's own docstring), so no
            # per-worker camera list needs to be tracked separately here.
            detector_name = worker_key.split("#", 1)[0]
            detector_config = self.config.detectors[detector_name]
            camera_names = [c.name for c in self.config.cameras.values() if c.enabled and c.detector == detector_name]
            queue = self.detection_queues[detector_name]

            def factory(dc=detector_config, cn=camera_names, dq=queue):
                return DetectorProcess(
                    detector_config=dc, detection_queue=dq, camera_names=cn,
                    detector_pub_addr=self.detector_pub_addr, stop_event=self.stop_event,
                )

            self.watchdog.register(f"detector:{worker_key}", process, factory, self._on_detector_restarted(worker_key))

    def _on_detector_restarted(self, worker_key: str):
        def callback(new_process):
            self.detector_processes[worker_key] = new_process
        return callback

    # ------------------------------------------------------------------
    # Result consumption
    # ------------------------------------------------------------------

    def _result_consumer_loop(self) -> None:
        previous_ids_by_camera: dict[str, set[str]] = {}

        while not self.stop_event.is_set():
            try:
                camera_name, frame_name, frame_time, tracked_objects, motion_boxes, regions, frame_jpeg = (
                    self.detected_frames_queue.get(timeout=DETECTED_FRAMES_QUEUE_TIMEOUT)
                )
            except queue_module.Empty:
                continue
            except (OSError, EOFError):
                break

            camera = self.config.cameras.get(camera_name)
            if camera is None:
                continue

            # Merge in any still-live open-vocab synthetic tracks (see
            # mirage.openvocab.dispatcher.SyntheticTrack's docstring) BEFORE
            # EventProcessor/ReviewSegmentMaintainer run, so a real OWLv2 match for a
            # camera's Track objects word (e.g. "animals", auto-provisioned via
            # mirage.api.routers.config._sync_track_object_queries) or a manual saved
            # query drives a real Event/ReviewSegment through the exact same
            # start/update/end diff logic a closed-vocab detection does -- rather than
            # being a second, disconnected notification path (QueryMatch rows alone,
            # which is all that existed before this bridge). Built as a SEPARATE dict,
            # not a mutation of `tracked_objects` itself: the raw tracker-produced dict
            # is still what openvocab_dispatcher.process_frame needs below (it decides
            # whether to check GATE 1 using each object's real is_false_positive
            # status, and synthetic entries are never gate-1 candidates themselves --
            # they're the dispatcher's OUTPUT, not its input).
            objects_for_review = tracked_objects
            synthetic_frame_jpegs: dict[str, bytes] = {}
            if self.openvocab_dispatcher is not None:
                synthetic = self.openvocab_dispatcher.synthetic_tracked_objects(camera_name, frame_time)
                if synthetic:
                    objects_for_review = {**tracked_objects, **synthetic}
                    # MUST be called after synthetic_tracked_objects() above for this
                    # same (camera_name, frame_time) -- see synthetic_frame_jpegs()'s
                    # own docstring. Gives EventProcessor a per-object clean-frame
                    # override for direct-frame-mode open-vocab matches, which were
                    # detected in a completely separate frame than `frame_jpeg`
                    # (TODO_FIX_LIST.md item 9/11 -- open-vocab Events previously never
                    # got a real snapshot box at all).
                    synthetic_frame_jpegs = self.openvocab_dispatcher.synthetic_frame_jpegs(camera_name)

            try:
                self.event_processor.process(
                    camera, frame_time, objects_for_review, frame_jpeg=frame_jpeg,
                    object_frame_jpegs=synthetic_frame_jpegs,
                )

                # Rule triggers (crowd count / dwell-time -- mirage.tracking.rules)
                # feed ONLY ReviewSegmentMaintainer, deliberately NOT EventProcessor:
                # per the user's explicit choice, a rule firing surfaces as a
                # ReviewSegment (severity="rule") on the existing Review page, not as
                # a new kind of Event -- a crowd/dwell condition isn't "one tracked
                # object," so it doesn't fit Events' per-object model the way an
                # open-vocab match (a real, single detected thing) does. Built on
                # objects_for_review (already includes any openvocab synthetic
                # entries) so a rule can, in principle, also fire on a synthetic
                # open-vocab-detected object's dwell time.
                rule_triggers = self.rules_engine.process(camera, frame_time, objects_for_review)
                objects_for_review_with_rules = (
                    {**objects_for_review, **rule_triggers} if rule_triggers else objects_for_review
                )
                self.review_maintainer.process(camera, frame_time, objects_for_review_with_rules, frame_jpeg=frame_jpeg)
            except Exception:
                logger.exception("%s: error consuming tracked-object result", camera_name)

            if self.openvocab_dispatcher is not None:
                try:
                    self.openvocab_dispatcher.process_frame(
                        self.config, camera, frame_name, frame_time, tracked_objects, motion_boxes=motion_boxes,
                    )
                except Exception:
                    logger.exception("%s: error dispatching to openvocab", camera_name)

                previous_ids = previous_ids_by_camera.get(camera_name, set())
                current_ids = set(tracked_objects.keys())
                for ended_id in previous_ids - current_ids:
                    self.openvocab_dispatcher.forget_object(ended_id)
                previous_ids_by_camera[camera_name] = current_ids

            # Species dispatch itself already happened synchronously inside
            # EventProcessor._on_start above (at Event-creation time) -- this is just
            # draining whatever results the SpeciesProcess worker has finished since
            # the last iteration, same non-blocking-poll shape as every other queue
            # drained in this loop. Not camera-specific (a single shared worker serves
            # every camera's Animal/Bird events), so this runs once per loop
            # iteration regardless of which camera's frame just came through.
            if self.species_dispatcher is not None:
                try:
                    self.species_dispatcher.drain_results()
                except Exception:
                    logger.exception("error draining species classification results")

            self._drain_activity_log()
            self._drain_notify_queue()

    def _drain_notify_queue(self) -> None:
        """Non-blocking drain of every NotifyEvent any pipeline process/thread has put
        onto notify_queue since the last iteration -- see this class's notify_queue
        docstring and mirage.notify_bus's module docstring for why only this loop ever
        touches NotifyLogWriter.
        """
        while True:
            try:
                event: NotifyEvent = self.notify_queue.get_nowait()
            except queue_module.Empty:
                break
            except (OSError, EOFError):
                break
            try:
                self.notify_writer.append(event)
            except Exception:
                logger.exception("error appending to notify log")

    def _drain_activity_log(self) -> None:
        """Non-blocking drain of every LogEvent any pipeline process/thread has put
        onto activity_log_queue since the last iteration -- see this class's
        activity_log_queue docstring and mirage.logging_bus's module docstring for why
        only this loop ever touches ActivityLogWriter.
        """
        while True:
            try:
                event: LogEvent = self.activity_log_queue.get_nowait()
            except queue_module.Empty:
                break
            except (OSError, EOFError):
                break
            try:
                self.activity_log_writer.append(event)
            except Exception:
                logger.exception("error appending to activity log")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        logger.info("mirage app shutting down")
        self.stop_event.set()
        self.record_stop_event.set()

        if self.watchdog is not None:
            self.watchdog.stop()

        for tracker in self.tracker_processes.values():
            tracker.join(timeout=10)
            if tracker.is_alive():
                tracker.terminate()
                tracker.join(timeout=5)

        for capture in self.capture_processes.values():
            capture.join(timeout=15)
            if capture.is_alive():
                capture.terminate()
                capture.join(timeout=5)

        for detector in self.detector_processes.values():
            detector.join(timeout=15)
            if detector.is_alive():
                detector.terminate()
                detector.join(timeout=5)

        if self.openvocab_process is not None:
            self.openvocab_process.join(timeout=15)
            if self.openvocab_process.is_alive():
                self.openvocab_process.terminate()
                self.openvocab_process.join(timeout=5)

        if self.species_process is not None:
            self.species_process.join(timeout=15)
            if self.species_process.is_alive():
                self.species_process.terminate()
                self.species_process.join(timeout=5)

        if self.record_thread is not None:
            self.record_thread.join(timeout=10)

        self.event_processor.close_dangling_events()
        self.review_maintainer.close_all_pending()

        for camera_name, ring_depth in self.camera_ring_depths.items():
            teardown_ring(self.frame_manager, camera_name, ring_depth)
            self.frame_manager.delete(detector_input_shm_name(camera_name))
            self.frame_manager.delete(detector_output_shm_name(camera_name))

        if self.zmq_proxy is not None:
            self.zmq_proxy.close()

        if self.go2rtc_process is not None:
            self.go2rtc_process.stop()

        if self.database is not None:
            close_database(self.database)

        self.manager.shutdown()
        logger.info("mirage app stopped")
