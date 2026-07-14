# Mirage — System Design

This document describes the architecture of the mirage NVR as it actually exists in this
repository today, not as originally planned. It is a snapshot, not a spec — where the
real implementation differs from `NVR_PIPELINE_IMPLEMENTATION_SPEC.md` or
`NVR_EXTENSION_MULTI_MODEL_ROUTING.md`, this document describes what the code does.
For unresolved gaps and planned work, see `TODO_FIX_LIST.md`. For local run instructions,
see `HOW_TO_RUN.md`.

Mirage is composed of three independently-runnable pieces:

- **`python -m mirage`** — the capture → detect → track → record pipeline (`MirageApp`)
- **`python -m mirage.api`** — a read/write HTTP API, a separate process that only
  shares the SQLite DB and disk paths with the pipeline
- **`frontend/`** — an Angular SPA talking to the API

Plus a fully standalone sibling tool, **`mock_cameras/`**, used to emulate real ONVIF/RTSP
cameras for local testing without hardware.

---

## 1. High-level process architecture

**File:** `mirage/app.py` (class `MirageApp`)

`MirageApp` is the top-level orchestrator for the whole capture→detect→track→record
pipeline. It runs as its own OS process, entirely separate from `mirage.api` — the two
only ever communicate indirectly, through the same SQLite DB file (WAL mode) and shared
disk paths (recordings, thumbnails, config).

### Processes and threads

| Component | Type | Purpose |
|---|---|---|
| `Go2rtcProcess` | plain `subprocess.Popen` (not `mp.Process`) | Vendored go2rtc binary; live view + ONVIF discovery proxy; restreams each camera's detect-role RTSP source as WebRTC/MSE/snapshot |
| `DetectorProcess` (one per configured `DetectorInstanceConfig`, **not** per camera) | `mp.Process` | Loads one ONNX model once; serves detection requests for every camera routed to it |
| `OpenVocabProcess` (zero or one, total) | `mp.Process` | Loads OWLv2 once; only spawned if at least one enabled `OpenVocabQuery` exists |
| `CameraCapture` (one per enabled camera) | `mp.Process` | Owns the camera's ffmpeg subprocess(es) (detect role, plus optionally record/audio roles); runs a watchdog thread supervising ffmpeg |
| `CameraTracker` (one per enabled camera) | `mp.Process` | Runs the full per-frame pipeline: motion → regions → tensor → remote detect → tracking → lifecycle |
| Record-maintainer thread (`_record_loop`) | `threading.Thread`, main process | Runs `RecordingMaintainer` every ~5s: promotes ffmpeg cache segments to permanent storage, writes `Recordings` rows |
| Result-consumer thread (`_result_consumer_loop`) | `threading.Thread`, main process | Drains `detected_frames_queue`; feeds `EventProcessor`, `ReviewSegmentMaintainer`, and (if active) `OpenVocabDispatcher` |
| `ServiceWatchdog` | `threading.Thread`, main process | Every 10s, restarts any registered `mp.Process` that died non-cleanly (currently: detector processes), with a circuit breaker (max 5 restarts/60s) |
| `ZmqProxy` | background thread (in-process `zmq.proxy`) | XSUB/XPUB broker so any number of `DetectorProcess` publishers and `CameraTracker` subscribers can connect without direct sockets |

### IPC mechanisms

**1. `multiprocessing.Queue`** (via `mp.Manager().Queue()` for cross-process-safe queues,
plain `mp.Queue()` for direct ones):

- `detection_queues[detector_name]` — `CameraTracker` → `DetectorProcess`; carries just
  the camera name, i.e. a "go" signal
- `detected_frames_queue` — `CameraTracker` → main process; carries
  `(camera_name, frame_name, frame_time, tracked_objects, motion_boxes, regions)`
- `openvocab_request_queue` / `openvocab_result_queue` — main process ↔
  `OpenVocabProcess`; carries JPEG bytes directly (not SHM — see section 4)
- `frame_queue` (one per camera) — `CameraCapture` → `CameraTracker`; carries
  `(frame_name, frame_time)`

**2. Shared memory** (`mirage/util/shm.py`, `SharedMemoryFrameManager` wrapping
`UntrackedSharedMemory`):

- Works around a real CPython bug ([python/cpython#82300](https://github.com/python/cpython/issues/82300)):
  `multiprocessing.resource_tracker` unlinks a `SharedMemory` segment as soon as *any*
  process that opened it exits, even if others still hold it open — so tracking is
  disabled and the unlink lifecycle is managed manually by this codebase instead.
- Per-camera **frame ring buffer**: N pre-allocated slots named `frame_name(camera, i)`,
  holding raw YUV420 planar frames. `calculate_shm_ring_depth()` computes ring depth
  (max 50) from available disk space and per-camera frame size (recommended minimum: 20
  slots).
- Per-camera **detector input/output SHM**: the input tensor, and a fixed-size output
  buffer (`MAX_DETECTIONS(20) × DETECTIONS_PER_ROW(6) × float32` = 480 bytes).
- Segment names are derived via a short-key helper: first 8 alphanumeric characters of
  the camera name plus an 8-character SHA1 hex digest — because macOS caps POSIX shm
  names at 30 characters total (confirmed empirically), and real camera names (e.g.
  `hikvision_ds_2cd1023g0e_i`) can overflow that once naming affixes are added.
- `write()`/`get()` always return exact-size-sliced views (never the raw, page-rounded
  buffer) and track the last view handed out per segment name so `close()`/`delete()`
  can release outstanding memoryviews first — otherwise `mmap.close()` raises
  `BufferError: cannot close exported pointers exist`. An `atexit` hook releases (but
  does not unlink) views on interpreter shutdown, so an unclean process exit doesn't spam
  that error either.

**3. ZMQ pub/sub** (`mirage/ipc/zmq_pubsub.py`):

- `ZmqProxy` — one XSUB bind + one XPUB bind, both unix-domain-socket-backed
  (`ipc://` addresses under `CACHE_DIR`), bridged via `zmq.proxy()` on a background
  thread.
- `Publisher` (used by `DetectorProcess`) publishes on topic `object_detector/<camera>`
  with an empty payload the instant inference for that camera's SHM write completes — a
  pure "done" signal; the actual detections travel via the output SHM segment, not the
  ZMQ message.
- `Subscriber` (used by `RemoteObjectDetector`, one per camera, filtered to that
  camera's topic) blocks up to 5.0s waiting for the signal, and drains any stale
  messages before each request as a defensive measure.
- A short settle-sleep after `Publisher.connect()` works around ZMQ's PUB "slow
  joiner" problem.

### Startup sequence (`MirageApp.start()`, in order)

1. Ensure directories exist; create the DB path's parent directory.
2. `init_database(db_path)`.
3. `EventProcessor.close_dangling_events()` — force-close any `Event` rows left with
   `end_time IS NULL` from an unclean previous shutdown.
4. Create `ZmqProxy` (binds sockets).
5. If go2rtc is enabled: start `Go2rtcProcess` (writes `go2rtc.yaml`, downloads the
   binary if needed, spawns the subprocess).
6. Pre-allocate SHM: compute ring depth from disk space, pre-create every enabled
   camera's frame-ring segments plus detector input/output segments.
7. Start detectors: group enabled cameras by `camera.detector`, spawn one
   `DetectorProcess` per configured `DetectorInstanceConfig` (one process per config
   entry, regardless of whether any camera currently uses it).
8. Start open-vocab: only if at least one enabled `OpenVocabQuery` exists — resolve the
   torch device (`resolve_device("auto")`), spawn `OpenVocabProcess`, construct
   `OpenVocabDispatcher`.
9. Start the record-maintainer thread.
10. Start cameras: for each enabled camera, spawn `CameraCapture` first, then
    `CameraTracker` (wired to that camera's detector's queue and the shared
    `detected_frames_queue`).
11. Start the result-consumer thread.
12. Create `ServiceWatchdog` (10s tick), register detector processes with restart
    factories, start it.

### Shutdown sequence (`MirageApp.stop()`, in order)

1. Set the shared `stop_event` and `record_stop_event`.
2. `watchdog.stop()` — blocks until the watchdog thread has actually exited, preventing
   a race where it "helpfully" restarts a process mid-teardown.
3. Join+terminate (10s/5s timeouts) every `CameraTracker`.
4. Join+terminate (15s/5s) every `CameraCapture`.
5. Join+terminate (15s/5s) every `DetectorProcess`.
6. Join+terminate (15s/5s) `OpenVocabProcess`, if it exists.
7. Join the record-maintainer thread (10s).
8. `EventProcessor.close_dangling_events()` + `ReviewSegmentMaintainer.close_all_pending()`
   — force-close any still-open `Event`/`ReviewSegment` rows.
9. Tear down every per-camera SHM ring and detector input/output segment (unlink).
10. Close the ZMQ proxy.
11. Stop go2rtc (SIGTERM to its process group, SIGKILL after a 10s timeout).
12. Close the database.
13. Shut down the `multiprocessing.Manager`.

---

## 2. Per-frame pipeline (the hot path)

**Files:** `mirage/tracking/camera_tracker.py` (process entry point
`camera_tracker_main`), `mirage/tracking/orchestration.py` (`CameraOrchestrator.process_frame`).

Each `CameraTracker` process loop:

1. `os.nice(PROCESS_PRIORITY_HIGH)` (best-effort).
2. Constructs a `RemoteObjectDetector`, `MotionDetector`, and `ObjectTracker`, wrapped
   in a `CameraOrchestrator`.
3. Loop: pull `(frame_name, frame_time)` off `frame_queue` (1.0s poll timeout); read the
   raw YUV frame out of SHM; call `orchestrator.process_frame(...)`; detach (not
   unlink) the SHM slot; push the result onto `detected_frames_queue` non-blocking
   (dropped on `Full` — this is the pipeline's backpressure valve).

`CameraOrchestrator.process_frame()`, in order:

1. **Motion detection** (`mirage/motion/detector.py`) — unconditional every frame. A
   hand-rolled algorithm, not OpenCV's `BackgroundSubtractorMOG2`: downsizes to a small
   working resolution (default 100px tall), optional contrast stretch (percentile
   clipping against a rolling average), a rasterized polygon mask, gaussian blur, and a
   diff against an exponential-moving-average background (`cv2.accumulateWeighted`),
   thresholded/dilated/contoured into boxes. Background updates are debounced (motion
   must persist ≥10 consecutive frames before bleeding into the average). A
   configurable `skip_motion_threshold` drops all boxes and forces recalibration on a
   huge spike (e.g. an IR-cut switch); a separate `lightning_threshold` forces
   recalibration without dropping boxes.
2. If `camera.detect.enabled` is `False`: the tracker still runs with zero detections
   (so existing tracks age out normally), and the function returns early with empty
   regions.
3. **Region selection** (`mirage/regions/selection.py`) — always includes clustered
   regions around every currently *confirmed* tracked object's box, plus
   `_pending_candidate_boxes` (unconfirmed norfair candidates carried forward from the
   previous frame specifically so a brand-new object keeps being re-scanned long enough
   to accumulate norfair's `initialization_delay` and get promoted — without this, a new
   object would only ever be scanned once). If not calibrating and no PTZ move is in
   progress, clustered regions around any motion box not already covered by a
   tracked-object region are added too. If no regions exist at all and this is the very
   first frame ever processed for this camera, falls back to a single full-frame region.
4. **Tensor construction** (`mirage/detection/tensor.py`) — for each region: convert the
   full YUV420 frame to RGB/BGR per the model's pixel format, crop the region, resize to
   the model's input size if needed, expand to an NHWC batch, cast per the configured
   input dtype (int passthrough / float÷255 / float, no scaling).
5. **Remote detector call** (`mirage/detection/remote.py`) — drain stale ZMQ messages;
   write the tensor into the camera's input SHM segment (zero-copy); enqueue the camera
   name (signals the shared `DetectorProcess`); block up to 5s on the ZMQ "done" signal;
   on timeout, return an empty result (the caller just retries next frame); on success,
   read the fixed `(20, 6)` float32 output array, stopping at the first row scoring
   below threshold (rows are sorted descending, zero-padded), producing
   `(label, score, normalized box)` tuples.
6. Boxes are denormalized to full-frame pixel coordinates, clamped to frame bounds,
   filtered by `camera.objects.track` membership and per-label score/area/aspect-ratio
   filters, and collected as raw detections.
7. **Reduce/consolidate** (`mirage/regions/reduce.py`) — per-label NMS, with an
   edge-clip confidence floor applied first (protects a legitimately-cropped-off object
   at a region boundary from being unfairly suppressed for a low score), followed by
   containment suppression (drops a smaller same-label box mostly contained in a larger
   one, unless the smaller box is negligibly small).
8. **Tracking** (`mirage/tracking/tracker.py`, wraps [norfair](https://github.com/tryolabs/norfair))
   — one norfair `Tracker` per label, Kalman-filter-based, with per-label tuning
   (distance thresholds, process/measurement noise). Every label's tracker advances
   every frame, even with zero detections, so hit-counter aging/expiry stays correct.
   Each confirmed track maps to mirage's own object id, wrapped in a
   `TrackedObjectState` dataclass (id, label, box, score, a `StationaryClassifier`,
   frame_time, `is_false_positive`). The `StationaryClassifier`
   (`mirage/tracking/stationary.py`) uses asymmetric IoU thresholds (different
   thresholds for active→stationary vs. stationary→active) specifically to prevent
   boundary oscillation, and force-deregisters objects that have been stationary too
   long.
9. **Lifecycle / false-positive classification** (`mirage/tracking/lifecycle.py`) —
   maintains a rolling score history (a non-detected frame appends a zero, dragging the
   median down); once the median score clears a configured threshold, the object's
   `is_false_positive` flips to `False` **permanently** (sticky — never reverts). This
   flag is mirrored onto the plain `TrackedObjectState.is_false_positive` field
   specifically because the richer `ObjectLifecycle` object itself never leaves the
   `CameraTracker` process, but `TrackedObjectState` does cross into the main process.
10. Consolidated detection boxes are carried forward as next frame's pending-candidate
    boxes.
11. The result — camera name, frame time, tracked objects, motion boxes, regions — is
    put onto `detected_frames_queue`, crossing into the main process. There,
    `_result_consumer_loop` drains it and feeds `EventProcessor.process()`,
    `ReviewSegmentMaintainer.process()`, and (if configured)
    `OpenVocabDispatcher.process_frame()`.

---

## 3. Detector plugin system

**Files:** `mirage/detection/registry.py`, `api.py`, `plugins/onnx_yolov8.py`,
`execution_providers.py`, `mirage/config/schema.py`.

`DetectionApi` (`api.py`) is the abstract base — one subclass per inference backend. The
contract: `detect_raw(tensor_input)` must return a fixed `(20, 6)` float32 array, rows
`[class_id, score, y_min, x_min, y_max, x_max]` normalized to `[0, 1]`, sorted descending
by score, zero-padded trailing rows. Each implementation owns its own NMS if the raw
model output isn't already NMS'd.

**Registry** (`registry.py`): scans `mirage.detection.plugins` via `pkgutil`, importing
each module inside a `try/except ImportError` (so a missing runtime library on a given
platform doesn't crash the whole app), then collects every `DetectionApi` subclass keyed
by its `type_key` class attribute. `create_detector(config)` looks it up by
`config.device`; `available_backends()` exposes the registered keys for the
config-write API's validation.

**`OnnxYolov8Detector`** (`type_key = "onnx_yolov8"`, the only backend implemented today)
— loads an `onnxruntime.InferenceSession` with providers from `resolve_providers()`
(below). YOLOv8's ONNX export outputs `[1, 4+num_classes, num_anchors]`, *not*
pre-NMS'd: the plugin does its own argmax over class scores, score-threshold filtering,
box-format conversion, and NMS before packing into the canonical `(20, 6)` output.

**One-process-per-detector-instance model**: `DetectorProcess` loads exactly one model
per `DetectorInstanceConfig`, regardless of how many cameras use it. Camera tracker
processes never load a model themselves — they hold only a `RemoteObjectDetector`
client (SHM handles plus a ZMQ subscriber). Per-camera routing is via
`CameraConfig.detector: str`, a name key into `MirageConfig.detectors`; at startup,
`MirageApp` groups enabled cameras by that field and spawns one `DetectorProcess` per
detector config entry, each pre-provisioned with SHM output segments for exactly the
camera names routed to it.

**Execution provider resolution** (`execution_providers.py`): `ExecutionProvider`
(`auto`/`cpu`/`coreml`/`cuda`) is a *separate* axis from `DetectorInstanceConfig.device`
— `device` picks the plugin class; `execution_provider` picks which onnxruntime hardware
backend that plugin's session actually runs on. `resolve_providers()` builds a
priority-ordered `providers=[...]` list (onnxruntime does automatic per-op fallback
within a single session, so this is not an either/or choice): `cpu` → CPU only; `auto`
→ prefers CUDA, then CoreML, whichever is actually installed, with CPU appended as the
guaranteed fallback; an explicit `coreml`/`cuda` request still gets CPU appended after
it. `available_execution_providers()` reports which choices will genuinely work on the
current machine, so the Manage Detectors UI never offers a choice that would silently
no-op back to CPU.

**Real measured benchmark** (documented directly in `ExecutionProvider`'s docstring):
on Apple Silicon, against the real shipped `yolov8n.onnx` — **CPU averaged 9.90ms/frame,
CoreML averaged 1.75ms/frame, a genuine ~5.6x speedup**, not a theoretical one.

**Key config shapes** (`mirage/config/schema.py`):
- `DetectorInstanceConfig`: `{name, model: ModelConfig, device: str = "onnx"}`
- `ModelConfig`: `{width, height, input_dtype, pixel_format, layout, model_path,
  labelmap_path, execution_provider}`

---

## 4. Open-vocabulary detection subsystem

**Files:** `mirage/openvocab/process.py`, `gating.py`, `dispatcher.py`, `device.py`.

This is a second, independent detection subsystem sitting alongside the closed-vocab
YOLOv8n pipeline, added specifically to answer queries the fixed COCO label set can't —
"a person carrying a red backpack," "a dog off-leash," anything phrased as free text
rather than picked from a fixed class list.

### Model loading (`process.py`)

`OpenVocabProcess`'s entry point loads `google/owlv2-base-patch16-ensemble` via
HuggingFace `transformers` (`Owlv2Processor` / `Owlv2ForObjectDetection`), moved to a
resolved torch device. Loaded exactly once, shared across every camera and every query —
the same "one process, one loaded model" principle as `DetectorProcess`.

### Why this doesn't reuse `DetectorProcess`'s SHM+ZMQ transport

OWLv2 calls are multi-second (~1-1.7s measured, see below) and happen at most a
handful of times per minute, versus YOLOv8n's millisecond-scale, many-times-per-second
calls. A plain `multiprocessing.Queue` carrying small JPEG-encoded crops directly is
simpler and entirely sufficient at this call volume — SHM's zero-copy benefit doesn't
matter here. An `OpenVocabRequest` carries the request id, camera name, object id, JPEG
bytes, the crop's box in full-frame coordinates (so a match's box can be reported back
in the same coordinate space the caller understands), and the list of
`(query_id, query_text)` pairs to check. An `OpenVocabResult` echoes the crop bytes back
so the dispatcher can save a match thumbnail without holding the original bytes across
the async round-trip itself.

One real bug caught during end-to-end testing: OWLv2's regressed box coordinates can
slightly overshoot the crop's own pixel bounds (ordinary regression imprecision). The
fix clamps to the crop's own size *before* offsetting back into full-frame coordinates.

### Device resolution (`device.py`)

Kept separate from `execution_providers.py` because OWLv2 runs on PyTorch
(`torch.device("cpu"/"mps"/"cuda")`), a differently-shaped API from onnxruntime's
provider list. `resolve_device("auto")` prefers CUDA, then MPS, then CPU, and silently
falls back to CPU if a specific unavailable device is explicitly requested.

**Real measured benchmark**: on the same Apple Silicon dev machine, MPS gave only a
**~1.21x speedup over CPU** (1423.8ms vs. 1720.7ms average, against the real OWLv2
checkpoint on a real test photo) — MPS only accelerates GPU-core matrix ops, it does
*not* reach the Apple Neural Engine the way onnxruntime's CoreML provider does for
YOLOv8n, so nowhere near CoreML-YOLOv8n's ~5.6x. This is the reason the open-vocab path
is architected around heavy gating rather than continuous per-frame checking — even
with GPU acceleration, a ~1-2 second call is far too slow to run on every frame.

### Two-gate dispatch system (`gating.py`, `OpenVocabGate`)

Both gates must pass before a crop is ever sent to OWLv2:

- **Gate 1 (track confirmed)** — checked by the caller (the dispatcher), not inside
  `OpenVocabGate` itself: only ever consider an object once `TrackedObjectState
  .is_false_positive` has flipped to `False` — the same signal that already gates real
  `Event`/`ReviewSegment` creation elsewhere in the pipeline.
- **Gate 2 (perceptual-hash dedup)** — `should_dispatch(key, crop_rgb, now)` computes a
  from-scratch average-hash (an 8×8 downsample → grayscale → threshold-against-mean →
  64-bit integer, the same size `python-imagehash`'s default uses, deliberately *not* a
  CLIP embedding encoder — a cheaper substitute chosen specifically to avoid adding a
  third heavy model) and compares its Hamming distance against the hash last sent for
  this key. Dispatches again only if the distance clears a threshold, or a periodic
  forced-recheck interval has elapsed even if visually unchanged. An in-flight set also
  prevents a second dispatch while a request for the same key is still outstanding.

### Two dispatch modes (`dispatcher.py`, selected per-camera via `CameraConfig.openvocab_direct_frame`)

1. **Confirmed-object mode** (default, `openvocab_direct_frame = False`) — for every
   tracked object that passes Gate 1 and has at least one enabled query scoped to this
   camera, crop its box out of the frame (synchronously, from SHM), check Gate 2 keyed
   by object id, and if it passes, JPEG-encode and dispatch. This only ever sees crops
   of things the closed-vocab detector already found — it enriches/refines an existing
   detection, it can't discover something the detector was never trained to recognize
   as an object at all.
2. **Direct-frame mode** (`openvocab_direct_frame = True`) — skips the confirmed-object
   requirement entirely, gating on motion-box presence instead. Sends the *whole* frame
   (not a crop) to OWLv2. Gate 2 is keyed by camera name instead of object id, since
   there's no tracked object to key it by. This is for queries about things the
   closed-vocab detector was never trained to recognize as an object at all, so mode 1
   would never produce a confirmed object for it to enrich.

Crucially, in both modes the crop/frame must be read out of SHM and JPEG-encoded
*synchronously in the same call*, never deferred — the ring buffer is finite-depth
(20-50 slots) and would be overwritten multiple times before a multi-second OWLv2 call
returns.

### Query scoping and match persistence

`OpenVocabQuery`: `{id, text, cameras: list[str] (empty means all cameras), enabled}`.
`MirageConfig` has a cross-field validator rejecting a query scoped to an unknown camera
name. The dispatcher filters queries down to `enabled and (not cameras or camera.name in
cameras)` *before* doing any SHM read at all — a camera with no applicable queries skips
the frame entirely, at zero cost.

Every frame, the dispatcher first drains any completed results off the result queue, and
for each real match creates a `QueryMatch` row — a dedicated DB table (not folded into
`Event.data`) so matches are independently queryable/listable/filterable by
query/camera/time without JSON-scanning every event. Thumbnails are written from the
echoed crop bytes to disk, best-effort (a failure returns `None` rather than raising —
never blocks match persistence).

`OpenVocabProcess` is only spawned at all if at least one enabled query exists,
specifically to avoid loading a ~600MB model for the common case of zero configured
queries.

---

## 5. Config system

**Files:** `mirage/config/schema.py`, `mirage/config/store.py`.

### The full `MirageConfig` tree

```
MirageConfig
├── detectors: dict[str, DetectorInstanceConfig]
├── cameras:   dict[str, CameraConfig]
└── queries:   list[OpenVocabQuery]
```

Cross-field validators enforce that every `camera.detector` references a real entry in
`detectors`, and every `query.cameras` entry references a real entry in `cameras`.

**`CameraConfig`**: `{name, enabled, ffmpeg: FfmpegConfig, motion: MotionConfig, detect:
DetectConfig, objects: ObjectsConfig, record: RecordConfig, review: ReviewConfig,
detector: str, openvocab_direct_frame: bool}`. Computed properties (`frame_shape`,
`frame_shape_yuv`, `frame_size_bytes`, `min_initialized`, `max_disappeared`,
`stationary_threshold`) all fall back to fps-derived defaults if not explicitly set.

**`FfmpegConfig`**: one or more `CameraInputConfig` entries, each assigned one or more
roles (`detect` / `record` / `audio`). A validator auto-assigns `[record, detect]` if
there's exactly one input with no roles set, and enforces that exactly one input carries
the `detect` role. `CameraInputConfig.rtsp_transport` (`tcp` / `udp`) is a real
per-camera escape hatch — confirmed against a real Hikvision camera whose RTSP-over-TCP
session reset every ~2 seconds, resolved by switching that camera to UDP.

**`MotionConfig`**, **`DetectConfig`**, **`ModelConfig`**, **`DetectorInstanceConfig`**,
**`ObjectFilterConfig`** / **`ObjectsConfig`**, **`RetainConfig`** / **`RecordConfig`**,
**`ReviewLabelConfig`** / **`ReviewConfig`**, **`OpenVocabQuery`** — each holds the
tuning knobs for its respective subsystem (see sections 2-4 above for how they're
consumed). `RecordConfig.segment_seconds` defaults to 10 (matching Frigate's own real
default) with a hard ceiling of 60 (coarser segments make retention granularity worse
and increase data-loss risk on an ungraceful shutdown).

### Persistence (`mirage/config/store.py`)

Config lives as a single JSON blob in the `AppConfig` singleton DB row (`id = 1`,
enforced via a `CHECK (id = 1)` SQL constraint — not just convention). Loading seeds a
default config (one `general` ONNX YOLOv8n detector, no cameras) if the row doesn't
exist yet; saving is a Pydantic `model_dump(mode="json")` upserted via peewee's
`on_conflict`.

Config is **not** hot-reloaded: `--config <path>` (in `mirage/__main__.py`) is a
one-time YAML import used only when the DB has zero config rows (a genuinely fresh
install) — every subsequent run reads exclusively from the DB, and writes made through
the API only take effect in the running pipeline after a manual restart. This is a real,
recurring operational constraint — see section 12.

---

## 6. Database schema

**File:** `mirage/db/models.py` — peewee ORM, SQLite in WAL mode.

| Table | Key fields | Purpose |
|---|---|---|
| `Event` | `id, label, sub_label, camera, start_time, end_time, score, top_score, false_positive, zones, has_clip, has_snapshot, snapshot_path, data (json)` | One tracked object's detection lifecycle (start → update → end), written by `EventProcessor` |
| `Recordings` | `id, camera, path (unique), start_time, end_time, duration, motion, objects, regions, dBFS, segment_size_mb` | One permanently-stored (promoted) recording segment |
| `ReviewSegment` | `id, camera, start_time, end_time, severity, thumb_path (unique), data (json: detections/objects/zones/...)` | A human-facing aggregated activity window, possibly spanning multiple tracked objects |
| `Timeline` | `timestamp, camera, source, source_id, class_type, data (json)` | Optional scrubbable activity timeline (e.g. "entered zone" events) |
| `Regions` | `camera (pk), grid (json), last_update` | Learned per-camera object-size grid |
| `AppConfig` | `id (pk, CHECK id=1), data (json — the whole MirageConfig tree), updated_at` | Singleton row backing config persistence |
| `QueryMatch` | `id, query_id, query_text (denormalized snapshot), camera, object_id, matched_at, score, box (json), thumb_path` | One saved open-vocab query matching a confirmed tracked object |

---

## 7. Event, review, and recording pipelines

### `EventProcessor` (`mirage/events/processor.py`)

Diffs the tracker's current confirmed object-id set against the previous frame's set,
per camera, driving `Event` row lifecycle:

- **Start** — creates the row (`false_positive` set from the object's current state,
  snapshot captured via a thumbnail fetcher).
- **Update** — DB writes are throttled: immediate if `top_score` just increased, a
  forced heartbeat write at least once a minute regardless, otherwise at most once every
  5 seconds. Critically, the in-memory tracking of `top_score`/`false_positive` is
  always kept current even on throttled-out frames — without this, a false-positive
  flag flipping to real on a throttled frame right before the object disappears would
  never reach the database at all, since the end-of-life write has no live object state
  left to read from.
- **End** — writes the final `end_time`/`top_score`/`false_positive` from the last-known
  in-memory state, the last chance to persist a true final status.
- **Startup recovery** — any row left with `end_time IS NULL` from an unclean shutdown
  is force-closed.

### `ReviewSegmentMaintainer` (`mirage/events/review.py`)

Aggregates tracked-object activity into human-reviewable windows, distinct from
individual events. A label counts as `alert` severity if it's in the camera's configured
alert-label list, `detection` severity if in the (typically broader) detections list,
else it doesn't qualify at all — stationary and false-positive objects are excluded from
qualifying regardless of label. Per camera, one segment is pending at a time: a
qualifying object starts a new segment (or upgrades an existing detection-severity
segment to alert — never downgrades) and keeps it open; with no qualifying activity, the
segment closes once a configurable cutoff (default 30 seconds) of inactivity elapses.
Shutdown force-closes any still-open segments.

### Recording segment promotion (`mirage/recording/segments.py`, `maintainer.py`)

Looped every ~5 seconds: lists raw ffmpeg cache segments, enforces cache backpressure
(force-deletes oldest excess beyond a per-camera cap), then per segment — skips a
disabled camera's segment (deleting it), leaves a still-open (still being written by
ffmpeg) segment for the next pass, discards an invalid-duration segment, applies the
retention policy, and if retained, remuxes it (stream copy, `+faststart`) into permanent
storage under `RECORD_DIR/<date>/<hour>/<camera>/<mm.ss>.mp4`, writing a `Recordings`
row.

### Retention (`mirage/recording/retention.py`)

Pure decision logic, no DB access: retain if continuous retention days is set and the
segment falls within that window, or motion retention days is set and the segment has
activity within its (always-at-least-as-long-as-continuous) window. A segment that
overlaps an active `ReviewSegment`'s time window can instead be governed by that
segment's own retention mode.

### Review-clip stitching (`mirage/recording/stitch.py`)

A review segment's own duration (up to 30+ seconds of continuous activity) almost always
spans *multiple* separate fixed-length recording files (10-60s each), never exactly one
— so "play the footage that led to this alert" needs to find every recording overlapping
the segment's time window and stitch them together. `recordings_overlapping()` finds
every recording row whose own time range overlaps the given window (not just ones that
started inside it — a recording that started slightly before the window still contains
the first moments of it). `stitch_recordings()` writes an ffmpeg concat-demuxer list
file and runs a stream-copy concat (no re-encode, since all of one camera's recordings
share codec/resolution) into the destination path. Exposed via `GET
/api/review/{id}/clip`, cached on disk keyed by segment id (a segment's window and its
underlying recordings are immutable once it's ended, so repeat requests reuse the
already-stitched file).

---

## 8. API layer

**File:** `mirage/api/app.py` (`create_app()`) plus every router under
`mirage/api/routers/`.

A separate FastAPI process from `MirageApp` — reads the same SQLite DB (safe under WAL)
but does not own or supervise the pipeline. Every route reads config fresh from the DB
per-request (except when a fixed config override is passed, test-only), so newly-saved
config is visible through the API immediately even though the pipeline itself needs a
restart to act on it.

| Router | Prefix | Endpoints |
|---|---|---|
| `cameras.py` | `/api/cameras` | `GET ""` (credential-free camera list), `GET "/{name}"` |
| `config.py` | `/api/config` | Detector CRUD (`GET/POST/PUT/DELETE /detectors[/{name}]`), `GET /execution-providers`, camera-config CRUD with raw RTSP URL (`GET/POST/PUT/DELETE /cameras[/{name}]`), query CRUD (`GET/POST/PUT/DELETE /queries[/{id}]`) |
| `events.py` | `/api/events` | `GET ""` (filterable, excludes false positives by default), `GET "/{id}"`, `GET "/{id}/snapshot"` |
| `live.py` | `/api/live` | `GET "/{camera}/snapshot.jpg"` and `WS "/{camera}/ws"`, both proxying go2rtc so the frontend only ever talks to the mirage API origin |
| `onvif.py` | `/api/onvif` | `GET "/scan"` (proxies go2rtc's WS-Discovery probe), `GET "/resolve"` (calls the device's real ONVIF Media service directly) |
| `query_matches.py` | `/api/query-matches` | `GET ""` (filterable), `GET "/{id}"`, `GET "/{id}/thumbnail"` |
| `recordings.py` | `/api/recordings` | `GET ""` (filterable), `GET "/{id}"`, `GET "/{id}/clip"` (single raw clip) |
| `review.py` | `/api/review` | `GET ""` (filterable), `GET "/{id}"`, `GET "/{id}/thumbnail"`, `GET "/{id}/clip"` (the stitched multi-recording clip) |

Notable design decisions:

- **Credential separation**: the general-purpose `GET /api/cameras` returns a
  credential-free shape (no RTSP URL at all), while `GET /api/config/cameras` (used only
  by the camera edit form) returns the raw RTSP URL including embedded credentials —
  deliberately different response models, so the broadly-consumed listing endpoint (used
  by the Live page's camera grid) never leaks credentials to every reader.
- **ONVIF two-step flow**: `/scan` proxies go2rtc's own WS-Discovery client (no
  credentials needed, matches ONVIF's own discovery semantics); `/resolve` deliberately
  bypasses go2rtc for the actual stream URI, calling the device's ONVIF Media service
  directly via `onvif-zeep-async`, because go2rtc's own resolve mode returns an internal
  `onvif://...` URL that mirage's ffmpeg-based capture process has no way to consume —
  it needs a plain `rtsp://` URL.
- **Restart-required mutations**: every config-write endpoint's response includes
  `restart_required: true` explicitly, rather than silently implying the change is live.

---

## 9. go2rtc integration

**Files:** `mirage/go2rtc/download.py`, `config.py`, `process.py`.

go2rtc is a separate Go binary, not vendored in this repository — `ensure_go2rtc_binary()`
downloads the correct platform release asset from GitHub on first use and caches it
locally (idempotent; no network I/O once cached). Its config is generated per-run,
containing only the `streams:`/`api:`/`webrtc:` sections actually needed: each enabled
camera's detect-role ffmpeg input is registered as a go2rtc stream keyed by the camera's
own name, so the frontend can request `?src=<camera_name>` with no separate name
mapping. When a camera's `rtsp_transport` is `udp`, the generated source URL gets a
`#transport=udp` suffix — go2rtc holds its *own* independent RTSP connection to the
camera, separate from mirage's own ffmpeg capture connection, so it needs to be told the
same transport choice explicitly.

go2rtc runs as a plain `subprocess.Popen` (not `mp.Process` — there's no Python code
running inside it to share memory with), given its own process group so it can be
SIGTERM'd (then SIGKILL'd after a timeout) as a unit.

Its role in the rest of the system is entirely as a proxied backend: the live-view API
proxies snapshot/WebSocket traffic to it so the browser never talks to it directly, and
the ONVIF scan endpoint proxies its WS-Discovery client — but the ONVIF *resolve* step
bypasses it entirely (see section 8).

---

## 10. Frontend architecture

Angular, standalone components, zoneless change detection (`provideZonelessChangeDetection`).

### Routes (all children of a single `Shell` layout)

| Path | Page | Purpose |
|---|---|---|
| `/review` (default) | Review | Alert/detection activity feed, filterable by camera/severity; click a card for detail plus stitched-clip playback |
| `/live` | Live | Camera tiles with tiered MSE → WebRTC → snapshot playback |
| `/events` | Events | Every individual tracked-object detection, filterable by camera/label/time range, lightbox snapshot view |
| `/recordings` | Recordings | Stored clips grouped by date, filterable by camera, hover preview, lightbox playback |
| `/add-camera` | Add Camera | Wizard: ONVIF scan+resolve *or* manual RTSP entry, then a details form; also handles edit mode via a query param |
| `/cameras` | Manage Cameras | List, edit, enable/disable, delete cameras |
| `/detectors` | Manage Detectors | List, add, edit, delete detector instances; execution-provider picker |
| `/queries` | Queries | Add/edit/delete saved open-vocab text queries, plus a live "recent matches" list with lightbox |

### API service

A single injectable `ApiService` wraps `HttpClient`, with methods mapping roughly 1:1
onto every backend router (cameras, events, recordings, review, live snapshot/WS URLs,
detectors, execution providers, camera-config CRUD, ONVIF scan/resolve, query CRUD,
query-match listing/thumbnails). The frontend's model types are deliberately kept in
sync field-for-field with the backend's Pydantic response schemas.

### Live view tiered playback

Mirrors Frigate's own fallback chain: prefer MSE (speaks go2rtc's handshake, then
receives raw binary fmp4 fragments over a proxied WebSocket), fall back to WebRTC
(standard offer/answer/ICE-candidate signaling over the same WebSocket) if MSE is
unsupported or fails, and fall back to a 1-second-polled, cache-busted JPEG snapshot
poster if neither video tier connects. The snapshot poll runs continuously regardless of
video-tier state, since it also serves as the poster during the initial connecting
window.

### Shared components

- **Lightbox** — a generic full-screen image viewer (URL, alt text, projected caption),
  closable via close button, backdrop click, or Escape.
- **VideoLightbox** — the same, but for a full-screen in-page `<video controls>`
  overlay with a separate download affordance; auto-plays on open, silently absorbing
  autoplay-block errors.
- **Icon** — an inline-SVG icon set with no external font/library dependency.
- **FilterSelect** — a generic labeled dropdown, used for camera/label/severity/time
  filters across several pages.

---

## 11. `mock_cameras` — standalone ONVIF/RTSP test double

**Directory:** `mock_cameras/` (sibling to `mirage/`, at the repo root).

A fully standalone helper app — its own `requirements.txt`, its own virtualenv — with
**zero shared code with mirage**. Even its go2rtc download/process logic is a
deliberate vendored copy rather than an import, specifically to avoid pulling in
mirage's entire dependency tree (FastAPI, onnxruntime, OpenCV, ...) just to reuse a
small amount of subprocess/download logic, and so this tool has no dependency on the
mirage package existing on disk at all.

It exists so the add-camera wizard's ONVIF scan+resolve flow and the live-view pipeline
can be exercised end-to-end, offline, without a real camera.

- **Config**: a minimal, dependency-light YAML loader — a video directory plus a list
  of camera entries (name, optional explicit path override); each camera's video file is
  resolved by searching the video directory for a matching filename if no explicit path
  is given.
- **Stream generation**: one go2rtc `streams:` entry per configured camera, each set to
  loop its source video forever at real-time pace via go2rtc's `ffmpeg:` source syntax.
  A **named** custom ffmpeg input template had to be registered in go2rtc's own config
  (rather than an inline literal), because an inline template string containing spaces
  was confirmed, through actual testing, to silently produce zero media tracks — despite
  go2rtc's own documentation showing that inline form as valid.
- **WS-Discovery responder**: a hand-rolled UDP multicast listener on the real WS-Discovery
  group and port, replying to Probe messages with a spec-shaped SOAP ProbeMatches
  envelope per configured camera, unicast back to the sender (matching real WS-Discovery
  semantics: only the probe is multicast, replies are unicast). De-duplicates by the
  probe's own message id, since go2rtc's own prober doesn't dedupe and a multi-interface
  host can deliver one multicast probe more than once.
- **ONVIF SOAP service**: a hand-rolled ONVIF Device Management + Media SOAP service, one
  HTTP server per camera, verified directly against the real bundled WSDL/XSD schemas and
  against the actual SOAP client library's binding-selection and response-parsing code —
  not just "looks like valid XML." Notably, it emits SOAP **1.2** envelopes even though
  the WSDL's `transport` attribute literally names the SOAP 1.1 URI, because the WSDL's
  actual binding element uses the 1.2 binding namespace, which is what the real client
  library's binding-selection logic actually keys off. It deliberately does *not*
  implement `GetServices` (always answers with a SOAP fault), forcing every client
  through the documented `GetServices → GetCapabilities` fallback path, so there's only
  one place advertising service endpoints to keep in sync.
- **Entry point**: a single asyncio event loop starts go2rtc, then the WS-Discovery
  responder, then one ONVIF server per configured camera, each on its own port; shuts
  down in reverse order on SIGINT/SIGTERM.

---

## 12. Known limitations and operational notes

See `TODO_FIX_LIST.md` for full detail and file-level pointers. In brief:

- **No config hot-reload** — the single most recurring source of confusion in practice.
  Adding, editing, or deleting a camera/detector/query via the API takes effect in the
  database (and is visible through `mirage.api`) immediately, but the running pipeline
  process has no way to know until it's manually restarted. The UI currently gives no
  indication of this distinction — a newly-added camera just shows "Signal lost"
  indefinitely, indistinguishable from a genuinely unreachable camera.
- **Mock camera stream stutter** — one of the mock cameras' source videos is real
  50fps/1280×544 footage, captured by mirage at only 5fps (a 10:1 discard ratio), which
  can visibly stutter on the Live page even though no frames are actually being dropped
  at the network level.
- **Crowd detection** was scoped alongside the open-vocab work as a pure counting/geometry
  feature (no new model needed) but is not yet implemented.
- **MQADet-style full free-text frame search** (a heavier, MLLM-driven alternative to
  the current OWLv2-based query matching) was explored and deliberately deferred as
  infeasible for continuous monitoring on current hardware.
- **OWLv2 is CPU/MPS-only today** — no CoreML/Neural-Engine export has been attempted,
  and no real CUDA hardware has ever exercised the `cuda` code path, though the
  execution-provider-style abstraction is already in place so that's expected to be a
  config change rather than a rewrite once dedicated hardware is available.

### Local run order (`HOW_TO_RUN.md`)

Four independent processes, started in this order: an optional synthetic test stream
(not needed with a real camera or with `mock_cameras`), `python -m mirage` (the
pipeline), `python -m mirage.api` (the read/write API), and `npm start` in `frontend/`
(the Angular dev server). The pipeline's `--config` flag only ever imports on a
genuinely empty database — every subsequent run reads exclusively from the DB, and
picking up any config change made through the wizard/API always requires restarting
`python -m mirage` specifically (the API process itself needs no restart to reflect a
config change, since it reads fresh from the DB on every request).
