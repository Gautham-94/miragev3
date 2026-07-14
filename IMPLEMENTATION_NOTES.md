# Implementation Notes / Deviations from the Spec

Tracks places where building the real thing surfaced a bug or ambiguity in
`NVR_PIPELINE_IMPLEMENTATION_SPEC.md` / `NVR_EXTENSION_MULTI_MODEL_ROUTING.md`, and what was
actually done differently and why. Read this alongside the spec docs, not instead of them.

## 0. Build status summary

The full pipeline described in the spec is implemented end-to-end in `mirage/` (53 source
files) and covered by 205 tests (28 test files) in `tests/`, all passing. This includes
real integration tests (not mocked) exercising: real ffmpeg subprocesses reading actual
network video streams, a real exported ONNX model (yolov8n) performing genuine object
detection against real photos, the real multi-process architecture (separate OS processes
for capture/tracking/detection, verified via `is_alive()`/`exitcode` checks), real SQLite
DB writes, and clean shutdown with explicit post-test sweeps confirming zero leaked
processes or shared-memory segments after every integration test.

Key end-to-end scenarios validated:
- Single camera: capture -> motion -> region selection -> ONNX detection -> tracking ->
  event/review DB persistence -> recording (`tests/test_app_integration.py`).
- Two cameras sharing ONE detector process (the spec's core "N cameras, 1 loaded model"
  claim), each independently producing recordings while the shared detector process
  count stays at exactly 1 regardless of camera count
  (`tests/test_multi_camera_integration.py`).
- Every individual module (shm, ffmpeg presets, motion, tracker, region selection,
  reduce/consolidate, event lifecycle, review segments, retention, watchdog, zmq
  pub/sub, time utils) also has focused unit tests independent of the full pipeline.

Sections below are numbered roughly in the order they were discovered during
development, not by severity -- read the whole file if extending this codebase, since
several of these (the SHM lifecycle fix, the datetime/peewee fix, the
is_open_by_ffmpeg fix) are foundational and affect multiple call sites.

## 1. `mirage/util/shm.py` -- SharedMemory page-rounding and exported-buffer bugs

The spec's Appendix B.1 skeleton (`FrameManager.write(name) -> memoryview`) has two real bugs
that only show up under actual use, not casual reading:

1. **Page rounding.** `multiprocessing.shared_memory.SharedMemory` rounds an allocation up to
   the OS page size, so `shm.buf` can be *larger* than the size originally passed to
   `create()`. The spec's capture-loop skeleton does `frame_buffer[:] = ffmpeg_process.stdout
   .read(frame_size)` -- a memoryview slice-assignment that requires the lvalue and rvalue
   to have the exact same length. If `write()` returns the raw (page-rounded) `shm.buf`
   instead of a view sliced to the caller's actual frame size, this assignment raises
   `ValueError: memoryview assignment: lvalue and rvalue have different structures` for
   basically every real YUV420 frame size (they're essentially never exact multiples of the
   page size).

   Fix: `write(name, size)` now takes an explicit `size` argument (same value the camera
   always uses for that segment, e.g. its known YUV420 frame size) and returns
   `shm.buf[:size]`, never the raw buffer. `get(name, shape, dtype)` already had the
   information needed (shape+dtype -> byte count) and does the same slicing. This is a
   **signature change** from the spec's `write(name)` -- worth knowing if you're
   cross-referencing the spec's capture loop pseudocode, which calls `frame_manager.write
   (frame_name)` with no size arg.

2. **Exported-pointer close() failure.** Python's `mmap.close()` (called from `SharedMemory
   .close()`) raises `BufferError: cannot close exported pointers exist` if any
   memoryview/ndarray obtained from that mmap's buffer is still alive. A completely
   reasonable-looking caller pattern -- keep the `buf` returned by `write()`, write into it,
   then call `frame_manager.close(name)` while a local variable still references that
   `buf` -- triggers this every time. The spec's skeleton doesn't call this out at all.

   Fix: `SharedMemoryFrameManager` now tracks the last view (memoryview or ndarray) it
   handed out per segment name, and `close()`/`delete()`/`cleanup()` explicitly release
   that view before closing the underlying `SharedMemory`. Callers no longer need to
   manage memoryview lifetimes themselves -- requesting a new view (via `write()`/`get()`)
   or closing/deleting is sufficient; there's no need to call `.release()` manually.

Both are covered by regression tests in `tests/test_shm.py`
(`test_single_process_create_write_get_roundtrip`, `test_segment_survives_child_process_exit`,
`test_multiple_independent_frame_managers_share_segment`) which failed before the fix and
pass after.

## 2. Environment/paths

The spec assumes a Linux production host (`/tmp/cache` as tmpfs, `/media/nvr`, `/dev/shm`).
This build runs on macOS during development, so `mirage/const.py` resolves all of the spec's
path constants relative to the project root by default, fully overridable via environment
variables (`MIRAGE_CACHE_DIR`, `MIRAGE_BASE_DIR`, etc.) for a real Linux deployment. No
logic depends on being on Linux specifically -- `multiprocessing.shared_memory` and
`subprocess`/ffmpeg both work the same way on macOS, so the only actual change needed is
where files/sockets live, not how the pipeline behaves.

## 3. Detector backend for first working build

Per user decision: ONNX Runtime (CPU) is the first `DetectionApi` backend implemented, using
a small pretrained COCO model. Other backends (OpenVINO, EdgeTPU/TFLite, etc.) are not
implemented in this pass but the `DetectionApi` plugin interface is built to support adding
them later without touching any other module.

**Model provenance**: `models/yolov8n.onnx` was exported from a `yolov8n.pt` checkpoint the
user already had locally (`/Users/gauthamkrishna/Documents/Projects/python/epoch1/
yolov8n.pt`), via `ultralytics`' own `model.export(format="onnx", imgsz=320, simplify=True)`
(installed into the project venv for this one-time export only -- `ultralytics`/`torch` are
NOT runtime dependencies of `mirage` and are not in `requirements.txt`; only `onnxruntime`
is needed to actually run the exported model). `models/coco_labelmap.txt` was generated
from the same checkpoint's `model.names` dict (80 standard COCO classes, index-prefixed
format). The ONNX export's output tensor shape is `[1, 84, N]` (4 box-regression values +
80 class scores, transposed/anchor-last layout) -- `mirage/detection/plugins/
onnx_yolov8.py` decodes specifically this YOLOv8 export layout; a differently-exported
model (e.g. YOLOv5, a native-NMS SSD) would need its own decode path behind the same
`DetectionApi` interface, per spec section 3.3.2.

**Test images**: `media/bus.jpg` and `media/zidane.jpg` are ultralytics' own bundled
example assets (shipped inside the `ultralytics` pip package's `assets/` directory),
copied in for the real end-to-end detection test in `tests/test_onnx_yolov8_e2e.py` --
these are standard, widely-used YOLO demo images (a bus with pedestrians; two people),
used here purely to verify the full pipeline (YUV420 conversion -> tensor construction ->
ONNX inference -> decode -> label mapping) correctly identifies real "bus" and "person"
objects, not just that the code runs without error.

## 3a. Unix domain socket path length limit (real constraint, not just a test artifact)

`AF_UNIX`/`ipc://` socket paths are capped at roughly 103-107 bytes
(`sizeof(sockaddr_un.sun_path)`) on both macOS and Linux. `mirage/ipc/zmq_pubsub.py`'s
`ZmqProxy` binds real filesystem paths under `CACHE_DIR`, and this limit was hit
immediately by pytest's own deeply-nested `tmp_path` fixture during test development
(paths like `.../pytest-of-<user>/pytest-<N>/<full-test-function-name>0/proxy_pub`
routinely exceed it) -- tests use a short fixed-prefix temp dir instead (see
`tests/test_zmq_pubsub.py`).

This is not just a test-scaffolding quirk: `mirage/const.py`'s development-mode default
`CACHE_DIR` (nested under the project directory, e.g. `.../frigate_nvr/mirage/media/cache`)
produces an `ipc://` address around 85 characters -- inside the limit, but with little
margin. **In a real deployment, set `MIRAGE_CACHE_DIR` to something short** (the spec's own
recommendation of `/tmp/cache` is exactly right for this reason, independent of its
tmpfs-performance rationale) rather than a long nested application-install path, or the
ZMQ proxy/IPC sockets will fail to bind with `ZMQError: ipc path ... is longer than ...`
at startup.

## 3b. ZMQ PUB socket "slow joiner" message drop

`mirage/ipc/zmq_pubsub.py`'s `Publisher.__init__` connects a `zmq.PUB` socket to the
proxy's XSUB endpoint. `connect()` returns before the underlying transport handshake has
actually completed -- if `publish()` is called immediately afterward (which a fresh
detector-signal `Publisher` legitimately might be, right after construction), the message
can be silently dropped with no error, since PUB sockets never buffer for a
not-yet-connected subscriber. This is ZMQ's well-known "slow joiner" problem, not a bug
specific to this codebase, but it reproduced concretely and consistently here (see
`tests/test_zmq_pubsub.py::test_cross_thread_publish_subscribe_timing`, which failed 100%
of the time before the fix). Fixed with a short (0.1s) fixed settle delay after `connect()`
inside `Publisher.__init__` -- there is no portable "connection established" event to wait
on instead. This cost is paid once per `Publisher`'s lifetime (at construction), not per
`publish()` call, and every real `Publisher` in this system (the detector process's
result-ready signal, any config/event broadcaster) is long-lived -- created once per
process, reused for every subsequent message -- so this has no steady-state throughput
impact; it only guards the very first message after a fresh connection.

## 3c. Stationary classifier: trimmed-box-vs-trimmed-box comparison doesn't work

`mirage/tracking/stationary.py`'s `StationaryClassifier` (spec section 5.7) maintains a
rolling percentile ("trimmed") box over the last 10 raw detection boxes, and the spec text
describes comparing "this trimmed box's IoU against the previous trimmed box" to decide
active/stationary. Implemented literally, **this produces false "stationary" positives for
any object moving at a normal, steady pace** (walking speed, driving speed at typical
detect-fps) -- confirmed both with an aggressive synthetic rate (30px/frame) and a much
gentler one (2px/frame; still flagged stationary within ~2 frames either way).

Root cause: a fixed-size sliding window's 15th/85th-percentile bounds track a linearly
translating box almost exactly -- two *consecutive* trimmed boxes only differ by the one
oldest sample leaving and one newest sample entering the window, so they stay at high IoU
with each other (~0.6-0.8 in testing) regardless of how fast the underlying object is
actually moving, as long as the per-frame displacement is smaller than the trimmed box's
own width (which is itself inflated to roughly the object's total travel distance across
the whole 10-frame window, i.e. often several times the object's own size).

**Fix**: compare the CURRENT raw single-frame box against the rolling trimmed
(percentile-smoothed) history box, not trimmed-vs-previous-trimmed. This correctly
distinguishes the two cases: under jitter, each new raw box stays within the established
smoothed range (high IoU, correctly not flagged active); under real sustained motion, the
fresh raw box quickly diverges from a history box anchored to now-stale older positions
(low IoU, correctly flagged active). Covered by
`tests/test_tracker.py::test_stationary_classifier_stays_active_with_movement` and
`test_object_tracker_moving_object_never_flagged_stationary`, both of which failed with
the literal trimmed-vs-trimmed comparison and pass with this fix.

This is a case where the spec document's prose (itself a summary of a real system's
behavior, not verbatim source) didn't fully specify the exact comparison operands, and the
literal reading produced an incorrect result -- worth flagging if this spec is used as a
reference again: "trimmed box vs previous trimmed box" is the wrong comparison; "raw
current box vs trimmed history box" is correct.

## 3d. Region selection must carry forward UNCONFIRMED candidate detections, not just confirmed tracks

`mirage/tracking/orchestration.py`'s `CameraOrchestrator.process_frame` (spec section 7)
builds this frame's detector regions from confirmed tracked-object boxes (via
`ObjectTracker.current_states()`) plus fresh motion boxes. Built literally that way, **a
genuinely new object can only ever be detected once and then never again**, because:
norfair's `initialization_delay` (`min_initialized`, e.g. 2 frames at 5fps) requires
multiple *consecutive* matching detections before a candidate track is promoted to
"confirmed" and returned by `current_states()` -- but region selection only re-scans
confirmed tracks' locations (plus wherever fresh motion happens to trigger next). If the
object's location isn't independently re-triggered by motion on the very next frame
(realistically common: motion can settle, or the object's motion contour can shift
slightly and fall just outside the previous frame's region), there's nothing left to make
the detector look at that location again, so the candidate can never accumulate its
second confirming detection and is silently lost forever after one frame.

This reproduced concretely with a real static test photo: the very first frame's one-shot
"startup scan" (full-frame region) found the photo's real bus/person objects, but every
subsequent frame produced zero regions (motion settles to zero on a truly static input,
and there were no confirmed tracks yet to build regions from) -- so the tracker stayed
permanently empty, confirmed via
`tests/test_orchestration.py::test_orchestrator_produces_tracked_objects_from_real_photo_sequence`,
which failed before the fix.

**Fix**: `CameraOrchestrator` now also carries forward every box from the *previous
frame's consolidated (post-NMS) detections* -- confirmed or not -- as additional
tracked-object-style regions for the next frame, via a `_pending_candidate_boxes` list
populated at the end of every `process_frame()` call. This guarantees a real object,
once detected anywhere even once, keeps being re-scanned in its last-known location every
subsequent frame until either it's confirmed (and then covered by `current_states()`
directly) or genuinely stops appearing there (no detection there next frame -> it drops
out of `_pending_candidate_boxes` naturally, and norfair's own `hit_counter_max` still
governs how long an unconfirmed candidate is kept alive internally).

## 3e. peewee DateTimeField silently fails to parse timezone-aware datetimes

Every write path that stored `datetime.datetime.now(datetime.timezone.utc)` (a
timezone-AWARE datetime) directly into a peewee `DateTimeField` hit the same real bug:
`peewee.DateTimeField.formats` is `['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
'%Y-%m-%d']` -- none of these patterns include a timezone-offset specifier. SQLite
serializes a tz-aware datetime with a `+00:00` suffix, which then fails to match any of
peewee's parse formats on read-back, and `playhouse`'s `format_date_time()` helper
silently falls back to returning the raw, unparsed string instead of raising -- so the
field comes back as a `str`, not a `datetime.datetime`, and any arithmetic on it
(`event.start_time + timedelta(...)`) raises `TypeError: can only concatenate str (not
"datetime.timedelta") to str` at the point of use, often far from the actual write.

This was caught concretely in `mirage/events/processor.py`'s `close_dangling_events()`
(`tests/test_event_processor.py::test_close_dangling_events_on_startup` failed with
exactly that TypeError) and independently affected `mirage/recording/maintainer.py`
(`Recordings.start_time`/`end_time`) and `mirage/db/models.py`'s `Regions.last_update`
default -- all three called `datetime.datetime.now(datetime.timezone.utc)` directly as a
field value or default.

**Fix**: added `mirage/util/time.py` (`utcnow()`, `utc_from_timestamp()`,
`as_naive_utc()`), all of which strip `tzinfo` before returning -- the convention adopted
everywhere in this codebase is that **every stored datetime is naive and implicitly
UTC**. Every call site that used to write `datetime.datetime.now(datetime.timezone.utc)`
or `datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)` directly into a model
field now goes through one of these helpers instead. A regression test
(`tests/test_db.py::test_naive_utc_datetime_survives_roundtrip_as_a_real_datetime`) pins
down the exact mechanism (asserting the naive path round-trips as a real `datetime` and
the tz-aware path does NOT, documenting why) so this can't silently regress.

**If you're extending this codebase**: never call `datetime.datetime.now(datetime.timezone.utc)`
or construct a tz-aware datetime directly as a value passed to a peewee model field or
compared against one read back from the DB -- always go through
`mirage.util.time.utcnow()` / `utc_from_timestamp()` / `as_naive_utc()`.

## 3f. RecordingMaintainer never actually implemented the "skip in-progress segments" check

Spec section 8.1 point 3 explicitly calls for skipping cache segments still being
actively written by ffmpeg, rather than treating them as invalid. The initial
implementation of `mirage/recording/maintainer.py`'s `_process_segment` never actually
did this -- it just ran `probe_duration()` once and deleted the file on any failure. A
segment-muxer output file has no moov atom (and is therefore unprobeable / "invalid")
until ffmpeg rolls over to the *next* segment or exits cleanly, so **every single segment
was being deleted on its very first maintainer poll, before ffmpeg ever finished writing
it** -- this reproduced 100% of the time once the full multi-process app (capture +
tracker + detector all running together, real CPU load from ONNX inference competing with
ffmpeg) was tested end-to-end, even though `RecordingMaintainer` in isolation (tested
against pre-existing, already-complete files) never caught it.

Diagnosed by tracing real segment files through several maintainer poll cycles directly
(bypassing the maintainer's own deletion) via `scripts/debug_app_recording.py`: a segment
that looked permanently broken (`moov atom not found`) at t=10s after it first appeared
was, in fact, a completely valid 10-second segment by t=15s -- it simply hadn't been
finalized yet at the moment of the first check. The recording pipeline had no bug in its
segment *creation*; the bug was in *reacting too eagerly* to a transient, expected state.

**Fix**: added `mirage/recording/segments.py`'s `is_open_by_ffmpeg(path)`, which uses
`psutil.process_iter()` to check whether any running `ffmpeg` process currently has the
exact file open, matching the spec's literal suggestion. `RecordingMaintainer
._process_segment` now checks this FIRST and simply skips (does not delete, does not
probe) a still-open file, leaving it for a later poll once ffmpeg has actually finished
writing it. Covered by `tests/test_recording_segments.py::test_is_open_by_ffmpeg_true_while_ffmpeg_holds_the_file_open`,
which starts a real ffmpeg process holding a real file open and asserts detection, then
asserts detection correctly flips to False once that process exits.

A secondary, purely-cosmetic finding along the way: the synthetic `lavfi testsrc + libx264
-preset ultrafast` source used in the app-level integration test defaults to a very long
GOP (~25s observed) with no explicit `-g`/keyframe forcing, and ffmpeg's segment muxer can
only cut a new segment at a keyframe boundary -- so with `-segment_time 10` and a 25s GOP,
no segment would ever complete within a realistic test window regardless of the
is_open_by_ffmpeg fix. `tests/test_app_integration.py`'s video server fixture now forces a
keyframe every 2 seconds (`-g 20 -force_key_frames 'expr:gte(t,n_forced*2)'`), matching how
a real IP camera's encoder is virtually always configured (short GOP specifically so
recording/segmenting/seeking works well).

## 3g. ServiceWatchdog.stop() didn't actually block, racing shutdown

Found running the real CLI (`python -m mirage`, added as `mirage/__main__.py` after the
initial build so the app could be launched outside of pytest) by hand and hitting Ctrl+C:
the log showed `detector:general died (exitcode=1), restarting` *after* the shutdown
sequence had already begun -- the watchdog resurrected a detector process that
`MirageApp.stop()` was in the middle of tearing down.

Root cause: `MirageApp.stop()` calls `self.watchdog.stop()` first, specifically so no
further restarts can happen once it starts terminating processes -- but the old
`ServiceWatchdog.stop()` only did `self._stop_event.set()` and returned immediately. The
watchdog thread's own loop is `while not self._stop_event.wait(self.tick_seconds):
self._check_all()`, and `Event.wait()` only re-checks the event after it wakes up (up to
`tick_seconds`, default 10s) -- so there's a window of up to 10 seconds, right after
`stop()` "returns", during which the watchdog thread can still wake up, see a process the
caller is mid-terminate on as simply "dead" (indistinguishable from a real crash from the
watchdog's point of view), and restart it. This is exactly the kind of bug that's
essentially invisible in tests unless a test happens to stop() the app while a genuinely
slow shutdown step is in flight, but was immediately visible running the real CLI by hand
against a live process tree.

**Fix**: `ServiceWatchdog.stop()` now joins its own thread (bounded by `tick_seconds + 5`)
before returning, so a caller can rely on "stop() returned" meaning "no future tick can
fire." Covered by `tests/test_watchdog.py::test_stop_blocks_until_thread_exits_and_prevents_late_restart`,
which uses a short `tick_seconds` so the race would reproduce reliably if this regressed.
Note this does not (and can't cheaply) abort a tick already *in progress* at the moment
`stop()` is called -- an in-flight `_check_all()` reads live process state and completing
it is fine; the bug was specifically the thread sleeping obliviously past the stop signal.

## 4. Test source for the capture pipeline

No external RTSP server dependency (e.g. mediamtx) was installed. Several synthetic
network-source options were tried against this machine's ffmpeg (8.1, Homebrew) before
landing on one that works reliably:

- **RTSP `-listen 1` server mode**: accepted by the CLI (`-f rtsp -rtsp_flags listen` /
  `-listen 1` on the rtsp muxer) but this build's RTSP muxer never actually binds/accepts
  a connection in practice -- `ffprobe`/client connections get "Connection refused"
  indefinitely. Not usable as a test source on this build.
- **RTMP `-listen 1` with a looped mp4 file source** (`-stream_loop -1 -i file.mp4 ... -f
  flv -listen 1 rtmp://...`): the server itself binds and accepts fine, but decoding
  reliably fails with `Packet mismatch` from the FLV demuxer at small test resolutions
  (64x48) -- reproduced consistently, root cause not fully isolated (suspected libx264
  `-preset ultrafast` + tiny resolution + `-re` real-time pacing interacting badly with
  FLV's tag-size verification), but not worth pursuing further given a working alternative.
- **Raw TCP MPEG-TS with an `lavfi` testsrc generator** (`ffmpeg -re -f lavfi -i
  testsrc=... -f mpegts 'tcp://host:port?listen'`): this is what's actually used
  (`tests/test_network_stream_integration.py`). Reproduced reliably with no decode errors.
  A natively-infinite generator source also sidesteps any loop-boundary timestamp
  discontinuity entirely, unlike looping a short file.
- In all three `-listen`-mode cases, the server process accepts exactly ONE connection
  and then exits once that client disconnects -- not a real camera's behavior (a real
  camera/RTMP/RTSP source accepts reconnects indefinitely). Test fixtures wrap the server
  invocation in a shell `while true; do ffmpeg ...; done` loop to approximate that.

**A real platform-specific SIGTERM-delivery quirk was found and is relevant beyond
testing**: when ffmpeg is actively reading from a **live, still-connected network socket**
(confirmed with the TCP/mpegts source above; not reproduced against a local file source),
`Popen.terminate()` (and even an explicit `send_signal(SIGTERM)`) reliably fails to stop
the process within several seconds on this machine -- the process remains alive and
interruptible (not stuck in an uninterruptible D-state), but SIGTERM appears to not be
acted on promptly while ffmpeg is blocked in the socket read. `SIGKILL` (`Popen.kill()`)
reliably terminates it within the same test in under a second. This is **not a bug in
`mirage.capture.process_utils.stop_ffmpeg`** -- its existing design (SIGTERM, wait up to
`timeout` seconds, then SIGKILL) already handles this correctly; it just means the
`timeout` (spec default: 30s) is a real, sometimes-hit ceiling in production for a camera
whose ffmpeg is mid-read on a live connection when a restart is triggered, not merely a
theoretical worst case. Integration tests that trigger a stop/restart against a live
network source should budget their own deadlines accordingly (see the shortened
`retry_interval` used in `test_network_stream_integration.py`), rather than assuming
termination is always near-instant.

## 5. go2rtc integration for live view (frontend work, post-Task-15)

The original spec/build (Tasks 1-15) covered capture/detect/track/record with no HTTP or
UI surface at all. Adding a web frontend (Angular) meant first deciding how "live view"
would actually work, since mirage has no restreaming layer of its own. Researched what the
real Frigate project does (its actual source, not assumption): go2rtc (a separate Go
binary) restreams each camera into WebRTC/MSE/MJPEG/snapshot-consumable formats; Frigate's
frontend tiers itself MSE -> WebRTC -> JSMpeg -> JPEG-poster depending on browser support
and whether a given stream is go2rtc-restreamed at all.

Chose to replicate this rather than build something simpler, specifically to match
Frigate's real architecture. Key implementation decisions:

- **No vendoring**: go2rtc ships as a single OS-specific binary; `mirage/go2rtc/download.py`
  downloads the correct release asset (verified directly against go2rtc's GitHub releases
  API, not guessed -- macOS/Windows/FreeBSD assets are zipped, Linux/mips assets are raw
  executables, confirmed by both listing real release assets and testing the actual
  download+extract+chmod+run path end-to-end against a live release) into `mirage/bin/`
  on first use, then reuses the cached binary on every subsequent run (verified the second
  call is ~15 microseconds vs. a real network fetch on the first).
- **Config generation**: `mirage/go2rtc/config.py` builds go2rtc's `streams:`/`api:`/
  `webrtc:` YAML sections from `MirageConfig`, registering each enabled camera's `detect`
  role input path as a stream under the camera's own name (so the frontend can request
  `?src=<camera_name>` directly with no separate name-mapping table). Deliberately picks
  the `detect`-role input specifically (not `record`), confirmed via a dedicated test,
  since a camera can have separate detect/record ffmpeg inputs and go2rtc should restream
  whichever one is actually meant for live viewing.
- **Process supervision**: `mirage/go2rtc/process.py`'s `Go2rtcProcess` follows the same
  process-group SIGTERM-then-SIGKILL pattern as the ffmpeg helpers in
  `capture/process_utils.py`, launched/stopped from `MirageApp.start()`/`stop()` behind a
  new `enable_go2rtc` constructor flag (default True; existing integration tests set it
  False since they test the capture/detect/record pipeline specifically and don't need
  the extra network-facing process).
- **Verified for real, not just "process starts"**: the integration test
  (`tests/test_go2rtc_integration.py`) launches a real go2rtc against a real live TCP test
  stream and asserts on go2rtc's own `/api/streams` and `/api/frame.jpeg?src=...`
  endpoints -- confirming actual JPEG bytes (`\xff\xd8` SOI marker) come back, i.e. go2rtc
  is genuinely decoding the live stream, not just that its process object reports alive.

## 6. FastAPI read API (mirage/api)

Runs as a third, independent process (`python -m mirage.api --config ... --db-path ...`),
separate from both `MirageApp` (the capture/detect/track/record pipeline) and go2rtc. It
only ever reads: the same SQLite DB file (safe to open concurrently under WAL mode --
already the mode `mirage/db/database.py` uses) and the same `MirageConfig` YAML, and
serves recording clips / review thumbnails straight off disk via `FileResponse`. It never
writes to the DB and holds no reference to any capture/tracker/detector process, so it can
be started, stopped, or crashed independently of the running pipeline with zero
coordination needed.

Response models are hand-written Pydantic schemas (`mirage/api/schemas.py`), deliberately
decoupled from the peewee ORM models (`mirage/db/models.py`) rather than serializing them
directly -- this converts naive-UTC `DateTimeField` values to epoch-seconds floats (via
`dt.replace(tzinfo=utc).timestamp()`, matching the naive-UTC storage convention documented
in `mirage/util/time.py`) and avoids committing internal storage details (e.g. `Regions`
schema) as part of the public API contract.

Verified with both `TestClient`-based integration tests inserting real rows via the peewee
models directly (`tests/test_api.py`, 14 tests) AND a real subprocess smoke test: launching
`python -m mirage.api` for real against the actual `config/mirage.yaml`, then curling
`/api/health`, `/api/cameras`, and the auto-generated `/docs` Swagger UI, confirming the
CLI entrypoint itself (argument parsing, uvicorn startup, DB init via the FastAPI lifespan
context) works end-to-end, not just the app object in isolation.

## 7. Live-view proxy (mirage/api/routers/live.py) + a real single-client test-source bug

Added two endpoints so the Angular frontend only ever talks to one backend origin (the
mirage API), never go2rtc's port directly: `GET /api/live/{camera}/snapshot.jpg` (proxies
go2rtc's `/api/frame.jpeg?src=...`) and `WS /api/live/{camera}/ws` (a full bidirectional
relay to go2rtc's own `/api/ws?src=...` signaling endpoint, which the MSE/WebRTC player
code in the frontend needs to speak directly). Both verified against a real go2rtc
restreaming a real live test source, including a real WebSocket round-trip through the
relay (`tests/test_api_live.py`).

**Found a real full-stack bug only visible when running mirage + go2rtc + the API
together for real** (not in any isolated test): a full-stack smoke test (test video server
+ `python -m mirage` with go2rtc enabled + `python -m mirage.api`) got a permanently empty
response (`200 OK`, 0 bytes) from the snapshot proxy, indefinitely -- while the exact same
proxy code passed its own isolated integration test reliably. Root cause: the synthetic
test video server (`scripts/run_test_stream.sh`) uses ffmpeg's TCP `-listen` muxer, which
only accepts **one** client connection at a time (a real IP camera's RTSP server accepts
many concurrent clients -- this is purely a synthetic-test-source limitation, not
something a real deployment would ever hit). Mirage's own `CameraCapture` process
connected to the single-client test server first; go2rtc's later connection attempt to
the *same port* was silently left with nothing to read, producing an indefinitely "alive
but empty" stream. The isolated go2rtc test passed because it never had a competing
mirage capture process connecting to the same port.

**Fix**: `scripts/run_test_stream.sh` now runs TWO independent looping server instances
(port 19500 for mirage's own capture, port 19501 for go2rtc), both looping the same
source file. `mirage/go2rtc/config.py`'s `build_go2rtc_config`/`write_go2rtc_config` gained
an optional `stream_overrides: dict[str, str]` parameter (threaded through
`Go2rtcProcess.__init__` and `MirageApp.__init__`'s new `go2rtc_stream_overrides` param,
and exposed on the CLI as repeatable `--go2rtc-source CAMERA=URL`) so a camera's go2rtc
stream can be pointed at a different URL than its own detect-role ffmpeg input --
needed ONLY for this local-testing scenario; in production go2rtc simply reads the same
detect-role URL mirage's own capture reads, since a real camera has no such single-client
limit. Re-ran the exact same full-stack smoke test after the fix: snapshot proxy succeeded
on the very first attempt.

## 8. Angular frontend (mirage/frontend)

New Angular 21 app (standalone components, zoneless change detection, SCSS), scaffolded
with `ng new` then built out with: an API model/service layer
(`core/models/api.models.ts`, `core/services/api.service.ts`) mirroring
`mirage/api/schemas.py` field-for-field, a sidebar+router shell (`shell/`), and four
routed pages (Review, Live, Events, Recordings) that fetch real data from the FastAPI
backend built in Tasks 17-18. Live view uses the JPEG-snapshot-polling tier (1s interval,
cache-busted per request) for now; the MSE/WebRTC tiers (Task 20) build on top of the same
camera tile component.

Design direction: dark "control-room / operational instrument" aesthetic -- cool
near-black neutrals (never pure black/grey), Archivo (display/UI) paired with IBM Plex
Mono (data: timestamps, camera resolution/fps, durations, file sizes -- anywhere digits
need to align), a single warm signal-orange accent reserved for interactive/active state,
and a separate semantic palette for severity (alert/detection/info) so severity color
never gets confused with the interactive accent. Both typefaces self-hosted via
`@fontsource` (no external font CDN dependency). Design tokens live in
`frontend/src/styles/_tokens.scss` as CSS custom properties.

Two build-time SCSS/TS issues were caught immediately by `ng build` and fixed (not
theoretical, both reproduced on the very first build attempt):
- Sass's `@use` must appear before any `@import` in the same file -- had the fontsource
  CSS imports first; reordered.
- A generic `Record<string, unknown>` parameter type doesn't structurally accept the
  `EventListParams`/`RecordingListParams`/`ReviewListParams` interfaces (no index
  signature) -- loosened `toHttpParams`'s parameter type to a plain `object`.

Verified genuinely end-to-end, not just "it compiles": ran the real backend stack (test
video source, `python -m mirage` with go2rtc enabled, `python -m mirage.api`) alongside
`ng serve`, and used headless Chrome (`--headless --screenshot`) to capture and visually
inspect the actual rendered pages -- confirming the Live page's camera tile displays a
real decoded JPEG frame from the live test camera (fetched through the full mirage capture
-> go2rtc -> mirage.api -> Angular HttpClient -> browser `<img>` chain), and the
Recordings page lists real segment rows produced by the actual running recording
pipeline, with correct monospace numeral alignment. `ng build --configuration production`
and the default Vitest unit test both pass.

## 9. Tiered live player (MSE -> WebRTC -> JPEG poster)

`camera-tile/` gained two browser-side player classes
(`pages/live/players/mse-player.ts`, `webrtc-player.ts`) implementing go2rtc's actual
WebSocket signaling protocol directly, verified against go2rtc's own reference client
source AND Frigate's frontend implementation (not guessed):

- **MSE**: client sends `{"type":"mse","value":"<comma-joined supported codec mimes>"}`
  once the `MediaSource` fires `sourceopen`; server replies with the chosen codec string,
  client creates a matching `SourceBuffer`; all further messages are **binary** raw fmp4
  fragments appended via `appendBuffer()` (queued locally while the buffer is mid-update).
- **WebRTC**: client builds a recvonly `RTCPeerConnection`, sends
  `{"type":"webrtc/offer","value":"<sdp>"}` and each local ICE candidate as
  `{"type":"webrtc/candidate","value":"..."}`; server replies with `webrtc/answer` and its
  own `webrtc/candidate` messages.

`CameraTile` orchestrates the tiering exactly like Frigate's `LivePlayer`: try MSE first
(if `MediaSource` is supported), fall back to WebRTC on error, fall back to the
already-built JPEG snapshot poster if neither connects. The poster keeps polling
independently the whole time (not just as a fallback) since it's also what's shown during
the initial "connecting" window.

**Verified genuinely decoding and playing real video, not just that the code compiles**:
headless Chrome's `--screenshot` CLI flag turned out to NOT reliably capture a `<video>`
element's actual rendered frame (a real, confirmed headless-Chrome limitation, unrelated
to whether video is playing) -- so verification instead used the Chrome DevTools Protocol
directly (a small raw-websocket Python script against `--remote-debugging-port`) to: (a)
capture the real WebSocket frames flowing through the proxy, confirming actual fmp4
`ftyp`/`moov`/`moof`/`mdat` boxes arriving; (b) run `Runtime.evaluate` against the live
page to read the real `<video>` element's state directly
(`readyState: 4`/`HAVE_ENOUGH_DATA`, `videoWidth/videoHeight` matching the real source
video's dimensions, `paused: false`, badge text `"LIVE"`); and (c) use
`Page.captureScreenshot` (the CDP method, not the CLI flag) to get an actual compositor
screenshot showing the real decoded frame on-screen with the red LIVE badge. Also
observed and correctly handled a real transient failure mode during this verification:
the synthetic test source's single-client `-listen` TCP server briefly refuses new
connections in the gap between one client disconnecting and the loop's next ffmpeg
instance coming up -- go2rtc surfaced this as `{"type":"error","value":"mse: streams:
dial tcp ... connection refused"}`, and the tile correctly fell back to the "STILL" JPEG
poster rather than getting stuck, matching the designed fallback behavior exactly.

## 10. Real thumbnail capture for review segments AND events (backend gap found while building Task 21's UI)

Building the Review/Events UI surfaced a real backend gap: `ReviewSegment.thumb_path` and
`Event.has_snapshot` existed as DB fields per the original spec, but nothing ever actually
wrote a thumbnail file anywhere -- the `/api/review/{id}/thumbnail` endpoint built in Task
17 had nothing real to serve. Rather than build the frontend against a permanently-broken
image tag, added real capture:

- New shared helper `mirage/util/thumbnail.py`: `capture_thumbnail(fetcher, thumb_dir,
  filename_stem, camera_name)` -- best-effort (any failure returns None rather than
  propagating, since a missing thumbnail should never break event/review-segment
  creation), used identically by both `ReviewSegmentMaintainer` and `EventProcessor`.
- `ReviewSegmentMaintainer._start_segment` and `EventProcessor._on_start` both now call
  this with an injected `thumbnail_fetcher` callable, kept decoupled from any concrete
  HTTP client (same injection-seam pattern already used for `get_lifecycle`) so neither
  module has a direct dependency on go2rtc/httpx and both stay independently testable.
- `MirageApp` wires the real fetcher: a synchronous `httpx.get()` against go2rtc's own
  `/api/frame.jpeg?src=<camera>` (the same mechanism the live-view snapshot proxy in
  `mirage/api/routers/live.py` uses) -- deliberately synchronous/simple rather than async,
  since a segment/event START is a rare event (at most a handful per minute), not a
  per-frame operation.
- **Schema change**: `Event` gained a new `snapshot_path` column (previously only the
  `has_snapshot` boolean existed with nothing setting it meaningfully). No migration
  framework is wired up yet (`peewee_migrate` is a listed dependency but unused) --
  `create_tables(safe=True)` only creates missing tables, so an existing dev DB from
  before this change needs to be deleted and recreated, not migrated in place. Acceptable
  at this pre-release stage; worth revisiting before any real upgrade-in-place deployment.
- New API endpoint `GET /api/events/{id}/snapshot`, mirroring the existing
  `GET /api/review/{id}/thumbnail` pattern exactly (`FileResponse`, 404 if no
  snapshot_path set, 410 if the path is set but the file no longer exists on disk).

Covered by real tests, not just "the field exists": `tests/test_review.py` and
`tests/test_event_processor.py` each gained 2-3 tests using a fake injected fetcher
(success, no-fetcher, fetcher-raises-exception-doesn't-break-record-creation), and
`tests/test_api.py` gained the same three-case coverage (404/410/200-with-real-bytes) for
the new event snapshot endpoint that the review thumbnail endpoint already had.

## 11. Review/Events/Recordings page build-out (Task 21)

Built real filtering, thumbnails, and interaction on all three pages (not just visual
polish), per explicit scope confirmation:

- **Review**: camera + severity filter dropdowns (`shared/filter-select/`, a small
  reusable styled `<select>` component used across all three pages), a card grid showing
  each segment's real thumbnail (or a placeholder icon if none), and a click-through
  detail overlay showing the segment's contributing objects from its `data.objects` field.
- **Events**: camera + label + time-range filter dropdowns (camera/label options are
  derived live from whatever's actually in the fetched event set, via Angular `computed()`
  signals, rather than hardcoded), a small snapshot thumbnail column per row.
- **Recordings**: camera filter, date-grouped sections (grouped client-side via a
  `computed()` signal keyed by locale date string), and a real inline `<video>` preview
  that plays the actual recording clip on row hover (not just a static thumbnail).

Verified end-to-end against the real running stack once more: real go2rtc-backed review
segments/thumbnails where available, plus directly-inserted DB rows (bypassing the
detection pipeline) used specifically to exercise the filter/detail-panel/hover-preview UI
paths that the test video's real (but sparse/low-confidence) animal detections wouldn't
reliably trigger on their own -- confirmed via headless-Chrome CDP scripting: a real
`Runtime.evaluate`-driven click opening the detail overlay with correct data, and a real
`Input.dispatchMouseEvent` mouseMoved-driven hover showing the actual recorded video
playing inline in the Recordings row, screenshotted via `Page.captureScreenshot`.

## 12. Task 22 final validation: a real intermittent thumbnail-capture bug found running the whole stack for 10+ minutes

Ran the complete real stack together (test video source, `python -m mirage` with go2rtc,
`python -m mirage.api`, `ng serve`) continuously for over 10 minutes -- long enough for the
real detection pipeline to genuinely classify the test video's animal as "person"/"dog"
with real (if transient/low) confidence, producing real `Event` and `ReviewSegment` rows
for the first time outside of directly-inserted test data.

**Found**: every real `Event` got a real thumbnail file, but every real `ReviewSegment`
in that same window got `thumb_path = None` -- despite both using the literal same
`_fetch_go2rtc_thumbnail` closure via the same shared `capture_thumbnail()` helper (Task
21, section 10). No exception was logged for the review path (confirmed by grepping the
full debug log), which narrowed it to `capture_thumbnail`'s `if not jpeg_bytes: return
None` branch -- i.e. the fetcher call itself returned empty bytes for review specifically,
not an error. Root-caused to the same transient go2rtc behavior already documented in
section "3f"/Task 18 testing: `/api/frame.jpeg` can return `200 OK` with an EMPTY body
during a brief window (no buffered keyframe yet), and review-segment starts and
event starts fire at very slightly different moments for the same real-world detection,
occasionally landing review's call inside that empty-response window by chance of timing.

**Fix**: `capture_thumbnail()` now retries once (0.5s pause) if the fetcher returns empty
bytes on the first attempt, via a new `_fetch_with_retry` helper -- covered by
`tests/test_thumbnail.py` (new file, 6 tests: success, no-fetcher, empty-response,
fetcher-raises, retry-recovers-on-second-attempt, gives-up-after-two-failures). Verified
the fix for real, not just via the unit test: restarted the live `MirageApp` process
(clean shutdown in ~1s, confirmed) with the fix applied, let 4 more real review segments
fire naturally, and confirmed via direct sqlite query + `file` command that all 4 got a
real, valid JPEG `thumb_path` this time -- while the 4 pre-fix segments from before the
restart correctly remained `None` (proving the fix didn't touch old rows, only new
capture attempts). Screenshotted the real Review page afterward showing exactly this
split: the 4 newest cards render real thumbnails of the actual test camera's footage, the
4 older cards still correctly show the "no thumbnail" placeholder.

This is the kind of bug that is close to invisible in any test shorter than several
minutes against real (not synthetic-pattern) detection content -- it never reproduced in
any of the Task 16-21 test runs, all of which either used directly-inserted DB rows or
ran for under a minute against content that rarely triggered real detections at all.
Runs like this -- long enough for the full, real, un-mocked pipeline to naturally produce
several independent instances of the same code path -- are exactly what surfaces
low-probability transient-timing bugs that per-component unit/integration tests, however
thorough, structurally cannot catch on their own.

Also confirmed during this same validation pass: a `<video autoplay muted>` element that
appeared "stuck" (paused, badge stuck on "STILL") when driven via CDP without
`--autoplay-policy=no-user-gesture-required` turned out to be a headless-Chrome-CLI-only
artifact, not a real bug -- the exact same page, same CDP driver, with that one flag
added, correctly showed the video playing (`paused: false`, `readyState: 4`, real
`videoWidth`/`videoHeight`, badge "LIVE"). The component already sets `muted` on the
`<video>` element, which is the standard, correct mitigation for real browsers' autoplay
restrictions -- nothing needed to change here, but worth recording so a future session
doesn't mistake this specific headless-testing quirk for a real regression again.

## 13. Config moves from hand-edited YAML to the database (add-camera wizard prerequisite)

Building an "add camera" wizard exposed a real architectural gap: config lived only in a
YAML file, read once at `MirageApp`/`mirage.api` startup via `MirageConfig.from_yaml_file`
-- there was no way for a UI to change it short of generating YAML text and asking the
user to hand-edit a file. Moved to a DB-backed model instead:

- New singleton table `AppConfig` (`mirage/db/models.py`): one row (`id=1`, enforced by a
  `CHECK (id = 1)` constraint, not just convention), holding the entire validated
  `MirageConfig` tree as JSON. Deliberately NOT normalized into per-field columns/tables
  -- Pydantic already owns validation/shape for the whole nested tree
  (`CameraConfig`/`DetectorInstanceConfig`/`FfmpegConfig`/etc.), and re-deriving that as
  SQL schema would be a much larger, riskier change for no real benefit at this stage.
- `MirageConfig.from_db()` / `.save_to_db()` (`mirage/config/schema.py`, delegating to new
  `mirage/config/store.py` to keep the Pydantic schema module itself free of any direct
  peewee/DB import) mirror `from_yaml_file()`'s shape exactly. Both assume
  `init_database()` has already been called by the caller -- `AppConfig` shares the same
  `db_proxy` global connection as every other model; there's no separate path/connection
  for config specifically.
- `MirageConfig.default()`: the config a fresh install actually starts from -- the same
  `general` ONNX YOLOv8n detector `config/mirage.yaml` has always shipped, but
  `cameras: {}`. `from_db()` seeds this automatically (and persists the seed) the first
  time it's called against a database with no `AppConfig` row yet, so every caller always
  gets a usable, non-empty-of-detectors config back rather than needing separate
  first-run-vs-not branching logic of its own.

Covered by `tests/test_config_store.py` (6 tests): default-config shape, fresh-DB
auto-seed (and that the seed is actually persisted, not just returned in-memory),
repeated `from_db()` calls don't create duplicate rows, a full add-camera-then-reload
round trip, `save_to_db()` idempotency (multiple saves don't multiply rows), and adding a
second detector doesn't clobber the first.

Still to come (later tasks in this same effort): wiring the `mirage`/`mirage.api` CLI
entrypoints to actually use `from_db()` (currently only `from_yaml_file()` is wired to the
CLIs -- this section covers the storage layer only, not yet the switch-over), and the
config-write API endpoints + wizard UI that let a user add a camera through the DB path
instead of hand-editing YAML at all.

## 14. Switching `mirage`/`mirage.api` CLIs over to the DB-backed config

Both CLI entrypoints now use `MirageConfig.from_db()`/`.save_to_db()` (section 13)
instead of `.from_yaml_file()`:

- `python -m mirage --config <path>` -- `--config` changed from *required, read every
  run* to a **one-time import**: it only has any effect if the database has no
  `AppConfig` row yet (a genuinely fresh install), in which case that YAML file is
  imported once and saved to the DB. On every subsequent run, `--config` is accepted but
  ignored -- the DB is always the source of truth once it has one. This required
  reordering the CLI's own startup: it now calls `init_database()` itself, BEFORE
  constructing `MirageApp`, specifically to decide what config to load; `MirageApp.start()`
  still calls `init_database()` again internally on the same path afterward, which is
  safe (confirmed directly: re-initializing the same SQLite file just opens a fresh
  connection object bound to the same underlying WAL file, no data loss/duplication --
  this is the same one-DB-file/many-independent-connections pattern every other
  multi-process part of this system already relies on).
- `python -m mirage.api` -- `--config` was dropped from its CLI entirely (it now always
  reads config from `--db-path`, no YAML path ever accepted). More significantly,
  `create_app()` no longer bakes a single static `MirageConfig` into `app.state` at
  startup -- every route now calls `app.state.get_config()`, which by default is
  `MirageConfig.from_db` itself (re-reading the DB fresh on every single request). This
  is the mechanism that actually makes "add a camera through the API/wizard" work at
  all: without it, a request arriving even a second after the API process started would
  still see the config as it was AT STARTUP, never reflecting anything saved
  afterward -- exactly the kind of bug that's invisible in a quick manual check (you'd
  restart the API to test your change anyway) but would confuse every real user of the
  wizard. `create_app(config=...)` still accepts a literal fixed `MirageConfig` as an
  override, used only by tests that construct one directly without a real on-disk DB.

Covered by three new tests in `tests/test_api_db_config.py`: API reflects the seeded
default config on a fresh DB with zero manual setup; API reflects a camera saved to the
DB *before* the app process started; and -- the one that actually matters for the
wizard -- API reflects a camera saved to the DB *while already running*, proving config
isn't cached/frozen at app startup. Also verified by hand against the real CLI (not just
tests): ran `python -m mirage --config <yaml>` once, confirmed the `AppConfig` row was
seeded correctly; ran it again with `--config` completely omitted and the YAML file
deleted, confirmed it still started with the same camera/detector, read entirely from
the DB.

## 15. Config-write API endpoints (mirage/api/routers/config.py)

The actual "add a camera" save path the wizard (Task in progress) calls:

- `GET /api/config/detectors` -- lists configured detectors for the wizard's dropdown.
- `POST /api/config/cameras` -- create.
- `PUT /api/config/cameras/{name}` -- update (including rename, by changing the request
  body's `name` field relative to the path segment).
- `DELETE /api/config/cameras/{name}` -- remove.

All three mutating routes go through a flat, wizard-friendly `CameraWriteRequest`
(`name`, `rtsp_url`, `detector`, `track_objects`, `width`/`height`/`fps`, etc.) rather
than the full nested `CameraConfig` tree -- a `_build_camera_config()` helper assembles
the real `CameraConfig`/`FfmpegConfig`/`DetectConfig`/etc. from it, so it goes through
exactly the same Pydantic validation a hand-typed YAML camera block always has (confirmed
directly: `test_created_camera_config_is_actually_valid_mirage_config` round-trips a
created camera back through `MirageConfig.from_db()` and asserts on the real nested
config object, not just the flat API response). Every mutating response includes
`restart_required: true` explicitly, since none of this takes effect in an
already-running `MirageApp` -- config changes only apply on the next `python -m mirage`
restart (ties back to the "hot-reload was explicitly deferred" decision in section 13).

One real FastAPI gotcha hit and fixed immediately: `@router.delete(..., status_code=204)`
on a function annotated `-> None` raised `AssertionError: Status code 204 must not have a
response body` at import time -- FastAPI treats a bare `-> None` return annotation as an
inferred (non-`None`) response model unless `response_model=None` is passed explicitly;
204 responses can't have any body at all, so the two conflict. Fixed by adding
`response_model=None` explicitly to that one route.

Covered by 10 tests in `tests/test_api_config.py` (list detectors; create + immediately
visible via `/api/cameras`; duplicate name -> 409; unknown detector -> 422; missing
required field -> 422; update persists; update unknown -> 404; delete removes it; delete
unknown -> 404; full round-trip through real `MirageConfig.from_db()` validation) plus a
manual smoke test against the real running `python -m mirage.api` process confirming the
same flow works end-to-end outside of `TestClient`.

## 16. ONVIF discovery + stream resolution (mirage/api/routers/onvif.py)

Two endpoints, deliberately split across two different mechanisms rather than one,
because neither one alone covers the whole flow (confirmed against real go2rtc source,
not assumed -- see the research recorded in this session):

- `GET /api/onvif/scan` -- proxies go2rtc's own `GET /api/onvif` (no `src` param), which
  performs the actual WS-Discovery UDP multicast probe and returns whatever ONVIF devices
  answer. No credentials needed (discovery itself doesn't require auth). go2rtc's real
  "no sources" response is a plain-text 404 (`internal/api/api.go`'s `ResponseSources`),
  not JSON -- confirmed directly from source, and handled explicitly (translated to a
  clean empty `[]` rather than surfaced as an error).
- `GET /api/onvif/resolve?ip=&port=&username=&password=` -- calls the ONVIF device's own
  Media service (`GetProfiles` + `GetStreamUri`) directly via `onvif-zeep-async`, **NOT**
  via go2rtc. This split exists because go2rtc's own `?src=onvif://...` resolve mode
  returns an `onvif://...?subtype=N` URL (go2rtc's own internal protocol, meant for
  go2rtc itself to consume) -- not a plain `rtsp://` URL. mirage's own ffmpeg-based
  capture process has no `onvif://` protocol support, only rtsp/http/tcp/etc., so the
  wizard specifically needs the REAL underlying stream URI the camera exposes, which
  requires actually performing the ONVIF Media `GetStreamUri` SOAP operation. If the
  camera's stream URI comes back without embedded credentials (the common case -- ONVIF
  device-service auth and RTSP auth are often configured together on the same
  username/password even though they're technically separate), the resolve endpoint
  splices the provided credentials into the URI so the returned `rtsp_url` is directly
  usable as a `CameraInputConfig.path` with no further editing needed.

**Honest test-coverage gap, noted explicitly rather than glossed over**: the scan
endpoint is verified against a REAL running go2rtc process (confirmed it correctly
returns `[]` for go2rtc's real empty-scan 404 case -- this test setup has no real ONVIF camera to scan for, since none
exists on this development network). The resolve endpoint's real ONVIF success path
(actually calling `GetProfiles`/`GetStreamUri` against a real camera and getting back a
real playable `rtsp://` URL) is **not** verified against real hardware in this test
suite -- only its deterministic failure paths are (unreachable device -> 502, missing
required query params -> 422). The implementation was written by reading the installed
`onvif-zeep-async` package's actual API surface directly (confirmed
`ONVIFCamera.create_media_service()`/`.close()`, and the `GetProfiles`/`GetStreamUri`
operation names/parameter shapes from the library's own WSDL-derived service methods),
not guessed from memory -- but "written correctly against the documented API" and
"verified against a real camera" are different claims, and only the former is true here.
Recommend testing the resolve endpoint against a real ONVIF camera (or a lightweight
ONVIF simulator) before relying on it in a real deployment.

## 17. Add-camera wizard (frontend) + a real zoneless-signals bug found while testing it

New page `pages/add-camera/` (route `/add-camera`, reachable via a dashed "Add camera"
button in the sidebar, styled distinctly from the regular nav items since it's an action
not a destination): entry choice (scan vs. manual RTSP entry) -> for scan, a device list
from `GET /api/onvif/scan` -> credentials step -> `GET /api/onvif/resolve` -> a shared
"camera details" step (name, RTSP URL, detector dropdown populated from
`GET /api/config/detectors`, tracked objects, resolution/fps, recording/retention) ->
`POST /api/config/cameras` -> a "saved, restart required" confirmation screen.

**A real bug, not just a test artifact, found while verifying the save flow end-to-end**:
the "Save camera" button stayed permanently disabled even after filling in every
required field with real simulated keystrokes (via CDP `Input.dispatchKeyEvent`, not a
synthetic DOM event -- ruled out as a test-harness fidelity issue first). Root cause:
this app runs zoneless change detection (`provideZonelessChangeDetection`), and the
initial implementation bound form fields as plain mutable class properties via
`[(ngModel)]="cameraName"` etc., then gated the submit button on a `computed(() =>
this.cameraName.trim().length > 0 && ...)`. `computed()` only recomputes when a *signal*
it read during its last run changes -- reading a plain property inside it creates no
such dependency, so it silently never re-evaluates after its first (empty-string) run,
regardless of how the underlying property later changes. This is invisible in the most
common testing shortcut (manually clicking through the UI while developing, where you
naturally reload the page after most edits) but reproduces 100% of the time from a clean
page load, which is exactly how a real user experiences it once.

**Fix**: every form-bound field (`cameraName`, `resolvedRtspUrl`, `selectedDetector`,
`onvifIp`/`onvifUsername`/`onvifPassword`, `width`/`height`/`fps`/`recordEnabled`/
`retainDays`/`trackObjectsInput`) converted to a real `signal()`, bound via
`[ngModel]="x()" (ngModelChange)="x.set($event)"` instead of `[(ngModel)]="x"` directly
on a plain property -- this gives `computed()` an actual reactive dependency to track.
Verified the fix for real: reproduced the bug first (confirmed `button.disabled === true`
via CDP `Runtime.evaluate` immediately after real-keystroke-filling every required field,
with the DOM inputs showing the correct values), applied the fix, then re-ran the exact
same real-typing-and-submit script and confirmed `disabled === false`, a real
`POST /api/config/cameras` firing, a real row landing in `AppConfig`'s JSON (verified via
direct sqlite query, full valid nested `CameraConfig` structure, not just the flat wizard
input), and the "saved" confirmation screen rendering -- screenshotted via
`Page.captureScreenshot`, not just asserted in code.

This is a good general lesson for this specific codebase going forward (zoneless Angular
+ template-driven `ngModel` forms): a `computed()` gating a submit/save action must be
built on top of signals, not plain properties, even when those plain properties are
correctly bound and correctly updating on-screen -- the disabled-button symptom looks
identical to "the value didn't reach the component" but is actually "the component never
noticed its own value changed."

## 18. Final validation: the whole wizard-to-running-pipeline loop, closed for real (Task 28)

The thing no single test in sections 13-17 exercised together: does a camera added
through the config-write API (the same call the wizard makes) actually make it all the
way to a real, running `CameraCapture`/`CameraTracker` pair producing real recordings,
with zero YAML file involved anywhere in the chain? Verified directly, against a clean
scratch database, in this order:

1. `POST /api/config/cameras` against a real running `mirage.api` process -- confirmed
   the camera appears immediately via `GET /api/cameras`.
2. Stopped that API process entirely, then started the REAL `python -m mirage` pipeline
   against that same database file with **no `--config` flag and no YAML file present
   at all** -- log line `mirage app started: 1 camera(s), 1 detector(s)` confirmed it
   read the wizard-saved camera purely from the DB.
3. Let it run against the real synthetic test stream -- confirmed a real 10-second
   recording landed in the database and on disk, and verified with `ffprobe` that the
   file has a real, valid `duration: 10.000000` (not a placeholder/corrupt file).
4. Separately, exercised the full create -> update -> delete lifecycle directly against
   the config-write endpoints (change `fps`/`track_objects` via `PUT`, confirm the change
   is visible via `GET`, then `DELETE` and confirm the camera list goes back to empty) --
   proving edits and removals round-trip correctly through the same DB-backed config
   path, not just creation.

This is the closing proof for the whole effort (sections 13-18): a user can genuinely add
a camera through the wizard UI, restart `python -m mirage`, and have that camera actually
start recording -- with the entire chain (browser -> API -> DB -> next process
restart -> real ffmpeg capture -> real recording file) verified as a connected whole, not
just as isolated, independently-passing pieces.

All 267 backend tests and the full Angular test/production-build suite pass as of this
final validation pass.

## 19. Two real bugs found and fixed against a real camera, plus the ONVIF resolve gap closed

The user added a real Hikvision DS-2CD1023G0E-I camera on their LAN through the actual
running stack (not a test) -- this surfaced two real bugs that every synthetic test in
this codebase had missed, both traced back to the same root cause: every existing test
fixture uses short camera names ("test_cam", "cam1", ...), and this was the first time a
realistically-long real-world camera name ("hikvision_ds_2cd1023g0e_i") went through the
system.

**Bug 1 -- SHM segment names overflow macOS's POSIX limit.** `python -m mirage` crashed
at startup with `OSError: [Errno 63] File name too long: '/hikvision_ds_2cd1023g0e_i_frame0'`.
Confirmed empirically that macOS caps a POSIX shared-memory name at exactly 30
characters (29-char names beyond the leading `/` succeed, 31 fails) -- far tighter than
Linux's ~255-char limit, and easily overflowed once this module's own `_frame<N>`/`out-`
affixes are added to a longer real name. Fixed in `mirage/util/shm.py` by adding
`_shm_key()`: derives a short (16-char), deterministic, collision-resistant key from any
camera name (sanitized 8-char prefix + 8-char sha1 hex digest) and uses that everywhere
`frame_name()`/`detector_input_shm_name()`/`detector_output_shm_name()` are called,
instead of the raw camera name. Verified for real: created actual SHM ring buffers for
the long camera name post-fix, and added `test_shm_names_stay_under_macos_posix_length_limit_for_long_camera_names`.

**Bug 2 -- a real latent naming mismatch this fix exposed.** Once `_shm_key()` started
producing a genuinely different string than the raw camera name (previously
`detector_input_shm_name()` was just `return camera_name`, an identity function), a
second, pre-existing bug became visible: `mirage/detection/process.py`'s
`detector_process_main` read/closed the input tensor SHM segment using the **raw**
`camera_name` from the detection queue, while `mirage/detection/remote.py`'s
`RemoteObjectDetector` (the writer) correctly called `detector_input_shm_name(camera_name)`.
These two sides had only ever "agreed" by coincidence, because the old
`detector_input_shm_name()` was a no-op. The moment it became a real transform, the
reader and writer used different segment names, and detection silently stopped for that
camera (`WARNING: detector general: no input SHM for camera hikvision_ds_2cd1023g0e_i`
-- frames were captured and recorded fine, only detection was silently broken). Fixed by
applying `detector_input_shm_name(camera_name)` at the read/close call sites in
`process.py` too. Added `test_detector_process_serves_a_camera_with_a_long_real_world_name`
to `tests/test_detector_process_integration.py` -- a real DetectorProcess + real
RemoteObjectDetector + real yolov8n.onnx inference against a long camera name,
specifically chosen to catch this exact class of write/read-mismatch bug if it recurs
(a short name wouldn't have caught it, as this whole incident demonstrated).

**The ONVIF resolve gap (section 16) is now closed for real.** Ran
`GET /api/onvif/resolve` against the user's actual Hikvision camera on their real LAN:
returned a genuine `rtsp://admin:***@192.168.1.99:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1`
-- the camera's real ONVIF-reported stream path (notably different from, and more
"correct" than, the path the user had found manually by guessing Hikvision's common
default, `/h264/ch1/main/av_stream` -- ONVIF's GetStreamUri returned the camera's own
actual preferred profile path instead), plus the real device name
("HIKVISION DS-2CD1023G0E-I") and profile name ("mainStream"), and correctly spliced in
the provided credentials since the camera's own URI came back without them. This is now
verified against real hardware, not just written-correctly-per-the-library's-API as
section 16 originally had to caveat.

The `/api/onvif/scan` test previously asserted the network has zero ONVIF devices
(`== []`) -- true when originally written, but the user's real camera being
scan-discoverable afterwards is a correctness proof, not a regression. Updated
`test_scan_against_real_go2rtc_with_no_devices_returns_empty_list` to
`test_scan_against_real_go2rtc_returns_a_well_shaped_list`, asserting response shape
only, not device count, so the test doesn't depend on whatever happens to be reachable
on whichever machine runs it.

## 20. Per-camera RTSP transport (TCP/UDP) + a "Manage cameras" page for editing existing cameras

Two follow-on features requested directly from the real Hikvision-camera incident
(section 19): (a) make the TCP-vs-UDP choice a per-camera config option instead of a
hardcoded default, so a camera with the same TCP instability that camera showed can be
switched to UDP without code changes, and (b) a way to edit (or delete) a camera's config
after it's already been added -- the wizard only ever had a create path before this.

**Schema**: new `RtspTransport` enum (`tcp` | `udp`) and a `rtsp_transport` field on
`CameraInputConfig` (`mirage/config/schema.py`), defaulting to `tcp`. Threaded through:
- `mirage/capture/ffmpeg_presets.py`'s `build_ffmpeg_cmd_for_input` now passes
  `ffmpeg_input.rtsp_transport.value` into `build_input_args`'s existing (previously
  hardcoded-default) `transport` param, so mirage's own ffmpeg capture actually uses it.
- `mirage/go2rtc/config.py`'s `_detect_role_source` appends go2rtc's own
  `#transport=udp` URL-fragment syntax when a camera's input is `rtsp://` AND set to
  udp. This matters because go2rtc holds an entirely SEPARATE RTSP connection to the
  camera (for live view/snapshots) from mirage's own ffmpeg capture -- without this,
  switching mirage's own capture to UDP would leave go2rtc's connection still fighting
  the same TCP instability. The exact fragment syntax was confirmed against go2rtc's
  actual source (`internal/rtsp/rtsp.go` parses the URL fragment as query params and
  sets `conn.Transport` from a `transport=` key; `pkg/rtsp/client.go`'s `Dial()`
  switches SETUP to UDP mode when that's `"udp"`), not assumed from general RTSP
  knowledge.

**API**: `CameraWriteRequest` (mirage/api/routers/config.py) gained `rtsp_transport:
RtspTransport = RtspTransport.tcp`, applied in `_build_camera_config`. A NEW response
model, `CameraConfigOut`, was added specifically for this router's own use -- it
includes the raw `rtsp_url` (which embeds credentials) plus `rtsp_transport`,
`retain_days`, and the label lists, needed to pre-fill an edit form with a camera's real
existing settings. This is deliberately NOT merged into the existing `CameraOut` used by
`GET /api/cameras` (mirage/api/routers/cameras.py) -- that endpoint is consumed more
broadly (e.g. the Live page's camera grid) and must not leak RTSP credentials to every
reader. Two new endpoints expose `CameraConfigOut`: `GET /api/config/cameras` (list, for
the Manage Cameras page) and `GET /api/config/cameras/{name}` (single, for the edit
form's pre-fill) -- covered by
`test_general_cameras_endpoint_does_not_leak_rtsp_credentials`, which asserts the
credential string literally does not appear anywhere in `GET /api/cameras`'s response.

**Frontend**: 
- The add-camera wizard's details step gained a TCP/UDP toggle (`.transport-toggle`,
  two buttons, TCP labeled "Recommended", UDP labeled "If TCP keeps disconnecting").
- `AddCameraPage` now supports an edit mode: reached via `/add-camera?camera=<name>`,
  it skips the entry/scan/credentials steps entirely, loads the camera's current config
  via the new `GET /api/config/cameras/{name}`, pre-fills every field (including the
  transport toggle), and calls `updateCamera()` (PUT) instead of `createCamera()`
  (POST) on submit. Copy adapts throughout ("Edit camera" / "Save changes" / "apply
  these changes" instead of "Add camera" / "Save camera" / "start capturing this
  camera").
- New page `pages/manage-cameras/` (route `/cameras`, "Cameras" in the sidebar nav)
  lists every configured camera with its detector/resolution/fps/transport/recording
  status, and edit (pencil) / delete (trash) icon actions per row. Delete uses an
  inline confirm/cancel (not a browser `confirm()` dialog) before actually calling
  `DELETE /api/config/cameras/{name}`, and shows the same "restart required" banner
  pattern used elsewhere in this config-write surface.

**Verified for real** (not just unit tests): seeded a real camera via the API, loaded
the Manage Cameras page in a real headless-Chrome + CDP session, screenshotted it,
clicked the edit icon, confirmed the wizard genuinely pre-filled from that camera's
saved config (including the real RTSP URL with embedded credentials), clicked the UDP
toggle, submitted, and confirmed via direct API query that `rtsp_transport: "udp"`
actually persisted to the database. Then repeated for delete: clicked the trash icon,
screenshotted the inline "Delete front_door? Confirm/Cancel" state, clicked Confirm,
confirmed the "Camera removed. Restart..." banner rendered and the camera list actually
went back to empty via direct API query.

All 280 backend tests and the full Angular test/production-build suite pass as of this
pass.

## 21. Real bug: false_positive status never actually reached the database (Events page was permanently empty)

Investigated live (against the real Hikvision camera, walking in front of it and
tracing real per-frame scores with temporary debug logging -- reverted after) why the
Events page showed nothing despite 84 real, well-detected people in the database
(`top_score` up to 0.92). The false-positive median-score gate itself (section 6.1,
`ObjectLifecycle`) was working correctly -- a live trace showed `computed_score`
genuinely reaching 0.864, well clear of the 0.7 threshold. The bug was architectural,
and affected BOTH `EventProcessor` (events) and `ReviewSegmentMaintainer` (review
segments), for the same underlying reason:

`ObjectLifecycle` instances (which own `is_false_positive`) are created and updated
entirely inside `CameraOrchestrator`, which runs in a separate per-camera `CameraTracker`
OS process (`mirage/tracking/camera_tracker.py`). Only the plain, already-cross-process
`TrackedObjectState`/`FrameResult` dataclasses are ever sent back to the main process via
a multiprocessing queue -- `ObjectLifecycle` itself never crosses that boundary.
`EventProcessor.process`/`ReviewSegmentMaintainer.process` (both running in the MAIN
process) took a `get_lifecycle` callback for this exact reason, but `MirageApp`'s actual
wiring (`_get_lifecycle_stub`) was a **permanent** stub that always returned `None` --
not a temporary placeholder, a real, by-design dead end, per its own docstring
("ObjectLifecycle isn't picklable/shared across the process boundary... Fall back to no
lifecycle info here"). Concretely, this meant:

- **Events**: `false_positive` was only ever written once, at `Event.create()` time
  (always `True`) -- `_on_update`'s `if lifecycle is not None: update_fields[...]`
  branch could never fire, since `lifecycle` was always `None`. Every event stayed
  `false_positive=True` forever, regardless of how strong its real detections were.
- **Review**: `qualifies_for_review`'s `if lifecycle is not None and
  lifecycle.is_false_positive: return False` could also never fire -- the opposite
  failure mode, silently letting every tracked object (including genuine false
  positives) count toward review segments.

**Fix**: added `is_false_positive: bool = True` directly to `TrackedObjectState`
(`mirage/tracking/tracker.py`) -- a plain field on the dataclass that ALREADY crosses
the process boundary safely, unlike `ObjectLifecycle` itself. `CameraOrchestrator.
_update_lifecycles` (`mirage/tracking/orchestration.py`) now writes `state.
is_false_positive = lifecycle.is_false_positive` right after computing it, each frame --
since `ObjectTracker._states` holds mutable references and `_update_lifecycles` runs on
the same `tracked` dict `process_frame` returns, this correctly reaches the caller with
no extra wiring. `EventProcessor`/`ReviewSegmentMaintainer` (`mirage/events/processor.py`,
`mirage/events/review.py`) had their `get_lifecycle` parameter removed entirely and now
read `state.is_false_positive` directly -- no cross-process lookup needed at all anymore.
`MirageApp._get_lifecycle_stub` and both call sites in `mirage/app.py` were deleted.

**A second, related bug found and fixed in the same pass**: even with the lifecycle
status now correctly flowing through, `EventProcessor._on_end` (called when a tracked
object disappears) only ever wrote `end_time` to the database -- never `false_positive`
or `top_score`. Since `_on_update`'s own DB write is throttled (at most once per 5s,
see `_should_update_db`), a `false_positive -> true_positive` flip that happens on a
throttled-out frame, immediately followed by the object disappearing before the next
scheduled write, would be silently lost -- the row stays at whatever it last wrote
(often still the initial `Event.create()` value). Fixed by having `_on_end` write the
final `false_positive`/`top_score` (tracked in `self._active[obj_id]`, kept current on
every `_on_update` call regardless of whether that call's DB write was throttled) before
closing the row.

**Verified for real, end to end**: reproduced the original bug's live symptom first
(temporary debug logging in `lifecycle.py`, confirmed a real tracked person's
`computed_score` genuinely reached 0.864 in-process while its DB row stayed
`false_positive=1` forever -- proving the gate itself wasn't the problem), applied both
fixes, restarted the real pipeline against the real camera, asked the user to walk in
front of it again, and confirmed via `GET /api/events` that a new event landed with
`"false_positive": false, "top_score": 0.924` and is now correctly returned by the
default (false-positives-excluded) query the Events page uses. Also confirmed review
segments are now being created with the correct `alert` severity for `person`.

Updated `tests/test_event_processor.py` and `tests/test_review.py` for the new
`get_lifecycle`-free signature, and added three new regression tests:
`test_false_positive_flip_on_a_throttled_frame_still_persists_when_object_ends`
(reproduces the exact throttle-then-disappear race), `test_false_positive_objects_do_not_start_a_segment`
(confirms review segments now actually exclude false positives, previously dead code),
and `test_orchestration.py`'s real end-to-end test now also asserts at least one tracked
object gets promoted out of false-positive status by its final frame (using the real
yolov8n.onnx model against a real repeated photo, not a mock). All 282 backend tests
pass.
