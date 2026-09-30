# Graph Report - miragev3  (2026-09-23)

## Corpus Check
- 227 files · ~189,358 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 36 file(s) not represented in the graph (top: .scss 18, (none) 9, .onnx 2)

## Summary
- 2336 nodes · 6256 edges · 128 communities (94 shown, 34 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 493 edges (avg confidence: 0.92)
- Token cost: 483,659 input · 0 output

## Community Hubs (Navigation)
- Detection API & Execution Providers
- Recording Maintainer & Retention
- Camera Config & Orchestration Tests
- API App Factory & DB
- Config API Router
- API App & Routers Bootstrap
- Detector Management UI
- Review Segment Maintainer
- Angular Build Config
- Live View Proxy API
- Event Processor
- Frontend Core Services
- FFmpeg Command Presets
- Mirage Config Schema
- Object Tracker Distance Function
- Thumbnail Capture Utility
- Live View Players
- Region Selection
- Config API Tests
- Object Lifecycle Filtering
- Add-Camera Models
- Review & Logs Services
- API Entrypoint & Path Constants
- Frontend Package Config
- Review Severity Classification
- Review API Router
- ZMQ IPC Pub/Sub
- Capture Process Utils & Watchdog
- Shared-Memory Frame Ring
- API Read Tests
- Region Reduce & NMS
- go2rtc Process Supervisor
- Rules Engine (Alerts)
- Service Watchdog
- Camera Capture & Frame Ring
- Motion Detector
- Detector Process & Labelmap
- Species Result Dispatcher
- PTZ Client
- Recording Maintainer Loop
- Docker Compose Deployment
- Capture Read Loop & Rate Counters
- Pipeline Supervisor
- Species Classifier Registry
- App Bootstrap & Activity Log
- MirageApp Orchestrator
- Implementation Notes & DB Config Migration
- PTZ Config & Presets
- Tracker Tuning Config
- Recordings Page
- Frontend Dependencies
- Events Page
- System Status API
- Datetime Utilities
- Shared-Memory Frame Manager
- Species Classifier Abstraction
- Notify Bus (SSE Source)
- go2rtc Config Generation
- Detector E2E Tests
- Camera Tracker Snapshot Encoding
- Architecture Diagram (Docs)
- Open-Vocabulary Detection & Known Issues
- go2rtc Binary Download
- Detector Process Entrypoint
- Event Processor Internals
- PTZ API Tests
- Detector Scaling & Optimization Docs
- Bug-Fix & Process Audit Docs
- Dwell & Crowd Rules
- Tensor Construction
- Config Schema & Store
- Manage Cameras Page
- Rate & Event Counters
- Tensor Construction Tests
- PTZ Poller
- SpeciesNet Worker Process
- System API Tests
- How-To-Run & Sample Config
- System Design Doc & Frontend Architecture
- Video Lightbox Component
- Deployment & Dependency Docs
- Logs Page
- App Shell & System Status
- Notifications Tests
- SpeciesNet Classifier Client
- Integration Test Fixtures
- Design Decisions (SDD)
- Supervisor Pipeline Lifecycle
- Process-Group Test Fixtures
- Video Server Test Fixtures
- Paged List Utility
- Lightbox Component
- Events-Per-Second Counter
- PTZ Client & Poller
- Tracking Frame Orchestration
- PTZ Poller Retry Test
- Bus Test Fixture Image
- Camera/Detector Config Validators
- Camera Tracker Process Entrypoint
- Detector Labelmap Files
- Test Stream Script
- Snapshot Fallback Fetcher Test
- Frontend Environment Files
- Zidane Test Fixture Image
- Dangling Event Cleanup
- Thumbnail Fetcher Internals
- Thumbnail Fetcher Internals
- Docker Image Export Script
- Mirage CLI Entrypoint

## God Nodes (most connected - your core abstractions)
1. `CameraConfig` - 115 edges
2. `ModelConfig` - 76 edges
3. `CameraInputConfig` - 72 edges
4. `FfmpegConfig` - 70 edges
5. `Event` - 66 edges
6. `MirageConfig` - 65 edges
7. `ApiService` - 58 edges
8. `SharedMemoryFrameManager` - 58 edges
9. `DetectConfig` - 54 edges
10. `RecordConfig` - 53 edges

## Surprising Connections (you probably didn't know these)
- `test_event_crud_with_json_data_field()` --uses--> `Event`  [INFERRED]
  tests/test_db.py → mirage/db/models.py
- `test_event_end_time_nullable()` --uses--> `Event`  [INFERRED]
  tests/test_db.py → mirage/db/models.py
- `test_recordings_unique_path_constraint()` --uses--> `Recordings`  [INFERRED]
  tests/test_db.py → mirage/db/models.py
- `test_review_segment_severity_and_data()` --uses--> `ReviewSegment`  [INFERRED]
  tests/test_db.py → mirage/db/models.py
- `Single DetectorProcess-per-config scaling ceiling` --semantically_similar_to--> `Open-vocab match draining does synchronous disk+DB I/O in shared loop`  [INFERRED] [semantically similar]
  DETECTOR_SCALING.md → OPTIMIZATION_OPPORTUNITIES.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Live-view WebRTC/MSE delivery chain across docs, compose config, and frontend tile** — live_view_and_capture_fixes_doc, implementation_notes_go2rtc_integration, docker_compose_webrtc_udp_fix, frontend_src_app_pages_live_camera_tile_camera_tile [INFERRED 0.75]
- **Gated open-vocabulary enrichment architecture (OWLv2, gating, benchmark, pending-request leak)** — todo_fix_list_dual_detector_openvocab_feature, system_design_openvocab_subsystem, todo_fix_list_owlv2_apple_silicon_benchmark, optimization_opportunities_openvocab_pending_no_timeout [INFERRED 0.80]
- **DB-backed config plus manual-restart architecture decision (no hot-reload)** — implementation_notes_config_db_migration, todo_fix_list_hot_reload_reverted, todo_fix_list_apply_changes_button, how_to_run_db_backed_config [INFERRED 0.85]
- **Video clip playback via shared video-lightbox component** — frontend_src_app_pages_recordings_recordings_page, frontend_src_app_pages_review_review_page, frontend_src_app_shared_video_lightbox_video_lightbox [INFERRED 0.85]
- **Filterable list pages sharing the filter-select control** — frontend_src_app_pages_recordings_recordings_page, frontend_src_app_pages_review_review_page, frontend_src_app_shared_filter_select_filter_select [INFERRED 0.85]
- **Detector labelmap taxonomy family** — models_coco_labelmap, models_megadetector_labelmap, models_wildlife_labelmap, models_zilodetector_labelmap [INFERRED 0.80]

## Communities (128 total, 34 thin omitted)

### Community 0 - "Detection API & Execution Providers"
Cohesion: 0.06
Nodes (54): ExecutionProvider, Which onnxruntime execution provider to run this detector's InferenceSession on…, mirage_detection, DetectionApi, empty_detection_output(), ABC, ndarray, Detector abstraction -- one interface, multiple ML backends. Spec reference:… (+46 more)

### Community 1 - "Recording Maintainer & Retention"
Cohesion: 0.07
Nodes (62): RetainMode, default_activity_stats_provider(), datetime, RecordingMaintainer: moves validated ffmpeg cache segments to permanent storage…, Section 8.1 point 4: if more than MAX_SEGMENTS_IN_CACHE unprocessed segments…, continuous_expire_date(), motion_expire_date(), overlaps() (+54 more)

### Community 2 - "Camera Config & Orchestration Tests"
Cohesion: 0.10
Nodes (51): DetectConfig, InputDType, ObjectsConfig, PixelFormat, RecordConfig, RetainConfig, CameraOrchestrator, Owns one camera's motion detector, object tracker, and per-object lifecycle… (+43 more)

### Community 3 - "API App Factory & DB"
Cohesion: 0.06
Nodes (50): fastapi_testclient, create_app(), lifespan(), `config`, if given, is used as a FIXED override for the lifetime of this app --…, close_database(), init_database(), main(), pytest (+42 more)

### Community 4 - "Config API Router"
Cohesion: 0.10
Nodes (48): delete, _build_camera_config(), _camera_config_out(), _camera_out(), CameraConfigOut, CameraWriteRequest, ConfigMutationResponse, create_camera() (+40 more)

### Community 5 - "API App & Routers Bootstrap"
Cohesion: 0.08
Nodes (37): asyncio, contextlib, FastAPI, fastapi_middleware_cors, fastapi_responses, FastAPI app factory for the mirage read API. Runs as its own process, separate…, mirage_api_routers, get_camera() (+29 more)

### Community 6 - "Detector Management UI"
Cohesion: 0.07
Nodes (13): Camera, Detector, DetectorWriteRequest, ExecutionProvider, SystemCapabilities, ApiService, Injectable, ConfigPage (+5 more)

### Community 7 - "Review Segment Maintainer"
Cohesion: 0.16
Nodes (41): Spec section 9/10.1: human-facing activity-aggregation window., ReviewSegment, ReviewSegmentMaintainer, _active_state(), _camera(), Regression test for a REAL bug the user caught live via a screenshot: a…, A real object's label (e.g. "person") still accumulates in the historical…, rule outranks alert (see mirage.events.review._SEVERITY_RANK) -- an already-… (+33 more)

### Community 8 - "Angular Build Config"
Cohesion: 0.05
Nodes (41): build, serve, test, builder, configurations, defaultConfiguration, options, cli (+33 more)

### Community 9 - "Live View Proxy API"
Cohesion: 0.08
Nodes (37): httpx, get_snapshot(), _go2rtc_api_base(), live_ws_proxy(), get, Request, Response, Live-view backend surface: proxies to go2rtc so the Angular frontend only ever… (+29 more)

### Community 10 - "Event Processor"
Cohesion: 0.16
Nodes (38): Event, Spec section 10.1: one tracked object's detection lifecycle., EventProcessor, Section 6.3: diffs the tracker's current confirmed object-id set against the…, _camera(), Real bug the user found: a snapshot only ever showed a box for the ONE object…, An object still mid initialization_delay (is_false_positive still True) must…, Real bug caught live: Event.update(data=...) REPLACES the whole JSON field, so… (+30 more)

### Community 11 - "Frontend Core Services"
Cohesion: 0.15
Nodes (19): visibleInterval(), SseMessage, SPECIES_ENRICHABLE_LABELS, TIME_RANGE_OPTIONS, PlayerMode, CATEGORY_LABELS, PROVIDER_LABELS, SEVERITY_OPTIONS (+11 more)

### Community 12 - "FFmpeg Command Presets"
Cohesion: 0.10
Nodes (32): functools, build_all_ffmpeg_cmds(), build_detect_scale_args(), build_ffmpeg_cmd_for_input(), build_input_args(), build_record_output_args(), _libavformat_major(), ffmpeg argv construction. Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md… (+24 more)

### Community 13 - "Mirage Config Schema"
Cohesion: 0.15
Nodes (37): CameraInputConfig, DetectorInstanceConfig, FfmpegConfig, MirageConfig, ModelConfig, Section 3.3.1. One instance per configured detector (see…, NVR_EXTENSION_MULTI_MODEL_ROUTING.md section 1., The config a fresh install starts from: the same 'general' ONNX YOLOv8n… (+29 more)

### Community 14 - "Object Tracker Distance Function"
Cohesion: 0.09
Nodes (32): collections, box_to_points(), points_to_box(), ndarray, Custom distance function for the object tracker. Spec reference:…, Both are [[x1,y1],[x2,y2]] point pairs (top-left, bottom-right corners). Uses…, Converts an (x1,y1,x2,y2) box into norfair's expected Nx2 points array., tracking_distance() (+24 more)

### Community 15 - "Thumbnail Capture Utility"
Cohesion: 0.10
Nodes (35): capture_thumbnail(), crop_jpeg_to_box(), draw_boxes_on_jpeg_bytes(), _fetch_with_retry(), ThumbnailFetcher, Shared "fetch a JPEG snapshot and write it to disk" helper, used by both…, Writes `frame_jpeg` to disk completely unmodified -- the CLEAN snapshot file…, Best-effort: any failure (fetcher raising, empty response, disk write failure)… (+27 more)

### Community 16 - "Live View Players"
Cohesion: 0.09
Nodes (8): CameraTile, Component, ViewChild, CANDIDATE_CODECS, MsePlayer, supportedCodecsValue(), ICE_SERVERS, WebRtcPlayer

### Community 17 - "Region Selection"
Cohesion: 0.11
Nodes (35): box_center(), box_inside_extent(), box_overlaps_extent(), build_regions(), _clamp_region_preserving_size(), cluster_boxes(), cluster_extent(), denormalize_box() (+27 more)

### Community 18 - "Config API Tests"
Cohesion: 0.06
Nodes (3): client(), fixture, Tests for the config-write endpoints (mirage/api/routers/config.py) -- the add-…

### Community 19 - "Object Lifecycle Filtering"
Cohesion: 0.12
Nodes (25): dataclasses, ObjectFilterConfig, is_object_filtered(), ObjectLifecycle, Object lifecycle / event state machine: true-positive vs false-positive…, Tracks the score history and false-positive state for one tracked object across…, Pass None for a frame where the object wasn't actually re-detected (only…, Section 6.2: applied at raw-detection time. Returns True if the detection… (+17 more)

### Community 20 - "Add-Camera Models"
Cohesion: 0.08
Nodes (12): CameraConfigDetail, CameraWriteRequest, ConfigMutationResponse, OnvifDevice, OnvifResolveResult, ReviewSeverity, RtspTransport, SpeciesStatus (+4 more)

### Community 21 - "Review & Logs Services"
Cohesion: 0.08
Nodes (9): EventListParams, LogEntry, ReviewListParams, ReviewSegment, toHttpParams(), SseService, Injectable, ReviewPage (+1 more)

### Community 22 - "API Entrypoint & Path Constants"
Cohesion: 0.08
Nodes (25): argparse, main(), CLI entrypoint: `python -m mirage.api --db-path path/to/mirage.db`. Starts the…, _env_path(), Path, Global path/priority constants. Spec reference:…, Database connection setup. Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md…, Database schema. Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section… (+17 more)

### Community 23 - "Frontend Package Config"
Cohesion: 0.07
Nodes (28): name, packageManager, private, scripts, build, ng, start, test (+20 more)

### Community 24 - "Review Severity Classification"
Cohesion: 0.09
Nodes (24): classify_severity(), _normalized_boxes_at_thumb_time(), PendingReviewSegment, Enum, str, qualifies_for_review(), ReviewSegmentMaintainer: aggregates tracked-object activity into human-…, Section 9: an object is "alert"-worthy if its label is in the configured alert-… (+16 more)

### Community 25 - "Review API Router"
Cohesion: 0.12
Nodes (28): get_review_segment(), get_review_segment_clip(), get_review_thumbnail(), list_review_segments(), FileResponse, get, Request, Stitches together every permanent recording clip overlapping this review… (+20 more)

### Community 26 - "ZMQ IPC Pub/Sub"
Cohesion: 0.10
Nodes (21): Context, Publisher, ZeroMQ pub/sub wrapper classes. Spec reference:…, Non-blocking: discards any already-pending messages (defensive cleanup before a…, Blocks up to `timeout` seconds for a message matching this subscriber's topic…, Subscriber, proxy_addrs(), fixture (+13 more)

### Community 27 - "Capture Process Utils & Watchdog"
Cohesion: 0.11
Nodes (14): logging, Popen, ffmpeg subprocess launch/stop helpers. Spec reference:…, Section 1.7 graceful-stop helper. If drain_output is False, skip trying to read…, start_ffmpeg(), stop_ffmpeg(), CameraWatchdog, CameraWatchdog: supervises the ffmpeg subprocess(es) for one camera. Spec… (+6 more)

### Community 28 - "Shared-Memory Frame Ring"
Cohesion: 0.10
Nodes (24): calculate_shm_ring_depth(), Release outstanding views WITHOUT closing/unlinking the segments themselves --…, Detach only. Does NOT unlink -- the segment persists for the next ring lap., Detach AND unlink -- destroys the underlying shared memory object. Only call…, Exact YUV420 planar frame byte size: Y plane (w*h) + U+V planes combined…, Spec section 1.5.3. Returns (ring_depth, recommended_min_shm_mb).…, Each process instantiates its own; the dict below is a process-local cache of…, SharedMemoryFrameManager (+16 more)

### Community 29 - "API Read Tests"
Cohesion: 0.11
Nodes (26): Naive datetime representing `ts` (a time.time()-style Unix timestamp) in UTC --…, utc_from_timestamp(), client(), fixture, Integration tests for the FastAPI read API (mirage/api). Uses FastAPI's…, Boxes are rendered on demand at request time from Event.data["snapshot_boxes"]…, _real_jpeg(), test_get_event_roundtrip() (+18 more)

### Community 30 - "Region Reduce & NMS"
Cohesion: 0.16
Nodes (28): box_area(), get_consolidated_object_detections(), intersection_area(), is_clipped_at_region_edge(), Box, Reduce/consolidate detections across all regions scanned in a frame. Spec…, The full spec section 4 reduce/consolidate pipeline: per-label NMS with edge-…, True if `box` touches a region boundary that is NOT also a frame edge (i.e. the… (+20 more)

### Community 31 - "go2rtc Process Supervisor"
Cohesion: 0.11
Nodes (19): Go2rtcProcess, Go2rtcProcess: launches and supervises the go2rtc binary as a subprocess.…, os, signal, subprocess, client(), _kill_process_group(), _port_open() (+11 more)

### Community 32 - "Rules Engine (Alerts)"
Cohesion: 0.20
Nodes (27): One instance per MirageApp (not per camera) -- all per-camera state is keyed by…, RulesEngine, _camera(), _person(), Tests for mirage/tracking/rules.py -- RulesEngine, the crowd-count and dwell-…, A stable object id across frames is what lets ReviewSegmentMaintainer treat…, Explicit design choice: dwell time is NOT gated on state.stationary -- a person…, Regression test for a REAL production bug: ObjectTracker.update()… (+19 more)

### Community 33 - "Service Watchdog"
Cohesion: 0.13
Nodes (16): MonitoredProcess, ServiceWatchdog: restarts crashed service processes (detector, recording,…, Signals the watchdog thread to stop and blocks until it has actually exited.…, ServiceWatchdog, FakeProcess, crash_immediately=True models a process that dies right after start() is called…, test_multiple_registered_services_checked_independently(), test_stop_blocks_until_thread_exits_and_prevents_late_restart() (+8 more)

### Community 34 - "Camera Capture & Frame Ring"
Cohesion: 0.16
Nodes (23): atexit, hashlib, CameraCapture, CameraCapture: the per-camera OS process that owns ffmpeg subprocess(es). Spec…, preallocate_ring(), Shared-memory frame ring buffer, safe for use across independent processes.…, Numpy shape for a YUV420 planar buffer, laid out as one 2D uint8 array., Pre-create all N named SHM segments for a camera before spawning its… (+15 more)

### Community 35 - "Motion Detector"
Cohesion: 0.18
Nodes (21): MotionConfig, MotionDetector, ndarray, rasterize_mask_polygons(), Adaptive background-average motion detector. Spec reference:…, Debounced background accumulation: motion must persist >=10 consecutive frames…, Converts normalized [0,1] polygon coordinates into a uint8 mask image of shape…, luma_frame: the Y (luma) plane only, shape (full_height, full_width), uint8. (+13 more)

### Community 36 - "Detector Process & Labelmap"
Cohesion: 0.11
Nodes (19): load_labels(), Labelmap loading. Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section…, Supports both index-prefixed ("0 person") and plain newline-delimited (implicit…, DetectorProcess, ndarray, RemoteObjectDetector: the per-camera client that talks to the shared detector…, Detach only -- does NOT unlink the SHM segments. In the real system these…, Fully destroys the SHM segments. Only call this if THIS instance is also the… (+11 more)

### Community 37 - "Species Result Dispatcher"
Cohesion: 0.15
Nodes (20): Fire-and-forget -- called synchronously from EventProcessor._on_start, so this…, Non-blocking drain of every result currently available, applying each via a…, SpeciesDispatcher, _process_request(), SpeciesRequest, SpeciesResult, _FakeQueue, _make_event() (+12 more)

### Community 38 - "PTZ Client"
Cohesion: 0.10
Nodes (13): PtzClient, One instance per PTZ-enabled camera. Must call connect() before any other…, _Attrs, Tests for mirage/ptz/client.py (PtzClient). Following the same philosophy as…, A bare attribute holder -- stands in for the zeep-generated response objects…, test_connect_against_unreachable_device_raises(), test_continuous_move_passes_velocity_through(), test_get_presets_maps_tokens_and_names() (+5 more)

### Community 39 - "Recording Maintainer Loop"
Cohesion: 0.20
Nodes (20): ActivityStatsProvider, Spec section 8.1/10.1: one permanently-stored recording segment., Recordings, Runs maintainer.run_once() repeatedly, pacing so each cycle starts roughly…, RecordingMaintainer, run_recording_maintainer_loop(), _camera(), _make_cache_segment() (+12 more)

### Community 40 - "Docker Compose Deployment"
Cohesion: 0.11
Nodes (24): api service (prebuilt image), Client-distribution docker-compose (prebuilt images), frontend service (prebuilt image), pipeline service (prebuilt image), Multi-container mirage stack (docker-compose.yml), --go2rtc-source CAM1_RTSP_URL override (go2rtc owns physical camera connection), pipeline service (build from Dockerfile.pipeline, CUDA image), 8555/tcp + 8555/udp WebRTC port publish fix (+16 more)

### Community 41 - "Capture Read Loop & Rate Counters"
Cohesion: 0.17
Nodes (13): capture_frames(), The capture read loop: reads raw YUV420 frames off ffmpeg's stdout into the…, Runs in a dedicated thread inside the per-camera capture process. Reads exactly…, EventsPerSecond, Rolling-window events-per-second counter. Used for camera_fps / skipped_fps /…, frame_name(), Naming convention for ring buffer slots (spec section 1.5.2):…, FakeProcess (+5 more)

### Community 42 - "Pipeline Supervisor"
Cohesion: 0.15
Nodes (19): main(), PipelineSupervisor, Supervises the `python -m mirage` pipeline subprocess so the frontend's "Apply…, Only callable from the process's real main thread (a hard Python/OS constraint…, `pipeline_command`, if given, REPLACES the default `[sys.executable, "-m",…, sys, crashing_pipeline_command(), fake_pipeline_command() (+11 more)

### Community 43 - "Species Classifier Registry"
Cohesion: 0.16
Nodes (20): importlib, Mirage V3: one shared species-classification worker process for every camera's…, SpeciesClassifierConfig, mirage_species, available_backends(), create_classifier(), Species classifier plugin registry -- mirrors mirage/detection/registry.py…, Every registered `device` key a SpeciesClassifierConfig can validly use -- e.g.… (+12 more)

### Community 44 - "App Bootstrap & Activity Log"
Cohesion: 0.13
Nodes (14): Main application bootstrap. Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md…, ActivityLogWriter, _log_path(), LogEvent, Path, Structured activity log, shared between the `python -m mirage` pipeline process…, Lives in the main process's result-consumer thread (see mirage.app.MirageApp)…, SpeciesDispatcher: lives in the main process's result-consumer loop… (+6 more)

### Community 45 - "MirageApp Orchestrator"
Cohesion: 0.13
Nodes (8): MirageApp, factory(), Only spawns SpeciesProcess if config.species_classifier.enabled -- see…, Real thumbnail source for ReviewSegmentMaintainer: a JPEG snapshot from…, Non-blocking drain of every NotifyEvent any pipeline process/thread has put…, Non-blocking drain of every LogEvent any pipeline process/thread has put onto…, ipc_addr(), Build an ipc:// zmq address rooted at `cache_dir` (default: the module-level…

### Community 46 - "Implementation Notes & DB Config Migration"
Cohesion: 0.16
Nodes (22): Add-camera wizard zoneless-signals disabled-button bug fix, Angular 21 frontend scaffold (zoneless, standalone), mirage/mirage.api CLIs switched to DB-backed config, Config moves from hand-edited YAML to DB (AppConfig singleton), Config-write API endpoints (add-camera wizard save path), Implementation Notes / Deviations from the Spec (doc), go2rtc integration for live view, Live-view proxy endpoints (snapshot + WS relay) (+14 more)

### Community 47 - "PTZ Config & Presets"
Cohesion: 0.21
Nodes (15): PtzConfig, PtzPresetConfig, One ONVIF PTZ preset this camera can be commanded to -- token is the ONVIF…, Mirage V3 PTZ hybrid scheduling + ONVIF PTZ control (mirage/ptz/,…, PtzStatus, _FakePtzClient, Tests for mirage/ptz/poller.py (PtzPoller) -- the background thread that lives…, Records calls made to it and returns a scripted sequence of statuses --… (+7 more)

### Community 48 - "Tracker Tuning Config"
Cohesion: 0.14
Nodes (15): max_disappeared(), min_initialized(), Per-object-type tracker tuning table. Spec reference:…, stationary_threshold_frames(), TrackerTuning, tuning_for_label(), norfair_distance(), Adapter matching norfair's expected distance_function(Detection, TrackedObject)… (+7 more)

### Community 49 - "Recordings Page"
Cohesion: 0.13
Nodes (5): Recording, RecordingListParams, RecordingGroup, RecordingsPage, Component

### Community 50 - "Frontend Dependencies"
Cohesion: 0.10
Nodes (19): dependencies, @angular/common, @angular/compiler, @angular/core, @angular/forms, @angular/platform-browser, @angular/router, @fontsource/archivo (+11 more)

### Community 51 - "Events Page"
Cohesion: 0.15
Nodes (3): Event, EventsPage, Component

### Community 52 - "System Status API"
Cohesion: 0.18
Nodes (18): get_capabilities(), get_logs(), get_status(), LogEntryOut, get, Path, post, Request (+10 more)

### Community 53 - "Datetime Utilities"
Cohesion: 0.17
Nodes (16): datetime, as_naive_utc(), datetime, Datetime helpers for values that will be written to the database. IMPORTANT:…, Naive datetime representing the current UTC time -- safe to store in / compare…, Normalizes any datetime (naive-assumed-UTC, or timezone-aware) down to a naive…, utcnow(), Regression test for a real bug: peewee's DateTimeField.formats has no timezone-… (+8 more)

### Community 54 - "Shared-Memory Frame Manager"
Cohesion: 0.12
Nodes (6): memoryview, FrameManager, ABC, ndarray, A SharedMemory that does not register itself with…, UntrackedSharedMemory

### Community 55 - "Species Classifier Abstraction"
Cohesion: 0.19
Nodes (12): Backend-agnostic model settings for the (optional) async species classifier --…, SpeciesModelConfig, ABC, ndarray, Species classifier abstraction -- one interface, multiple pluggable backends…, Result of classifying one crop. species_common/species_scientific/taxonomy are…, One subclass per species-classification backend. Every implementation must load…, SpeciesClassification (+4 more)

### Community 56 - "Notify Bus (SSE Source)"
Cohesion: 0.24
Nodes (13): _log_path(), NotifyEvent, NotifyLogWriter, Path, Structured "a row was created/updated" bus, shared between the `python -m…, Lives in the main process's result-consumer thread (see mirage.app.MirageApp)…, Reads every notify entry at or past line `since_offset` in the file's CURRENT…, read_notify_events_since() (+5 more)

### Community 57 - "go2rtc Config Generation"
Cohesion: 0.24
Nodes (15): build_go2rtc_config(), _detect_role_source(), _env_webrtc_candidates(), Generates a go2rtc.yaml config from a MirageConfig. Only the `streams:`,…, Comma-separated `host:port` ICE candidates for go2rtc to advertise, from…, stream_overrides lets a caller point go2rtc at a DIFFERENT source URL than the…, write_go2rtc_config(), _minimal_config() (+7 more)

### Community 58 - "Detector E2E Tests"
Cohesion: 0.20
Nodes (15): create_tensor_input(), Crops `region` out of the full-resolution YUV420 frame, resizes to the model's…, skipif, test_coreml_configured_detector_still_detects_correctly(), _bgr_image_to_yuv420(), detector(), labels(), fixture (+7 more)

### Community 59 - "Camera Tracker Snapshot Encoding"
Cohesion: 0.26
Nodes (15): _has_qualifying_object(), _maybe_encode_frame(), Cheap proxy for "might EventProcessor/ReviewSegmentMaintainer want a snapshot…, Encodes a plain (no box burned in yet -- that happens later, once…, ndarray, Tests for mirage.tracking.camera_tracker's frame-snapshot-encoding helpers --…, _state(), test_has_qualifying_object_false_when_all_objects_are_false_positive() (+7 more)

### Community 60 - "Architecture Diagram (Docs)"
Cohesion: 0.27
Nodes (16): Mirage Application Flow Architecture Diagram, CameraCapture process, CameraTracker process, detection_queue (IPC), DetectorProcess (shared ONNX model process), EventProcessor thread, frame_queue (IPC, drop-oldest), Main process (+8 more)

### Community 61 - "Open-Vocabulary Detection & Known Issues"
Cohesion: 0.23
Nodes (15): false_positive status never reached DB bug fix (TrackedObjectState.is_false_positive), Open-vocabulary detection subsystem (OWLv2), track_objects -> open-vocabulary bridge (SyntheticTrack), Two-gate dispatch system (track-confirmed + perceptual-hash dedup), Explicit user-triggered pipeline restart (Apply changes button, mirage.supervisor), Known issues to fix (doc), Dual-detector architecture: open-vocabulary enrichment (implemented), Camera hot-reload built then reverted by user decision (+7 more)

### Community 62 - "go2rtc Binary Download"
Cohesion: 0.16
Nodes (13): io, _binary_path(), ensure_go2rtc_binary(), Path, Downloads and caches the go2rtc binary for the host platform. go2rtc…, Returns the path to a working go2rtc binary, downloading + extracting it into…, _resolve_asset(), UnsupportedPlatformError (+5 more)

### Community 63 - "Detector Process Entrypoint"
Cohesion: 0.20
Nodes (11): detector_process_main(), DetectorProcess: the single dedicated OS process that loads a model once and…, Runs as the single dedicated OS process for one configured…, detector_input_shm_name(), detector_output_shm_name(), Derives a short, stable, filesystem-safe key for a camera name, for use in…, Section 3.2: input tensor SHM segment name for a camera's detection request., Section 3.2: output detections SHM segment name, 'out-<camera>'. (+3 more)

### Community 64 - "Event Processor Internals"
Cohesion: 0.17
Nodes (7): _confirmed_snapshot_boxes(), _event_id(), datetime, EventProcessor: consumes detected-frame results and maintains Event DB rows…, Section 6.3: throttle DB writes to at most once per…, Every Gate-1-confirmed (is_false_positive is False) object's box in this frame,…, _to_datetime()

### Community 65 - "PTZ API Tests"
Cohesion: 0.29
Nodes (14): TestClient, _client(), _config_with_ptz_camera(), Tests for the PTZ control endpoints (mirage/api/routers/ptz.py). Following the…, PtzMoveRequest's fields are all optional (default 0.0) -- confirmed by posting…, test_goto_preset_against_unreachable_device_returns_502(), test_move_against_unreachable_device_returns_502(), test_move_defaults_all_axes_to_zero_when_omitted() (+6 more)

### Community 66 - "Detector Scaling & Optimization Docs"
Cohesion: 0.18
Nodes (14): Shared detection_queues[detector_name] queue, Detector Process Scaling Ceiling (doc), Single DetectorProcess-per-config scaling ceiling, RecordingMaintainer premature segment deletion fix (is_open_by_ffmpeg), MirageConfig.from_db() re-validates on every API request (uncached), Optimization opportunities audit (doc), Event.false_positive missing index fix, is_open_by_ffmpeg per-cycle-not-per-segment scan fix (+6 more)

### Community 67 - "Bug-Fix & Process Audit Docs"
Cohesion: 0.15
Nodes (13): Region selection must carry forward unconfirmed candidates fix, ServiceWatchdog.stop() non-blocking race fix, Stationary classifier trimmed-vs-trimmed comparison bug fix, ZMQ PUB socket slow-joiner message drop fix, motion/detector.py scipy->cv2 GaussianBlur perf fix (~7.15x), OpenVocabDispatcher._pending / gate _in_flight no-timeout leak, StationaryClassifier dead second trimmed_box() computation, Process audit -- mirage stack (doc) (+5 more)

### Community 68 - "Dwell & Crowd Rules"
Cohesion: 0.19
Nodes (7): _DwellStart, RulesEngine: camera-level derived-condition alerts -- crowd count and dwell-…, Forgets an object id's dwell-start bookkeeping only once it's been ABSENT for…, Returns synthetic rule-trigger entries for THIS frame only -- a rule condition…, Public accessor for the tracker's current confirmed tracked-object states,…, TrackedObjectState, _state()

### Community 69 - "Tensor Construction"
Cohesion: 0.26
Nodes (10): cv2, crop_yuv_region(), ndarray, Tensor construction: crop a region out of a YUV420 frame and build the model…, yuv_frame: full YUV420 planar buffer, shape (height*3//2, width). frame_shape:…, Converts the full frame to RGB/BGR once, then crops the region out of it.…, yuv420_to_bgr(), yuv420_to_rgb() (+2 more)

### Community 70 - "Config Schema & Store"
Cohesion: 0.26
Nodes (9): Enum, str, Configuration schema (Pydantic models). Spec references: -…, TensorLayout, load_or_seed_config(), Reads/writes the singleton AppConfig DB row that backs MirageConfig.from_db()/…, save_config(), AppConfig (+1 more)

### Community 72 - "Rate & Event Counters"
Cohesion: 0.22
Nodes (3): Event, BoundedWindowCounter, Counts timestamped events within a rolling window (e.g. reconnects/stalls per…

### Community 73 - "Tensor Construction Tests"
Cohesion: 0.29
Nodes (10): clamp_region_to_frame(), ndarray, A flat YUV420 planar buffer: Y plane at y_value, U/V planes at neutral 128…, _synthetic_yuv_frame(), test_clamp_region_to_frame(), test_create_tensor_input_float_denorm_no_scaling(), test_create_tensor_input_float_normalization(), test_create_tensor_input_no_resize_when_region_already_model_size() (+2 more)

### Community 74 - "PTZ Poller"
Cohesion: 0.22
Nodes (4): PtzPoller, One instance per PTZ-enabled camera, owned by that camera's CameraTracker…, Thread-safe read of the current moving state -- this is the ptz_moving_fn…, Runs its own asyncio event loop on this thread -- PtzClient's methods are all…

### Community 75 - "SpeciesNet Worker Process"
Cohesion: 0.27
Nodes (8): base64, json, SpeciesNet backend -- Google/CalTech's camera-trap species classifier…, _build_response(), _parse_label(), Standalone SpeciesNet worker -- runs under a SEPARATE Python venv from the rest…, SpeciesNet's label convention:…, _run_server()

### Community 76 - "System API Tests"
Cohesion: 0.20
Nodes (3): client_and_cache_dir(), fixture, Tests for /api/system/* (mirage/api/routers/system.py) -- the "Apply changes"…

### Community 77 - "How-To-Run & Sample Config"
Cohesion: 0.25
Nodes (9): Sample mirage config (config/mirage.yaml), "general" ONNX YOLOv8n detector config entry, "test_cam" sample camera config entry, Adding a camera via wizard/API (no YAML editing), DB-backed config: --config as one-time import, Running mirage locally (doc), Synthetic test video source (run_test_stream.sh), systemd services (mirage-pipeline, mirage-api) (+1 more)

### Community 78 - "System Design Doc & Frontend Architecture"
Cohesion: 0.31
Nodes (9): Frontend README (Angular CLI usage), App root component (router-outlet shell), Camera-tile snapshot polling never stops fix (setMode gating), API layer (mirage.api routers), Detector plugin system (DetectionApi, OnnxYolov8Detector), Mirage System Design (doc), Frontend architecture (Angular routes, tiered playback), go2rtc integration (stream config, restreaming) (+1 more)

### Community 79 - "Video Lightbox Component"
Cohesion: 0.25
Nodes (4): Component, HostListener, ViewChild, VideoLightbox

### Community 80 - "Deployment & Dependency Docs"
Cohesion: 0.25
Nodes (8): Dockerfile.api (CPU-only image), Dockerfile.pipeline (CUDA build), go2rtc (external live-view proxy service), mirage/api web API module, NVR_PIPELINE_IMPLEMENTATION_SPEC.md, requirements.txt (core dependency pins), requirements-gpu.txt (GPU dependency pins), requirements-speciesnet.txt (SpeciesNet venv deps)

### Community 81 - "Logs Page"
Cohesion: 0.25
Nodes (3): LogCategory, LogsPage, Component

### Community 82 - "App Shell & System Status"
Cohesion: 0.29
Nodes (4): SystemState, SystemStatus, Shell, Component

### Community 83 - "Notifications Tests"
Cohesion: 0.36
Nodes (7): _fetch_and_serialize(), notify_tailer_loop(), Runs for the lifetime of the API process (started/cancelled from the app's…, Tests for the notify tailer (mirage/api/routers/notifications.py) -- the…, test_fetch_and_serialize_existing_event(), test_fetch_and_serialize_missing_row_returns_none(), test_fetch_and_serialize_unknown_table_returns_none()

### Community 84 - "SpeciesNet Classifier Client"
Cohesion: 0.32
Nodes (3): ndarray, subprocess.stdout has no built-in readline timeout -- runs the blocking read in…, SpeciesNetClassifier

### Community 85 - "Integration Test Fixtures"
Cohesion: 0.25
Nodes (8): app_environment(), _free_port(), _kill_process_group(), _port_open(), fixture, Popen, A fresh, currently-unused TCP port -- video_server() used to hardcode a single…, video_server()

### Community 86 - "Design Decisions (SDD)"
Cohesion: 0.33
Nodes (7): On-demand clip stitching design decision, Mirage NVR Software Design Document (doc), EventProcessor detailed design (read-modify-write Event.data), Multi-process pipeline + separate API process strategy, Notify bus / SSE bridge design (file-tail IPC to SSE), PagedList<T> frontend pagination + live-prepend design, Events/Review/Recordings pages visibleInterval tab-visibility fix

### Community 88 - "Process-Group Test Fixtures"
Cohesion: 0.29
Nodes (7): _kill_process_group(), _port_open(), fixture, Popen, SIGKILL the whole process group (the sh -c wrapper and every ffmpeg child it…, short_ipc_dir(), tcp_video_server()

### Community 89 - "Video Server Test Fixtures"
Cohesion: 0.38
Nodes (7): app_environment(), _kill_process_group(), _port_open(), fixture, Popen, _start_video_server(), two_video_servers()

### Community 91 - "Lightbox Component"
Cohesion: 0.40
Nodes (3): Lightbox, Component, HostListener

### Community 93 - "PTZ Client & Poller"
Cohesion: 0.33
Nodes (3): PtzPreset, ONVIF PTZ client wrapper -- reuses onvif-zeep-async's ONVIFCamera exactly as…, PtzPoller: a background thread run INSIDE each PTZ-enabled camera's own…

### Community 94 - "Tracking Frame Orchestration"
Cohesion: 0.47
Nodes (3): FrameResult, ndarray, The shared "run the detector over `regions`, consolidate, feed the tracker"…

### Community 96 - "Bus Test Fixture Image"
Cohesion: 0.83
Nodes (4): bus.jpg (test fixture image), Bus (electric minibus, EMT Madrid), Object detection test scene (vehicle + pedestrian classes), Pedestrians (people crossing street)

### Community 98 - "Camera Tracker Process Entrypoint"
Cohesion: 0.50
Nodes (3): camera_tracker_main(), log_fn(), Runs as the per-camera tracker OS process. Pulls (frame_name, frame_time) off…

### Community 99 - "Detector Labelmap Files"
Cohesion: 0.67
Nodes (4): COCO Labelmap (80 classes), MegaDetector Labelmap (animal/person/vehicle), Wildlife Labelmap (animal/bird/person/vehicle), ZiloDetector Labelmap (animal/person/vehicle)

## Knowledge Gaps
- **99 isolated node(s):** `$schema`, `version`, `packageManager`, `newProjectRoot`, `projectType` (+94 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 754 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CameraConfig` connect `FFmpeg Command Presets` to `Recording Maintainer & Retention`, `Camera Config & Orchestration Tests`, `API App Factory & DB`, `Config API Router`, `Review Segment Maintainer`, `Event Processor`, `Mirage Config Schema`, `Object Lifecycle Filtering`, `Review Severity Classification`, `Capture Process Utils & Watchdog`, `API Read Tests`, `go2rtc Process Supervisor`, `Rules Engine (Alerts)`, `Camera Capture & Frame Ring`, `Recording Maintainer Loop`, `go2rtc Config Generation`, `Event Processor Internals`, `PTZ API Tests`, `Dwell & Crowd Rules`, `Tensor Construction`, `Config Schema & Store`, `Rate & Event Counters`, `Events-Per-Second Counter`, `Camera Tracker Process Entrypoint`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `ApiService` connect `Detector Management UI` to `Manage Cameras Page`, `Frontend Core Services`, `Live View Players`, `Recordings Page`, `App Shell & System Status`, `Events Page`, `Add-Camera Models`, `Review & Logs Services`, `Logs Page`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `requirements-speciesnet.txt (SpeciesNet venv deps)` connect `Deployment & Dependency Docs` to `SpeciesNet Worker Process`, `Frontend Core Services`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `CameraConfig` (e.g. with `_camera_config_out()` and `_camera_out()`) actually correct?**
  _`CameraConfig` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ModelConfig` (e.g. with `DetectionApi` and `OnnxMegadetectorDetector`) actually correct?**
  _`ModelConfig` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `CameraInputConfig` (e.g. with `_primary_input()` and `build_ffmpeg_cmd_for_input()`) actually correct?**
  _`CameraInputConfig` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FfmpegConfig` (e.g. with `config()` and `test_camera_capture_produces_frames_from_local_file()`) actually correct?**
  _`FfmpegConfig` has 2 INFERRED edges - model-reasoned connections that need verification._