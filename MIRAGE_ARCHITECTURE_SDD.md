# Mirage NVR — Software Design Document

Status: reflects the codebase as of 2026-07-26 (branch `dev`). Supersedes/summarizes
`NVR_PIPELINE_IMPLEMENTATION_SPEC.md` (a lower-level implementation blueprint) at the
architecture level, and includes subsystems added since that spec was written
(species classification, PTZ control, live SSE notifications).

## 1. Introduction

### 1.1 Purpose

This document describes the system architecture of Mirage, a self-hosted network
video recorder (NVR) with real-time AI object detection, tracking, and open-vocabulary
search. It is intended for engineers working on the pipeline, API, or frontend who
need a component-level map of the system before making changes, plus anyone onboarding
to the codebase who needs the "why" behind its multi-process design.

### 1.2 Scope

Covered: the capture/detection/tracking/recording pipeline (`python -m mirage`), the
HTTP+SSE API server (`python -m mirage.api`), the Angular frontend, the SQLite schema,
and the cross-process IPC mechanisms that tie them together.

Not covered: deployment/Docker packaging details (`docker/`), the mock camera test
harness (`mock_cameras/`), and per-endpoint request/response field-level docs (see the
FastAPI-generated OpenAPI schema at runtime for that level of detail).

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Event | One tracked object's full detection lifecycle (start → updates → end), stored in the `Event` table |
| Review Segment | A human-facing aggregation window grouping nearby events for the Review page, with a severity (`alert`/`detection`) |
| Query Match | A single open-vocabulary text query matching a tracked object's crop |
| SHM ring | A fixed-depth shared-memory ring buffer of raw frames, used to pass bulk pixel data between processes without copying through a queue |
| go2rtc | The embedded RTSP/WebRTC restreaming server used for low-latency live view |
| Notify bus | The file-based IPC mechanism that lets the pipeline process push "a row changed" events to the API process for SSE fan-out |
| PagedList | The frontend's offset-pagination + live-prepend list helper |

### 1.4 References

- `NVR_PIPELINE_IMPLEMENTATION_SPEC.md` — original low-level implementation spec (ffmpeg command construction, SHM layout, motion algorithm detail, detector I/O contract)
- `NVR_EXTENSION_MULTI_MODEL_ROUTING.md` — multi-model/multi-detector routing extension notes
- In-repo module docstrings (`mirage/app.py`, `mirage/notify_bus.py`, `mirage/api/routers/system.py`, etc.) — treated as living design authority; this document summarizes them rather than duplicating them verbatim

## 2. System Overview

### 2.1 System Context

Mirage ingests RTSP streams from IP cameras, runs motion-gated object detection and
tracking on each stream, persists events/recordings/review segments to a local SQLite
database and local disk, and exposes both a REST+SSE API and an Angular single-page
frontend for live viewing, historical browsing, and configuration. It has no external
cloud dependency — inference, storage, and serving all run on the local host. It
optionally talks to cameras over ONVIF for PTZ control and stream discovery.

### 2.2 High-Level Architecture

```
                         ┌─────────────────────────────────────────┐
                         │         Angular frontend (SPA)           │
                         │  Live / Events / Review / Recordings /   │
                         │  Queries / Config / Manage Cameras        │
                         └───────────────┬───────────────────────────┘
                                         │ HTTP + SSE
                         ┌───────────────▼───────────────────────────┐
                         │        mirage.api  (FastAPI process)      │
                         │  routers: cameras, config, events, live,  │
                         │  notifications, onvif, ptz, query_matches,│
                         │  recordings, review, system                │
                         └───────┬───────────────────────┬───────────┘
                                 │ reads/writes            │ tails
                          ┌──────▼──────┐           ┌──────▼───────┐
                          │  SQLite DB  │           │ notify_log /  │
                          │  (WAL mode) │           │ activity_log  │
                          └──────▲──────┘           └──────▲───────┘
                                 │ writes                  │ writes
                         ┌───────┴──────────────────────────┴────────┐
                         │        python -m mirage  (MirageApp)      │
                         │                                            │
                         │  per camera: CameraCapture + CameraTracker │
                         │  shared:     DetectorProcess(es)           │
                         │              RecordProcess                 │
                         │              OpenVocabProcess               │
                         │              SpeciesProcess                 │
                         │              go2rtc (restream)              │
                         │  main proc:  EventProcessor,                │
                         │              ReviewSegmentMaintainer,        │
                         │              ServiceWatchdog                │
                         └───────────────┬────────────────────────────┘
                                         │ RTSP / ONVIF
                                 ┌───────▼────────┐
                                 │  IP cameras     │
                                 └─────────────────┘
```

### 2.3 Key Components

- **`mirage` pipeline (`python -m mirage`, `MirageApp`)**: the always-on detection/recording engine. Owns every camera's capture and tracking, all ML inference processes, and the recording/retention lifecycle.
- **`mirage.api` (FastAPI + Uvicorn)**: stateless-per-request HTTP server. Never runs inference or touches cameras directly; it only reads/writes the same SQLite DB and cache-dir files the pipeline uses, plus proxies a couple of live/PTZ operations to go2rtc/ONVIF.
- **`mirage.supervisor`** (optional): a thin restart-orchestration layer that can start/stop/restart the pipeline process and report its status to the API via a status file. Not required for local development (`python -m mirage` can be run directly), but is what makes the frontend's "Apply changes" / restart button actually take effect.
- **Frontend (Angular)**: single-page app served by its own dev server (or a static build in production), talking only to `mirage.api`.
- **go2rtc**: vendored/embedded restreaming server giving the frontend a low-latency live view without proxying raw RTSP through the API process.

## 3. Design Considerations

### 3.1 Assumptions

- Single-host deployment: pipeline, API, and DB all run on one machine with a shared local filesystem.
- Camera count and detector throughput are small enough that one or a few shared `DetectorProcess` workers (not one detector per camera) suffice.
- SQLite (not a client/server RDBMS) is acceptable given the single-host assumption and read-heavy, moderate-write-volume workload.

### 3.2 Constraints

- **No shared memory/objects across the pipeline/API process boundary.** They are separate OS processes started independently (`python -m mirage`, `python -m mirage.api`); all cross-process communication must go through disk (SQLite, JSONL log files) or a network socket (ZeroMQ, HTTP).
- Real-time constraint on the per-frame orchestration path (capture → motion gate → region select → detect → track) — this path must stay allocation-light and cannot block on disk or network I/O.
- Must degrade gracefully when optional subsystems are disabled (no detector configured, OpenVocab disabled, species classifier disabled, camera has no PTZ) — every one of these is an additively-gated code path, never a hard dependency.

### 3.3 Dependencies

- FastAPI + Uvicorn (API server), peewee + `playhouse.sqlite_ext` (ORM, SQLite JSON fields)
- ffmpeg (capture decode + record + on-demand clip stitching), go2rtc (restreaming)
- ONNX Runtime / equivalent inference runtime for detector plugins, OWLv2 for open-vocabulary matching, a pluggable species-classification backend (`mirage/species/plugins/`)
- norfair (or equivalent) for multi-object tracking
- ONVIF (via a Python ONVIF client) for camera discovery/resolution and PTZ control
- Angular (frontend framework), RxJS (reactive polling/SSE composition)

### 3.4 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| SQLite write contention under concurrent pipeline + API writes | Med | WAL mode; pipeline is the primary writer, API writes are rare (config, restart-request) |
| File-based IPC (logging/notify bus) adds latency vs. a true message bus | Low | Acceptable at current scale (250ms notify poll, sub-second logging poll); documented as a deliberate simplicity/latency tradeoff, not an oversight |
| Running the pipeline without `mirage.supervisor` leaves the frontend's restart-status UI permanently stale | Low | Known, accepted tradeoff for the current dev workflow; `mirage.supervisor` exists and fixes this when restart-status accuracy matters |
| PTZ hybrid scheduling changes the single hottest per-frame code path (`CameraOrchestrator.process_frame`) | Med | Gated entirely behind `ptz.enabled`; non-PTZ cameras take a byte-for-byte unchanged path |
| Ffmpeg RTSP-port misconfiguration (e.g., camera configured for one port, actual stream on another) causes silent per-camera restart loops | Low | Watchdog restarts the affected camera's processes; does not affect other cameras |

## 4. Architectural Strategies

### 4.1 Strategy Selection

Mirage is built as a **multi-process pipeline** (one OS process per camera capture,
one or a few shared detector processes, one recording process, one main orchestration
process) rather than a single-process multi-threaded design, and a **separate API
process** rather than embedding the HTTP server inside the pipeline. This mirrors
Frigate's own architecture and is chosen because: (a) Python's GIL makes true
parallel frame decode + inference impractical in one process; (b) isolating ffmpeg
subprocess management and ML inference into their own OS processes means one
camera's ffmpeg crash or one detector's inference hang cannot take down the whole
pipeline (the watchdog restarts only the affected part); (c) keeping the API process
separate means it can be restarted, redeployed, or briefly down without interrupting
active recording/detection — the two have different availability requirements.

### 4.2 Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Single process, threads only | Simpler IPC (shared memory natively) | GIL serializes CPU-bound decode/inference; one crash takes down everything | Rejected |
| API embedded in pipeline process | No IPC needed for API↔pipeline state | Couples API uptime to pipeline uptime; blocks pipeline's real-time loop with HTTP request handling | Rejected |
| Message queue / broker (Redis, RabbitMQ) for pipeline↔API notifications | Lower latency, richer delivery guarantees | New external dependency for a single-host app that already has a working file-based pattern (`logging_bus.py`) | Rejected — file-poll bridge reused instead |
| One detector process per camera | Simpler per-camera isolation | Duplicates model load cost N times; wastes GPU/accelerator memory | Rejected — shared `DetectorProcess` pool, `num_workers`-configurable |
| Full ORM/RDBMS (Postgres) | Better concurrent-write story | Unjustified operational complexity for a single-host app | Rejected — SQLite + WAL sufficient |

### 4.3 Key Architectural Decisions

- **File-based IPC bridge for cross-process notifications** (`mirage/logging_bus.py`, `mirage/notify_bus.py`): the pipeline process appends small JSON lines to a file; the API process tails it on a short poll interval (250ms for notify, similar for activity log). Chosen over a socket/queue because it required no new dependency and directly reused an already-proven pattern in this codebase.
- **SSE, not WebSockets, for live push to the frontend**: one-directional (server→client) notifications are all that's needed (new Event/ReviewSegment/QueryMatch rows); plain FastAPI `StreamingResponse` with hand-formatted `data: {json}\n\n` frames avoids a new dependency (`sse-starlette`) and browsers' native `EventSource` auto-reconnect removes the need for custom reconnect logic.
- **Notify payloads carry only `{table, id, op}`, never the full row**: the API re-fetches and re-serializes fresh from SQLite at broadcast time. This guarantees clients always see current state — important for `ReviewSegment`, which is saved multiple times over its life (create → severity upgrade → end) — rather than a stale creation-time snapshot.
- **Offset-based pagination (`PagedList<T>`) composed with SSE, not a replacement for it**: SSE delivers new rows as they're created; pagination's "Load more" handles browsing further into history. The two share one abstraction (`prependLive()` for push, `loadMore()` for pull) rather than two separate list-management code paths per page.
- **On-demand clip stitching, not pre-rendered per-event/per-review clips**: `mirage/recording/stitch.py`'s `recordings_overlapping()` + `stitch_recordings()` finds and ffmpeg-concatenates whatever permanent recording segments overlap a requested time window (padded), only at request time, cached to disk after first render. Avoids doing stitching work for events nobody ever asks to watch.
- **`Content-Disposition` deliberately omitted from clip/recording `FileResponse`s**: setting `filename=` forces browsers to treat the response as a download-only attachment, which silently blocks inline `<video>` playback. The frontend's own download button sets its own `[download]` attribute instead, so the same URL serves both inline playback and explicit download.
- **Species classification and PTZ control are both async/additively-gated extensions, not core-path changes**: species classification runs in its own `SpeciesProcess`, dispatched from `EventProcessor` only for enrichable labels, writing results back into `Event.data` without blocking event creation. PTZ hybrid scheduling only engages for cameras with `ptz.enabled=true`; all other cameras run the original, unmodified per-frame orchestration path.

## 5. System Architecture

### 5.1 Component Diagram

```
mirage/
├── capture/        ffmpeg process management, frame ingest into SHM rings
├── motion/          adaptive background-differencing motion detector
├── regions/          bridges motion output to detector input (region selection/clustering)
├── detection/         pluggable detector backend abstraction + DetectorProcess
├── tracking/          norfair-based multi-object tracker, per-frame orchestration, PTZ hybrid scheduling
├── openvocab/         OWLv2-backed open-vocabulary query matching (async worker)
├── species/           pluggable species-classification backend (async worker)
├── ptz/                ONVIF PTZ client (status/move/preset/patrol)
├── events/             EventProcessor (event state machine) + ReviewSegmentMaintainer
├── recording/           recording maintainer (retention) + on-demand clip stitching
├── review/              (review-specific helpers used by events/review.py)
├── ipc/                  ZeroMQ pub/sub wrapper for control/signaling
├── db/                    peewee models + database bootstrap
├── config/                Pydantic config schema, DB-backed (no YAML authoring)
├── util/                   SHM frame manager, time helpers
├── logging_bus.py          activity-log file-IPC (pipeline → API)
├── notify_bus.py            row-change-notification file-IPC (pipeline → API)
├── watchdog.py               per-process health monitor + restart
├── supervisor.py              optional pipeline restart orchestration
├── app.py                      MirageApp: process bootstrap/lifecycle
└── api/
    ├── app.py                   FastAPI app factory, lifespan (DB + SSE tailer task)
    ├── schemas.py                 Pydantic response models (*Out)
    └── routers/                    cameras, config, events, live, notifications,
                                     onvif, ptz, query_matches, recordings, review, system
```

### 5.2 Data Flow

**Detection → Event → frontend (the core write path):**

1. `CameraCapture` reads ffmpeg's stdout, writes decoded frames into a per-camera SHM ring.
2. `CameraTracker`'s `CameraOrchestrator.process_frame` reads the latest frame, runs motion detection, selects candidate regions (or, for a moving PTZ camera, samples the whole frame directly), and dispatches crops to the shared `DetectorProcess` pool over its request queue.
3. Detection results feed the tracker (norfair), which maintains `TrackedObjectState` per object across frames and applies the start/update/end lifecycle rules (true/false-positive filtering, zone evaluation).
4. Lifecycle callbacks are enqueued to the main process's result-consumer loop, which drives `EventProcessor` (writes `Event` rows) and `ReviewSegmentMaintainer` (writes `ReviewSegment` rows).
5. On a qualifying event, `EventProcessor` optionally dispatches an async species-classification request (`SpeciesDispatcher` → `SpeciesProcess`) and/or the tracked crop is checked by `OpenVocabDispatcher` → `OpenVocabProcess` against enabled saved queries, writing `QueryMatch` rows on a hit.
6. Every create/update to `Event`/`ReviewSegment`/`QueryMatch` also calls a `_notify(...)` helper, which appends a `{table, id, op}` line to the notify-bus log file.
7. `mirage.api`'s background tailer task polls that file every 250ms, re-fetches the changed row from SQLite, serializes it with the matching `*Out` schema, and fans it out to every connected SSE client.
8. The frontend's `SseService` delivers the message to the relevant page, which calls `PagedList.prependLive(item)` to insert/update the row in place — no full re-fetch required. A slow 30s reconciliation poll (`visibleInterval`) self-heals anything missed during a brief SSE reconnect gap.

**On-demand clip playback:**

1. Frontend requests `GET /api/{events|review}/{id}/clip` (or `/api/recordings/{id}/clip`).
2. The API looks up the row's time window (padded), finds overlapping permanent `Recordings` rows, and — if not already cached on disk — stitches them with ffmpeg's concat demuxer.
3. Response is served as an inline-playable `video/mp4` (no `Content-Disposition`), with the stitched file cached under the app's export directory so repeat requests skip re-stitching.

### 5.3 API Design

#### 5.3.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cameras` | List configured cameras + live status |
| GET | `/api/cameras/{name}` | Single camera detail |
| GET | `/api/events` | Paginated event list (filterable: camera, label, time range, false_positive) |
| GET | `/api/events/{id}` | Single event detail |
| GET | `/api/events/{id}/snapshot` | Event snapshot image |
| GET | `/api/events/{id}/clip` | Stitched inline-playable clip for the event's time window |
| GET | `/api/events/stream` | SSE stream — live Event/ReviewSegment/QueryMatch notifications |
| GET | `/api/review` | Paginated review segment list |
| GET | `/api/review/{id}` | Single review segment detail |
| GET | `/api/review/{id}/thumbnail` | Review segment thumbnail image |
| GET | `/api/review/{id}/clip` | Stitched inline-playable clip for the segment |
| GET | `/api/recordings` | Paginated recording list |
| GET | `/api/recordings/{id}` | Single recording detail |
| GET | `/api/recordings/{id}/clip` | Inline-playable recording file |
| GET | `/api/query-matches` | Paginated open-vocab query match list |
| GET | `/api/query-matches/{id}` | Single match detail |
| GET | `/api/query-matches/{id}/thumbnail` | Match thumbnail image |
| GET | `/api/live/{camera}/snapshot.jpg` | Current live snapshot |
| GET / POST / PUT / PATCH / DELETE | `/api/config/detectors[...]` | Detector CRUD + enable/disable + worker count |
| GET / POST / PUT / DELETE | `/api/config/cameras[...]` | Camera config CRUD |
| GET / POST / PUT / DELETE | `/api/config/queries[...]` | Saved open-vocab query CRUD |
| GET / PATCH | `/api/config/openvocab[...]` | OpenVocab subsystem settings |
| GET | `/api/onvif/scan` | Discover ONVIF devices on the network |
| GET | `/api/onvif/resolve` | Resolve stream URL + capabilities for one device |
| GET | `/api/ptz/{camera}/presets` | List PTZ presets |
| GET | `/api/ptz/{camera}/status` | Current pan/tilt/zoom + moving state |
| POST | `/api/ptz/{camera}/goto-preset/{token}` | Move to a saved preset |
| POST | `/api/ptz/{camera}/move` | Continuous-move (joystick-style) |
| POST | `/api/ptz/{camera}/stop` | Stop any in-progress move |
| POST | `/api/ptz/{camera}/patrol/start` \| `/stop` | Start/stop preset-cycling patrol |
| GET | `/api/system/status` | Pipeline running/stopped/restarting state (via `mirage.supervisor`, if running) |
| POST | `/api/system/restart` | Request a pipeline restart |
| GET | `/api/system/logs` | Recent activity log entries |
| GET | `/api/system/capabilities` | Host CPU/memory, for Config page sizing guidance |

All list endpoints accept `limit`/`offset` for pagination; `events`/`review`/`query-matches` additionally support `since` (SSE reconciliation) and entity-specific filters.

#### 5.3.2 Request/Response Schemas

Response shapes are defined as Pydantic models in `mirage/api/schemas.py` (`EventOut`,
`ReviewSegmentOut`, `RecordingOut`, `QueryMatchOut`, `CameraOut`, etc.), each built via
a `from_model(row)` classmethod that maps a peewee row (including relevant `data` JSON
blob keys) onto a stable, named API field set. The SSE stream at `/api/events/stream`
emits frames shaped `{"type": "event" | "review_segment" | "query_match", "data": {...}}`,
where `data` is exactly the same `*Out` shape returned by the corresponding REST detail
endpoint.

### 5.4 Data Models

| Model | Key fields | Notes |
|-------|-----------|-------|
| `Event` | `id` (PK), `label`, `sub_label`, `camera`, `start_time`, `end_time` (nullable), `score`, `top_score`, `false_positive` (indexed, default `true`), `zones` (JSON), `has_clip`, `has_snapshot`, `snapshot_path`, `data` (JSON: box, region, attributes, path history, max_severity, species fields) | One tracked object's full lifecycle |
| `Recordings` | `id` (PK), `camera`, `path` (unique), `start_time`, `end_time`, `duration`, `motion`, `objects`, `regions`, `dBFS`, `segment_size_mb` | Permanent recording segments, correlated to detection activity by timestamp only |
| `ReviewSegment` | `id` (PK), `camera`, `start_time`, `end_time` (nullable), `severity` (`alert`\|`detection`), `thumb_path` (unique, nullable), `data` (JSON: detections, objects, zones, audio, thumb_time, metadata) | Human-facing aggregation window |
| `Timeline` | `timestamp`, `camera`, `source`, `source_id`, `class_type`, `data` (JSON) | Optional scrubbable activity timeline |
| `Regions` | `camera` (PK), `grid` (JSON), `last_update` | Learned per-camera object-size grid |
| `AppConfig` | `id` (PK, CHECK id=1), `data` (JSON — full `MirageConfig` tree), `updated_at` | Singleton row; config is DB-authored, not hand-edited YAML |
| `QueryMatch` | `id` (PK), `query_id`, `query_text` (denormalized), `camera`, `object_id`, `matched_at`, `score`, `box` (JSON), `thumb_path` | One open-vocab query hit against a tracked object |

All tables use SQLite in WAL mode via peewee. Indexes are placed on every field used
as a default list-endpoint filter (`camera`, `label`, `start_time`, `end_time`,
`false_positive` on `Event`) — the last of these was added specifically because
`false_positive=false` is the default (and near-universal) filter on the
most-frequently-polled endpoint, `GET /api/events`.

## 6. Policies and Tactics

### 6.1 Security

- Single-host, LAN-oriented deployment model — no built-in multi-tenant auth; access control is expected to be handled at the network/reverse-proxy layer if exposed beyond localhost/LAN.
- Camera and ONVIF/PTZ credentials are stored in the same `AppConfig` JSON blob as the rest of configuration (consistent with existing practice; flagged in the original spec as a candidate for future credential-hardening, not yet implemented).
- No data-at-rest encryption; recordings and snapshots are plain files on local disk.

### 6.2 Error Handling

- API endpoints return standard HTTP status codes: `404` for unknown IDs, `410` when a referenced recording file no longer exists on disk (expired/purged), `500` only for genuinely unexpected failures (e.g., ffmpeg stitch failure).
- Pipeline-side subsystems (species dispatch, notify-log writes, OpenVocab dispatch) wrap their side-effect calls in try/except-and-log rather than propagating — a failure to enrich or notify must never block or crash the core event-creation path.
- `mirage.watchdog.ServiceWatchdog` monitors every managed subprocess (detector, record, per-camera capture/tracker, OpenVocab, species) and restarts on unexpected exit; a single camera's ffmpeg failure (e.g., unreachable RTSP endpoint) restarts only that camera's processes, not the whole pipeline.
- A read racing the supervisor's atomic status-file rename is treated as `"unknown"` state rather than a hard error — the next poll a second later resolves it.

### 6.3 Logging and Monitoring

- `mirage/logging_bus.py`: pipeline-wide activity log, appended to a shared JSONL file, read incrementally by `GET /api/system/logs` (`since` param avoids re-fetching the whole tail).
- `mirage/notify_bus.py`: same file-append/tail pattern, purpose-built for row-change notifications rather than human-readable log lines; kept as a separate file from the activity log since the schema and consumer differ.
- `GET /api/system/status`: reports pipeline running/stopped/restarting state and PID, sourced from `mirage.supervisor`'s status file when that process is running; reports `"unknown"` (treated as healthy/no-banner by the frontend) when the pipeline runs directly without a supervisor.

### 6.4 Performance

- Bulk frame data moves through fixed-depth SHM rings, never through a serialized queue — the only cross-process payloads on the hot path are small structured messages (detection requests/results) over `multiprocessing.Queue`/ZeroMQ.
- Detector inference is centralized in one (or a small `num_workers`-configurable pool of) `DetectorProcess`, avoiding per-camera model-load duplication.
- List endpoints paginate (`limit`/`offset`); the frontend's initial page size was deliberately reduced (100 → 20) on Events/Review/Recordings specifically because each card triggers a real image (or, for Recordings, a real `<video>` poster) fetch — a large initial page means dozens of simultaneous media requests on first load.
- Clip stitching is on-demand and cache-backed (`export_dir/{event,review}_clips/{id}.mp4`) — never pre-rendered for events nobody watches, but never re-stitched once rendered once.
- Live notifications use a 250ms file-tail poll (SSE) rather than the previous 5-10s full-list HTTP poll; each page keeps a slow 30s reconciliation poll as a self-healing fallback, not a primary delivery mechanism.

## 7. Detailed Design

### 7.1 EventProcessor Design

#### 7.1.1 Responsibilities
- Consume tracked-object lifecycle callbacks (start/update/end) from the per-camera orchestration loop.
- Apply true/false-positive and zone-based filtering rules before persisting.
- Create/update `Event` rows; on qualifying label types, dispatch async species classification and open-vocab query matching.
- Emit notify-bus events on every create/update so the API's SSE tailer can broadcast fresh state to the frontend.

#### 7.1.2 Interface
```python
class EventProcessor:
    def __init__(self, notify_queue: "mp.Queue | None" = None, ...): ...
    def _on_start(self, state: TrackedObjectState, frame_jpeg: bytes) -> None: ...
    def _on_update(self, state: TrackedObjectState, frame_jpeg: bytes) -> None: ...
    def _on_end(self, state: TrackedObjectState) -> None: ...
    def _notify(self, table: str, row_id: str, op: str) -> None: ...
```

#### 7.1.3 Implementation Notes
- `_on_update` must read-modify-write the `Event.data` JSON blob (never a blind full replace) — a throttled heartbeat update (box/snapshot_boxes) landing after an async species result would otherwise silently clobber the completed classification. This is the single highest-risk correctness point in the species-enrichment design.
- Species dispatch fires at most once per event's lifetime (at creation, for enrichable labels only), not per-frame — unlike OpenVocab's perceptual-hash-gated per-frame dedup, which exists because OpenVocab checks run continuously against a live crop, not once at creation.

### 7.2 Notify Bus / SSE Bridge Design

#### 7.2.1 Responsibilities
- Bridge row-change events from the pipeline process to the API process without shared memory.
- Fan out each change to every connected SSE client with fresh, re-fetched row data.

#### 7.2.2 Interface
```python
# mirage/notify_bus.py
@dataclass
class NotifyEvent:
    table: str; id: str; op: str; timestamp: float

class NotifyLogWriter:
    def append(self, event: NotifyEvent) -> None: ...

def read_notify_events_since(cache_dir: str, since_offset: int) -> tuple[list[dict], int]: ...

# mirage/api/routers/notifications.py
async def notify_tailer_loop(cache_dir: str) -> None: ...   # background task, 250ms poll
@router.get("/stream")
def stream_notifications(request: Request) -> StreamingResponse: ...
```

#### 7.2.3 Dependencies
- SQLite (`Event`/`ReviewSegment`/`QueryMatch` tables) — for re-fetch-and-serialize at broadcast time.
- FastAPI `StreamingResponse` — no additional SSE library dependency.

#### 7.2.4 State Management
- A module-level `set[asyncio.Queue]` tracks connected SSE clients; one background `asyncio.Task` (started in the API app's `lifespan`) does the file-tail polling and fan-out. A slow client is dropped from delivery (queue-full) rather than allowed to block broadcast to everyone else.

#### 7.2.5 Implementation Notes
- Router registration order matters: `notifications.router` (`GET /stream`) must be registered before `events.router` (`GET /{event_id}`) since both share the `/api/events` prefix — otherwise `events.router`'s catch-all `{event_id}` path parameter greedily matches the literal `/stream` segment and 404s.
- A ~15s `: keepalive\n\n` comment frame is sent periodically to survive idle-timeout proxies between real notifications.

### 7.3 PagedList (Frontend) Design

#### 7.3.1 Responsibilities
- Manage one page's worth of offset-paginated rows plus "load more" expansion.
- Accept SSE-pushed live rows and merge them without disturbing already-loaded pagination state or causing duplicate/out-of-place cards.

#### 7.3.2 Interface
```typescript
class PagedList<T extends { id: string }> {
  items(): T[];
  hasMore(): boolean;
  loadingMore(): boolean;
  setFirstPage(items: T[], hasMore: boolean): void;
  loadMore(fetchNextPage: (offset: number) => Promise<{items: T[]; hasMore: boolean}>): Promise<void>;
  prependLive(item: T): void;  // insert new id at index 0, or replace-in-place if id already present
}
```

#### 7.3.3 Implementation Notes
- `prependLive`'s replace-in-place behavior for an already-known id is what makes repeated notifications for the same row (e.g., a `ReviewSegment` saved at start, severity-upgrade, and end) safe to deliver without ever producing a duplicate card or a jarring position jump.

## 8. Appendix

### 8.1 Sequence Diagram — Live Event Notification

```
CameraTracker   EventProcessor   notify_log.jsonl   API tailer task   SSE clients   Frontend
     |                |                  |                 |                |            |
     |-- on_start --->|                  |                 |                |            |
     |                |-- Event.create ->| (SQLite)         |                |            |
     |                |-- append notify->|                 |                |            |
     |                |                  |<--- poll 250ms--|                |            |
     |                |                  |-- new line ---->|                |            |
     |                |                  |                 |-- refetch/     |            |
     |                |                  |                 |   serialize -->|            |
     |                |                  |                 |-- data:{...} ->|-- onmessage->
```

### 8.2 State Diagram — ReviewSegment Notify Ops

```
        _start_segment                _upgrade_segment              _end_segment
(no segment) ───────────────► [open, notify="create"] ──────► [open, notify="update"]
                                        ▲                                │
                                        │ (severity upgrade,             │ end_time set
                                        │  repeated frames with          ▼
                                        │  no change: notify=None)  [closed, notify="update"]
```

### 8.3 Glossary

- **Region selection**: the bridge module (`mirage/regions/`) that turns raw motion-detector output into a set of candidate crops to hand the detector, avoiding full-frame inference on every frame.
- **Hybrid PTZ scheduling**: for PTZ-enabled cameras only — while panning/moving, bypass motion-gated region selection and sample the whole frame directly at a low fixed rate; resume normal motion-gated detection once stationary.
- **Perceptual-hash gating**: OpenVocab's per-frame dedup check — only send a tracked object's crop to the (expensive) OWLv2 process if it's changed enough visually since the last check.
