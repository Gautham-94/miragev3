# Known issues to fix

Running log of real bugs found during manual testing (with the real Hikvision camera and
the `mock_cameras` standalone app), written with enough detail to pick up and fix cold,
without re-deriving the investigation.

---

## 1. Mock camera stream looks laggy/stuttery in the Live page

**Status:** open, not yet fixed (deprioritized to look at later per user).

**Symptom:** the `front_door` mock camera (backed by a real ~4-minute mp4, not a
synthetic clip) visibly stutters/lags when viewed on the Live page, compared to the real
Hikvision camera tile.

**Root cause (suspected, not fully confirm, needs the investigation below):**
`mock_cameras/videos/front_door.mp4` is a real-world video at **1280x544, 50fps, ~2 Mbps
H.264 + AAC audio** (confirmed via `ffprobe`). `mock_cameras` streams it through go2rtc
with `-re -stream_loop -1` (real-time loop, no `-r` fps override), so go2rtc emits it at
its native 50fps. But the mirage camera config for this stream is set to capture at only
**5fps** (see the Live tile badge: `640x480 . 5fps`) -- a 10:1 discard ratio between what
ffmpeg has to decode and what mirage actually keeps. ffprobe also logged `co located POCs
unavailable` against this file, which suggests non-trivial B-frame reordering in the
source encode; combined with the heavy discard ratio, this is a plausible cause of visible
jank even though no packets are being dropped at the network level (confirmed via
`http://127.0.0.1:1984/api/streams` on mirage's own go2rtc -- steady producer byte/packet
counts, no errors).

By contrast, `backyard.mp4` and `garage.mp4` are lighter (5fps and 30fps respectively) and
were not reported as laggy.

**What's been ruled out:**
- Not a network/RTSP-drop issue -- go2rtc's own `/api/streams` endpoint shows continuous,
  error-free producer flow for the mock camera.
- Not a capture-pipeline error -- `grep`ing `/tmp/mirage_pipeline.log` for
  `mockcameras_mc_1000`/`front_door` shows zero errors/warnings/drops logged.
- Not specific to the "STILL" live badge -- STILL vs LIVE is a separate, correct-by-design
  indicator (no motion in frame keeps it on the polled JPEG snapshot tier); the lag
  complaint is about actual playback smoothness when the tile IS on the MSE/WebRTC tier.

**Suggested fixes (pick one, not yet decided by user):**
1. **Cap `mock_cameras`' output fps to match a realistic camera fps** (e.g. 15fps) by
   adding an explicit `-r 15` to the go2rtc ffmpeg source command generated in
   `mock_cameras/mock_cameras/go2rtc.py`. Closest to how a real IP camera would actually
   be configured (most cameras don't push 50fps).
2. **Re-encode `front_door.mp4` itself** to a lower, steadier fps/bitrate one time, e.g.:
   ```
   ffmpeg -i front_door.mp4 -r 15 -bf 0 -c:v libx264 -b:v 1M front_door_15fps.mp4
   ```
   Keeps `mock_cameras`' go2rtc config untouched; fixes the asset itself.
3. Investigate further first -- e.g. check actual browser-side MSE `SourceBuffer`
   buffering/backpressure behavior in devtools during playback, and host CPU load while
   this stream is being decoded, before committing to an fps-based fix (in case the real
   cause turns out to be something else, e.g. main-thread contention in the Angular MSE
   player itself).

**Relevant files:**
- `mock_cameras/mock_cameras/go2rtc.py` -- generates the go2rtc stream config / ffmpeg
  source command per mock camera
- `mock_cameras/videos/front_door.mp4` -- the source asset
- `mirage/frontend/src/app/pages/live/players/mse-player.ts` -- browser-side MSE playback,
  worth checking for buffering behavior if pursuing option 3

---

## 2. Adding/editing any camera requires a manual pipeline restart -- easy to forget, no UI warning

**Status:** open, working as documented but a repeated source of confusion; worth a
proper fix (hot-reload) or at minimum a UI nudge.

**Symptom:** multiple times in this session, a newly-added camera (via the wizard's ONVIF
scan+resolve flow) saved successfully to the database and appeared in `GET
/api/cameras`, but showed **"Signal lost"** on the Live page indefinitely, with zero
capture attempts ever logged for it in the pipeline's log file. This happened repeatedly:
after adding `backyard`, then again after adding `garage` and `front_door` in the same
session.

**Root cause (this is NOT a bug in the traditional sense -- it's documented, but the UI
gives no indication of it, so it reads as broken):** `python -m mirage` reads camera
config **once at startup** and does not hot-reload it. `mirage/HOW_TO_RUN.md` states this
explicitly ("Adding/editing cameras after that first run... restart `python -m mirage` to
pick up the change (config changes aren't hot-reloaded into an already-running pipeline
yet)"), but:
- The add-camera wizard's success screen does not mention this at all.
- The Manage Cameras page does not mention this at all.
- The Live page's "Signal lost" state is visually indistinguishable between "this camera
  is genuinely unreachable" and "the pipeline just doesn't know this camera exists yet" --
  there's no way for a user to tell these apart from the UI alone.

**Confirmed reproduction pattern:** every time, `GET /api/cameras` showed the new camera
correctly, but `grep "serving cameras" /tmp/mirage_pipeline.log | tail -1` showed the
pipeline's actual camera list was stale (missing the newest additions) until the pipeline
process was manually killed (`kill -INT <pid>`) and restarted
(`python3 -m mirage -v`, from `mirage/`), at which point the log line
`mirage.app: mirage app started: N camera(s), 1 detector(s)` correctly reflected the new
count and the new camera immediately started showing signal.

**Suggested fixes (pick one or combine, not yet decided by user):**
1. **Proper hot-reload**: have the running `python -m mirage` process detect config
   changes (e.g. poll the `AppConfig`/camera tables for a version/updated_at bump, or have
   the config-write API endpoints in `mirage/api/routers/config.py` publish a signal the
   running pipeline process can observe) and dynamically spin up/tear down
   `CameraTracker`/capture processes for added/removed/edited cameras without a full
   restart. This is the "real" fix but is the most invasive -- touches
   `mirage/app.py`, `mirage/tracking/orchestration.py`, and the config API.
2. **Minimal UI nudge (much cheaper, good stopgap)**: after a successful camera
   save in `mirage/frontend/src/app/pages/add-camera/add-camera-page.ts` (the `save()`
   method's success path) and in the Manage Cameras page's edit/delete flows, show a
   persistent banner/toast: "Camera saved. Restart the mirage pipeline to start
   capturing from it." Cheap, and immediately removes the "is this broken?" confusion
   even without solving hot-reload.
3. Have the API expose whether the running pipeline's camera list is stale relative to
   the DB (e.g. a `GET /api/health` or new endpoint comparing `AppConfig` cameras vs. an
   in-memory "pipeline last-loaded camera list" the pipeline process periodically reports
   back into the DB or a shared file), and have the frontend surface that mismatch
   directly on the Live page tile instead of a generic "Signal lost".

**Relevant files:**
- `mirage/HOW_TO_RUN.md` -- already documents the restart requirement (for a human
  reading the docs, not enforced/surfaced anywhere in the app itself)
- `mirage/mirage/app.py` -- where the pipeline reads config once at startup
- `mirage/api/routers/config.py` -- camera create/update/delete endpoints, would need to
  either trigger reload signaling (option 1) or nothing extra (option 2/3)
- `mirage/frontend/src/app/pages/add-camera/add-camera-page.ts` (`save()` method) --
  where a post-save banner would go (option 2)
- `mirage/frontend/src/app/pages/manage-cameras/manage-cameras-page.ts` -- same, for
  edit/delete flows
- `mirage/frontend/src/app/pages/live/camera-tile/camera-tile.ts` -- Live tile's
  connecting/error state, relevant if pursuing option 3

---

## 3. (Fixed, keeping for reference) Mock camera ONVIF resolve used device_name instead of profile_name for auto-derived camera name

**Status:** fixed in this session -- keeping a short record since it explains data
currently sitting in the DB.

**What happened:** `mock_cameras`' ONVIF SOAP service returns the same hardcoded
`device_name` ("MockCameras MC-1000") for every mock camera, since `device_name` models
the physical hardware, not the individual stream -- while `profile_name` is set per-camera
correctly (`front_door`/`backyard`/`garage`). The wizard's auto-derived camera-name logic
in `add-camera-page.ts`'s `resolveStream()` preferred `device_name` first, so every mock
camera added through the wizard was named `mockcameras_mc_1000`, and since camera name is
the identity key for `POST/PUT /api/config/cameras/{name}`, each new save silently
overwrote the previous mock camera row instead of creating a new one.

**Fix applied:** swapped the preference order to `profile_name ?? device_name ?? ...`.

**Leftover cleanup possibly still needed:** check `GET /api/cameras` for any stray
`mockcameras_mc_1000` entry left over from before the fix (should be manually deleted via
the Manage Cameras page if still present, since it's a name collision artifact, not a
real camera).

**Also worth fixing properly later:** `mock_cameras`' ONVIF `GetDeviceInformation`
handler could just as easily set a per-camera-derived model/serial string (e.g. include
the camera name in `SerialNumber`) so `device_name` itself is unique too, removing the
sharp edge entirely rather than relying on the consuming wizard always preferring the
right field. See `mock_cameras/mock_cameras/onvif_server.py`.

---

## 4. (IMPLEMENTED) Dual-detector architecture: open-vocabulary enrichment
   for "animal"/beyond-COCO detection, plus derived crowd detection

**Status:** the open-vocabulary enrichment half (OWLv2 + saved text queries + gating) is
now BUILT and verified working live against real camera footage -- see the new
"Implementation" subsection at the end of this item. The crowd-detection half (pure
count/geometry logic, no new model) is still NOT built -- that part remains open.

Original status before implementation: open, deferred -- user said "we will revisit this
again," not yet scheduled. Later revisited and built in full (see below).

**Motivating problem:** mirage's only detector today is `yolov8n.onnx`, a **closed-vocabulary**
model -- its output layer is fixed at training time to exactly the 80 COCO classes in
`models/coco_labelmap.txt`. `track_objects` matching is plain string equality against
whatever label that model emits (`mirage/tracking/orchestration.py`: `if label not in
self.camera.objects.track: continue`). This means:
- A user-configured `track_objects: ["animal"]` is a **silently dead entry** -- COCO has no
  generic "animal" class, only 10 specific species (`bird, cat, dog, horse, sheep, cow,
  elephant, bear, zebra, giraffe`). Confirmed directly against the code (`orchestration.py`,
  `mirage/detection/remote.py`, `mirage/config/schema.py`) -- no aliasing/grouping table
  exists anywhere.
- Vehicle detection (`car`/`truck`/`bus`/`motorcycle`/`bicycle`) and person detection
  already work today with zero new architecture -- they're native COCO classes.
- "Crowd detection" isn't a model output at all in any object-detection architecture --
  it's a derived/aggregate condition (how many confirmed `person` tracks exist at once,
  optionally their spatial clustering), computed from tracker state mirage already has.

**User's specific idea for the animal-detection gap (this conversation, verbatim intent):**
run a **dual-detector pipeline**, inspired by an unrelated NVR design doc
(`DigiNetra`, seen in `~/Desktop/ai_inference_abhisk/docs/SYSTEM_DESIGN.md`) that pairs a
fast closed-vocabulary detector (their RTMDet) for every-frame tracking with a slow
open-vocabulary detector (Grounding DINO / OWLv2) for text-prompted enrichment
(e.g. `"person wearing red shirt"`) of already-tracked objects. The user's refinement on
top of that base idea, arrived at through discussion in this session:

1. **Trigger stage**: motion detection (already runs every frame today,
   `mirage/motion/detector.py`) feeds the existing fast detector (`yolov8n.onnx`) and
   tracker (norfair, `mirage/tracking/tracker.py`) exactly as today -- unchanged.
2. **Gate 1 -- track confirmation** (proposed by assistant, not yet user-confirmed as
   final): only consider dispatching to the open-vocab model once
   `ObjectLifecycle.is_false_positive` flips `True -> False` for a track (i.e. the object
   survived norfair's `initialization_delay` and `is_object_filtered`'s score/area
   checks) -- avoids burning the expensive model on motion noise (leaves, headlight
   glare) that the existing pipeline already filters for free.
3. **Gate 2 -- embedding-similarity dedup** (user's specific idea, this turn): don't
   re-run the open-vocab model on every subsequent frame of an object that's just
   sitting there unchanged (e.g. an animal standing still for a minute would otherwise
   hammer the slowest model in the stack for no new information). Concretely: run a
   *cheap* embedding encoder (proposed: CLIP's image encoder specifically, since
   Grounding DINO/OWLv2 already operate in a CLIP-aligned embedding space, so reusing it
   is architecturally coherent rather than an unrelated third model) on each tracked
   object's current crop, keep a rolling "last embedding sent to open-vocab" per
   `TrackedObjectState` (would need a new field there, following the same
   already-established pattern used for `is_false_positive` crossing the
   `CameraTracker` process boundary), and only dispatch to open-vocab when cosine
   distance from the last-sent embedding exceeds a threshold (object changed pose,
   lighting shifted, a second object entered the crop, etc).
4. **Open-vocab dispatch**: crop passes to a new, separate process (parallel in spirit
   to how `mirage/detection/remote.py` already isolates the YOLO detector) running
   OWLv2/Grounding DINO with a text-prompt list derived from any `track_objects` entries
   that aren't literal COCO labels (this is where `"animal"` becomes a real query
   instead of a dead config value). Runs asynchronously, does NOT need to keep up with
   real-time frame rate (this is the "Frame Sampler"/"Scheduler" role from the DigiNetra
   doc) -- worst case on a slow/backlogged result is a delayed or missing enrichment,
   never a stalled recording/tracking pipeline, since this is purely additive.
5. **Write-back**: result attaches as enrichment onto the *same* `Event` row, e.g. a new
   `data["open_vocab_label"]` / `data["attributes"]` key -- `Event.data` is already a
   flexible JSON blob (`mirage/db/models.py`), so no schema migration needed for this
   part.

**Crowd detection (separate, much cheaper sub-feature, discussed same session):** pure
counting/geometry logic on tracker state mirage already maintains -- no new model.
Count confirmed `person`-label `TrackedObjectState`s per camera per frame/window; cross a
configurable threshold (e.g. `crowd_threshold: 5`) to emit a new severity/condition.
Natural fit: extend `ReviewSegment.severity` (currently `"alert"`/`"detection"`) with a
`"crowd"` value, or add a boolean/count field to `ReviewSegment.data`, rather than a new
table. If density/clustering (people grouped tightly vs. spread across frame) is wanted
later, that needs each track's box centroid and a spatial clustering/distance check
across the current frame's confirmed boxes -- still pure geometry, no ML model, but a
meaningfully more involved step than a flat count threshold. **Not yet decided which of
count-only vs. count+density the user wants** -- flagged as an open question when this
was discussed, not yet answered.

**Known, explicitly-flagged feasibility risk (not yet resolved):** this whole design adds
a *third* model to the pipeline (fast detector + CLIP embedding encoder + open-vocab
detector). Grounding DINO/OWLv2 are transformer vision-language models, meaningfully
heavier than `yolov8n.onnx` even at their smallest checkpoints -- multi-second per-image
latency is typical on CPU-only inference. Everything in this session so far (ONNX
YOLOv8n, ffmpeg, go2rtc) has been running on this Mac's CPU with no GPU mentioned
anywhere -- **before building this, check whether a CPU-runnable/ONNX-exportable OWLv2 or
Grounding DINO checkpoint gets acceptable-enough latency to be useful as enrichment**
(a few seconds of delay is fine; tens of seconds with events queueing up faster than
that is not, and would need explicit throttling/sampling akin to DigiNetra's
"Scheduler" component). This feasibility check was proposed but not yet done when the
user said to defer this whole item.

**Relevant files if picking this up:**
- `mirage/tracking/orchestration.py` -- where the new gate-1/gate-2 dispatch logic would
  slot in, right after the existing `_update_lifecycles` call
- `mirage/tracking/tracker.py` (`TrackedObjectState`) -- would need a new
  `last_openvocab_embedding` (or similar) field, same cross-process pattern as
  `is_false_positive`
- `mirage/detection/remote.py` -- existing pattern for isolating a detector as its own
  process/service, to mirror for the new open-vocab detector
- `mirage/config/schema.py` (`ObjectsConfig`) -- where non-COCO `track_objects` entries
  would need to be recognized/routed as open-vocab prompts instead of hitting the dead
  membership-check path
- `mirage/db/models.py` (`Event.data`, `ReviewSegment.severity`/`data`) -- both already
  flexible enough (JSON blobs) to hold the new enrichment/crowd-condition data without a
  migration

### Implementation (built, not just designed)

Built as "saved text queries" rather than routing through `track_objects` directly --
functionally the same idea (open-vocab matching beyond COCO's fixed classes), shipped as
its own first-class feature (a Queries page) since that's a clearer UX than overloading
`track_objects` with non-COCO strings. Actual design ended up slightly different from
the original sketch above in a few ways, noted below.

**What was built, end to end:**
- `mirage/config/schema.py`: `OpenVocabQuery` (id, text, cameras scope, enabled) +
  `MirageConfig.queries: list[OpenVocabQuery]`, with a cross-field validator rejecting a
  query scoped to an unknown camera name.
- `mirage/db/models.py`: new `QueryMatch` table (NOT folded into `Event.data` as
  originally sketched -- a dedicated table was chosen instead so matches are
  independently queryable/listable/filterable by query/camera/time without JSON-scanning
  every event row, which is what the Queries page's match list actually needs).
- `mirage/openvocab/` (new package):
  - `process.py` -- `OpenVocabProcess`, the dedicated OS process loading OWLv2
    (`google/owlv2-base-patch16-ensemble`) exactly once via `transformers`/`torch`, NOT
    reusing `DetectorProcess`'s SHM+ZMQ IPC (that's tuned for millisecond-scale,
    fixed-shape closed-vocab calls) -- instead a plain `multiprocessing.Queue` carrying
    small JPEG-encoded crops, since OWLv2 calls are multi-second and infrequent.
  - `device.py` -- `resolve_device`/`available_devices`, the torch-specific analog of
    `mirage/detection/execution_providers.py` (separate because torch's device API
    differs from onnxruntime's `providers=[...]` list). `auto` prefers cuda, then mps,
    then cpu.
  - `gating.py` -- `OpenVocabGate`: Gate 1 (track confirmed, i.e.
    `TrackedObjectState.is_false_positive is False`) is checked by the caller; Gate 2
    (dedup) uses a **from-scratch average-hash** (NOT a CLIP embedding encoder as
    originally sketched -- agreed with the user in this session to use a cheap
    perceptual hash instead, avoiding a third heavy model) comparing each object's
    current crop against the hash last actually sent to OWLv2 for that object id, plus a
    periodic forced-recheck interval.
  - `dispatcher.py` -- `OpenVocabDispatcher`, living in the main process's
    `_result_consumer_loop` (mirage/app.py) alongside `EventProcessor`/
    `ReviewSegmentMaintainer`. Crops a confirmed object's box out of the SHM frame ring
    and JPEG-encodes it **synchronously, immediately** (not deferred) since the ring is
    finite-depth and would otherwise be overwritten before a multi-second OWLv2 call
    returns. Drains completed results and writes real matches as `QueryMatch` rows.
- `mirage/app.py`: `_start_openvocab()` only spawns `OpenVocabProcess` if at least one
  enabled query exists (no point loading a ~600MB model otherwise); wired into the
  startup sequence, result-consumer loop, and graceful shutdown.
- API (`mirage/api/routers/config.py`): full `GET/POST/PUT/DELETE /api/config/queries`
  CRUD, same shape/validation pattern as the existing detector endpoints.
- API (`mirage/api/routers/query_matches.py`, new): `GET /api/query-matches` (filterable
  by camera/query_id, most-recent-first) + `GET /api/query-matches/{id}`.
- Frontend: new Queries page (`frontend/src/app/pages/queries/`) -- add/edit/delete saved
  queries with a per-camera checkbox scope picker, plus a live "Recent matches" list
  below it. New nav entry.
- Dependencies added: `torch==2.12.*`, `transformers==5.13.*` in `requirements.txt` (were
  previously only installed ad hoc for the item 6 benchmark spike, now a real pinned
  dependency since the feature is real code).

**Real bug caught and fixed during this build (not just "wrote code, moved on"):** OWLv2's
regressed box coordinates can slightly overshoot the crop's own pixel bounds (normal
detector regression imprecision) -- a genuine e2e test against the real model on
`media/bus.jpg` caught this (`x1 <= x2 <= width` assertion failed by ~6px), fixed by
clamping to the crop's own size in `process.py`'s `_process_request` before offsetting
into full-frame coordinates, mirroring the same clamp `mirage/tracking/orchestration.py`
already does for the closed-vocab detector's boxes.

**Real, live verification performed (not just unit tests):** registered a real query ("a
person", scoped to the `front_door` mock camera) via the live API, restarted
`python -m mirage`, confirmed in the real log output: `openvocab: started (device=mps), 1
enabled query`, model loaded in 5.3s, and -- watching real mock-camera footage (a person
walking) -- a real match was logged and persisted: `openvocab: match found -- camera=
front_door object=... query='a person' score=0.311`. Confirmed via
`GET /api/query-matches` and a live screenshot of the Queries page showing the real saved
query and its real matches (score 0.34, 0.31) with correct camera/timestamp.

**Test coverage:** `tests/test_openvocab_gating.py` (10 tests, pure gating logic),
`tests/test_openvocab_dispatcher.py` (8 tests, real SHM ring + fake queues, including a
crop-offset correctness test), `tests/test_openvocab_process_e2e.py` (2 tests, the REAL
OWLv2 model against `media/bus.jpg`, skipped if `transformers` isn't installed),
`tests/test_api_config.py` (9 new query CRUD tests), `tests/test_api_query_matches.py` (7
tests). Full suite: 342 passed.

**Thumbnails for matches -- IMPLEMENTED (was the one gap flagged right after the initial
build; closed same session once the user reported clicking a match showed no photo):**
`OpenVocabResult` now echoes the exact crop bytes (`crop_jpeg: bytes`) back from
`OpenVocabProcess`/`_process_request`, so `OpenVocabDispatcher._save_thumb` writes the
real crop OWLv2 actually checked to `QUERY_MATCH_THUMB_DIR` (no longer a no-op stub).
New `GET /api/query-matches/{id}/thumbnail` (mirrors the existing event-snapshot
FileResponse pattern in `mirage/api/routers/events.py`). Frontend: match rows in the
Queries page now show a real 48x48 thumbnail preview and are clickable (reusing the
existing shared `Lightbox` component, same pattern as the Events page) to view the full
crop with a caption (query text / camera / time / score). Verified live: a real match's
thumbnail file confirmed non-empty JPEG on disk, fetched via the new endpoint (200,
`image/jpeg`, real decodable image showing an actual person from `front_door` footage),
and the lightbox click-to-view flow screenshotted end-to-end in a real browser. New
tests: 2 more in `test_openvocab_dispatcher.py` (real thumb write + the empty-crop
defensive case), 4 more in `test_api_query_matches.py` (no-thumb 404, unknown-match 404,
missing-file-on-disk 410, real-bytes 200). Full suite: 347 passed.

One real bug hit and fixed during this: the new `/thumbnail` route returned a generic
404 at first even though the DB row and file were both genuinely correct -- turned out
to be the already-running `mirage.api` process simply not having reloaded the new route
(same "restart required" pattern as every other config/code change in this app, see item
2 -- not a code bug, just needed `python -m mirage.api` restarted to pick up the new
endpoint).

**Not yet built / explicitly out of scope:**
- Crowd detection (the other half of this item) -- still not built, see the original
  design notes above this Implementation section.
- CoreML/ANE acceleration for OWLv2 (still MPS-only per item 6's findings -- GPU cores,
  not the Neural Engine).
- Real CUDA testing (device resolution supports it, `resolve_device`/`available_devices`
  handle it, but never exercised against a real NVIDIA GPU -- relevant once the user's
  planned move to dedicated hardware happens, see the "Deployment plan update" note
  further down this file).

---

## 5. (Feature, deferred by user) MQADet-style saved text-query matching ("find this
   description across any camera")

**Status:** open, deferred -- user said "add it to todo," not yet scheduled. Heavier
variant of item 4 above (open-vocab enrichment) -- read that entry first, this builds on
the same trigger/gating design.

**User's request (this conversation, verbatim intent):** let the user save a list of
free-text descriptions (e.g. "person carrying a red backpack", "a dog off-leash"),
optionally scoped to specific cameras, and have the system flag a "match found" whenever
any camera's feed matches one of the saved queries -- referencing the MQADet paper
(arXiv:2502.16486, "MQADet: A Plug-and-Play Paradigm for Enhancing Open-Vocabulary Object
Detection via Multimodal Question Answering") as the architecture to base this on.

**What MQADet actually is (confirmed by reading the real paper via PMC12905182, not
guessed):** a training-free, plug-and-play 3-stage pipeline sitting on top of an
open-vocabulary detector:
1. **TASE** (Text-Aware Subject Extraction) -- an MLLM (LLaVA-1.5-7B or GPT-4o in the
   paper) parses a complex free-text query (e.g. "teddy bear with checkered design on one
   foot and bumble bee design on the other") into a structured subject list.
2. **TMOP** (Text-Guided Multimodal Object Positioning) -- feeds those subjects into an
   open-vocab detector (the paper evaluates Grounding DINO, YOLO-World, and OmDet-Turbo --
   detector-agnostic) to generate candidate bounding boxes, each numbered and rendered
   onto a "marked image."
3. **MOOS** (MLLMs-driven Optimal Object Selection) -- the MLLM reasons over the marked
   image + original query a second time to pick which candidate box, if any, actually
   matches the description.

**Real measured cost (from the paper, RTX 4090, NOT this Mac's hardware):** ~1000.9ms
total per image with the cheaper LLaVA-1.5-7B config (TASE 87.6ms + TMOP 34.7ms + MOOS
878.6ms -- the final MLLM reasoning step dominates). Heavier MLLM configs are far worse:
Qwen2-VL-2B measured 1951.7ms/image, Gemini-2.0-Flash-Lite measured 7364.9ms/image. Input
queries in the paper's benchmarks average 8.4-24.2 words (RefCOCOg / Ref-L4), so this is
built for genuinely compositional, multi-attribute descriptions, not single keywords.

**Why this can't be "watch every camera continuously" (the literal ask), and what the
realistic scope is instead:** ~1+ second per single query per single image, on a discrete
NVIDIA GPU, is 2-3 orders of magnitude too slow for continuous/per-frame monitoring across
multiple cameras -- and critically, **this Mac has no CUDA path at all** (confirmed
earlier this session via `onnxruntime.get_available_providers()`, see item 4's CoreML
work), so realistic latency here would be meaingfully worse than the paper's own
already-slow baseline; a 7B-parameter MLLM on CPU/CoreML could plausibly take tens of
seconds per query. The only realistic integration is the same gated-enrichment design
already sketched in item 4, NOT a live watcher:
- Gate 1 (track confirmed) + Gate 2 (embedding-diff dedup) from item 4 still apply --
  don't run MQADet on every motion event, only on tracks the fast YOLOv8n detector has
  already confirmed as real and meaningfully changed since the last check.
- Saved queries are the TASE input; matches get written back onto the same `Event`/
  `ReviewSegment` row as a `data["query_match"]` (or similar) field -- `Event.data` is
  already a flexible JSON blob (`mirage/db/models.py`), no schema migration needed, same
  as item 4's enrichment write-back.
- Per-camera query scoping maps onto camera config the same shape as `track_objects`
  today -- e.g. a new `queries: list[str]` field on `ObjectsConfig` or a sibling config
  section.
- Expect a real, honest lag between an event happening and a "match found" surfacing --
  seconds to low tens-of-seconds per checked event on this hardware, not real-time. This
  needs to be communicated in the UI (e.g. "checking..." state on a review card) rather
  than presented as instant.

**Two heavy models chained, not one:** unlike item 4 (a single open-vocab detector), this
needs BOTH an open-vocab detector AND a full multimodal LLM (multi-GB weights) at
inference time -- meaningfully bigger scope/resource footprint than item 4. Worth
prototyping item 4 first and only building this on top once that's proven feasible on
real hardware, per item 4's own build-order note.

**Relevant references:**
- Paper: https://arxiv.org/abs/2502.16486 / https://pmc.ncbi.nlm.nih.gov/articles/PMC12905182/
- Item 4 above (this file) -- shares the trigger-gating design and detector-agnostic
  TMOP stage overlaps directly with item 4's own open-vocab detector plan
- Local MLLM options worth investigating if this is picked up: something CoreML/MPS-
  runnable on Apple Silicon (this repo's dev machine has no CUDA), e.g. a quantized
  LLaVA/Qwen-VL variant -- not yet researched, flagged as a real open question before
  starting implementation

---

## 6. Real, measured OWLv2-on-Apple-Silicon benchmark -- informs items 4 and 5's detector choice

**Status:** research done, real numbers captured, no code built yet. This directly
updates item 4 ("which open-vocab detector to build a plugin for") and item 5 (MQADet's
TMOP stage, which is detector-agnostic and could use OWLv2).

**What was tested:** whether OWLv2 (a real open-vocabulary detector candidate, alongside
Grounding DINO) can run on this Mac's Apple Silicon GPU via PyTorch's MPS backend, and
how much that actually helps versus CPU -- measured directly on this machine, not taken
from the OWLv2 paper or someone else's hardware.

**Setup:** `google/owlv2-base-patch16-ensemble` (OWLv2's smallest HuggingFace checkpoint,
~607MB cached), loaded via `transformers` (newly installed into `mirage/.venv` for this
test -- `pip install transformers`, now present at v5.13.0; NOT yet added to
`requirements.txt`, since no plugin consumes it yet). PyTorch 2.12.1 in this venv already
had MPS support built in and available (`torch.backends.mps.is_built()` /
`.is_available()` both `True` with zero extra setup). Test image: `media/bus.jpg` (the
same fixture `tests/test_onnx_yolov8_e2e.py` already uses for the YOLOv8n baseline).
Queries: `["a bus", "a person"]`. 5 timed runs per device after a warm-up run; `mps`
runs used `torch.mps.synchronize()` before stopping the clock (GPU work is async
otherwise and would under-measure).

**Real results:**

| Device | Avg latency (5 runs) | Min | Max |
|---|---|---|---|
| CPU | 1720.7 ms | 1674.9 ms | 1812.0 ms |
| MPS (Apple GPU) | 1423.8 ms | 1345.8 ms | 1686.5 ms |

**Speedup: ~1.21x.** Both devices correctly detected the same objects at the same
confidence (bus: 0.552; 3 people: 0.133-0.143) -- MPS is numerically correct, not just
faster-and-wrong.

**How this compares to what's already known:**
- This is a MUCH smaller speedup than YOLOv8n's CoreML result measured earlier in item 4
  (~5.6x: 9.90ms CPU vs 1.75ms CoreML on the same machine). Reasoning: CoreML can target
  the Apple Neural Engine (ANE) as well as the GPU, and `onnxruntime`'s
  `CoreMLExecutionProvider` gets that dispatch for free. `torch.device("mps")` only
  targets the GPU cores -- **it does NOT use the ANE at all**. A proper CoreML export of
  OWLv2 (via `coremltools`, converting the PyTorch model rather than just moving tensors
  to an `mps` device) might do meaningfully better by reaching the ANE, but that's a
  separate, nontrivial conversion step for a transformer this size -- NOT yet attempted,
  flagged as the next thing to try if pursuing this further.
- ~1.4-1.7 seconds per single image/query on this Mac is in the same ballpark as the
  MQADet paper's own RTX 4090 numbers (item 5's TMOP-equivalent stage would be one part
  of that pipeline, not the whole ~1s budget) -- i.e. Apple Silicon MPS is NOT a shortcut
  around the "this is slow, gate it, don't run it live" conclusion already reached in
  items 4 and 5. The gated-enrichment design (track-confirmed + embedding-diff triggers,
  not per-frame) remains the only realistic integration path on this hardware.

**Practical conclusion for item 4's "which detector" question:** OWLv2's simpler,
single-encoder architecture (ViT + detection head, vs. Grounding DINO's separate
Swin-backbone + frozen-BERT + cross-modal-fusion design) is still the better starting
point for a first plugin attempt, per the earlier reasoning -- this benchmark doesn't
change that recommendation, it just confirms OWLv2 is real, loadable, and correctly
functional on this exact machine via a well-trodden path (`transformers` + MPS), which
lowers the risk of picking it as the first open-vocab plugin to build.

**Cleanup note:** the downloaded OWLv2 checkpoint remains cached at
`~/.cache/huggingface` (~607MB) from this test -- harmless to leave (HuggingFace's
standard cache location, reused automatically if this work is picked up again) or safe
to delete if reclaiming disk space matters before then.

**Deployment plan update (user, this conversation):** the target deployment will later
move to a system with dedicated hardware (presumably a real GPU, likely CUDA-capable
given "dedicated hardware" is being contrasted with this Mac's CPU/MPS-only setup) --
higher inference latency on this current dev Mac is acceptable/expected for now and
should NOT be treated as a blocker for building items 4/5/6's open-vocab and MQADet
features. Practical implications for whoever picks this up:
- Don't gate the decision to start building the open-vocab plugin (item 4) or MQADet
  integration (item 5) on hitting a specific latency target on THIS machine -- the
  ~1-1.7s/query numbers measured here (item 6) are acceptable for current
  development/testing purposes, not a hard ceiling to optimize against.
- Do still build the execution-provider abstraction generally (same pattern as
  `mirage/detection/execution_providers.py` already does for the YOLOv8n plugin) so that
  once real dedicated hardware (likely CUDA) is available, switching is a config change
  (`execution_provider: cuda`), not a rewrite -- this is already how the existing
  detector plugin/execution-provider system is designed, so no new architecture is
  needed for this to just work, only testing against the real hardware once it exists.
- The gated-enrichment design (track-confirmed + embedding-diff triggers, not per-frame)
  from items 4/5 is still the right architecture regardless of hardware -- even on fast
  dedicated hardware, there's no reason to run an expensive open-vocab/MLLM pass on every
  frame of every camera when the fast YOLOv8n detector can filter first. Hardware speed
  changes how tolerable the LATENCY is, not whether the gating design is needed.
