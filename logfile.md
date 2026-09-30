# Session log — 2026-09-25: recovery, live-view fix, Detections page, NVDEC + SpeciesNet GPU

Continuation of the session below (GPU/TensorRT notes) plus everything else from a long
follow-up session. Grouped by area; each item verified live against the running stack,
not assumed.

## Repo recovery (start of session)

- Picked up from `Mirage Handoff.html` (crashed prior session, OneDrive Files-On-Demand
  choked under 6-camera + inference + dual-dev-server load). Working repo was already
  relocated to `C:\Users\gauth\Documents\projects\miragev3` (outside OneDrive) via a
  fresh `git clone`, but the entire `frontend/src/app/{core,pages,shared,shell}` subtree
  (~40 files, including all of the prior session's live-view fixes) was missing from
  disk. Recovered by copying from the old OneDrive path, which had rehydrated in the
  meantime — verified byte-for-byte against expected content (e.g. `camera-tile.ts`'s
  `refreshOnReactivation`) before trusting it.
- `.venv` and `frontend/node_modules` were both silently corrupted (same
  partial-hydration pattern as the source recovery — `pydantic._internal`,
  `uvicorn.middleware`, and `@angular/cli`'s own bin/ were all missing despite the
  packages "being installed"). Fixed via `uv pip install --reinstall -r
  requirements-windows-gpu.txt` and `npm ci` respectively.

## Live-view bug: tiles going black and reconnecting in lockstep every ~10s

Root cause: `CameraTile`'s `LiveActivation` effect (constructor, `camera-tile.ts`) read
`this.mode()` *inside* `refreshOnReactivation()`, called from within the effect body —
Angular tracks any signal read during an effect's execution as a dependency of that
effect, even transitively. Since `refreshOnReactivation()` → `startVideoTiers()`
eventually changes `mode()` itself once a new connection lands, that change re-triggered
the *same* effect, forcing another full reconnect, forever — a self-sustaining loop with
no real reactivation ever happening. Proved via temporary logging: the effect fired
repeatedly with an *identical* `activatedAt` value, showing `mode()` was driving it, not
a genuine tab/route reactivation.

**Fix**: wrapped `refreshOnReactivation()` in `untracked()` so its internal signal reads
don't leak into the effect's own dependency set. Confirmed fixed live.

Also fixed, while chasing an earlier (real but separate) symptom on the way to this one:
`MsePlayer.destroy()` (`mse-player.ts`) revoked its blob URL but never cleared
`videoEl.src`, leaving a stale reference that the browser would later fail to reload
(`net::ERR_FILE_NOT_FOUND` on a `blob:` URL) — same class of bug `WebRtcPlayer.destroy()`
already guarded against for `srcObject`. Also added `SourceBuffer`/`MediaSource` error
listeners and a `readyState` check in `appendChunk`'s catch, since a decode error was
previously swallowed silently (WS stays open, so the old code's only error path never
fired) — this used to leave a tile permanently black with no recovery.

## New page: Detections (`frontend/src/app/pages/detections/`)

Flat, chronological list of every distinct sighting (one row per `Event` — no
cross-object suppression, so N same-species animals in one long scene each get their own
row), not grouped by scene per user preference. Each row: boxed snapshot (click for
full-size lightbox), label + species badge, camera/confidence/timestamp (explicitly
labeled), a clip-play button, and — when the sighting resolved to a review "scene" — an
inline severity chip plus a direct scene-clip-play button (no intermediate panel/click
just to see metadata).

Backend: two new, purely additive endpoints — `GET /api/review/{id}/events` (events
overlapping a scene's time window) and `GET /api/events/scenes?ids=...` (batched reverse
lookup: which scene, if any, each of a page of events belongs to) — both derived via
time-range joins over existing data, no schema changes. `review-page.ts`/`events-page.ts`
and their own routes are untouched.

## Diagnostic logging added while investigating a missed porcupine detection (park3)

**Still active in the codebase as of this entry — all clearly marked `TEMP diagnostic`
in comments, gated behind `label == "animal"` and/or `logger.isEnabledFor(DEBUG)` to
keep cost near-zero outside `-v`, but worth removing once no longer needed:**
- `mirage/tracking/orchestration.py` — Gate 0 (per-camera `min_score`) rejections, Gate 1
  (sticky `threshold`) confirmations/still-unconfirmed/evicted-while-unconfirmed status,
  and per-frame motion/region coverage (throttled).
- `mirage/detection/plugins/onnx_yolov8.py` — near-miss sub-threshold candidates (the
  model's own hard 0.4 floor is otherwise a black box; this surfaces what scored just
  below it, e.g. a real frame with five separate `animal=0.20–0.39` candidates that
  would otherwise be invisible).
- `mirage/detection/process.py` — one line correlating detector-plugin log lines back to
  a camera name (the plugin itself has no camera context, shared across every camera
  routed to that detector).

**Finding**: `park1` is the only camera with no explicit `objects.filters.animal`
override, so it silently falls back to the strictest defaults in the fleet
(`min_score=0.5, threshold=0.7` vs. siblings' `0.4/0.4`) — confirmed live via repeated
Gate 0 rejections in the 0.40–0.499 range that would pass cleanly on `park2`–`park4`.
**Not yet fixed** — real, cheap fix (align `park1`'s filter with its siblings, or
lower `threshold`), deliberately deferred by user request ("we will look into this
later").

## Hardware decode (NVDEC) — confirmed working fleet-wide

`-hwaccel cuda` (via `CameraInputConfig.hwaccel_args`, no UI/API exposure yet — set
directly via `MirageConfig.from_db()`/`save_to_db()`) works cleanly on all 6 cameras
simultaneously. An earlier "only 0-1 of 6 actually get NVDEC" finding was **wrong** — it
was a verification bug, not a real hardware/driver limit: `nvidia-smi`'s process list
only reliably attributes CUDA *compute* ("C") contexts, not pure NVDEC decode-engine
usage, especially across several simultaneous short-lived Windows sessions. Full verbose
ffmpeg output (`-v verbose`) is the trustworthy check — confirmed `NVDEC capabilities...
format supported: yes` + `pix_fmt: cuda` + 0 decode errors on all 6 at once, and
sustained nonzero GPU decoder utilization while the real pipeline runs. Matches public
info: NVDEC has no artificial session cap on GeForce, unlike NVENC.

## SpeciesNet GPU — confirmed working, three real bugs found and fixed

Set up `.venv-speciesnet` (Windows: `python -m venv .venv-speciesnet`, per
`LINUX_INSTALL.md`'s documented Linux path — no Windows path existed before this). GPU
support required force-reinstalling PyTorch against a CUDA index after the plain
`pip install speciesnet` (which pulls CPU-only torch by default on Windows):
```
.venv-speciesnet/Scripts/pip install torch torchvision --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126
```
Verified via `python -m speciesnet.scripts.gpu_test` (CUDA 12.6, cuDNN, RTX 3050
detected) before wiring into mirage. Model auto-downloads anonymously via kagglehub
(~214MB, no Kaggle auth needed), cached under `~/.cache/kagglehub` afterward.

Three separate real bugs found along the way, all fixed in
`mirage/species/plugins/speciesnet.py`:
1. **Windows relative-path bug**: `subprocess.Popen([venv_python_path, ...])` with a
   relative path raises `FileNotFoundError: [WinError 2]` via `_winapi.CreateProcess`,
   even when `os.path.exists()` on that exact path (same cwd) returns `True` — Windows
   does not resolve a relative `argv[0]` against cwd the way POSIX `execve` does. Fixed
   by resolving `venv_python_path` via `os.path.abspath()` before use (a no-op, and
   correct, on POSIX too — no config changes needed for existing setups).
2. **stderr PIPE deadlock risk**: same class of bug already documented and fixed once in
   this codebase (`mirage/util/proc.py`'s own docstring, a *different* mechanism but the
   same "don't PIPE a stream nothing drains in real time" lesson) — `stderr=PIPE` while
   only ever actively reading `stdout` risks the child blocking on a full stderr pipe
   buffer while stdout never arrives either. Fixed by redirecting stderr to a real
   `tempfile.TemporaryFile()` instead (stdin/stdout stay as pipes — needed live for the
   per-request JSON protocol; stderr's contents are only ever needed after the fact, on
   failure).
3. **Startup-ordering race — the actual root cause of an intermittent native crash**:
   `SpeciesProcess` used to start right after `_start_detectors()`, both cold-starting
   *separate* CUDA contexts (onnxruntime and PyTorch) at nearly the same moment. This
   reliably crashed with a native Windows `STATUS_THREADPOOL_HANDLE_EXCEPTION`
   (`0xC000070A`, decoded from the raw `returncode`) — empty stderr, no Python
   traceback, no Windows Error Reporting record (confirmed via `Get-WinEvent`), i.e. a
   genuine low-level driver-level exception, not anything catchable in Python. **This is
   not a "GPU can't run two models" limitation** — constructing the exact same classifier
   against an *already-warm* pipeline (detector fully loaded, cameras already running)
   succeeded cleanly every time in isolation. Fixed by moving `_start_species_worker()`
   to run last in `MirageApp.start()`, after detectors/recording/cameras are all already
   launched, giving the detector's own CUDA cold start time to finish before PyTorch's.
   Confirmed fixed live, repeatedly, with `-hwaccel cuda` also active on all 6 cameras —
   real species classifications now complete end-to-end (e.g. `tiger` 99.97%,
   `white-tailed deer` 98.8%, `wild boar` 95.2%, full taxonomy attached to the Event).

Final confirmed-working state: `nvidia-smi`'s process list shows exactly 2 CUDA compute
contexts (the `zilo` detector and the SpeciesNet worker) coexisting cleanly, 68% VRAM
still free.

## Still open / not addressed this session

- **`homecam` RTSP instability** (`Error number -10054`, connection reset by the remote
  side, roughly every ~10s) — real, camera/network-side, not caused by or fixed in this
  session, deliberately deferred (external, needs the camera's own admin UI or network
  investigation, not a code fix).
- **`park1`'s missing `animal` filter override** — identified above, not yet applied.
- **Test suite** — `pytest tests/ -q` was never re-run this session (carried over as an
  open item from the original handoff doc, predating this session's own work).
- **TEMP diagnostic logging** (see above) — still active, should be removed once no
  longer needed for the `park1`/porcupine-style investigation.
- **RAM headroom is tight**: 93.4% system RAM used (806 MB free of 12.2 GB) with the
  full stack running (6 cameras, hwaccel decode, GPU detection, GPU species
  classification, plus the usual desktop environment). Not caused by anything broken —
  just the real cost of everything running at once on a 12GB box — but worth watching;
  a memory leak or an added process could tip this into real trouble sooner than CPU/GPU
  would.

# GPU acceleration status & TensorRT follow-up

Working notes from getting CUDA/TensorRT wired up for the RTX 3050. Everything here was
verified directly against this machine's `.venv`, not assumed.

## Current status (as of this session)

- **Package swapped**: `.venv` now has `onnxruntime-gpu==1.22.0` (was the CPU-only
  `onnxruntime==1.22.1`). Matches `requirements-gpu.txt`'s pin. The two packages import
  as the same `onnxruntime` module and conflict if both installed — never both at once.
- **The "Acceleration" dropdown on Manage Detectors now offers NVIDIA GPU** —
  `GET /api/config/execution-providers` returns `["cpu","auto","cuda"]` (was
  `["cpu","auto"]` before the swap). This list comes from
  `mirage/detection/execution_providers.py`'s `available_execution_providers()`, which
  is driven by `onnxruntime.get_available_providers()`.
- **Selecting it initially did NOT actually run on GPU** — caught by testing a real
  `InferenceSession(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`
  against the shipped `models/yolov8n.onnx`, not by trusting
  `get_available_providers()`:
  ```
  [E:onnxruntime ...] Error loading "...\onnxruntime_providers_cuda.dll" which depends
  on "cublasLt64_12.dll" which is missing. (Error 126)
  [W:onnxruntime ...] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and
  CUDA 12.*, and the latest MSVC runtime.
  ```
  It silently fell back to `CPUExecutionProvider` and still ran correctly — no crash,
  just no acceleration. **The gotcha this exposed**:
  `ort.get_available_providers()` reports what the *wheel was compiled with support
  for*, not what will actually work on this machine — the only trustworthy check is a
  real session + `sess.get_providers()` after the fact.
  **Update: this is now fixed** — see "CUDA now genuinely works" below. The dropdown's
  own docstring claim ("only ever offer an accelerator choice that will really work")
  is accurate again now that the runtime deps are actually in place.

## CUDA now genuinely works — confirmed, with a caveat

Installed the pip CUDA runtime wheels into `.venv` (the lighter of the two options
below), matching `requirements-windows-gpu.txt`:
```
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cublas-cu12==12.9.2.10
nvidia-cudnn-cu12==9.26.0.51
nvidia-cufft-cu12==11.4.1.4       # not obvious up front -- onnxruntime's CUDA EP needs
                                   # this too; only surfaced as a second missing-DLL
                                   # error after cublas was fixed, see the discovery
                                   # trail below
nvidia-cuda-nvrtc-cu12==12.9.86   # transitive
nvidia-nvjitlink-cu12==12.9.86    # transitive
```

**Verified for real** — not just `get_available_providers()` (still unreliable, see
above) but an actual `InferenceSession(providers=["CUDAExecutionProvider", ...])`
against `models/yolov8n.onnx`, with the six `nvidia/*/bin` directories added to PATH:
```
session providers actually in use: ['CUDAExecutionProvider', 'CPUExecutionProvider']
inference OK, output shapes: [(1, 84, 2100)]
```
Genuinely running on the RTX 3050 now, confirmed by the provider list onnxruntime
reports back *after* session creation (the only trustworthy check — see above).

**Discovery trail**, since it wasn't a single clean install:
1. `onnxruntime-gpu` alone → `CUDAExecutionProvider` failed to load,
   `cublasLt64_12.dll` missing.
2. Installed `nvidia-cuda-runtime-cu12` + `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` →
   different error, `cufft64_11.dll` missing (cuFFT isn't obviously "needed for object
   detection," but onnxruntime's CUDA EP links it regardless of whether the specific
   model's ops use FFT).
3. Installed `nvidia-cufft-cu12` → works.

This is exactly the kind of iterative discovery `packaging/README.md` already warned
to expect for PyInstaller-frozen ML packages — same shape of problem, just hit during
plain `.venv` install instead of a frozen build this time.

**The caveat — PATH is not handled automatically:**
Pip installing these wheels does NOT put their DLLs on PATH. They land under
`.venv\Lib\site-packages\nvidia\<component>\bin\`, and nothing in Windows or these
packages adds that to the process's DLL search path automatically. Without it, the
exact same `cublasLt64_12.dll`-missing error above reappears even with everything
installed — verified this directly (first test attempt used a Git-Bash-mangled PATH
and failed identically to before installing anything, second attempt via PowerShell
with the directories explicitly prepended succeeded).

**Follow-up, not done yet**: wire this into mirage's own startup instead of requiring
it be set manually before every run. Two options:
- Simplest: prepend the six directories to `os.environ["PATH"]` early in
  `mirage/__main__.py` / `mirage/api/__main__.py` (or a small shared helper both call),
  guarded to Windows only (`sys.platform == "win32"`), resolving the paths relative to
  `sys.prefix`/the venv rather than hardcoding.
  - Windows 3.8+, but not currently used anywhere else in this codebase.
- Whichever is chosen, `mirage/desktop/paths.py`'s `configure_environment()` (the
  packaged-app launcher) needs the same treatment — same gap applies there, and a
  client machine has even less chance of noticing PATH is the missing piece than a dev
  running this file's own list manually.

Requirements file: `requirements-windows-gpu.txt` (repo root) — full self-contained
list (base deps + `onnxruntime-gpu` + the CUDA runtime wheels above), mirroring
`requirements-gpu.txt`'s existing convention of being a complete alternative rather
than a diff to layer on top.

## TensorRT specifically

`ort.get_available_providers()` already listed `TensorrtExecutionProvider` right after
the plain `onnxruntime-gpu` swap (before CUDA was even confirmed working) — the same
wheel has both compiled in. So **no separate pip package is needed for TensorRT** vs.
plain CUDA — but two more things are needed beyond what CUDA itself needs:

1. **NVIDIA TensorRT SDK**, a separate, heavier native install from CUDA/cuDNN.
   Historically required an NVIDIA developer account to download; newer TensorRT
   versions also have `pip install tensorrt` wheels for some CUDA versions — check
   what's compatible with CUDA 12.x + onnxruntime 1.22 specifically before choosing.
   **Version matching between installed TensorRT and what this onnxruntime-gpu build
   expects is the single most common failure point in practice** — a mismatch
   typically means `TensorrtExecutionProvider` just doesn't work at session-creation
   time (same silent-fallback-to-CPU behavior seen above with CUDA), not a clear error
   naming the mismatch.
2. **Code changes in this repo** — TensorRT is not wired up at all today:
   - `mirage/config/schema.py`'s `ExecutionProvider` enum has only `auto`, `cpu`,
     `coreml`, `cuda`. Needs a `tensorrt = "tensorrt"` value added.
   - `mirage/detection/execution_providers.py`'s `_PROVIDER_NAMES` dict maps `cuda` and
     `coreml` only. Needs `ExecutionProvider.tensorrt: "TensorrtExecutionProvider"`
     added, and `resolve_providers()`'s `auto` branch's preference order updated
     (TensorRT should likely be preferred over plain CUDA when both are available,
     given it's faster on Ampere — see the batching/GPU-performance discussion this
     followed from).
   - **FP16 is a provider option, not a separate package or model requirement**:
     `InferenceSession(..., providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider", "CPUExecutionProvider"])`
     — the tuple form passing provider-specific options. This repo's detector plugins
     (`mirage/detection/plugins/onnx_yolov8.py`,
     `mirage/detection/plugins/onnx_megadetector.py`) currently call
     `resolve_providers()` and pass a plain list of provider name strings with no
     options dict — would need extending to support passing per-provider options
     through, at minimum a `trt_fp16_enable` flag sourced from `ModelConfig`.
   - **TensorRT's first-run engine-build cost**: unlike CUDA, TensorRT compiles an
     optimized engine for the exact model+shape+precision combination the first time a
     session is created with a given cache directory, then reuses it on later runs
     (via `trt_engine_cache_enable` + `trt_engine_cache_path` provider options — also
     not currently wired). Worth setting an explicit cache path under `CACHE_DIR`
     rather than letting it default, both for packaging (see
     `packaging/README.md`) and so this rebuild cost isn't silently paid on every
     detector process restart.

## Suggested order of work

1. Get plain CUDA verified working first (the runtime-DLL gap above) — this alone was
   already identified as most of the win in the earlier batching/GPU-performance
   discussion, and it's a prerequisite for TensorRT's own CUDA dependency anyway.
   2. Batch inference across cameras (`mirage/detection/process.py` — see that
      discussion; still the single highest-leverage change, independent of TensorRT).
3. FP16 on plain CUDA, if the CUDA EP supports it adequately for this model — cheaper
   than TensorRT's engine-build complexity, worth measuring before reaching for step 4.
4. TensorRT + FP16, only if still compute-bound after 1-3, given the SDK
   install/version-matching cost and the code changes listed above.
