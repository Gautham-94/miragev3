# Process audit — mirage stack

A point-in-time snapshot of every OS process the mirage stack spawns when running with
3 active cameras (`front_door`, `garage` via mock_cameras; `hikvision_ds_2cd1023g0e_i`
real), 1 enabled detector (`megadetector` at 320x320), 2 enabled open-vocab queries, and
the species classifier enabled (`speciesnet` backend). Captured to answer "what's
actually running and why," not a permanent architecture doc -- re-run the audit
commands below if the process/config mix changes.

Total at capture time: **23 processes, ~434 MB RSS combined**.

## How to reproduce this audit

```
ps aux | grep -iE "mirage|mock_camera|go2rtc|ng serve|esbuild" | grep -v grep
```

Multiprocessing children spawned via Python's `multiprocessing` module show up as
generic `spawn_main(...)` command lines, not descriptive names (mirage does not call
`setproctitle` for these) -- map each PID to its real role via `ps -o pid,ppid,command`
against the parent `-m mirage` PID, cross-referenced with spawn order in
`MirageApp.start()` (`mirage/app.py`): `mp.Manager()` (in `__init__`, before `start()`)
-> `_start_detectors()` -> `_start_openvocab()` -> `_start_species_worker()` ->
`_start_cameras()` (CameraCapture + CameraTracker pair per enabled camera, in camera
iteration order).

## Process table

| Process | RSS | CPU% | Why it exists |
|---|---|---|---|
| `mirage.supervisor` | 2.9 MB | 0% | Watches/restarts the pipeline subprocess on config changes (`mirage/supervisor.py`); does almost nothing itself |
| `mirage` (main pipeline process) | 50.3 MB | 5.2% | Owns the result-consumer loop, `EventProcessor`, `ReviewSegmentMaintainer`, `ServiceWatchdog`, DB writes |
| `mp.Manager()` server | 9.5 MB | 0.2% | Backs shared `mp.Queue`/`mp.Value` objects (frame queues, fps counters) used across camera processes |
| go2rtc (mirage's own) | 10.1 MB | 2.7% | RTSP restreaming/live-view proxy for all configured cameras |
| `DetectorProcess` -- megadetector | 30.8 MB | 7.9% | Loads the 320x320 MegaDetector ONNX model, serves inference for every camera routed to this one detector |
| `OpenVocabProcess` (OWLv2) | 116.3 MB | 10.6% | Loads `google/owlv2-base-patch16-ensemble` on MPS -- **heaviest single process by far**; only spawned because >=1 open-vocab query is enabled (`MirageApp._start_openvocab`) |
| `SpeciesProcess` (dispatcher) | 4.8 MB | 0% | Lightweight -- only relays crop requests to/from the real SpeciesNet worker subprocess; does not hold the model itself |
| CameraCapture + CameraTracker (per camera, x3) | ~9-33 MB each | 0.2-3.7% | Frame capture + motion/tracking/detection orchestration; cost is proportional to camera count, not detector choice |
| ffmpeg (per camera's RTSP decode, x3) | 22-36 MB each | 6-8.4% | Real hikvision camera (1080p) costs noticeably more CPU than mock cameras (640x480) |
| **SpeciesNet worker** (isolated `.venv-speciesnet`) | 9.1 MB idle | 0% | Real SpeciesNet model + torch, kept in its OWN venv/process (see `mirage/species/plugins/speciesnet.py`'s module docstring) specifically to avoid an unresolvable numpy/opencv-python dependency conflict with norfair in the main venv. RSS is low here only because it was idle at capture time -- expect a real spike during actual classification, not reflected in this snapshot |
| `mirage.api` | 5.3 MB | 0.1% | Read-only REST API serving the frontend |
| `mock_cameras` + its own go2rtc | 0.5 + 4.7 MB | 0% + 0.4% | Standalone synthetic ONVIF camera emulator (separate project, own go2rtc instance on a different RTSP port to avoid colliding with mirage's own) |
| ffmpeg (mock_cameras video-file loops, x2) | 3.0 MB each | 0.2-0.3% | Loops `front_door.mp4`/`garage.mp4` as the RTSP source content mock_cameras serves |

## What's driving memory, ranked

1. **OWLv2/openvocab (116 MB)** -- dominant cost, a real transformer vision-language
   model held in memory purely because open-vocab queries are enabled. **Single
   biggest lever for cutting memory**: disabling every query stops this process from
   spawning at all (`MirageApp._start_openvocab` is fully conditional on
   `active_queries`).
2. **ffmpeg processes (~110 MB combined across 6)** -- one decode-per-camera-per-source
   is an unavoidable cost of having N cameras live, scales linearly with camera count.
3. **megadetector (31 MB)** -- modest at 320x320; would be substantially larger at its
   native 1280x1280 export (not re-measured here, but the model itself is ~140M
   parameters regardless of input resolution -- only activation memory scales down).
4. **3x CameraCapture+CameraTracker pairs (~115 MB combined)** -- proportional to
   camera count, not detector choice; adding a 4th camera adds roughly one more pair's
   worth.
5. **SpeciesNet worker (9 MB idle)** -- currently light only because idle; the isolated
   venv/subprocess design trades a small amount of IPC overhead (JSON+base64 over
   stdin/stdout, see `mirage/species/speciesnet_worker.py`) for keeping the main venv's
   pinned `norfair`/`opencv-python-headless`/numpy<2 stack completely untouched.

Notably absent from memory cost at capture time: two other registered detectors
(`general`, `custom detetctor`) were disabled via the Config page's per-detector
enable/disable toggle (`DetectorInstanceConfig.enabled`, see `mirage/app.py`'s
`_start_detectors`) -- correctly consuming **zero** processes/memory, confirming that
feature works as designed.
