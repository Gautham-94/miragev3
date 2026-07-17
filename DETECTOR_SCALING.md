# Detector process scaling: current ceiling and why it exists

This explains a specific known limitation already tracked in `TODO_FIX_LIST.md` item 7
("Single detector process per model, not a horizontally-scalable worker pool") in more
detail, with a concrete example.

## The setup

Say you have 8 cameras, all routed to the same `"general"` detector (`yolov8n.onnx` on
CoreML). Each camera runs at 5fps and needs detection on roughly 1-3 regions per frame.
Per the real measured benchmark already documented in `mirage/detection/
execution_providers.py`, CoreML inference takes ~1.75ms per call.

## Where the ceiling actually is

From `MirageApp._start_detectors()` (`mirage/app.py`): exactly **one** `DetectorProcess`
gets spawned per `DetectorInstanceConfig` entry, no matter how many cameras route to it.
Every camera's `CameraTracker` process pushes its detection requests onto the **same
shared queue** (`detection_queues[detector_name]`), and that single process pulls
requests off one at a time and runs them through the one loaded model, sequentially.

```
camera_1 ─┐
camera_2 ─┤
camera_3 ─┤
camera_4 ─┼──► detection_queues["general"] ──► ONE DetectorProcess ──► ONE onnxruntime session
camera_5 ─┤
camera_6 ─┤
camera_7 ─┤
camera_8 ─┘
```

## Why that's a ceiling

At 8 cameras × 5fps × ~2 regions/frame, that's roughly 80 inference calls/second all
funneling through one process. At 1.75ms each, that's ~140ms of actual GPU work per
second -- seems fine in isolation. But this is a **single OS process with a single
onnxruntime session**, so those calls aren't parallelized across CPU cores or the GPU's
own internal batching -- they're serviced one at a time off one queue. Add a 9th, 10th,
20th camera, and every camera's detection requests queue up behind whichever one is
currently running, even though the machine might have plenty of spare CPU/GPU capacity
sitting idle in other cores.

The scaling knob available today is **routing cameras to a second config entry** -- e.g.
defining a `"general_2"` detector loading the *same* model file, and moving half the
cameras to it -- which does spawn a second real process. But that's a manual
config-time decision the operator has to make and rebalance by hand; there's no
automatic "spin up worker #2 because worker #1 is falling behind" fan-out. If one
detector saturates, cameras routed to it silently fall behind (frames get dropped at the
`detected_frames_queue.put_nowait()` backpressure valve elsewhere in the pipeline)
rather than the system spreading load across more workers on its own.

## Why this matters practically

At small scale (a handful of cameras on one detector) this is a non-issue -- nowhere
close to saturating one detector process. It becomes relevant when scaling toward
Frigate-like deployments (dozens of cameras) or moving to hardware where the CPU has
many idle cores that a single-process model could never use. The fix sketched in
`TODO_FIX_LIST.md` item 7 (a real worker pool -- N processes pulling off one queue, or
auto-spawning additional detector instances under load) is exactly the kind of thing
that becomes worth building once past a handful of cameras on one detector, not before.

See also: `TODO_FIX_LIST.md` item 7 for the other architecture gaps found in the same
comparison (no message broker, no rules-engine layer, no INT8/batching, static frame
rate, no notification service) and the suggested highest-leverage next step (replacing
the ad-hoc `mp.Queue`+ZMQ signaling with a lightweight durable broker like Redis
Streams, which would also unlock config hot-reload).
