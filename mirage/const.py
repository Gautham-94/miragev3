"""Global path/priority constants.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 0.3.

The spec's defaults (/tmp/cache, /media/nvr, /dev/shm) assume a Linux production host.
This module keeps the same *names* and *roles* for each directory but resolves them
relative to the project root by default so the system runs unmodified on macOS during
development, while remaining fully overridable via environment variables for a real
Linux deployment (where you'd set MIRAGE_CACHE_DIR=/tmp/cache, MIRAGE_BASE_DIR=/media/nvr,
etc., ideally pointing CACHE_DIR at a tmpfs mount as the spec recommends).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# Project root = the `mirage/` directory that contains this package (i.e. mirage/mirage/../).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var_name: str, default: Path) -> str:
    return os.environ.get(var_name, str(default))


# tmpfs-equivalent scratch: in-flight ffmpeg segments, IPC unix sockets. On Linux, point
# this at a real tmpfs mount (e.g. /tmp/cache backed by tmpfs, or /dev/shm) for production.
CACHE_DIR = _env_path("MIRAGE_CACHE_DIR", _PROJECT_ROOT / "media" / "cache")

BASE_DIR = _env_path("MIRAGE_BASE_DIR", _PROJECT_ROOT / "media")
RECORD_DIR = _env_path("MIRAGE_RECORD_DIR", Path(BASE_DIR) / "recordings")
CLIPS_DIR = _env_path("MIRAGE_CLIPS_DIR", Path(BASE_DIR) / "clips")
REVIEW_THUMB_DIR = str(Path(CLIPS_DIR) / "review")
EVENT_SNAPSHOT_DIR = str(Path(CLIPS_DIR) / "events")
EXPORT_DIR = _env_path("MIRAGE_EXPORT_DIR", Path(BASE_DIR) / "exports")

CONFIG_DIR = _env_path("MIRAGE_CONFIG_DIR", _PROJECT_ROOT / "config")
DB_PATH = _env_path("MIRAGE_DB_PATH", Path(CONFIG_DIR) / "mirage.db")
MODEL_CACHE_DIR = _env_path("MIRAGE_MODEL_CACHE_DIR", _PROJECT_ROOT / "models")
GO2RTC_BIN_DIR = _env_path("MIRAGE_GO2RTC_BIN_DIR", _PROJECT_ROOT / "bin")


def resolve_model_path(path: str) -> str:
    """Resolves a detector/species ModelConfig's model_path/labelmap_path field against
    MODEL_CACHE_DIR -- these fields have always been stored as plain strings shaped
    "models/yolov8n.onnx" (see MirageConfig.default()'s seeded "general" detector), which
    only ever worked by coincidence: `python -m mirage`'s conventional invocation from the
    repo root makes a bare relative path like that resolve correctly against the process's
    cwd, since MODEL_CACHE_DIR's own dev-mode default (_PROJECT_ROOT / "models") happens to
    match. Neither assumption holds for the packaged desktop app -- there is no
    "repo root" cwd, and confirmed live, this exact gap made the frozen build's own
    SEEDED default detector unable to find its model at all, not just a newly-added one.

    An absolute path (e.g. a client Browse-ing to a custom model file outside the
    bundled set) passes through unchanged -- MODEL_CACHE_DIR is only ever a base for a
    *relative* path. Note MODEL_CACHE_DIR itself already points AT the models directory
    (not its parent), while the stored strings keep their own "models/" prefix (matching
    how they've always been written/seeded) -- so this resolves against
    MODEL_CACHE_DIR's PARENT, which is what actually makes "models/yolov8n.onnx" land at
    MODEL_CACHE_DIR/yolov8n.onnx in both dev (parent = repo root) and frozen (parent =
    %LOCALAPPDATA%\\Mirage, where mirage/desktop/paths.py copies the bundled models/ dir
    to) without a doubled "models/models/..." path or a behavior change for existing
    dev-tree configs.

    Every actual consumer of these fields (onnx_yolov8.py/onnx_megadetector.py's
    InferenceSession calls, load_labels()) must call this -- the raw string alone is not
    a usable path on its own, in dev OR frozen.
    """
    if os.path.isabs(path):
        return path
    return str(Path(MODEL_CACHE_DIR).parent / path)

# strftime pattern for raw ffmpeg segment filenames (before being moved to permanent storage).
CACHE_SEGMENT_FORMAT = "%Y%m%d%H%M%S%z"

MAX_SEGMENT_DURATION = 600  # sanity ceiling (s) for a "valid" recording segment
MAX_SEGMENTS_IN_CACHE = 6  # backpressure: max unprocessed cache segments/camera

# os.nice() values. macOS honors os.nice() the same as Linux (best-effort, may require
# privilege to go negative; 0/10/19 are all achievable as an unprivileged user).
PROCESS_PRIORITY_HIGH = 0
PROCESS_PRIORITY_MED = 10
PROCESS_PRIORITY_LOW = 19

# Detection output tensor contract (spec section 3.2 / 3.3.2) -- fixed regardless of model.
MAX_DETECTIONS = 20
DETECTIONS_PER_ROW = 6  # class_id, score, y_min, x_min, y_max, x_max
OUTPUT_SHM_DTYPE_SIZE = 4  # float32
OUTPUT_SHM_SIZE = MAX_DETECTIONS * DETECTIONS_PER_ROW * OUTPUT_SHM_DTYPE_SIZE  # 480 bytes

# IPC socket addresses -- unix domain sockets under CACHE_DIR, per spec section 11.2.
# libzmq's ipc:// transport is a thin wrapper over POSIX unix domain sockets and is not
# implemented on native Windows at all (zmq.error.ZMQError: "Protocol not supported" on
# bind) -- not a mirage bug, a genuine libzmq platform gap. tcp://127.0.0.1:<port> is the
# standard substitute; every one of these sockets is a purely local, same-machine
# broker/proxy connection (see mirage/ipc/zmq_pubsub.py), never exposed externally, so a
# loopback TCP port is exactly as private as the unix socket it replaces.
#
# The port is derived deterministically from (cache_dir, name) via a stable hash (NOT
# Python's built-in hash(), which is randomized per-process by default) rather than
# tracked in a registry: this file has exactly one call site today (mirage/app.py's
# detector_pub/detector_sub pair, one MirageApp per machine in production), and a stable
# per-(cache_dir, name) port means every test using its own tmp cache_dir still gets
# independent, collision-free ports for free, with no shared state to coordinate.
def _windows_ipc_port(base: str, name: str) -> int:
    digest = hashlib.sha256(f"{base}:{name}".encode()).hexdigest()
    return 50000 + (int(digest, 16) % 10000)  # 50000-59999: outside the well-known range


def ipc_addr(name: str, cache_dir: str | None = None) -> str:
    """Build a zmq address rooted at `cache_dir` (default: the module-level CACHE_DIR)
    -- ipc:// on POSIX, tcp://127.0.0.1:<port> on Windows (see the platform note above).
    Creates the directory if needed (still meaningful on Windows: CACHE_DIR itself must
    exist for everything else that writes ffmpeg segments etc. there).
    """
    base = cache_dir if cache_dir is not None else CACHE_DIR
    Path(base).mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        return f"tcp://127.0.0.1:{_windows_ipc_port(str(base), name)}"
    return f"ipc://{Path(base) / name}"


def ensure_dirs(*extra_dirs: str) -> None:
    """Create all directories this process might write to. Safe to call from every
    process. Pass explicit override paths (e.g. a test's own cache/record/db dirs) as
    extra_dirs to ensure those get created too, in addition to the module-level defaults.
    """
    all_dirs = (
        CACHE_DIR, BASE_DIR, RECORD_DIR, CLIPS_DIR, REVIEW_THUMB_DIR, EVENT_SNAPSHOT_DIR,
        EXPORT_DIR, CONFIG_DIR, MODEL_CACHE_DIR,
    ) + extra_dirs
    for d in all_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# The pip-installable CUDA/cuDNN runtime wheels (requirements-windows-gpu.txt) place
# their DLLs under <venv>/Lib/site-packages/nvidia/<component>/bin/ -- NOT on PATH by
# default, and onnxruntime-gpu's CUDAExecutionProvider fails to load without them
# discoverable there (verified directly: same "cublasLt64_12.dll is missing" error
# with everything pip-installed but this not yet called, see logfile.md's "Discovery
# trail"). This is a no-op everywhere except Windows, and a no-op even on Windows if
# these packages were never installed (plain CPU onnxruntime, requirements.txt) --
# every directory is checked with .is_dir() before being added, nothing is assumed.
_CUDA_DLL_COMPONENTS = ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn", "cufft", "nvjitlink")


def ensure_cuda_dll_directories_on_path() -> None:
    """Prepends each installed nvidia-*-cu12 wheel's bin/ dir to PATH, so
    onnxruntime-gpu's CUDAExecutionProvider can actually find cublasLt64_12.dll,
    cudnn64_9.dll, cufft64_11.dll, etc. at InferenceSession-creation time instead of
    silently falling back to CPU. Call once, early, in every entrypoint that might
    construct a CUDA-backed onnxruntime session (mirage/__main__.py,
    mirage/api/__main__.py) -- before any onnxruntime import, though calling it after
    is harmless too since this only affects DLL search, not anything already loaded.

    mirage/desktop/paths.py (the packaged-app launcher) intentionally does NOT import
    this -- it has its own self-contained copy of the same logic, since it must run
    before mirage.const itself is safe to import (see that module's own docstring).
    Keep both in sync if this list of components ever changes.
    """
    if sys.platform != "win32":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for component in _CUDA_DLL_COMPONENTS:
        bin_dir = nvidia_root / component / "bin"
        if bin_dir.is_dir():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
