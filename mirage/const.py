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

import os
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
def ipc_addr(name: str, cache_dir: str | None = None) -> str:
    """Build an ipc:// zmq address rooted at `cache_dir` (default: the module-level
    CACHE_DIR). Creates the directory if needed.
    """
    base = cache_dir if cache_dir is not None else CACHE_DIR
    Path(base).mkdir(parents=True, exist_ok=True)
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
