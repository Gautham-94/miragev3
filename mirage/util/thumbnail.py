"""Shared "fetch a JPEG snapshot and write it to disk" helper, used by both
ReviewSegmentMaintainer and EventProcessor. Kept independent of any concrete HTTP client
so both callers stay decoupled from go2rtc specifically -- see ThumbnailFetcher.

Also holds draw_boxes_on_jpeg_bytes -- box rendering follows Frigate's own model
(confirmed by reading its actual source, frigate/util/image.py's draw_snapshot_bounding_
boxes/draw_snapshot_overlay_boxes/get_snapshot_bytes): the file written to disk
(EVENT_SNAPSHOT_DIR/<event_id>.jpg) is always the CLEAN, unannotated frame -- boxes are
never burned into it. Box data (label + box, one entry per object confirmed in that
frame) is stored separately as JSON on Event.data["snapshot_boxes"], and
draw_boxes_on_jpeg_bytes is called fresh on every GET /api/events/{id}/snapshot request
(mirage/api/routers/events.py) to composite them on demand. This is deliberately NOT
"burn the box in once at Event-creation time" (an earlier version of this fix did that,
and hit a real wall: an open-vocab match is produced by a completely separate process on
a different frame than the closed-vocab tracker's own frame for the same result batch --
there's no single shared "the frame this was detected in" to burn into). Storing box
data separately sidesteps that entirely: whichever frame actually got saved as this
Event's snapshot, its own box data travels with it, rendered the same way regardless of
which pipeline path produced it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOX_COLOR_BGR = (24, 197, 245)  # amber/yellow, matches the frontend's own box color
_BOX_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.5
_LABEL_FONT_THICKNESS = 1


def draw_boxes_on_jpeg_bytes(
    frame_jpeg: bytes,
    boxes: list[tuple[str, tuple[float, float, float, float]]],
) -> bytes | None:
    """Decodes `frame_jpeg` (a CLEAN, unannotated snapshot -- never mutates the file on
    disk, always operates on bytes already read into memory), burns each
    (label, (x1, y1, x2, y2) full-frame pixel box) onto a COPY with cv2, re-encodes, and
    returns the new bytes. Called fresh on every snapshot HTTP request (see this
    module's own docstring for why) rather than once at write time. Returns None on any
    decode/encode failure rather than raising, so a broken/corrupt stored frame doesn't
    take down the whole request -- the caller can fall back to serving the raw bytes
    unboxed.
    """
    try:
        arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        height, width = image.shape[:2]
        for label, box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
            y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR_BGR, _BOX_THICKNESS)
            (text_w, text_h), _ = cv2.getTextSize(label.upper(), _LABEL_FONT, _LABEL_FONT_SCALE, _LABEL_FONT_THICKNESS)
            label_y1 = max(0, y1 - text_h - 6)
            cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 6, y1), _BOX_COLOR_BGR, -1)
            cv2.putText(
                image, label.upper(), (x1 + 3, y1 - 4), _LABEL_FONT, _LABEL_FONT_SCALE, (17, 17, 17),
                _LABEL_FONT_THICKNESS, cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return None
        return encoded.tobytes()
    except cv2.error:
        logger.exception("failed to draw boxes on snapshot")
        return None


def write_clean_snapshot(frame_jpeg: bytes, thumb_dir: str, filename_stem: str) -> str | None:
    """Writes `frame_jpeg` to disk completely unmodified -- the CLEAN snapshot file
    (see this module's own docstring). Distinct from capture_thumbnail below only in
    that its input is already-encoded bytes the caller obtained itself (e.g. from
    CameraTracker's synchronous frame capture, mirage.tracking.camera_tracker.
    _maybe_encode_frame), not a live fetch this function performs itself.
    """
    try:
        path = Path(thumb_dir) / f"{filename_stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame_jpeg)
        return str(path)
    except OSError:
        logger.exception("failed to write clean snapshot for %s", filename_stem)
        return None

# Given a camera name, return a JPEG snapshot's raw bytes, or None if unavailable.
ThumbnailFetcher = Callable[[str], Optional[bytes]]

# go2rtc's own frame.jpeg endpoint has been observed (see IMPLEMENTATION_NOTES.md) to
# return a 200 OK with an EMPTY body during a brief transient window -- specifically,
# whenever it hasn't buffered a real keyframe recently (e.g. right after a stream
# (re)connects). A single retry after a short pause absorbs this without meaningfully
# delaying event/review-segment creation (this call already happens at most a handful
# of times a minute, never per-frame).
_RETRY_DELAY_SECONDS = 0.5


def capture_thumbnail(
    fetcher: ThumbnailFetcher | None,
    thumb_dir: str,
    filename_stem: str,
    camera_name: str,
) -> str | None:
    """Best-effort: any failure (fetcher raising, empty response, disk write failure)
    returns None rather than propagating, since a missing thumbnail should never break
    the caller's own record-creation flow.
    """
    if fetcher is None:
        return None

    jpeg_bytes = _fetch_with_retry(fetcher, camera_name, filename_stem)
    if not jpeg_bytes:
        return None

    path = Path(thumb_dir) / f"{filename_stem}.jpg"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg_bytes)
    except OSError:
        logger.exception("%s: failed to write thumbnail for %s", camera_name, filename_stem)
        return None
    return str(path)


def _fetch_with_retry(fetcher: ThumbnailFetcher, camera_name: str, filename_stem: str) -> bytes | None:
    for attempt in range(2):
        try:
            jpeg_bytes = fetcher(camera_name)
        except Exception:
            logger.exception("%s: thumbnail fetch failed for %s (attempt %d)", camera_name, filename_stem, attempt + 1)
            jpeg_bytes = None
        if jpeg_bytes:
            return jpeg_bytes
        if attempt == 0:
            time.sleep(_RETRY_DELAY_SECONDS)
    return None
