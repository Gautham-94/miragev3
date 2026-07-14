"""Shared "fetch a JPEG snapshot and write it to disk" helper, used by both
ReviewSegmentMaintainer and EventProcessor. Kept independent of any concrete HTTP client
so both callers stay decoupled from go2rtc specifically -- see ThumbnailFetcher.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

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
