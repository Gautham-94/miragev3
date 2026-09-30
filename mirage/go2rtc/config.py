"""Generates a go2rtc.yaml config from a MirageConfig.

Only the `streams:`, `api:`, and `webrtc:` sections are needed (confirmed against go2rtc
source: internal/api/api.go, internal/webrtc/webrtc.go). Each enabled camera's "detect"
role input path is registered as a go2rtc stream under the camera's own name, so the
frontend can request MSE/WebRTC/snapshot for camera "front_door" via `?src=front_door`
without any extra name-mapping layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from mirage.config.schema import CameraConfig, CameraRole, MirageConfig, RtspTransport

DEFAULT_API_PORT = 1984
DEFAULT_WEBRTC_PORT = 8555

# Every enabled (non-test-source) camera also gets a second, low-resolution go2rtc
# stream registered under "<camera>_sub" -- see build_go2rtc_config's own docstring for
# why. Frontend side: frontend/src/app/pages/live/camera-tile/camera-tile.ts picks this
# suffix vs the bare camera name based on whether its tile is the maximized one.
SUB_STREAM_SUFFIX = "_sub"
SUB_STREAM_WIDTH = 320
SUB_STREAM_HEIGHT = 180


def _sub_stream_source(main_source: str) -> str:
    """go2rtc's own documented ffmpeg-transcode stream-source syntax (confirmed against
    go2rtc's wiki/source, not guessed): "ffmpeg:<input>#video=h264#width=W#height=H"
    spins up a SEPARATE go2rtc-managed ffmpeg process that reads the same input and
    re-encodes it down to WxH h264 before go2rtc ever sees it as a stream. Deliberately
    left as a plain software libx264 encode (no #hardware=... flag) rather than NVENC:
    at 320x180 the encode cost is trivial regardless, and a missing/misconfigured CUDA
    runtime silently breaking every substream (and therefore the whole grid view) is a
    far worse failure mode than a few extra points of CPU for cameras' worth of tiny
    encodes -- can revisit if the CPU cost ever actually matters.
    """
    return f"ffmpeg:{main_source}#video=h264#width={SUB_STREAM_WIDTH}#height={SUB_STREAM_HEIGHT}"


def _env_webrtc_candidates() -> list[str]:
    """Comma-separated `host:port` ICE candidates for go2rtc to advertise, from
    MIRAGE_GO2RTC_WEBRTC_CANDIDATES (empty/unset = advertise nothing extra, the previous
    behavior).

    Only matters when go2rtc is NOT on the same host as the browser -- most importantly
    inside Docker, where go2rtc gathers host candidates from the container's own bridge
    interface (172.x.x.x) and offers only those. The browser can't route to a container
    IP, so ICE has nothing usable to try, WebRTC fails, and every tile silently falls
    back to the MSE tier. Setting this to the address the browser actually reaches
    go2rtc on (e.g. "127.0.0.1:8555" for a browser on the Docker host, or the host's LAN
    IP for other devices) gives ICE a candidate that resolves. The published container
    port must match -- and be published for BOTH tcp and udp, since WebRTC needs udp.
    """
    raw = os.environ.get("MIRAGE_GO2RTC_WEBRTC_CANDIDATES", "")
    return [candidate.strip() for candidate in raw.split(",") if candidate.strip()]


def _detect_role_source(camera: CameraConfig) -> str | None:
    for inp in camera.ffmpeg.inputs:
        if CameraRole.detect in inp.roles:
            # go2rtc holds its OWN independent RTSP connection to the camera (for live
            # view/snapshots), entirely separate from mirage's own ffmpeg capture
            # process -- so it needs to honor the same per-camera transport choice, or a
            # camera whose TCP RTSP is unreliable (see CameraInputConfig.rtsp_transport's
            # docstring -- confirmed against a real Hikvision camera) would still have
            # go2rtc's connection fighting the same instability even after mirage's own
            # capture switches to UDP. `#transport=udp` is go2rtc's own documented,
            # source-confirmed URL-fragment syntax for forcing its RTSP client's
            # transport (internal/rtsp/rtsp.go parses the fragment as query params and
            # sets conn.Transport from `transport=`; only appended for rtsp:// sources
            # and only when non-default, since non-RTSP sources/synthetic test sources
            # don't understand it).
            if inp.path.lower().startswith("rtsp://") and inp.rtsp_transport == RtspTransport.udp:
                return f"{inp.path}#transport=udp"
            return inp.path
    return None


def build_go2rtc_config(
    config: MirageConfig,
    api_port: int = DEFAULT_API_PORT,
    webrtc_port: int = DEFAULT_WEBRTC_PORT,
    stream_overrides: dict[str, str] | None = None,
    webrtc_candidates: list[str] | None = None,
) -> dict:
    """stream_overrides lets a caller point go2rtc at a DIFFERENT source URL than the
    camera's own detect-role ffmpeg input, keyed by camera name. Real cameras (RTSP)
    accept many concurrent client connections, so in production go2rtc simply reads the
    same detect-role URL mirage's own capture process reads -- this parameter exists
    only for local/dev testing against a single-client-only synthetic TCP test source
    (see scripts/run_test_stream.sh), where mirage's capture process and go2rtc can't
    both connect to the same port at once.

    Also registers a low-resolution "<camera>_sub" stream alongside every enabled
    camera -- the live-view grid decodes this instead of the full-res stream for every
    tile except whichever one is maximized. This is the same fix Frigate's own docs
    land on for this exact problem: a browser showing N cameras at once is bottlenecked
    by decode capacity (confirmed here against a real deployment -- a consumer GPU's
    concurrent hardware-decode-session limit was reached well under 10 simultaneous
    full-res tiles), and that capacity is spent on tiles the user probably isn't even
    looking closely at. Two ways to get there: CameraConfig.live_sub_url, if the camera
    has a real native substream (go2rtc pulls it directly, no transcode); otherwise a
    software ffmpeg transcode of the main stream -- but only when the resolved main
    source is a genuine rtsp:// URL AND wasn't replaced by an override, since a
    single-client-only synthetic test source (see stream_overrides' own paragraph
    above, and _detect_role_source's non-rtsp handling) can't also support this extra
    connection, whether it got there via an override or was simply configured directly
    as a non-RTSP camera input.
    """
    overrides = stream_overrides or {}
    streams: dict[str, str] = {}
    for camera in config.cameras.values():
        if not camera.enabled:
            continue
        override = overrides.get(camera.name)
        source = override or _detect_role_source(camera)
        if source is None:
            continue
        streams[camera.name] = source
        if camera.live_sub_url:
            # A real, cheaper-than-transcoding substream URL is known for this camera
            # (see CameraConfig.live_sub_url's own docstring) -- go2rtc pulls it
            # directly, no ffmpeg transcode process needed at all. Registered
            # unconditionally (unlike the transcode fallback below): its connection
            # semantics are independent of whatever the MAIN stream's source turned out
            # to be, so neither the override-skip nor the rtsp-only guard applies here.
            streams[f"{camera.name}{SUB_STREAM_SUFFIX}"] = camera.live_sub_url
        elif override is None and source.lower().startswith("rtsp://"):
            streams[f"{camera.name}{SUB_STREAM_SUFFIX}"] = _sub_stream_source(source)

    candidates = _env_webrtc_candidates() if webrtc_candidates is None else webrtc_candidates
    webrtc_section: dict = {"listen": f":{webrtc_port}", "ice_servers": []}
    if candidates:
        webrtc_section["candidates"] = candidates

    return {
        "streams": streams,
        "api": {"listen": f":{api_port}"},
        # go2rtc defaults to its own public STUN servers for ICE candidate gathering.
        # The browser and go2rtc are always on the same LAN (or the same machine) in
        # this deployment, so a public STUN server only adds a slower, NAT-traversal
        # srflx candidate pair that Chrome can end up preferring over the direct host
        # candidate pair, stalling connection setup instead of just using the LAN route.
        # An empty ice_servers list disables that default so only host candidates are
        # offered -- plus any explicitly advertised address from webrtc_candidates /
        # MIRAGE_GO2RTC_WEBRTC_CANDIDATES (see _env_webrtc_candidates), which is what
        # makes this work at all from inside a container.
        "webrtc": webrtc_section,
        # go2rtc's own logging is noisy at default level; keep it to warnings so it
        # doesn't drown out mirage's own process logs when both run in the foreground.
        "log": {"format": "text", "level": "warn"},
    }


def write_go2rtc_config(
    config: MirageConfig,
    path: str,
    api_port: int = DEFAULT_API_PORT,
    webrtc_port: int = DEFAULT_WEBRTC_PORT,
    stream_overrides: dict[str, str] | None = None,
    webrtc_candidates: list[str] | None = None,
) -> str:
    payload = build_go2rtc_config(
        config, api_port=api_port, webrtc_port=webrtc_port, stream_overrides=stream_overrides,
        webrtc_candidates=webrtc_candidates,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path
