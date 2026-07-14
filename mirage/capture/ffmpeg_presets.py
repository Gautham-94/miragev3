"""ffmpeg argv construction.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md sections 1.2-1.3.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache

from mirage.config.schema import CameraConfig, CameraInputConfig, CameraRole

VERSION = "0.1.0"

FFMPEG_GLOBAL_ARGS_DEFAULT = ["-hide_banner", "-loglevel", "warning", "-threads", "2"]

DETECT_OUTPUT_ARGS = ["-threads", "2", "-f", "rawvideo", "-pix_fmt", "yuv420p"]

# Section 1.2.5: audio is ALWAYS transcoded to AAC, even though video is stream-copied --
# this sidesteps a real ffmpeg >=7.x regression where stream-copying RTSP audio through the
# segment muxer's threaded demux/encode/mux pipeline silently drops audio packets. Applying
# this unconditionally (rather than branching on ffmpeg version) means we don't need to
# detect the installed ffmpeg's exact release to decide the audio codec arg.
RECORD_OUTPUT_ARGS_BASE = [
    "-f", "segment", "-segment_format", "mp4",
    "-reset_timestamps", "1", "-strftime", "1",
    "-c:v", "copy", "-c:a", "aac",
]

FFMPEG_HVC1_ARGS = ["-tag:v", "hvc1"]


@lru_cache(maxsize=1)
def _libavformat_major(ffmpeg_path: str = "ffmpeg") -> int:
    """Detect the installed ffmpeg's libavformat major version, to decide whether the RTSP
    socket-timeout flag is spelled `-timeout` (libavformat >= 59, ffmpeg >= 5.0) or
    `-stimeout` (older). Cached since this never changes for a running process.
    """
    try:
        out = subprocess.run(
            [ffmpeg_path, "-version"], capture_output=True, text=True, timeout=10
        ).stdout
        m = re.search(r"libavformat\s+(\d+)\.", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 59  # assume a reasonably modern ffmpeg if detection fails


def rtsp_timeout_flag(ffmpeg_path: str = "ffmpeg") -> str:
    return "-timeout" if _libavformat_major(ffmpeg_path) >= 59 else "-stimeout"


def build_input_args(ffmpeg_path: str = "ffmpeg", transport: str = "tcp", input_path: str = "") -> list[str]:
    """Section 1.2.2, preset-rtsp-generic equivalent.

    RTSP-only flags (-user_agent, -rtsp_transport, the socket timeout flag) are rejected
    by ffmpeg's demuxer for non-network inputs (a local file, or any other non-rtsp
    scheme) -- `Option not found` is a hard error, not a warning, so these must only be
    included when the input path is actually an rtsp:// URL. This matters for local-file
    testing/synthetic sources as well as any camera config that isn't RTSP (e.g. an HTTP
    MJPEG source would need its own, different, preset -- see the main spec section 1.2.2
    for the other presets a full implementation should add).
    """
    if not input_path.lower().startswith("rtsp://"):
        return [
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts+discardcorrupt",
        ]
    return [
        "-user_agent", f"Mirage/{VERSION}",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts+discardcorrupt",
        "-rtsp_transport", transport,
        rtsp_timeout_flag(ffmpeg_path), "10000000",
        "-use_wallclock_as_timestamps", "1",
    ]


def build_detect_scale_args(fps: int, width: int, height: int) -> list[str]:
    """Section 1.2.4 / 1.3 -- the ENTIRE fps-limiting mechanism. ffmpeg drops/duplicates
    frames to hit `fps` and scales to (width, height) before a single byte reaches Python.
    Do not implement fps limiting in the Python read loop.
    """
    return ["-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}"]


def build_record_output_args(segment_seconds: int, apple_compatibility: bool = False) -> list[str]:
    args = ["-f", "segment", "-segment_time", str(segment_seconds), "-segment_format", "mp4",
            "-reset_timestamps", "1", "-strftime", "1", "-c:v", "copy", "-c:a", "aac"]
    if apple_compatibility:
        args = args + FFMPEG_HVC1_ARGS
    return args


def build_ffmpeg_cmd_for_input(
    camera: CameraConfig,
    ffmpeg_input: CameraInputConfig,
    cache_dir: str,
    segment_format: str = "%Y%m%d%H%M%S%z",
) -> list[str] | None:
    """Section 1.2.1/1.3: build one full ffmpeg argv for a single CameraInputConfig.

    Returns None if this input has neither the detect nor record role (nothing to do).
    """
    import os

    ffmpeg_path = camera.ffmpeg.ffmpeg_path
    cmd: list[str] = [ffmpeg_path] + list(camera.ffmpeg.global_args)

    has_detect = CameraRole.detect in ffmpeg_input.roles
    has_record = CameraRole.record in ffmpeg_input.roles and camera.record.enabled

    if not has_detect and not has_record:
        return None

    # hwaccel decode args ONLY apply when this input carries the detect role (section 1.2.3).
    if has_detect:
        cmd += list(ffmpeg_input.hwaccel_args)

    cmd += build_input_args(ffmpeg_path, transport=ffmpeg_input.rtsp_transport.value, input_path=ffmpeg_input.path)
    cmd += ["-i", ffmpeg_input.path]

    if has_record:
        output_path = f"{os.path.join(cache_dir, camera.name)}@{segment_format}.mp4"
        cmd += build_record_output_args(camera.record.segment_seconds, camera.record.apple_compatibility)
        cmd += [output_path]

    if has_detect:
        cmd += build_detect_scale_args(camera.detect.fps, camera.detect.width, camera.detect.height)
        cmd += DETECT_OUTPUT_ARGS + ["pipe:"]

    return [part for part in cmd if part != ""]


def build_all_ffmpeg_cmds(camera: CameraConfig, cache_dir: str) -> list[dict]:
    """Returns a list of {"roles": [...], "cmd": [...]} dicts, one per configured input that
    has at least one active role -- section 1.2/1.3, mirroring _build_ffmpeg_cmds.
    """
    result = []
    for ffmpeg_input in camera.ffmpeg.inputs:
        cmd = build_ffmpeg_cmd_for_input(camera, ffmpeg_input, cache_dir)
        if cmd is None:
            continue
        result.append({"roles": list(ffmpeg_input.roles), "cmd": cmd})
    return result
