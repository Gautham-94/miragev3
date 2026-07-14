from __future__ import annotations

from mirage.capture.ffmpeg_presets import (
    build_all_ffmpeg_cmds,
    build_detect_scale_args,
    build_ffmpeg_cmd_for_input,
    rtsp_timeout_flag,
)
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    CameraRole,
    DetectConfig,
    FfmpegConfig,
    RecordConfig,
    RtspTransport,
)


def _cam(record_enabled: bool = True, apple_compat: bool = False, roles=None) -> CameraConfig:
    inp = CameraInputConfig(path="rtsp://user:pass@host/stream1")
    if roles is not None:
        inp.roles = roles
    return CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[inp]),
        detect=DetectConfig(width=640, height=360, fps=5),
        record=RecordConfig(enabled=record_enabled, apple_compatibility=apple_compat),
    )


def test_rtsp_timeout_flag_modern_ffmpeg():
    # This host's ffmpeg is 8.1 / libavformat 62 -> must be "-timeout", not "-stimeout".
    assert rtsp_timeout_flag("ffmpeg") == "-timeout"


def test_detect_scale_args_are_the_fps_limiting_mechanism():
    args = build_detect_scale_args(fps=5, width=640, height=360)
    assert args == ["-r", "5", "-vf", "fps=5,scale=640:360"]


def test_single_input_merges_record_and_detect_in_one_command():
    cam = _cam(record_enabled=True)
    cmds = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")
    assert len(cmds) == 1
    cmd = cmds[0]["cmd"]

    # record's segment-muxer block must appear before detect's pipe: (order-of-clarity, not
    # semantically required by ffmpeg, but keep it deterministic and match the spec).
    assert cmd.count("-i") == 1
    assert "-f" in cmd
    assert "segment" in cmd
    assert cmd[-1] == "pipe:"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
    assert "front_door@" in " ".join(cmd)


def test_hwaccel_args_only_applied_to_detect_role_input():
    inp_detect_only = CameraInputConfig(path="rtsp://a/b", roles=[CameraRole.detect], hwaccel_args=["-hwaccel", "videotoolbox"])
    cam = CameraConfig(
        name="cam2",
        ffmpeg=FfmpegConfig(inputs=[inp_detect_only]),
        record=RecordConfig(enabled=False),
    )
    cmd = build_ffmpeg_cmd_for_input(cam, inp_detect_only, "/tmp/cache")
    assert "-hwaccel" in cmd
    assert cmd[-1] == "pipe:"


def test_hwaccel_args_absent_for_record_only_input():
    # A camera must still have a detect-role input SOMEWHERE (config validator enforces
    # this), so pair the record-only input with a separate detect input, and verify the
    # record-only input's own hwaccel_args are never applied to its own command.
    inp_detect = CameraInputConfig(path="rtsp://a/substream", roles=[CameraRole.detect])
    inp_record_only = CameraInputConfig(path="rtsp://a/mainstream", roles=[CameraRole.record], hwaccel_args=["-hwaccel", "videotoolbox"])
    cam = CameraConfig(
        name="cam3",
        ffmpeg=FfmpegConfig(inputs=[inp_detect, inp_record_only]),
        record=RecordConfig(enabled=True),
    )
    cmd = build_ffmpeg_cmd_for_input(cam, inp_record_only, "/tmp/cache")
    assert "-hwaccel" not in cmd
    assert "pipe:" not in cmd  # record-only input has no detect output block


def test_two_separate_inputs_produce_two_independent_commands():
    detect_input = CameraInputConfig(path="rtsp://a/substream", roles=[CameraRole.detect])
    record_input = CameraInputConfig(path="rtsp://a/mainstream", roles=[CameraRole.record])
    cam = CameraConfig(
        name="cam4",
        ffmpeg=FfmpegConfig(inputs=[detect_input, record_input]),
        record=RecordConfig(enabled=True),
    )
    cmds = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")
    assert len(cmds) == 2
    detect_cmd = next(c for c in cmds if CameraRole.detect in c["roles"])
    record_cmd = next(c for c in cmds if CameraRole.record in c["roles"])
    assert "substream" in " ".join(detect_cmd["cmd"])
    assert "mainstream" in " ".join(record_cmd["cmd"])
    assert detect_cmd["cmd"][-1] == "pipe:"
    assert "pipe:" not in record_cmd["cmd"]


def test_apple_compatibility_adds_hvc1_tag():
    cam = _cam(record_enabled=True, apple_compat=True)
    cmds = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")
    cmd = cmds[0]["cmd"]
    assert "-tag:v" in cmd
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"


def test_record_disabled_yields_detect_only_command():
    cam = _cam(record_enabled=False)
    cmds = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")
    assert len(cmds) == 1
    cmd = cmds[0]["cmd"]
    assert "segment" not in cmd
    assert cmd[-1] == "pipe:"


def test_rtsp_transport_defaults_to_tcp():
    cam = _cam()
    cmd = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")[0]["cmd"]
    assert "-rtsp_transport" in cmd
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"


def test_rtsp_transport_udp_is_threaded_through_to_the_real_ffmpeg_command():
    # Regression coverage for the real Hikvision camera issue this option exists to work
    # around (see CameraInputConfig.rtsp_transport's docstring): a camera whose
    # RTSP-over-TCP session gets reset almost immediately, but is stable over UDP.
    inp = CameraInputConfig(path="rtsp://user:pass@host/stream1", rtsp_transport=RtspTransport.udp)
    cam = CameraConfig(
        name="udp_cam",
        ffmpeg=FfmpegConfig(inputs=[inp]),
        detect=DetectConfig(width=640, height=360, fps=5),
        record=RecordConfig(enabled=True),
    )
    cmd = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")[0]["cmd"]
    assert "-rtsp_transport" in cmd
    assert cmd[cmd.index("-rtsp_transport") + 1] == "udp"


def test_rtsp_transport_flag_absent_for_non_rtsp_input():
    # A local file/synthetic tcp:// test source doesn't understand -rtsp_transport at
    # all (ffmpeg errors "Option not found" if it's passed) -- confirm the transport
    # setting has no effect and no flag is emitted for those inputs.
    inp = CameraInputConfig(path="tcp://127.0.0.1:19501", rtsp_transport=RtspTransport.udp)
    cam = CameraConfig(
        name="synthetic_cam",
        ffmpeg=FfmpegConfig(inputs=[inp]),
        detect=DetectConfig(width=640, height=360, fps=5),
        record=RecordConfig(enabled=False),
    )
    cmd = build_all_ffmpeg_cmds(cam, cache_dir="/tmp/cache")[0]["cmd"]
    assert "-rtsp_transport" not in cmd
