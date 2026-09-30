"""Unit tests for go2rtc config generation (mirage/go2rtc/config.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    CameraRole,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
    RtspTransport,
)
from mirage.go2rtc.config import build_go2rtc_config, write_go2rtc_config


def _minimal_config(**camera_overrides) -> MirageConfig:
    camera = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1:8554/front_door")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
        **camera_overrides,
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    return MirageConfig(detectors={"general": detector}, cameras={"front_door": camera})


def test_build_go2rtc_config_registers_detect_role_stream():
    config = _minimal_config()
    payload = build_go2rtc_config(config, api_port=1984, webrtc_port=8555)

    assert payload["streams"]["front_door"] == "rtsp://127.0.0.1:8554/front_door"
    assert payload["api"]["listen"] == ":1984"
    assert payload["webrtc"]["listen"] == ":8555"


def test_build_go2rtc_config_registers_a_low_res_sub_stream_too():
    config = _minimal_config()
    payload = build_go2rtc_config(config)

    assert payload["streams"]["front_door_sub"] == (
        "ffmpeg:rtsp://127.0.0.1:8554/front_door#video=h264#width=320#height=180"
    )


def test_build_go2rtc_config_uses_live_sub_url_directly_when_configured():
    # A real native substream (e.g. a Hikvision Channels/102 URL) -- go2rtc should pull
    # it directly, no ffmpeg transcode wrapper at all.
    config = _minimal_config(live_sub_url="rtsp://127.0.0.1:8554/front_door_sub_native")
    payload = build_go2rtc_config(config)

    assert payload["streams"]["front_door_sub"] == "rtsp://127.0.0.1:8554/front_door_sub_native"


def test_build_go2rtc_config_live_sub_url_registered_even_for_override_camera():
    # Unlike the ffmpeg-transcode fallback, an explicit live_sub_url's own connection
    # doesn't depend on where the main stream came from -- it should still be
    # registered even when the main stream is a single-client override.
    config = _minimal_config(live_sub_url="rtsp://127.0.0.1:8554/front_door_sub_native")
    payload = build_go2rtc_config(config, stream_overrides={"front_door": "tcp://127.0.0.1:19501"})

    assert payload["streams"] == {
        "front_door": "tcp://127.0.0.1:19501",
        "front_door_sub": "rtsp://127.0.0.1:8554/front_door_sub_native",
    }


def test_build_go2rtc_config_skips_disabled_cameras():
    config = _minimal_config(enabled=False)
    payload = build_go2rtc_config(config)

    assert payload["streams"] == {}


def test_build_go2rtc_config_picks_the_detect_role_input_not_record_role():
    camera = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(
            inputs=[
                CameraInputConfig(path="rtsp://127.0.0.1:8554/record_stream", roles=[CameraRole.record]),
                CameraInputConfig(path="rtsp://127.0.0.1:8554/detect_stream", roles=[CameraRole.detect]),
            ]
        ),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={"front_door": camera})

    payload = build_go2rtc_config(config)

    assert payload["streams"]["front_door"] == "rtsp://127.0.0.1:8554/detect_stream"


def test_build_go2rtc_config_stream_override_replaces_detect_role_source():
    # Real cameras (RTSP) accept many concurrent clients, so in production go2rtc reads
    # the same detect-role URL mirage's own capture reads. stream_overrides exists only
    # for local/dev testing against a single-client-only synthetic TCP source, where
    # mirage's capture process and go2rtc can't both connect to the same port at once.
    config = _minimal_config()
    payload = build_go2rtc_config(config, stream_overrides={"front_door": "tcp://127.0.0.1:19501"})

    # Exact equality (not just checking the "front_door" key) also proves no
    # "front_door_sub" got registered -- an override means a single-client-only
    # synthetic test source, which can't support the sub-stream's extra connection.
    assert payload["streams"] == {"front_door": "tcp://127.0.0.1:19501"}


def test_build_go2rtc_config_stream_override_only_applies_to_named_camera():
    camera1 = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1:8554/front_door")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    camera2 = CameraConfig(
        name="backyard",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1:8554/backyard")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={"front_door": camera1, "backyard": camera2})

    payload = build_go2rtc_config(config, stream_overrides={"front_door": "tcp://127.0.0.1:19501"})

    # front_door (overridden) gets no sub-stream; backyard (not overridden, a real
    # rtsp:// source) does.
    assert payload["streams"] == {
        "front_door": "tcp://127.0.0.1:19501",
        "backyard": "rtsp://127.0.0.1:8554/backyard",
        "backyard_sub": "ffmpeg:rtsp://127.0.0.1:8554/backyard#video=h264#width=320#height=180",
    }


def test_build_go2rtc_config_appends_transport_udp_fragment_when_configured():
    # go2rtc holds its own independent RTSP connection (for live view/snapshots),
    # separate from mirage's own ffmpeg capture -- it needs the same per-camera UDP
    # override or its connection would still fight the same TCP instability a camera's
    # rtsp_transport=udp setting exists to work around (see the real Hikvision-camera
    # incident documented in CameraInputConfig.rtsp_transport's docstring). `#transport=
    # udp` is go2rtc's own documented, source-confirmed URL-fragment syntax
    # (internal/rtsp/rtsp.go parses the fragment as query params and sets
    # conn.Transport from it) for forcing its RTSP client to use UDP.
    camera = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(
            inputs=[CameraInputConfig(path="rtsp://127.0.0.1:8554/front_door", rtsp_transport=RtspTransport.udp)]
        ),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={"front_door": camera})

    payload = build_go2rtc_config(config)

    assert payload["streams"]["front_door"] == "rtsp://127.0.0.1:8554/front_door#transport=udp"


def test_build_go2rtc_config_no_transport_fragment_for_default_tcp():
    config = _minimal_config()
    payload = build_go2rtc_config(config)

    assert payload["streams"]["front_door"] == "rtsp://127.0.0.1:8554/front_door"
    assert "#" not in payload["streams"]["front_door"]


def test_build_go2rtc_config_no_transport_fragment_for_non_rtsp_source():
    # A synthetic tcp:// test source doesn't understand go2rtc's RTSP-specific
    # transport fragment -- confirm it's never appended to a non-rtsp:// source, even
    # if rtsp_transport happens to be set to udp on that input.
    camera = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="tcp://127.0.0.1:19501", rtsp_transport=RtspTransport.udp)]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={"front_door": camera})

    payload = build_go2rtc_config(config)

    # Exact equality also proves no "front_door_sub" got registered -- a directly
    # configured non-rtsp:// input is exactly this codebase's single-client-only
    # synthetic test source pattern, same as an override (see build_go2rtc_config's
    # own docstring).
    assert payload["streams"] == {"front_door": "tcp://127.0.0.1:19501"}


def test_write_go2rtc_config_produces_valid_yaml_file():
    config = _minimal_config()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "go2rtc.yaml")
        write_go2rtc_config(config, path)

        with open(path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["streams"]["front_door"] == "rtsp://127.0.0.1:8554/front_door"
