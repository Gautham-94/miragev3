"""Tests for the PTZ control endpoints (mirage/api/routers/ptz.py).

Following the same philosophy as tests/test_api_onvif.py: the ONVIF-calling routes
(presets/status/goto-preset/move/stop) are only tested against real, deterministic
failure paths (an unreachable device, an unknown camera, PTZ not enabled) since a real
PTZ camera's IP/credentials aren't something this suite should hardcode. The
config-mutation routes (patrol/start, patrol/stop) are tested fully, since they're pure
config read/write with no real network call involved.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
    PtzConfig,
)

UNREACHABLE_HOST = "127.0.0.1"
UNREACHABLE_PORT = 9  # "discard" port -- connects then immediately refuses, same as test_api_onvif.py


def _config_with_ptz_camera(ptz: PtzConfig | None = None) -> MirageConfig:
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    camera = CameraConfig(
        name="ptz_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1/ptz_cam")]),
        detector="general",
        ptz=ptz or PtzConfig(enabled=True, onvif_host=UNREACHABLE_HOST, onvif_port=UNREACHABLE_PORT),
    )
    return MirageConfig(detectors={"general": detector}, cameras={"ptz_cam": camera})


def _client(config: MirageConfig) -> TestClient:
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(config=config, db_path=str(Path(tmp) / "test.db"))
        return TestClient(app)


def test_presets_unknown_camera_returns_404():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.get("/api/ptz/does-not-exist/presets")
    assert resp.status_code == 404


def test_presets_camera_without_ptz_enabled_returns_400():
    config = _config_with_ptz_camera(ptz=PtzConfig(enabled=False))
    with _client(config) as client:
        resp = client.get("/api/ptz/ptz_cam/presets")
    assert resp.status_code == 400
    assert "does not have PTZ enabled" in resp.json()["detail"]


def test_presets_against_unreachable_device_returns_502():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.get("/api/ptz/ptz_cam/presets")
    assert resp.status_code == 502
    assert "could not reach" in resp.json()["detail"].lower()


def test_status_against_unreachable_device_returns_502():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.get("/api/ptz/ptz_cam/status")
    assert resp.status_code == 502


def test_goto_preset_against_unreachable_device_returns_502():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.post("/api/ptz/ptz_cam/goto-preset/1")
    assert resp.status_code == 502


def test_move_against_unreachable_device_returns_502():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.post("/api/ptz/ptz_cam/move", json={"pan": 0.5, "tilt": 0.0, "zoom": 0.0})
    assert resp.status_code == 502


def test_stop_against_unreachable_device_returns_502():
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.post("/api/ptz/ptz_cam/stop")
    assert resp.status_code == 502


def test_move_defaults_all_axes_to_zero_when_omitted():
    """PtzMoveRequest's fields are all optional (default 0.0) -- confirmed by posting
    an empty body and still getting the expected 502 (unreachable device), not a 422
    validation error, proving the request body itself was accepted.
    """
    config = _config_with_ptz_camera()
    with _client(config) as client:
        resp = client.post("/api/ptz/ptz_cam/move", json={})
    assert resp.status_code == 502  # not 422 -- body was valid, just unreachable


# --------------------------------------------------------------------------------------
# Patrol start/stop -- pure config mutation, no network call, fully testable.
# --------------------------------------------------------------------------------------


def test_patrol_start_sets_patrol_enabled_true_and_persists(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(db_path=db_path)
        with TestClient(app) as client:
            # Seed a PTZ-enabled camera via the DB-backed config (create_app(config=...)
            # would freeze config, but patrol start/stop needs save_to_db() to actually
            # write through, so this test uses the real DB-backed path instead).
            config = MirageConfig.from_db()
            config.detectors["general"] = DetectorInstanceConfig(
                name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8",
            )
            config.cameras["ptz_cam"] = CameraConfig(
                name="ptz_cam", ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
                detector="general", ptz=PtzConfig(enabled=True, patrol_enabled=False),
            )
            config.save_to_db()

            resp = client.post("/api/ptz/ptz_cam/patrol/start")
            assert resp.status_code == 204

            reloaded = MirageConfig.from_db()
            assert reloaded.cameras["ptz_cam"].ptz.patrol_enabled is True


def test_patrol_stop_sets_patrol_enabled_false_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(db_path=db_path)
        with TestClient(app) as client:
            config = MirageConfig.from_db()
            config.detectors["general"] = DetectorInstanceConfig(
                name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8",
            )
            config.cameras["ptz_cam"] = CameraConfig(
                name="ptz_cam", ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
                detector="general", ptz=PtzConfig(enabled=True, patrol_enabled=True),
            )
            config.save_to_db()

            resp = client.post("/api/ptz/ptz_cam/patrol/stop")
            assert resp.status_code == 204

            reloaded = MirageConfig.from_db()
            assert reloaded.cameras["ptz_cam"].ptz.patrol_enabled is False


def test_patrol_start_unknown_camera_returns_404():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(db_path=str(Path(tmp) / "test.db"))
        with TestClient(app) as client:
            resp = client.post("/api/ptz/does-not-exist/patrol/start")
    assert resp.status_code == 404


def test_patrol_start_camera_without_ptz_enabled_returns_400():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(db_path=db_path)
        with TestClient(app) as client:
            config = MirageConfig.from_db()
            config.detectors["general"] = DetectorInstanceConfig(
                name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8",
            )
            config.cameras["no_ptz_cam"] = CameraConfig(
                name="no_ptz_cam", ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]), detector="general",
            )
            config.save_to_db()

            resp = client.post("/api/ptz/no_ptz_cam/patrol/start")
    assert resp.status_code == 400
