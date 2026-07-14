"""Tests for the ONVIF discovery/resolve endpoints (mirage/api/routers/onvif.py).

The scan endpoint is tested against a REAL running go2rtc process (not mocked). This
deliberately doesn't assert a specific device count either way: whatever ONVIF devices
are actually reachable on the network this test runs on is out of this test's control --
early in development this machine's LAN had none (go2rtc's real "no sources" 404 case,
confirmed against actual go2rtc source -- see mirage/api/routers/onvif.py's docstring),
but later a real Hikvision camera was added to the same LAN for real end-to-end wizard
testing and started showing up in scans, which is a GOOD sign (proof the scan endpoint
finds real hardware), not a regression. So this test only asserts the RESPONSE SHAPE is
always well-formed, not the device count.

The resolve endpoint (real ONVIF Media GetProfiles/GetStreamUri calls via
onvif-zeep-async) is tested only against its real, deterministic failure path -- an
unreachable device -- since a real ONVIF camera's exact IP/credentials aren't something
this automated suite should hardcode (it's real hardware on someone's real LAN, subject
to change). That success path was verified manually against the real Hikvision camera
during development (see IMPLEMENTATION_NOTES.md) but is not re-verified automatically on
every test run; note this gap explicitly rather than claiming untested code is proven
correct by an automated suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.go2rtc.process import Go2rtcProcess

API_PORT = 19586
WEBRTC_PORT = 19857


@pytest.fixture
def running_go2rtc():
    from mirage.config.schema import DetectorInstanceConfig, MirageConfig, ModelConfig

    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={})

    with tempfile.TemporaryDirectory(prefix="mrgonvif") as tmp:
        proc = Go2rtcProcess(config, cache_dir=tmp, api_port=API_PORT, webrtc_port=WEBRTC_PORT)
        proc.start()

        import socket
        import time

        def port_open(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(("127.0.0.1", port)) == 0

        deadline = time.time() + 15
        while time.time() < deadline and not port_open(API_PORT):
            time.sleep(0.2)
        assert port_open(API_PORT), "go2rtc API port never came up"

        yield proc
        proc.stop()


@pytest.fixture
def client(running_go2rtc):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(db_path=db_path, go2rtc_api_port=API_PORT)
        with TestClient(app) as c:
            yield c


def test_scan_against_real_go2rtc_returns_a_well_shaped_list(client):
    # Whatever this network currently has on it (nothing, or a real ONVIF camera -- see
    # this file's module docstring for why this test doesn't assume either way): the
    # response must always be a well-formed list of devices, each with a non-empty url,
    # whether go2rtc's real answer was an empty "no sources" 404 or a real `sources` list.
    resp = client.get("/api/onvif/scan")
    assert resp.status_code == 200
    devices = resp.json()
    assert isinstance(devices, list)
    for device in devices:
        assert "url" in device
        assert device["url"]


def test_resolve_against_unreachable_device_returns_502():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(db_path=str(Path(tmp) / "test.db"))
        with TestClient(app) as client:
            # Port 9 ("discard") on localhost: connects then immediately refuses/no
            # service -- a real, deterministic "unreachable ONVIF device" case.
            resp = client.get(
                "/api/onvif/resolve",
                params={"ip": "127.0.0.1", "port": 9, "username": "user", "password": "pass"},
            )
    assert resp.status_code == 502
    assert "could not reach" in resp.json()["detail"].lower()


def test_resolve_requires_username_and_password():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(db_path=str(Path(tmp) / "test.db"))
        with TestClient(app) as client:
            resp = client.get("/api/onvif/resolve", params={"ip": "127.0.0.1"})
    assert resp.status_code == 422
