"""Tests for mirage/ptz/client.py (PtzClient).

Following the same philosophy as tests/test_api_onvif.py: connect() against real ONVIF
network calls is only tested against its real, deterministic failure path (an
unreachable device), since a real PTZ camera's IP/credentials aren't something this
suite should hardcode. get_status()'s response-parsing logic and the other command
methods are tested directly against small fake service-response objects (plain
attribute holders, not a mocking framework) since that logic is pure and deterministic
regardless of what's on the network.

No async test plugin (pytest-asyncio/anyio) is installed in this project -- every other
async code path here (mirage/api/routers/onvif.py) is only tested through FastAPI's
synchronous TestClient, never bare `async def test_`. Following that same convention,
these tests stay plain `def test_` and drive PtzClient's async methods via
asyncio.run() themselves.
"""

from __future__ import annotations

import asyncio

import pytest

from mirage.ptz.client import PtzClient


class _Attrs:
    """A bare attribute holder -- stands in for the zeep-generated response objects
    ONVIF service calls return (which are attribute-access objects, not dicts).
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_connect_against_unreachable_device_raises():
    client = PtzClient(host="127.0.0.1", port=9, username="user", password="pass")
    try:
        with pytest.raises(Exception):
            # Port 9 ("discard") on localhost: connects then immediately refuses/no
            # service -- a real, deterministic "unreachable ONVIF device" case, same as
            # tests/test_api_onvif.py's own resolve-endpoint unreachable-device test.
            asyncio.run(client.connect())
    finally:
        asyncio.run(client.close())


def test_get_status_parses_moving_true():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"

    status = _Attrs(
        MoveStatus=_Attrs(PanTilt="MOVING"),
        Position=_Attrs(PanTilt=_Attrs(x=0.5, y=-0.2), Zoom=_Attrs(x=0.1)),
    )

    class _FakePtzService:
        async def GetStatus(self, params):
            assert params == {"ProfileToken": "token1"}
            return status

    client._ptz_service = _FakePtzService()
    result = asyncio.run(client.get_status())

    assert result.moving is True
    assert result.pan == 0.5
    assert result.tilt == -0.2
    assert result.zoom == 0.1


def test_get_status_parses_idle_and_missing_position():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"

    status = _Attrs(MoveStatus=_Attrs(PanTilt="IDLE"))

    class _FakePtzService:
        async def GetStatus(self, params):
            return status

    client._ptz_service = _FakePtzService()
    result = asyncio.run(client.get_status())

    assert result.moving is False
    assert result.pan is None
    assert result.tilt is None
    assert result.zoom is None


def test_get_presets_maps_tokens_and_names():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"

    class _FakePtzService:
        async def GetPresets(self, params):
            assert params == {"ProfileToken": "token1"}
            return [_Attrs(token="1", Name="Gate"), _Attrs(token="2")]

    client._ptz_service = _FakePtzService()
    presets = asyncio.run(client.get_presets())

    assert [(p.token, p.name) for p in presets] == [("1", "Gate"), ("2", None)]


def test_continuous_move_passes_velocity_through():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"
    calls = []

    class _FakePtzService:
        async def ContinuousMove(self, params):
            calls.append(params)

    client._ptz_service = _FakePtzService()
    asyncio.run(client.continuous_move(pan_speed=0.5, tilt_speed=-0.3, zoom_speed=0.1))

    assert calls == [{
        "ProfileToken": "token1",
        "Velocity": {"PanTilt": {"x": 0.5, "y": -0.3}, "Zoom": {"x": 0.1}},
    }]


def test_goto_preset_passes_preset_token():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"
    calls = []

    class _FakePtzService:
        async def GotoPreset(self, params):
            calls.append(params)

    client._ptz_service = _FakePtzService()
    asyncio.run(client.goto_preset("3"))

    assert calls == [{"ProfileToken": "token1", "PresetToken": "3"}]


def test_stop_stops_both_pan_tilt_and_zoom():
    client = PtzClient(host="x", port=80, username="u", password="p")
    client._media_profile_token = "token1"
    calls = []

    class _FakePtzService:
        async def Stop(self, params):
            calls.append(params)

    client._ptz_service = _FakePtzService()
    asyncio.run(client.stop())

    assert calls == [{"ProfileToken": "token1", "PanTilt": True, "Zoom": True}]
