"""ONVIF PTZ client wrapper -- reuses onvif-zeep-async's ONVIFCamera exactly as
mirage/api/routers/onvif.py's discovery/resolve wizard already does (same
create_media_service()/GetProfiles() call to obtain a media profile token), adding
create_ptz_service() for actual PTZ control: GetStatus, ContinuousMove, Stop,
GotoPreset, GetPresets.

Mirage V3 PTZ hybrid scheduling (mirage/tracking/orchestration.py) only needs
get_status()'s `moving` boolean; the rest (continuous_move/goto_preset/get_presets/stop)
back the manual-control and patrol-scheduling API routes (mirage/api/routers/ptz.py).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PtzStatus:
    moving: bool
    pan: float | None = None
    tilt: float | None = None
    zoom: float | None = None


@dataclasses.dataclass
class PtzPreset:
    token: str
    name: str | None = None


class PtzClient:
    """One instance per PTZ-enabled camera. Must call connect() before any other
    method -- kept separate from __init__ since ONVIF's update_xaddrs()/GetProfiles()
    are async network calls, and __init__ can't be async.
    """

    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._camera = None
        self._ptz_service = None
        self._media_profile_token: str | None = None

    async def connect(self) -> None:
        from onvif import ONVIFCamera

        self._camera = ONVIFCamera(self._host, self._port, self._username, self._password)
        await self._camera.update_xaddrs()

        media = await self._camera.create_media_service()
        profiles = await media.GetProfiles()
        if not profiles:
            raise ValueError(f"ONVIF device at {self._host} reported no media profiles")
        self._media_profile_token = profiles[0].token

        self._ptz_service = await self._camera.create_ptz_service()

    async def get_status(self) -> PtzStatus:
        status = await self._ptz_service.GetStatus({"ProfileToken": self._media_profile_token})
        move_status = getattr(status, "MoveStatus", None)
        pan_tilt_status = getattr(move_status, "PanTilt", "IDLE") if move_status is not None else "IDLE"
        moving = pan_tilt_status == "MOVING"

        position = getattr(status, "Position", None)
        pan_tilt = getattr(position, "PanTilt", None) if position is not None else None
        zoom = getattr(position, "Zoom", None) if position is not None else None
        return PtzStatus(
            moving=moving,
            pan=getattr(pan_tilt, "x", None) if pan_tilt is not None else None,
            tilt=getattr(pan_tilt, "y", None) if pan_tilt is not None else None,
            zoom=getattr(zoom, "x", None) if zoom is not None else None,
        )

    async def continuous_move(self, pan_speed: float, tilt_speed: float, zoom_speed: float = 0.0) -> None:
        await self._ptz_service.ContinuousMove({
            "ProfileToken": self._media_profile_token,
            "Velocity": {"PanTilt": {"x": pan_speed, "y": tilt_speed}, "Zoom": {"x": zoom_speed}},
        })

    async def stop(self) -> None:
        await self._ptz_service.Stop({"ProfileToken": self._media_profile_token, "PanTilt": True, "Zoom": True})

    async def goto_preset(self, preset_token: str) -> None:
        await self._ptz_service.GotoPreset({"ProfileToken": self._media_profile_token, "PresetToken": preset_token})

    async def get_presets(self) -> list[PtzPreset]:
        presets = await self._ptz_service.GetPresets({"ProfileToken": self._media_profile_token})
        return [PtzPreset(token=p.token, name=getattr(p, "Name", None)) for p in presets]

    async def close(self) -> None:
        if self._camera is not None:
            await self._camera.close()
