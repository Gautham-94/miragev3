"""ONVIF PTZ control for a camera's manual joystick/preset controls -- mirrors
mirage/api/routers/onvif.py's style (a fresh PtzClient connection per request, using
this camera's persisted camera.ptz.* ONVIF credentials, mirage/ptz/client.py).

/status makes a fresh one-shot ONVIF GetStatus call, same as every other route here --
it does NOT read the running PtzPoller's live cached state (mirage/ptz/poller.py, which
lives inside the pipeline's per-camera CameraTracker process, a separate OS process
from this API). Wiring a live cross-process status channel was considered and
deliberately deferred (see the Mirage V3 PTZ implementation plan's risk list) since
nothing in this app's existing architecture shares live state between the pipeline and
API processes today, and a fresh ONVIF call is simple, consistent with the other
routes, and accurate at the moment it's requested.

/patrol/start and /patrol/stop are config MUTATIONS (camera.ptz.patrol_enabled), not
live operational toggles -- consistent with every other config change in this app
(mirage/api/routers/config.py), they only take effect once the user hits "Apply
changes" (mirage/supervisor.py's restart-request sentinel), same as changing any other
per-camera setting. This was an explicit design choice (not the only valid one) to
avoid inventing new pipeline<->API process IPC for this feature.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mirage.ptz.client import PtzClient

router = APIRouter(prefix="/api/ptz", tags=["ptz"])


class PtzPresetOut(BaseModel):
    token: str
    name: str | None


class PtzStatusOut(BaseModel):
    moving: bool
    pan: float | None
    tilt: float | None
    zoom: float | None


class PtzMoveRequest(BaseModel):
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0


def _get_ptz_camera(request: Request, camera_name: str):
    config = request.app.state.get_config()
    camera = config.cameras.get(camera_name)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera_name!r}")
    if not camera.ptz.enabled:
        raise HTTPException(status_code=400, detail=f"camera {camera_name!r} does not have PTZ enabled")
    return camera


async def _connected_client(camera) -> PtzClient:
    client = PtzClient(
        host=camera.ptz.onvif_host, port=camera.ptz.onvif_port,
        username=camera.ptz.onvif_username, password=camera.ptz.onvif_password,
    )
    try:
        await client.connect()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not reach PTZ camera {camera.name!r}: {e}") from e
    return client


@router.get("/{camera_name}/presets", response_model=list[PtzPresetOut])
async def list_presets(camera_name: str, request: Request) -> list[PtzPresetOut]:
    camera = _get_ptz_camera(request, camera_name)
    client = await _connected_client(camera)
    try:
        presets = await client.get_presets()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ONVIF GetPresets failed for {camera_name!r}: {e}") from e
    finally:
        await client.close()
    return [PtzPresetOut(token=p.token, name=p.name) for p in presets]


@router.get("/{camera_name}/status", response_model=PtzStatusOut)
async def get_status(camera_name: str, request: Request) -> PtzStatusOut:
    camera = _get_ptz_camera(request, camera_name)
    client = await _connected_client(camera)
    try:
        status = await client.get_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ONVIF GetStatus failed for {camera_name!r}: {e}") from e
    finally:
        await client.close()
    return PtzStatusOut(moving=status.moving, pan=status.pan, tilt=status.tilt, zoom=status.zoom)


@router.post("/{camera_name}/goto-preset/{preset_token}", status_code=204, response_model=None)
async def goto_preset(camera_name: str, preset_token: str, request: Request) -> None:
    camera = _get_ptz_camera(request, camera_name)
    client = await _connected_client(camera)
    try:
        await client.goto_preset(preset_token)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ONVIF GotoPreset failed for {camera_name!r}: {e}") from e
    finally:
        await client.close()


@router.post("/{camera_name}/move", status_code=204, response_model=None)
async def move(camera_name: str, body: PtzMoveRequest, request: Request) -> None:
    camera = _get_ptz_camera(request, camera_name)
    client = await _connected_client(camera)
    try:
        await client.continuous_move(pan_speed=body.pan, tilt_speed=body.tilt, zoom_speed=body.zoom)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ONVIF ContinuousMove failed for {camera_name!r}: {e}") from e
    finally:
        await client.close()


@router.post("/{camera_name}/stop", status_code=204, response_model=None)
async def stop(camera_name: str, request: Request) -> None:
    camera = _get_ptz_camera(request, camera_name)
    client = await _connected_client(camera)
    try:
        await client.stop()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ONVIF Stop failed for {camera_name!r}: {e}") from e
    finally:
        await client.close()


@router.post("/{camera_name}/patrol/start", status_code=204, response_model=None)
def start_patrol(camera_name: str, request: Request) -> None:
    """Config mutation, not a live toggle -- see this module's own docstring for why.
    Takes effect on the next pipeline restart ("Apply changes"), same as any other
    per-camera config change.
    """
    config = request.app.state.get_config()
    camera = config.cameras.get(camera_name)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera_name!r}")
    if not camera.ptz.enabled:
        raise HTTPException(status_code=400, detail=f"camera {camera_name!r} does not have PTZ enabled")
    camera.ptz.patrol_enabled = True
    config.save_to_db()


@router.post("/{camera_name}/patrol/stop", status_code=204, response_model=None)
def stop_patrol(camera_name: str, request: Request) -> None:
    config = request.app.state.get_config()
    camera = config.cameras.get(camera_name)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera_name!r}")
    camera.ptz.patrol_enabled = False
    config.save_to_db()
