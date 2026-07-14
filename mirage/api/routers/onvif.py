"""ONVIF discovery + stream resolution for the add-camera wizard.

Two-step flow, split across two different tools since neither one alone covers both
steps (verified directly against source/docs, not assumed):

1. **Scan** (`GET /api/onvif/scan`) proxies go2rtc's own `GET /api/onvif` (no `src`
   param), which does the actual WS-Discovery multicast probe and returns whatever
   ONVIF-compliant devices answer on the local network -- name, hardware info, and a
   go2rtc-internal `onvif://` placeholder URL. No credentials needed for this step
   (matches go2rtc's own behavior: discovery doesn't require auth, only the ONVIF
   Device/Media calls after it do).

2. **Resolve** (`GET /api/onvif/resolve`) takes the IP the user picked plus real
   credentials, and calls the device's own ONVIF Media service (GetProfiles +
   GetStreamUri) DIRECTLY via `onvif-zeep-async` -- NOT via go2rtc. This is deliberate:
   go2rtc's own `?src=onvif://...` resolve mode returns an `onvif://...?subtype=N` URL
   (still go2rtc's internal protocol, meant for go2rtc itself to consume), not a plain
   `rtsp://` URL -- and mirage's own ffmpeg-based capture process has no `onvif://`
   protocol support, only rtsp/http/tcp/etc. So this step needs the REAL underlying
   rtsp:// stream URI the camera exposes, which requires actually calling the ONVIF
   Media GetStreamUri operation ourselves.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onvif", tags=["onvif"])


class OnvifDeviceOut(BaseModel):
    name: str | None
    info: str | None
    url: str


class OnvifResolveOut(BaseModel):
    rtsp_url: str
    profile_name: str | None
    device_name: str | None


@router.get("/scan", response_model=list[OnvifDeviceOut])
async def scan_onvif_devices(request: Request) -> list[OnvifDeviceOut]:
    go2rtc_api_port = request.app.state.go2rtc_api_port
    url = f"http://127.0.0.1:{go2rtc_api_port}/api/onvif"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"go2rtc unreachable: {e}") from e

    if resp.status_code == 404:
        # go2rtc's own "no sources" case (see internal/api/api.go ResponseSources) --
        # plain-text 404, not an error, just an empty result.
        return []
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"go2rtc returned {resp.status_code} scanning for ONVIF devices")

    data = resp.json()
    return [
        OnvifDeviceOut(name=src.get("name"), info=src.get("info"), url=src["url"])
        for src in data.get("sources", [])
    ]


@router.get("/resolve", response_model=OnvifResolveOut)
async def resolve_onvif_stream(
    ip: str = Query(..., description="ONVIF device IP address"),
    port: int = Query(80, description="ONVIF device service port (usually 80)"),
    username: str = Query(..., description="ONVIF device username"),
    password: str = Query(..., description="ONVIF device password"),
) -> OnvifResolveOut:
    from onvif import ONVIFCamera
    from onvif.exceptions import ONVIFError

    camera = ONVIFCamera(ip, port, username, password)
    try:
        try:
            await camera.update_xaddrs()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"could not reach ONVIF device at {ip}:{port}: {e}") from e

        device_name: str | None = None
        try:
            devicemgmt = await camera.create_devicemgmt_service()
            info = await devicemgmt.GetDeviceInformation()
            device_name = f"{info.Manufacturer} {info.Model}".strip() or None
        except Exception:
            logger.warning("%s: GetDeviceInformation failed (non-fatal, continuing)", ip)

        try:
            media = await camera.create_media_service()
            profiles = await media.GetProfiles()
        except ONVIFError as e:
            raise HTTPException(status_code=502, detail=f"ONVIF GetProfiles failed for {ip}: {e}") from e

        if not profiles:
            raise HTTPException(status_code=404, detail=f"ONVIF device at {ip} reported no media profiles")

        profile = profiles[0]
        try:
            stream_setup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri_response = await media.GetStreamUri(
                {"StreamSetup": stream_setup, "ProfileToken": profile.token}
            )
        except ONVIFError as e:
            raise HTTPException(status_code=502, detail=f"ONVIF GetStreamUri failed for {ip}: {e}") from e

        rtsp_url = uri_response.Uri
        if not rtsp_url.startswith("rtsp://"):
            raise HTTPException(
                status_code=502,
                detail=f"ONVIF device at {ip} returned a non-RTSP stream URI: {rtsp_url!r}",
            )

        # Splice the real credentials into the URI (ONVIF devices typically return the
        # stream URI WITHOUT embedded credentials, since RTSP auth is usually separate
        # from the ONVIF Device/Media service auth -- but many cameras use the same
        # credentials for both, which is the common case this optimizes for).
        if "@" not in rtsp_url.split("://", 1)[1]:
            scheme, rest = rtsp_url.split("://", 1)
            rtsp_url = f"{scheme}://{username}:{password}@{rest}"

        return OnvifResolveOut(rtsp_url=rtsp_url, profile_name=getattr(profile, "Name", None), device_name=device_name)
    finally:
        await camera.close()
