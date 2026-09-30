"""Live-view backend surface: proxies to go2rtc so the Angular frontend only ever talks
to one origin (the mirage API), never go2rtc's own port directly.

Two endpoints, matching the two tiers a camera tile needs (see
mirage/go2rtc: WebRTC/MSE for real video, JPEG snapshot as the poster/fallback):

- GET  /api/live/{camera}/snapshot.jpg  -- proxies go2rtc's GET /api/frame.jpeg?src=...
- WS   /api/live/{camera}/ws            -- bidirectionally relays to go2rtc's
  GET /api/ws?src=... (the signaling channel the MSE/WebRTC players speak over)
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from websockets.exceptions import ConnectionClosed

from mirage.go2rtc.config import SUB_STREAM_SUFFIX

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live"])


def _go2rtc_api_base(request: Request) -> str:
    return f"http://{request.app.state.go2rtc_host}:{request.app.state.go2rtc_api_port}"


def _known_camera(stream_name: str, config) -> bool:
    """`stream_name` is whatever go2rtc stream the frontend asked for -- either a real
    camera name (main, full-res tier) or "<camera>_sub" (the low-res grid tier go2rtc
    registers for it, see mirage.go2rtc.config.build_go2rtc_config). Both route through
    these same two endpoints unchanged; only the go2rtc `src=` value differs, so
    validation just needs to check the BASE camera name still exists.
    """
    base = stream_name.removesuffix(SUB_STREAM_SUFFIX)
    return base in config.cameras


@router.get("/{camera}/snapshot.jpg")
async def get_snapshot(camera: str, request: Request) -> Response:
    config = request.app.state.get_config()
    if not _known_camera(camera, config):
        raise HTTPException(status_code=404, detail=f"unknown camera {camera!r}")

    url = f"{_go2rtc_api_base(request)}/api/frame.jpeg"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={"src": camera}, timeout=10.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"go2rtc unreachable: {e}") from e

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"go2rtc returned {resp.status_code} for snapshot of {camera!r}")

    return Response(content=resp.content, media_type="image/jpeg")


@router.websocket("/{camera}/ws")
async def live_ws_proxy(websocket: WebSocket, camera: str) -> None:
    """Relays raw WebSocket frames both ways between the browser and go2rtc's own
    /api/ws?src=<camera> signaling endpoint, so the MSE/WebRTC player code in the
    frontend can speak go2rtc's protocol directly without knowing go2rtc's port/host.
    """
    config = websocket.app.state.get_config()
    if not _known_camera(camera, config):
        await websocket.close(code=4004, reason=f"unknown camera {camera!r}")
        return

    await websocket.accept()

    go2rtc_url = f"ws://{websocket.app.state.go2rtc_host}:{websocket.app.state.go2rtc_api_port}/api/ws?src={camera}"
    try:
        upstream = await websockets.connect(go2rtc_url, open_timeout=10)
    except OSError as e:
        logger.warning("could not connect to go2rtc for camera %s: %s", camera, e)
        await websocket.close(code=1011, reason="go2rtc unreachable")
        return

    async def browser_to_go2rtc():
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if "text" in message and message["text"] is not None:
                    await upstream.send(message["text"])
                elif "bytes" in message and message["bytes"] is not None:
                    await upstream.send(message["bytes"])
        except WebSocketDisconnect:
            pass
        finally:
            await upstream.close()

    async def go2rtc_to_browser():
        try:
            async for message in upstream:
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_bytes(message)
        except ConnectionClosed:
            pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass  # already closed by the other relay direction

    try:
        await asyncio.gather(browser_to_go2rtc(), go2rtc_to_browser())
    except Exception:
        logger.exception("%s: live view WS proxy error", camera)
