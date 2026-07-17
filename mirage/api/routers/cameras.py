from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from mirage.api.schemas import CameraOut

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraOut])
def list_cameras(request: Request) -> list[CameraOut]:
    config = request.app.state.get_config()
    return [
        CameraOut(
            name=cam.name,
            enabled=cam.enabled,
            width=cam.detect.width,
            height=cam.detect.height,
            fps=cam.detect.fps,
            detector=cam.detector,
            record_enabled=cam.record.enabled,
            track_objects=cam.objects.track,
            track_all=cam.objects.track_all,
        )
        for cam in config.cameras.values()
    ]


@router.get("/{name}", response_model=CameraOut)
def get_camera(name: str, request: Request) -> CameraOut:
    config = request.app.state.get_config()
    cam = config.cameras.get(name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")
    return CameraOut(
        name=cam.name,
        enabled=cam.enabled,
        width=cam.detect.width,
        height=cam.detect.height,
        fps=cam.detect.fps,
        detector=cam.detector,
        record_enabled=cam.record.enabled,
        track_objects=cam.objects.track,
        track_all=cam.objects.track_all,
    )
