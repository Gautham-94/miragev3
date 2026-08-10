"""Config-write endpoints: lets the add-camera wizard (and any other client) create,
update, or remove a camera without hand-editing config/mirage.yaml -- see
mirage/config/store.py for the underlying DB-backed persistence, and
IMPLEMENTATION_NOTES.md sections 13-14 for why this exists.

All writes go through the same CameraConfig/MirageConfig Pydantic validation a hand-typed
YAML file would have gone through -- this router builds a CameraConfig from a flatter,
wizard-friendly request shape (CameraCreate/CameraUpdate) and lets Pydantic's own
validators (e.g. FfmpegConfig's "exactly one detect-role input" rule) reject anything
invalid before it's persisted.

None of this takes effect in the currently-running capture/detect/record pipeline
(MirageApp) -- config changes are only picked up on the next `python -m mirage` restart,
which the user must now trigger explicitly via the "Apply changes" button on the
frontend (see mirage/api/routers/system.py, TODO_FIX_LIST.md item 2 -- hot-reload was
tried and reverted after it caused a real, hard-to-diagnose issue where a restarted
camera silently stopped producing new detections). Every mutating response here says so
explicitly via `restart_required: true` rather than silently implying otherwise.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from mirage.api.schemas import CameraOut
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    ExecutionProvider,
    FfmpegConfig,
    InputDType,
    ModelConfig,
    MirageConfig,
    ObjectFilterConfig,
    ObjectsConfig,
    PixelFormat,
    RecordConfig,
    RetainConfig,
    ReviewConfig,
    ReviewLabelConfig,
    RtspTransport,
    RulesConfig,
    TensorLayout,
)
from mirage.detection.execution_providers import available_execution_providers
from mirage.detection.registry import available_backends

router = APIRouter(prefix="/api/config", tags=["config"])


class DetectorOut(BaseModel):
    name: str
    device: str
    model_width: int
    model_height: int
    execution_provider: ExecutionProvider
    model_path: str
    labelmap_path: str
    enabled: bool
    cameras: list[str] = []
    num_workers: int = 1


class CameraWriteRequest(BaseModel):
    """Flat, wizard-friendly shape for adding/editing a camera -- deliberately not the
    full nested CameraConfig tree, since the wizard only ever collects a handful of
    fields; everything else (motion, review label lists, filter thresholds, ...) keeps
    mirage's existing defaults, same as a hand-typed minimal YAML camera block would.
    """

    name: str = Field(min_length=1)
    rtsp_url: str = Field(min_length=1)
    detector: str
    track_objects: list[str] = Field(default_factory=lambda: ["person"])
    # See ObjectsConfig.track_all's docstring -- when True, `track_objects` above is
    # ignored for the closed-vocab tracking gate (every routed detector label is
    # tracked instead), but is still saved/returned as-is so re-disabling this doesn't
    # lose whatever the user had typed there.
    track_all: bool = False
    width: int = 640
    height: int = 480
    fps: int = Field(default=5, ge=1, le=30)
    record_enabled: bool = True
    retain_days: float = Field(default=7, ge=0)
    # Matches Frigate's own real default (NVR_PIPELINE_IMPLEMENTATION_SPEC.md section
    # 1.2.5: "-segment_time 10 ... 10-second segments"), and RecordConfig's own schema
    # default -- the 60s ceiling exists specifically because coarser segments make
    # retention/deletion granularity worse and increase data-loss risk on an ungraceful
    # shutdown (see RecordConfig.segment_seconds's Field(le=60)).
    segment_seconds: int = Field(default=10, ge=1, le=60)
    alert_labels: list[str] | None = None
    detection_labels: list[str] | None = None
    enabled: bool = True
    # TCP is the safer default (see CameraInputConfig.rtsp_transport's docstring), but
    # some cameras' RTSP-over-TCP implementations are unreliable -- UDP is the escape
    # hatch, exposed here so the wizard can offer it as a per-camera choice.
    rtsp_transport: RtspTransport = RtspTransport.tcp
    # See RulesConfig's own docstring -- both None/0 (default) disable their respective
    # rule. crowd_threshold: alert once this many confirmed "person" tracks are present
    # at once. dwell_seconds: alert once ANY tracked object has been continuously
    # present for at least this long (covers both loitering and queue-wait-time).
    crowd_threshold: int | None = Field(default=None, ge=1)
    dwell_seconds: int | None = Field(default=None, ge=1)
    # See ObjectFilterConfig's own docstring -- min_score gates whether a raw detection
    # is tracked at all; threshold gates whether a tracked object's median score is ever
    # promoted from false_positive to true-positive (and therefore shown in Events/
    # Review). None (default) keeps ObjectFilterConfig's own schema defaults (0.5/0.7).
    # Exposed per-camera, not per-detector: two cameras routed to the identical model can
    # legitimately need different thresholds (e.g. one mounted far from the action scores
    # lower on genuine detections than one close-up), so this is a scene-confidence
    # tuning knob, not a property of the model itself.
    min_score: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)


class ConfigMutationResponse(BaseModel):
    ok: bool = True
    restart_required: bool = True
    camera: CameraOut


class CameraConfigOut(BaseModel):
    """Like CameraOut, but for the config-write router's own use ONLY -- includes the raw
    rtsp_url (which embeds credentials) so the edit form can pre-fill it. Deliberately NOT
    used by the general-purpose /api/cameras listing (mirage/api/routers/cameras.py),
    which stays on the credential-free CameraOut, since that endpoint's response is
    consumed more broadly (e.g. the Live page's camera grid) and shouldn't leak RTSP
    credentials to every reader.
    """

    name: str
    enabled: bool
    width: int
    height: int
    fps: int
    detector: str
    record_enabled: bool
    track_objects: list[str]
    track_all: bool
    rtsp_url: str
    rtsp_transport: RtspTransport
    retain_days: float
    segment_seconds: int
    alert_labels: list[str]
    detection_labels: list[str]
    crowd_threshold: int | None
    dwell_seconds: int | None
    min_score: float
    threshold: float


def _primary_input(cam: CameraConfig) -> CameraInputConfig | None:
    return cam.ffmpeg.inputs[0] if cam.ffmpeg.inputs else None


def _camera_out(cam: CameraConfig) -> CameraOut:
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


def _camera_config_out(cam: CameraConfig) -> CameraConfigOut:
    primary_input = _primary_input(cam)
    return CameraConfigOut(
        name=cam.name,
        enabled=cam.enabled,
        width=cam.detect.width,
        height=cam.detect.height,
        fps=cam.detect.fps,
        detector=cam.detector,
        record_enabled=cam.record.enabled,
        track_objects=cam.objects.track,
        track_all=cam.objects.track_all,
        rtsp_url=primary_input.path if primary_input else "",
        rtsp_transport=primary_input.rtsp_transport if primary_input else RtspTransport.tcp,
        retain_days=cam.record.continuous.days,
        segment_seconds=cam.record.segment_seconds,
        alert_labels=cam.review.alerts.labels,
        detection_labels=cam.review.detections.labels,
        crowd_threshold=cam.rules.crowd_threshold,
        dwell_seconds=cam.rules.dwell_seconds,
        # objects.filters is keyed per-label (ObjectsConfig.filter_for), not a single
        # camera-wide value -- this form edits one shared confidence bar applied to every
        # currently-tracked label uniformly (see _build_camera_config), so reading back
        # any one tracked label's filter (they're always written identically by this
        # form) reflects the same value. Untracked/never-customized labels fall back to
        # ObjectFilterConfig()'s own schema defaults via filter_for.
        min_score=cam.objects.filter_for(cam.objects.track[0]).min_score if cam.objects.track else ObjectFilterConfig().min_score,
        threshold=cam.objects.filter_for(cam.objects.track[0]).threshold if cam.objects.track else ObjectFilterConfig().threshold,
    )


def _build_camera_config(req: CameraWriteRequest) -> CameraConfig:
    review_kwargs = {}
    if req.alert_labels is not None:
        review_kwargs["alerts"] = ReviewLabelConfig(labels=req.alert_labels)
    if req.detection_labels is not None:
        review_kwargs["detections"] = ReviewLabelConfig(labels=req.detection_labels)

    filters_kwargs = {}
    if req.min_score is not None:
        filters_kwargs["min_score"] = req.min_score
    if req.threshold is not None:
        filters_kwargs["threshold"] = req.threshold
    # ObjectsConfig.filters is keyed per-label (see ObjectsConfig.filter_for) -- this
    # form edits one shared confidence bar, applied uniformly to every label this camera
    # tracks, rather than exposing per-label overrides the wizard has no UI for.
    shared_filter = ObjectFilterConfig(**filters_kwargs) if filters_kwargs else ObjectFilterConfig()
    filters = {label: shared_filter for label in req.track_objects}

    return CameraConfig(
        name=req.name,
        enabled=req.enabled,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=req.rtsp_url, rtsp_transport=req.rtsp_transport)]),
        detect=DetectConfig(width=req.width, height=req.height, fps=req.fps),
        objects=ObjectsConfig(track=req.track_objects, track_all=req.track_all, filters=filters),
        record=RecordConfig(
            enabled=req.record_enabled, segment_seconds=req.segment_seconds, continuous=RetainConfig(days=req.retain_days)
        ),
        review=ReviewConfig(**review_kwargs) if review_kwargs else ReviewConfig(),
        rules=RulesConfig(crowd_threshold=req.crowd_threshold, dwell_seconds=req.dwell_seconds),
        detector=req.detector,
    )


@router.get("/detectors", response_model=list[DetectorOut])
def list_detectors(request: Request) -> list[DetectorOut]:
    config = request.app.state.get_config()
    return [_detector_out(d, config) for d in config.detectors.values()]


@router.get("/execution-providers", response_model=list[str])
def list_execution_providers() -> list[str]:
    """Which ExecutionProvider values will actually work on the machine mirage.api is
    running on right now -- lets the Manage Detectors form only ever offer an
    accelerator choice that's real, rather than one that would silently no-op back to
    CPU (see mirage/detection/execution_providers.py).
    """
    return available_execution_providers()


class DetectorWriteRequest(BaseModel):
    """Registers a new detector instance by path, e.g. a wildlife/animal-specialist
    model, so cameras can be routed to it via CameraWriteRequest.detector (see
    NVR_EXTENSION_MULTI_MODEL_ROUTING.md). File-upload is deliberately out of scope for
    now -- mirage runs as a single local process on the same machine serving this API, so
    a path field pointing at a model file already placed on disk is sufficient; an upload
    endpoint can be added later without changing this shape.
    """

    name: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    labelmap_path: str = Field(min_length=1)
    device: str = "onnx_yolov8"
    width: int = Field(default=320, gt=0)
    height: int = Field(default=320, gt=0)
    input_dtype: InputDType = InputDType.int_
    pixel_format: PixelFormat = PixelFormat.rgb
    layout: TensorLayout = TensorLayout.nhwc
    # cpu by default -- deliberately not "auto", so registering a detector never silently
    # changes behavior based on what hardware happens to be available on this particular
    # machine; the caller opts into an accelerator explicitly (see ExecutionProvider's
    # docstring for the real CPU-vs-CoreML benchmark this is based on).
    execution_provider: ExecutionProvider = ExecutionProvider.cpu
    # False skips spawning a DetectorProcess for this detector entirely on next restart
    # (see DetectorInstanceConfig.enabled's docstring) -- lets a heavy/experimental
    # detector be registered and kept configured without paying its idle memory cost.
    enabled: bool = True
    # See DetectorInstanceConfig.num_workers's own docstring -- how many independent
    # worker processes serve cameras routed to this detector, all sharing one queue for
    # free least-busy-worker routing. 1 (default) is the original
    # single-process-per-detector behavior.
    num_workers: int = Field(default=1, ge=1, le=8)


def _validate_and_build_detector(req: DetectorWriteRequest) -> DetectorInstanceConfig:
    known_backends = available_backends()
    if req.device not in known_backends:
        raise HTTPException(
            status_code=422,
            detail=f"unknown detector backend {req.device!r}; available: {known_backends}",
        )
    if not os.path.isfile(req.model_path):
        raise HTTPException(status_code=422, detail=f"model_path {req.model_path!r} does not exist")
    if not os.path.isfile(req.labelmap_path):
        raise HTTPException(status_code=422, detail=f"labelmap_path {req.labelmap_path!r} does not exist")

    known_providers = available_execution_providers()
    if req.execution_provider.value not in known_providers:
        raise HTTPException(
            status_code=422,
            detail=f"execution provider {req.execution_provider.value!r} is not available on this machine; "
            f"available: {known_providers}",
        )

    try:
        return DetectorInstanceConfig(
            name=req.name,
            device=req.device,
            enabled=req.enabled,
            num_workers=req.num_workers,
            model=ModelConfig(
                width=req.width,
                height=req.height,
                input_dtype=req.input_dtype,
                pixel_format=req.pixel_format,
                layout=req.layout,
                model_path=req.model_path,
                labelmap_path=req.labelmap_path,
                execution_provider=req.execution_provider,
            ),
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _detector_out(detector: DetectorInstanceConfig, config: MirageConfig | None = None) -> DetectorOut:
    cameras = []
    if config is not None:
        cameras = sorted(cam.name for cam in config.cameras.values() if cam.detector == detector.name)
    return DetectorOut(
        name=detector.name, device=detector.device, model_width=detector.model.width, model_height=detector.model.height,
        execution_provider=detector.model.execution_provider,
        model_path=detector.model.model_path, labelmap_path=detector.model.labelmap_path,
        enabled=detector.enabled, cameras=cameras, num_workers=detector.num_workers,
    )


@router.post("/detectors", response_model=DetectorOut, status_code=201)
def create_detector(req: DetectorWriteRequest, request: Request) -> DetectorOut:
    config = request.app.state.get_config()

    if req.name in config.detectors:
        raise HTTPException(status_code=409, detail=f"detector {req.name!r} already exists")

    detector = _validate_and_build_detector(req)
    config.detectors[req.name] = detector
    config.save_to_db()

    return _detector_out(detector, config)


@router.put("/detectors/{name}", response_model=DetectorOut)
def update_detector(name: str, req: DetectorWriteRequest, request: Request) -> DetectorOut:
    """Edits an existing detector in place -- most commonly to change just
    execution_provider (e.g. a camera was set up on CPU and the user later wants to
    switch it to CoreML/CUDA once they've confirmed the accelerator works), but accepts
    the full write shape like create_detector so any field can be changed. Renaming
    (req.name != name) is intentionally NOT supported here -- every CameraConfig.detector
    reference points at the name, and silently rewriting all of them is a bigger, riskier
    operation than this endpoint's scope; delete+recreate under a new name if a rename is
    truly needed, same as the constraint already on delete_detector.
    """
    config = request.app.state.get_config()

    if name not in config.detectors:
        raise HTTPException(status_code=404, detail=f"unknown detector {name!r}")
    if req.name != name:
        raise HTTPException(
            status_code=422,
            detail="renaming a detector is not supported; delete and recreate under the new name instead",
        )

    detector = _validate_and_build_detector(req)
    config.detectors[name] = detector
    config.save_to_db()

    return _detector_out(detector, config)


class DetectorEnabledRequest(BaseModel):
    enabled: bool


@router.patch("/detectors/{name}/enabled", response_model=DetectorOut)
def set_detector_enabled(name: str, req: DetectorEnabledRequest, request: Request) -> DetectorOut:
    """Lightweight toggle for the config/control page -- flips just
    DetectorInstanceConfig.enabled without requiring the caller to resend every other
    field (model path, execution provider, ...) the way PUT /detectors/{name} does.
    """
    config = request.app.state.get_config()

    detector = config.detectors.get(name)
    if detector is None:
        raise HTTPException(status_code=404, detail=f"unknown detector {name!r}")

    detector.enabled = req.enabled
    config.save_to_db()

    return _detector_out(detector, config)


class DetectorNumWorkersRequest(BaseModel):
    num_workers: int = Field(ge=1, le=8)


@router.patch("/detectors/{name}/num-workers", response_model=DetectorOut)
def set_detector_num_workers(name: str, req: DetectorNumWorkersRequest, request: Request) -> DetectorOut:
    """Lightweight control for the Config page's worker-count stepper -- see
    DetectorInstanceConfig.num_workers's own docstring for what this actually changes
    (independent OS processes sharing one queue, serving this detector's routed
    cameras with free least-busy-worker routing; takes effect on next pipeline restart
    like every other config change here).
    """
    config = request.app.state.get_config()

    detector = config.detectors.get(name)
    if detector is None:
        raise HTTPException(status_code=404, detail=f"unknown detector {name!r}")

    detector.num_workers = req.num_workers
    config.save_to_db()

    return _detector_out(detector, config)


@router.delete("/detectors/{name}", status_code=204, response_model=None)
def delete_detector(name: str, request: Request) -> None:
    config = request.app.state.get_config()

    if name not in config.detectors:
        raise HTTPException(status_code=404, detail=f"unknown detector {name!r}")

    cameras_using_it = sorted(cam.name for cam in config.cameras.values() if cam.detector == name)
    if cameras_using_it:
        raise HTTPException(
            status_code=409,
            detail=f"detector {name!r} is still used by camera(s): {cameras_using_it}; reassign them first",
        )

    del config.detectors[name]
    config.save_to_db()


@router.get("/cameras", response_model=list[CameraConfigOut])
def list_camera_configs(request: Request) -> list[CameraConfigOut]:
    """Powers the "Manage cameras" page -- includes rtsp_url/rtsp_transport/retain_days/
    label lists, unlike GET /api/cameras (mirage/api/routers/cameras.py), so the edit form
    can pre-fill from a real, complete existing config rather than starting blank.
    """
    config = request.app.state.get_config()
    return [_camera_config_out(cam) for cam in config.cameras.values()]


@router.get("/cameras/{name}", response_model=CameraConfigOut)
def get_camera_config(name: str, request: Request) -> CameraConfigOut:
    config = request.app.state.get_config()
    cam = config.cameras.get(name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")
    return _camera_config_out(cam)


@router.post("/cameras", response_model=ConfigMutationResponse, status_code=201)
def create_camera(req: CameraWriteRequest, request: Request) -> ConfigMutationResponse:
    config = request.app.state.get_config()

    if req.name in config.cameras:
        raise HTTPException(status_code=409, detail=f"camera {req.name!r} already exists")
    if req.detector not in config.detectors:
        raise HTTPException(
            status_code=422,
            detail=f"unknown detector {req.detector!r}; known detectors: {sorted(config.detectors)}",
        )

    try:
        camera = _build_camera_config(req)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config.cameras[req.name] = camera
    config.save_to_db()

    return ConfigMutationResponse(camera=_camera_out(camera))


@router.put("/cameras/{name}", response_model=ConfigMutationResponse)
def update_camera(name: str, req: CameraWriteRequest, request: Request) -> ConfigMutationResponse:
    config = request.app.state.get_config()

    if name not in config.cameras:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")
    if req.detector not in config.detectors:
        raise HTTPException(
            status_code=422,
            detail=f"unknown detector {req.detector!r}; known detectors: {sorted(config.detectors)}",
        )
    if req.name != name and req.name in config.cameras:
        raise HTTPException(status_code=409, detail=f"camera {req.name!r} already exists")

    try:
        camera = _build_camera_config(req)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if req.name != name:
        del config.cameras[name]
    config.cameras[req.name] = camera
    config.save_to_db()

    return ConfigMutationResponse(camera=_camera_out(camera))


@router.delete("/cameras/{name}", status_code=204, response_model=None)
def delete_camera(name: str, request: Request) -> None:
    config = request.app.state.get_config()

    if name not in config.cameras:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")

    del config.cameras[name]
    config.save_to_db()
