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
from mirage.const import resolve_model_path
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
from mirage.detection.labelmap import load_labels
from mirage.detection.registry import available_backends

router = APIRouter(prefix="/api/config", tags=["config"])


def _detector_labels(detector: DetectorInstanceConfig) -> list[str]:
    """The real, distinct label names a detector's labelmap file declares -- excludes
    load_labels()'s "unknown" prefill entries (indices the labelmap file never actually
    assigns), so this only ever returns labels the model can genuinely emit (e.g.
    ["animal", "person", "vehicle"] for the MegaDetector-family plugins, COCO's 80 for
    "general"/yolov8n). Sorted for a stable, predictable UI row order regardless of the
    labelmap file's own line order or index gaps.
    """
    raw = load_labels(resolve_model_path(detector.model.labelmap_path))
    return sorted({label for label in raw.values() if label != "unknown"})


class LabelFilterOut(BaseModel):
    min_score: float
    threshold: float


class LabelFilterRequest(BaseModel):
    min_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


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
    # Optional cheap, low-resolution stream URL for the live-view grid tier -- see
    # mirage.config.schema.CameraConfig.live_sub_url's own docstring. Left unset
    # (None), go2rtc transcodes the main stream down for the grid instead -- always
    # works, just costs an extra ffmpeg process per camera.
    live_sub_url: str | None = None
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
    # Review). Keyed per-label (e.g. "animal"/"person"/"vehicle" for the MegaDetector-
    # family plugins -- see _detector_labels), NOT one shared value applied to every
    # tracked label uniformly: two labels on the same camera can legitimately need very
    # different bars (a real bug this replaces -- the single shared field used to write
    # to whichever label happened to be track_objects[0], silently leaving every other
    # label, including ones tracked only via track_all, stuck on the 0.5/0.7 schema
    # default with no way to change them). A label omitted here keeps its existing
    # filter untouched -- see _build_camera_config's merge, not a full replace.
    filters: dict[str, LabelFilterRequest] | None = None


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
    live_sub_url: str | None
    retain_days: float
    segment_seconds: int
    alert_labels: list[str]
    detection_labels: list[str]
    crowd_threshold: int | None
    dwell_seconds: int | None
    # Keyed by every label the camera's assigned detector's labelmap declares (see
    # _detector_labels), not just whatever's in track_objects -- a label only tracked
    # via track_all (e.g. "animal" on a camera whose track_objects lists just "person")
    # still needs its own editable filter row. A label never explicitly customized falls
    # back to ObjectFilterConfig()'s own schema defaults via filter_for.
    filters: dict[str, LabelFilterOut]


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


def _camera_config_out(cam: CameraConfig, config: MirageConfig) -> CameraConfigOut:
    primary_input = _primary_input(cam)
    detector = config.detectors.get(cam.detector)
    # Falls back to whatever labels this camera already has customized filters for if
    # the assigned detector is somehow missing (shouldn't happen in practice -- every
    # camera's detector is validated to exist at write time) or its labelmap file can't
    # be read right now, so this endpoint degrades rather than 500s.
    try:
        available_labels = _detector_labels(detector) if detector is not None else list(cam.objects.filters)
    except OSError:
        available_labels = list(cam.objects.filters)
    filters_out = {
        label: LabelFilterOut(min_score=cam.objects.filter_for(label).min_score, threshold=cam.objects.filter_for(label).threshold)
        for label in available_labels
    }
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
        live_sub_url=cam.live_sub_url,
        retain_days=cam.record.continuous.days,
        segment_seconds=cam.record.segment_seconds,
        alert_labels=cam.review.alerts.labels,
        detection_labels=cam.review.detections.labels,
        crowd_threshold=cam.rules.crowd_threshold,
        dwell_seconds=cam.rules.dwell_seconds,
        filters=filters_out,
    )


def _build_camera_config(req: CameraWriteRequest, existing: CameraConfig | None) -> CameraConfig:
    review_kwargs = {}
    if req.alert_labels is not None:
        review_kwargs["alerts"] = ReviewLabelConfig(labels=req.alert_labels)
    if req.detection_labels is not None:
        review_kwargs["detections"] = ReviewLabelConfig(labels=req.detection_labels)

    # Merge, not replace: start from whatever this camera already had (nothing, for a
    # brand-new camera), then overlay only the labels req.filters actually names. A
    # label this request doesn't mention keeps its existing filter untouched -- this is
    # the fix for a real bug (confirmed live): the old shared-single-value design
    # rebuilt ObjectsConfig.filters from scratch on every save using only
    # track_objects[0], silently discarding any other label's customized filter
    # (including labels only ever tracked via track_all, which aren't in track_objects
    # at all) even when the user never touched that label's own settings.
    filters = dict(existing.objects.filters) if existing is not None else {}
    for label, f in (req.filters or {}).items():
        # model_copy (not a fresh ObjectFilterConfig(...)) so a label's already-set
        # min_area/max_area/min_ratio/max_ratio (not exposed by this form at all) survive
        # a min_score/threshold-only edit, rather than silently resetting to schema
        # defaults.
        current = filters.get(label, ObjectFilterConfig())
        filters[label] = current.model_copy(update={"min_score": f.min_score, "threshold": f.threshold})

    return CameraConfig(
        name=req.name,
        enabled=req.enabled,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=req.rtsp_url, rtsp_transport=req.rtsp_transport)]),
        # Falsy-collapsed to None -- a form field left blank submits "" here, which
        # should mean "no override" (fall back to the ffmpeg-transcode grid tier), not a
        # literal empty-string stream URL go2rtc would fail to connect to.
        live_sub_url=req.live_sub_url or None,
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


@router.get("/detector-backends", response_model=list[str])
def list_detector_backends() -> list[str]:
    """Every registered detector plugin's type_key (mirage/detection/registry.py's
    available_backends(), which scans mirage/detection/plugins/ automatically) -- lets
    the Manage Detectors form's backend dropdown stay in sync with whatever plugins
    actually exist, rather than a hand-maintained list that silently drifts out of date
    whenever a plugin is added (e.g. onnx_yolo_nms alongside onnx_yolov8/onnx_megadetector).
    """
    return available_backends()


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
    # resolve_model_path: a bare "models/xyz.onnx"-style path (the convention every
    # bundled model is registered under, matching MirageConfig.default()'s own seeded
    # detector) is relative to MODEL_CACHE_DIR, not the API process's cwd -- see that
    # function's own docstring. An absolute path (a client Browse-ing to a custom model
    # file) passes through unchanged either way.
    if not os.path.isfile(resolve_model_path(req.model_path)):
        raise HTTPException(status_code=422, detail=f"model_path {req.model_path!r} does not exist")
    if not os.path.isfile(resolve_model_path(req.labelmap_path)):
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


class SpeciesConfigOut(BaseModel):
    enabled: bool


class SpeciesEnabledRequest(BaseModel):
    enabled: bool


@router.get("/species", response_model=SpeciesConfigOut)
def get_species_config(request: Request) -> SpeciesConfigOut:
    """Powers the Config page's species-identification toggle -- see
    SpeciesClassifierConfig's own docstring for why this defaults to enabled (the
    packaged build vendors everything it needs; this is an opt-out, not an opt-in).
    """
    config = request.app.state.get_config()
    return SpeciesConfigOut(enabled=config.species_classifier.enabled)


@router.patch("/species/enabled", response_model=SpeciesConfigOut)
def set_species_enabled(req: SpeciesEnabledRequest, request: Request) -> SpeciesConfigOut:
    """Lightweight toggle, same shape as PATCH /detectors/{name}/enabled -- takes
    effect on the next pipeline restart ("Apply changes"), like every other config
    change here.
    """
    config = request.app.state.get_config()
    config.species_classifier.enabled = req.enabled
    config.save_to_db()
    return SpeciesConfigOut(enabled=config.species_classifier.enabled)


class HwaccelConfigOut(BaseModel):
    enabled: bool


class HwaccelEnabledRequest(BaseModel):
    enabled: bool


@router.get("/hwaccel", response_model=HwaccelConfigOut)
def get_hwaccel_config(request: Request) -> HwaccelConfigOut:
    """Powers the Config page's hardware decode toggle -- see
    MirageConfig.hwaccel_enabled's own docstring for why this defaults to enabled and
    what it actually does (fills in NVIDIA decode args for cameras that don't already
    have their own, only when a real GPU is detected -- never forces it on a machine
    that can't do it).
    """
    config = request.app.state.get_config()
    return HwaccelConfigOut(enabled=config.hwaccel_enabled)


@router.patch("/hwaccel/enabled", response_model=HwaccelConfigOut)
def set_hwaccel_enabled(req: HwaccelEnabledRequest, request: Request) -> HwaccelConfigOut:
    """Lightweight toggle, same shape as PATCH /species/enabled -- takes effect on the
    next pipeline restart ("Apply changes"), like every other config change here.
    """
    config = request.app.state.get_config()
    config.hwaccel_enabled = req.enabled
    config.save_to_db()
    return HwaccelConfigOut(enabled=config.hwaccel_enabled)


@router.get("/cameras", response_model=list[CameraConfigOut])
def list_camera_configs(request: Request) -> list[CameraConfigOut]:
    """Powers the "Manage cameras" page -- includes rtsp_url/rtsp_transport/retain_days/
    label lists, unlike GET /api/cameras (mirage/api/routers/cameras.py), so the edit form
    can pre-fill from a real, complete existing config rather than starting blank.
    """
    config = request.app.state.get_config()
    return [_camera_config_out(cam, config) for cam in config.cameras.values()]


@router.get("/cameras/{name}", response_model=CameraConfigOut)
def get_camera_config(name: str, request: Request) -> CameraConfigOut:
    config = request.app.state.get_config()
    cam = config.cameras.get(name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")
    return _camera_config_out(cam, config)


@router.get("/detectors/{name}/labels", response_model=list[str])
def get_detector_labels(name: str, request: Request) -> list[str]:
    """Every real label a detector's labelmap declares -- powers the per-label Min
    score/Confirm score rows on the Add/Edit Camera form (see _detector_labels), fetched
    fresh whenever the form's selected detector changes so switching detectors (e.g.
    "general" -> "megadetector-e") immediately shows the right label set instead of a
    stale one.
    """
    config = request.app.state.get_config()
    detector = config.detectors.get(name)
    if detector is None:
        raise HTTPException(status_code=404, detail=f"unknown detector {name!r}")
    return _detector_labels(detector)


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
        camera = _build_camera_config(req, existing=None)
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
        camera = _build_camera_config(req, existing=config.cameras[name])
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
