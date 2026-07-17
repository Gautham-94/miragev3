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
import uuid

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
    ObjectsConfig,
    OpenVocabQuery,
    PixelFormat,
    RecordConfig,
    RetainConfig,
    ReviewConfig,
    ReviewLabelConfig,
    RtspTransport,
    TensorLayout,
    split_track_objects,
)
from mirage.detection.execution_providers import available_execution_providers
from mirage.detection.labelmap import load_labels
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
    # See CameraConfig.openvocab_direct_frame's docstring -- False (default) keeps
    # open-vocab queries scoped to crops of objects the closed-vocab detector already
    # confirmed; True checks the whole motion-triggered frame directly, for queries about
    # things that detector was never trained to recognize as an object at all.
    openvocab_direct_frame: bool = False


class ConfigMutationResponse(BaseModel):
    ok: bool = True
    restart_required: bool = True
    camera: CameraOut
    # Which of this camera's objects.track words were routed to the open-vocabulary
    # path (see split_track_objects/_sync_track_object_queries) rather than being a
    # real closed-vocab label the routed detector recognizes -- lets the Add/Edit
    # Camera form show an inline "checking X via open-vocabulary search instead" note
    # without the frontend needing its own copy of the detector's labelmap.
    open_vocab_terms: list[str] = []


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
    openvocab_direct_frame: bool


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
        openvocab_direct_frame=cam.openvocab_direct_frame,
    )


def _resolve_open_vocab_terms(camera: CameraConfig, config: MirageConfig) -> list[str]:
    """Which of this camera's objects.track words aren't in its routed detector's
    labelmap -- see split_track_objects()'s docstring. Best-effort: if the detector or
    its labelmap file is missing/unreadable (shouldn't happen for a validated config,
    but this runs at write-time before that's guaranteed), treat every track word as
    closed-vocab rather than raising, so a camera write never fails because of this
    auto-provisioning side effect.
    """
    detector = config.detectors.get(camera.detector)
    if detector is None:
        return []
    try:
        labels = set(load_labels(detector.model.labelmap_path).values())
    except OSError:
        return []
    _closed, open_vocab = split_track_objects(camera.objects.track, labels)
    return open_vocab


def _sync_track_object_queries(camera: CameraConfig, config: MirageConfig) -> list[str]:
    """Keeps config.queries' auto-provisioned (source="track_objects") entries for this
    camera in lockstep with its current objects.track list -- called after every
    camera create/update. This is what makes typing "animals" into Track objects
    actually DO something (TODO_FIX_LIST.md item 4's original gap: a non-COCO track
    word used to be a silently dead config value) without a separate trip to the
    Queries page: it's auto-wired to an open-vocab query scoped to this camera, and
    CameraConfig.openvocab_direct_frame is auto-enabled so that query is actually
    checked (against the whole motion-triggered frame, since there's no closed-vocab
    detection to crop for a word the detector doesn't know).

    Existing auto-provisioned queries for this camera that no longer match a current
    open-vocab track word are removed (e.g. "animals" deleted from Track objects should
    stop being checked, not linger as an orphaned query) -- manual (source="manual")
    queries are never touched here, regardless of camera scope.

    Returns the resolved open-vocab terms so the caller can surface them on
    ConfigMutationResponse.open_vocab_terms (the Add/Edit Camera form's inline info
    note) without recomputing.
    """
    open_vocab_terms = _resolve_open_vocab_terms(camera, config)

    other_queries = [
        q for q in config.queries if not (q.source == "track_objects" and q.cameras == [camera.name])
    ]
    existing_auto = {
        q.text: q for q in config.queries if q.source == "track_objects" and q.cameras == [camera.name]
    }

    synced = list(other_queries)
    for term in open_vocab_terms:
        existing = existing_auto.get(term)
        synced.append(
            OpenVocabQuery(
                id=existing.id if existing else uuid.uuid4().hex,
                text=term,
                cameras=[camera.name],
                enabled=True,
                source="track_objects",
            )
        )
    config.queries = synced

    # Auto-enable direct-frame mode the moment this camera has any open-vocab track
    # word -- otherwise the auto-provisioned query would silently sit unused (direct-
    # frame mode is required for it to ever be checked, since there's no confirmed
    # closed-vocab object to crop for a word outside that detector's label map). Leaves
    # the flag alone (doesn't turn it back off) if a camera had it manually enabled for
    # unrelated manual queries -- only ever turns it ON as a side effect here.
    if open_vocab_terms and not camera.openvocab_direct_frame:
        camera.openvocab_direct_frame = True

    # Auto-add each open-vocab track word to this camera's alert labels, same as a
    # native COCO track word (e.g. "person") already implicitly drives review severity
    # via ReviewSegmentMaintainer.classify_severity -- otherwise a real OWLv2 match for
    # "animals" would persist as a QueryMatch row but never surface as a review/alert,
    # which defeats the entire point of this auto-provisioning (see
    # mirage/openvocab/dispatcher.py's synthetic-tracked-object bridge, which is what
    # makes a match flow through the SAME alert path a real tracked object does). Only
    # ever adds -- never removes a label the user configured by hand, since alert_labels
    # can also hold entries unrelated to track_objects entirely.
    for term in open_vocab_terms:
        if term not in camera.review.alerts.labels:
            camera.review.alerts.labels.append(term)

    return open_vocab_terms


def _remove_track_object_queries_for_camera(camera_name: str, config: MirageConfig) -> None:
    """Cleanup counterpart to _sync_track_object_queries, called on camera delete --
    an auto-provisioned query scoped to a camera that no longer exists would otherwise
    linger forever (MirageConfig's own _validate_query_camera_refs validator would
    actually reject it on next load, since it references an unknown camera).
    """
    config.queries = [
        q for q in config.queries if not (q.source == "track_objects" and q.cameras == [camera_name])
    ]


def _build_camera_config(req: CameraWriteRequest) -> CameraConfig:
    review_kwargs = {}
    if req.alert_labels is not None:
        review_kwargs["alerts"] = ReviewLabelConfig(labels=req.alert_labels)
    if req.detection_labels is not None:
        review_kwargs["detections"] = ReviewLabelConfig(labels=req.detection_labels)

    return CameraConfig(
        name=req.name,
        enabled=req.enabled,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=req.rtsp_url, rtsp_transport=req.rtsp_transport)]),
        detect=DetectConfig(width=req.width, height=req.height, fps=req.fps),
        objects=ObjectsConfig(track=req.track_objects, track_all=req.track_all),
        record=RecordConfig(
            enabled=req.record_enabled, segment_seconds=req.segment_seconds, continuous=RetainConfig(days=req.retain_days)
        ),
        review=ReviewConfig(**review_kwargs) if review_kwargs else ReviewConfig(),
        detector=req.detector,
        openvocab_direct_frame=req.openvocab_direct_frame,
    )


@router.get("/detectors", response_model=list[DetectorOut])
def list_detectors(request: Request) -> list[DetectorOut]:
    config = request.app.state.get_config()
    return [
        DetectorOut(
            name=d.name, device=d.device, model_width=d.model.width, model_height=d.model.height,
            execution_provider=d.model.execution_provider,
            model_path=d.model.model_path, labelmap_path=d.model.labelmap_path,
        )
        for d in config.detectors.values()
    ]


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


def _detector_out(detector: DetectorInstanceConfig) -> DetectorOut:
    return DetectorOut(
        name=detector.name, device=detector.device, model_width=detector.model.width, model_height=detector.model.height,
        execution_provider=detector.model.execution_provider,
        model_path=detector.model.model_path, labelmap_path=detector.model.labelmap_path,
    )


@router.post("/detectors", response_model=DetectorOut, status_code=201)
def create_detector(req: DetectorWriteRequest, request: Request) -> DetectorOut:
    config = request.app.state.get_config()

    if req.name in config.detectors:
        raise HTTPException(status_code=409, detail=f"detector {req.name!r} already exists")

    detector = _validate_and_build_detector(req)
    config.detectors[req.name] = detector
    config.save_to_db()

    return _detector_out(detector)


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

    return _detector_out(detector)


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
    open_vocab_terms = _sync_track_object_queries(camera, config)
    config.save_to_db()

    return ConfigMutationResponse(camera=_camera_out(camera), open_vocab_terms=open_vocab_terms)


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
        _remove_track_object_queries_for_camera(name, config)
    config.cameras[req.name] = camera
    open_vocab_terms = _sync_track_object_queries(camera, config)
    config.save_to_db()

    return ConfigMutationResponse(camera=_camera_out(camera), open_vocab_terms=open_vocab_terms)


@router.delete("/cameras/{name}", status_code=204, response_model=None)
def delete_camera(name: str, request: Request) -> None:
    config = request.app.state.get_config()

    if name not in config.cameras:
        raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")

    del config.cameras[name]
    _remove_track_object_queries_for_camera(name, config)
    config.save_to_db()


class QueryOut(BaseModel):
    id: str
    text: str
    cameras: list[str]
    enabled: bool
    source: str


class QueryWriteRequest(BaseModel):
    """See mirage.config.schema.OpenVocabQuery / TODO_FIX_LIST.md items 4/6 -- a saved
    free-text description checked against confirmed tracked objects via OWLv2, e.g.
    "person carrying a red backpack". `cameras` empty/unset means "applies to every
    camera"; non-empty scopes it to just those camera names.
    """

    text: str = Field(min_length=1)
    cameras: list[str] = Field(default_factory=list)
    enabled: bool = True


def _query_out(query: OpenVocabQuery) -> QueryOut:
    return QueryOut(id=query.id, text=query.text, cameras=query.cameras, enabled=query.enabled, source=query.source)


@router.get("/queries", response_model=list[QueryOut])
def list_queries(request: Request) -> list[QueryOut]:
    config = request.app.state.get_config()
    return [_query_out(q) for q in config.queries]


@router.post("/queries", response_model=QueryOut, status_code=201)
def create_query(req: QueryWriteRequest, request: Request) -> QueryOut:
    config = request.app.state.get_config()

    unknown = [c for c in req.cameras if c not in config.cameras]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown camera(s) {unknown}; known cameras: {sorted(config.cameras)}")

    try:
        query = OpenVocabQuery(id=uuid.uuid4().hex, text=req.text, cameras=req.cameras, enabled=req.enabled)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config.queries.append(query)
    config.save_to_db()

    return _query_out(query)


@router.put("/queries/{query_id}", response_model=QueryOut)
def update_query(query_id: str, req: QueryWriteRequest, request: Request) -> QueryOut:
    config = request.app.state.get_config()

    existing = next((q for q in config.queries if q.id == query_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"unknown query {query_id!r}")
    if existing.source == "track_objects":
        raise HTTPException(
            status_code=409,
            detail=(
                f"query {query_id!r} was auto-created from camera "
                f"{existing.cameras[0] if existing.cameras else '?'!r}'s Track objects field; "
                "edit or remove the word there instead of editing this query directly"
            ),
        )

    unknown = [c for c in req.cameras if c not in config.cameras]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown camera(s) {unknown}; known cameras: {sorted(config.cameras)}")

    try:
        updated = OpenVocabQuery(id=query_id, text=req.text, cameras=req.cameras, enabled=req.enabled)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config.queries = [updated if q.id == query_id else q for q in config.queries]
    config.save_to_db()

    return _query_out(updated)


@router.delete("/queries/{query_id}", status_code=204, response_model=None)
def delete_query(query_id: str, request: Request) -> None:
    config = request.app.state.get_config()

    existing = next((q for q in config.queries if q.id == query_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"unknown query {query_id!r}")
    if existing.source == "track_objects":
        raise HTTPException(
            status_code=409,
            detail=(
                f"query {query_id!r} was auto-created from camera "
                f"{existing.cameras[0] if existing.cameras else '?'!r}'s Track objects field; "
                "remove the word there instead of deleting this query directly"
            ),
        )

    config.queries = [q for q in config.queries if q.id != query_id]
    config.save_to_db()
