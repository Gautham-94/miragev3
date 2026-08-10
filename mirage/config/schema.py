"""Configuration schema (Pydantic models).

Spec references:
  - NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.2 (ffmpeg/camera config), 2.2 (motion),
    3.3 (model config), 8.9-ish retention fields, 10.1 (schema-adjacent filter defaults).
  - NVR_EXTENSION_MULTI_MODEL_ROUTING.md section 1 (per-camera detector routing).

All field defaults match the values called out explicitly in the spec documents.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CameraRole(str, Enum):
    detect = "detect"
    record = "record"
    audio = "audio"


class InputDType(str, Enum):
    int_ = "int"  # raw uint8 passthrough
    float_ = "float"  # cast to float32, /255 normalized
    float_denorm = "float_denorm"  # cast to float32, no scaling


class PixelFormat(str, Enum):
    rgb = "rgb"
    bgr = "bgr"


class TensorLayout(str, Enum):
    nhwc = "nhwc"
    nchw = "nchw"


class RetainMode(str, Enum):
    all = "all"
    motion = "motion"
    active_objects = "active_objects"


class RtspTransport(str, Enum):
    tcp = "tcp"
    udp = "udp"


# --------------------------------------------------------------------------------------
# ffmpeg / camera input config
# --------------------------------------------------------------------------------------


class CameraInputConfig(BaseModel):
    path: str
    roles: list[CameraRole] = Field(default_factory=list)
    hwaccel_args: list[str] = Field(default_factory=list)
    # TCP is the correct default (in-order, lossless delivery, no dropped/corrupt frames)
    # -- but some cameras' RTSP-over-TCP implementations are unreliable (confirmed on a
    # real Hikvision DS-2CD1023G0E-I: a single, otherwise-idle TCP RTSP session was reset
    # by the camera after ~2 seconds every time, while the same stream over UDP ran
    # stable for the full test, with only the packet loss/corruption normally expected
    # from best-effort UDP delivery on a real WiFi link). UDP is the pragmatic escape
    # hatch for that class of camera, at the cost of tolerating occasional dropped or
    # corrupted frames.
    rtsp_transport: RtspTransport = RtspTransport.tcp


class FfmpegConfig(BaseModel):
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    global_args: list[str] = Field(
        default_factory=lambda: ["-hide_banner", "-loglevel", "warning", "-threads", "2"]
    )
    retry_interval: float = 10.0
    inputs: list[CameraInputConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _assign_default_roles_and_validate(self) -> "FfmpegConfig":
        if len(self.inputs) == 1 and not self.inputs[0].roles:
            self.inputs[0].roles = [CameraRole.record, CameraRole.detect]

        role_owners: dict[CameraRole, int] = {}
        for i, inp in enumerate(self.inputs):
            for role in inp.roles:
                if role in role_owners:
                    raise ValueError(
                        f"role {role} assigned to more than one input (inputs {role_owners[role]} and {i})"
                    )
                role_owners[role] = i

        if self.inputs and CameraRole.detect not in role_owners:
            raise ValueError("the 'detect' role must be assigned to exactly one input")
        return self


# --------------------------------------------------------------------------------------
# Motion config -- field names/defaults match spec section 2.2 exactly.
# --------------------------------------------------------------------------------------


class MotionConfig(BaseModel):
    enabled: bool = True
    threshold: int = Field(default=30, ge=1, le=255)
    contour_area: int = Field(default=10, ge=0)
    frame_height: int = Field(default=100, ge=1)
    frame_alpha: float = Field(default=0.01, gt=0.0)
    lightning_threshold: float = Field(default=0.8, ge=0.3, le=1.0)
    skip_motion_threshold: Optional[float] = None
    improve_contrast: bool = True
    mask: list[list[tuple[float, float]]] = Field(default_factory=list)  # polygons, normalized [0,1] coords


# --------------------------------------------------------------------------------------
# Detect / model config
# --------------------------------------------------------------------------------------


class DetectConfig(BaseModel):
    enabled: bool = True
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: int = 5
    min_initialized: Optional[int] = None  # default: max(fps/2, 2)
    max_disappeared: Optional[int] = None  # default: fps * 5
    stationary_threshold: Optional[int] = None  # default: fps * 10 (frames)
    stationary_interval: int = 10  # frames between periodic rechecks

    @model_validator(mode="after")
    def _both_or_neither(self) -> "DetectConfig":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must both be set or both omitted")
        return self


class ExecutionProvider(str, Enum):
    """Which onnxruntime execution provider to run this detector's InferenceSession on --
    a separate axis from DetectorInstanceConfig.device (which *plugin class* runs, e.g.
    "onnx_yolov8"); this picks which hardware backend that plugin's ONNX Runtime session
    uses. `auto` prefers the fastest available accelerator and falls back to CPU for any
    unsupported op or if the accelerator itself isn't present on this machine -- this
    mirrors onnxruntime's own `providers=[...]` list semantics directly (see
    mirage/detection/plugins/onnx_yolov8.py), not an either/or choice.

    Measured directly on Apple Silicon (this repo's dev machine) against the real
    yolov8n.onnx model already shipped here: CPU averaged 9.90ms/frame, CoreML averaged
    1.75ms/frame -- a real ~5.6x speedup, not a theoretical one.
    """

    auto = "auto"  # prefer the best available accelerator (coreml/cuda), fall back to cpu
    cpu = "cpu"
    coreml = "coreml"  # Apple Silicon/Intel Mac -- CoreML decides Neural Engine vs GPU (Metal) vs CPU per-op
    cuda = "cuda"  # NVIDIA GPU


class ModelConfig(BaseModel):
    """Section 3.3.1. One instance per configured detector (see DetectorInstanceConfig)."""

    width: int = 320
    height: int = 320
    input_dtype: InputDType = InputDType.int_
    pixel_format: PixelFormat = PixelFormat.rgb
    layout: TensorLayout = TensorLayout.nhwc
    model_path: str = ""
    labelmap_path: str = ""
    execution_provider: ExecutionProvider = ExecutionProvider.cpu


class DetectorInstanceConfig(BaseModel):
    """NVR_EXTENSION_MULTI_MODEL_ROUTING.md section 1."""

    name: str
    model: ModelConfig
    device: str = "onnx"  # backend key registered in the detector plugin registry
    # Lets a detector be registered (and cameras stay routed to it in config) without a
    # process actually being spawned for it -- e.g. a heavy/experimental detector like
    # MegaDetector, kept around for occasional testing but not paying its idle memory/
    # startup cost on every normal run. MirageApp._start_detectors skips disabled
    # detectors entirely; any camera still routed to one simply gets no detection
    # (logged, not an error) until reassigned or the detector is re-enabled.
    enabled: bool = True
    # How many independent OS processes (each loading its own full copy of the model)
    # serve cameras routed to this detector name -- 1 (default) preserves the original
    # "one process per configured detector" behavior byte-for-byte. Every camera routed
    # to this detector shares ONE queue (see MirageApp.detection_queues's own
    # docstring), and every one of this detector's num_workers processes blocks on
    # .get() against that SAME queue -- Python's mp.Queue already hands each queued
    # item to whichever consumer's blocking .get() unblocks first, so a worker that
    # just finished a fast inference naturally picks up the next request before a
    # still-busy sibling does. This gives free least-busy-worker routing with no
    # separate queue-depth tracking or broker process, and no static camera-to-worker
    # pinning (a camera can be served by a different worker frame to frame, depending
    # on which one happens to be free). Exists because a single shared detector process
    # serializes inference across every camera routed to it (confirmed via a real
    # measured failure: 3 cameras sharing one MegaDetector instance caused 5-second
    # per-call timeouts and silent frame drops on the busiest camera, see
    # PROCESS_AUDIT.md / this session's live debugging) -- adding worker processes
    # removes that single-consumer bottleneck at the cost of each extra worker's own
    # memory + a share of CPU/accelerator contention. Increasing this only makes sense
    # if the host actually has spare CPU cores/accelerator headroom -- see the Config
    # page's own capability-aware guidance for this field.
    num_workers: int = Field(default=1, ge=1, le=8)


class SpeciesModelConfig(BaseModel):
    """Backend-agnostic model settings for the (optional) async species classifier --
    mirrors ModelConfig's own "just enough to locate/load the model file" shape.
    Backend-specific extras (e.g. SpeciesNet's own geo-prior country/admin1 hints) can
    be added here later without a schema break, same as ModelConfig has done.
    """

    model_path: str = ""
    # SpeciesNet-specific (mirage/species/plugins/speciesnet.py): a Kaggle/HuggingFace
    # model identifier the `speciesnet` package resolves and auto-downloads itself (see
    # that plugin's own docstring) -- NOT a local path like model_path above, which
    # stays for any future backend that loads a plain local model file directly.
    model_name: str = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
    # Below this softmax score, a classification is treated as "no confident result"
    # (SpeciesClassification with every field None, species_status="skipped") rather
    # than reported -- SpeciesNet itself has no built-in cutoff; this is mirage's own
    # bar, matching the same "an over-approximation is fine, false negatives are safer
    # than false species labels" spirit as ObjectFilterConfig.threshold.
    confidence_threshold: float = 0.5
    # Path to the Python interpreter of a SEPARATE venv with `speciesnet` installed
    # (see mirage/species/plugins/speciesnet.py's module docstring for why this can't
    # share mirage's own main venv -- a real, unresolvable numpy/opencv dependency
    # conflict with norfair). Default assumes the documented sibling-venv setup
    # (`python3 -m venv .venv-speciesnet && .venv-speciesnet/bin/pip install
    # speciesnet`) run from the repo root.
    venv_python_path: str = ".venv-speciesnet/bin/python"


class SpeciesClassifierConfig(BaseModel):
    """Mirage V3: one shared species-classification worker process for every camera's
    Animal/Bird events (see mirage/species/, mirage/events/processor.py's
    SPECIES_ENRICHABLE_LABELS). Disabled by default -- species classification never
    runs, and every Animal/Bird Event's species_status stays "skipped", unless a user
    explicitly opts in here. Deliberately NOT per-camera (mirrors DetectorInstanceConfig
    being one shared thing referenced by camera.detector, not duplicated per camera) --
    unlike detectors, there's only ever one species worker, since Person/Vehicle
    filtering already happens at the label level, not the camera level.
    """

    enabled: bool = False
    device: str = "speciesnet"  # backend key registered in mirage.species.registry
    model: SpeciesModelConfig = Field(default_factory=SpeciesModelConfig)


# --------------------------------------------------------------------------------------
# Object filters -- section 6.1 / 6.2.
# --------------------------------------------------------------------------------------


class ObjectFilterConfig(BaseModel):
    min_score: float = 0.5
    threshold: float = 0.7
    min_area: float = 0
    max_area: float = 24_000_000
    min_ratio: float = 0
    max_ratio: float = 24_000_000


class ObjectsConfig(BaseModel):
    track: list[str] = Field(default_factory=lambda: ["person"])
    # When True, mirage/tracking/orchestration.py tracks EVERY label the routed
    # detector emits (e.g. all 80 COCO classes), bypassing the `track` membership
    # check entirely -- `track` itself is left untouched/ignored while this is on, so
    # toggling it back off restores whatever was previously configured without the
    # user needing to re-type it.
    track_all: bool = False
    filters: dict[str, ObjectFilterConfig] = Field(default_factory=dict)

    def filter_for(self, label: str) -> ObjectFilterConfig:
        return self.filters.get(label, ObjectFilterConfig())


# --------------------------------------------------------------------------------------
# Recording / retention config -- section 8.2.
# --------------------------------------------------------------------------------------


class RetainConfig(BaseModel):
    days: float = 0
    mode: RetainMode = RetainMode.all


class RecordConfig(BaseModel):
    enabled: bool = False
    segment_seconds: int = Field(default=10, le=60)
    expire_interval_minutes: int = 60
    continuous: RetainConfig = Field(default_factory=RetainConfig)
    motion: RetainConfig = Field(default_factory=RetainConfig)
    apple_compatibility: bool = False


class ReviewLabelConfig(BaseModel):
    labels: list[str] = Field(default_factory=list)


class ReviewConfig(BaseModel):
    alerts: ReviewLabelConfig = Field(default_factory=lambda: ReviewLabelConfig(labels=["person", "car"]))
    detections: ReviewLabelConfig = Field(default_factory=lambda: ReviewLabelConfig(labels=[]))
    cutoff_seconds: int = 30


class RulesConfig(BaseModel):
    """Camera-level derived-condition alerts (TODO_FIX_LIST.md item 7.4's "rules
    engine") -- distinct from ReviewConfig's per-label alert/detection classification,
    since these two rules are conditions computed from the CURRENT SET of tracked
    objects (a count, a duration), not any single object's label. Both are opt-in
    (None/0 = disabled) and off by default -- a camera with no rules configured
    behaves exactly as it did before this feature existed. See mirage/tracking/rules.py
    (RulesEngine) for the actual evaluation logic.
    """

    # Fires a "crowd" alert once the number of currently-confirmed `person` tracks on
    # this camera reaches this count. None/0 disables the rule.
    crowd_threshold: int | None = None
    # Fires a "loitering"/wait-time alert once ANY tracked object has been
    # continuously present for at least this many seconds -- deliberately
    # movement-agnostic (counts total time tracked, not just time spent motionless),
    # so the same rule covers both "person loitering at a doorway" and "customer
    # waiting in a queue" (a slowly-shuffling-forward track is not "stationary" in
    # mirage.tracking.stationary's sense, but should still count toward wait time).
    # None/0 disables the rule.
    dwell_seconds: int | None = None


class PtzPresetConfig(BaseModel):
    """One ONVIF PTZ preset this camera can be commanded to -- token is the ONVIF
    device's own preset identifier (from PtzClient.get_presets()/GetPresets), not
    something mirage assigns itself.
    """

    token: str
    name: str | None = None


class PtzConfig(BaseModel):
    """Mirage V3 PTZ hybrid scheduling + ONVIF PTZ control (mirage/ptz/,
    mirage/tracking/orchestration.py's ptz_moving_fn). Disabled by default -- a camera
    with ptz.enabled=False behaves EXACTLY as it did before this feature existed (no
    polling thread spawned, CameraOrchestrator's ptz_moving_fn stays None, byte-for-byte
    the same hardcoded ptz_moving=False control flow as every other fixed camera).

    Closes a real pre-existing gap: mirage/api/routers/onvif.py's ONVIF discovery/
    resolve wizard never persists the ONVIF username/password it uses (only the
    resulting rtsp:// URL is saved) -- ongoing PTZ control (status polling, move
    commands, presets) requires these credentials to be held onto going forward,
    unlike the one-shot wizard flow.
    """

    enabled: bool = False
    onvif_host: str = ""
    onvif_port: int = 80
    onvif_username: str = ""
    onvif_password: str = ""
    patrol_enabled: bool = False
    patrol_presets: list[PtzPresetConfig] = Field(default_factory=list)
    patrol_interval_seconds: int = 300  # dwell time at each preset before advancing
    # Grace period after issuing a move command before trusting GetStatus's
    # MoveStatus == IDLE -- some cameras report IDLE briefly before actually starting
    # to move, which would otherwise cause the hybrid-scheduling orchestration branch
    # to flip back to "stationary" one frame too early.
    move_settle_seconds: float = 2.0


# --------------------------------------------------------------------------------------
# Camera + top-level config
# --------------------------------------------------------------------------------------


class CameraConfig(BaseModel):
    name: str
    enabled: bool = True
    ffmpeg: FfmpegConfig
    motion: MotionConfig = Field(default_factory=MotionConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    ptz: PtzConfig = Field(default_factory=PtzConfig)
    detector: str = "default"  # which DetectorInstanceConfig this camera routes detection through

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) of the luma plane at detect resolution."""
        return self.detect.height, self.detect.width

    @property
    def frame_shape_yuv(self) -> tuple[int, int]:
        """(height, width) numpy shape for the full YUV420 planar buffer."""
        return self.detect.height * 3 // 2, self.detect.width

    @property
    def frame_size_bytes(self) -> int:
        return self.detect.width * self.detect.height * 3 // 2

    @property
    def min_initialized(self) -> int:
        return self.detect.min_initialized or max(self.detect.fps // 2, 2)

    @property
    def max_disappeared(self) -> int:
        return self.detect.max_disappeared or self.detect.fps * 5

    @property
    def stationary_threshold(self) -> int:
        return self.detect.stationary_threshold or self.detect.fps * 10


class MirageConfig(BaseModel):
    detectors: dict[str, DetectorInstanceConfig] = Field(default_factory=dict)
    cameras: dict[str, CameraConfig] = Field(default_factory=dict)
    species_classifier: SpeciesClassifierConfig = Field(default_factory=SpeciesClassifierConfig)

    @model_validator(mode="after")
    def _validate_camera_detector_refs(self) -> "MirageConfig":
        for cam in self.cameras.values():
            if cam.detector not in self.detectors:
                raise ValueError(
                    f"camera {cam.name!r} references unknown detector {cam.detector!r}; "
                    f"known detectors: {sorted(self.detectors)}"
                )
        return self

    @classmethod
    def from_yaml_file(cls, path: str) -> "MirageConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> "MirageConfig":
        """The config a fresh install starts from: the same 'general' ONNX YOLOv8n
        detector config/mirage.yaml has always shipped with, and no cameras yet (that's
        the whole point of the add-camera wizard -- there's nothing camera-specific to
        default to).
        """
        return cls(
            detectors={
                "general": DetectorInstanceConfig(
                    name="general",
                    device="onnx_yolov8",
                    model=ModelConfig(
                        width=320,
                        height=320,
                        input_dtype=InputDType.int_,
                        pixel_format=PixelFormat.rgb,
                        model_path="models/yolov8n.onnx",
                        labelmap_path="models/coco_labelmap.txt",
                    ),
                ),
            },
            cameras={},
        )

    @classmethod
    def from_db(cls) -> "MirageConfig":
        """Reads the singleton AppConfig row. Requires init_database() to already have
        been called (same assumption every other DB access in this codebase makes --
        see mirage.db.database). If the row doesn't exist yet (a genuinely fresh
        install), seeds it with default() first so callers always get a usable config
        back rather than an empty one with no detectors at all.
        """
        from mirage.config.store import load_or_seed_config

        return load_or_seed_config()

    def save_to_db(self) -> None:
        from mirage.config.store import save_config

        save_config(self)
