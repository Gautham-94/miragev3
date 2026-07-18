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


class OpenVocabQuery(BaseModel):
    """A saved free-text description to match against tracked objects via the
    open-vocabulary detector (see mirage/openvocab/), e.g. "person carrying a red
    backpack" or "a dog off-leash" -- TODO_FIX_LIST.md items 4/6. Not tied to COCO
    labels or any closed vocabulary; matched via OWLv2 (mirage/openvocab/process.py),
    gated so it only runs on already-confirmed tracked objects, not every frame.
    """

    id: str
    text: str
    # Empty/unset -- applies to every camera. Non-empty -- only these camera names.
    cameras: list[str] = Field(default_factory=list)
    enabled: bool = True
    # "manual" (default): created directly on the Queries page, for compositional
    # descriptions a single track_objects word can't express ("person carrying a red
    # backpack"). "track_objects": auto-provisioned because a camera's objects.track
    # list contained a word that isn't in its detector's labelmap (e.g. "animals") --
    # see split_track_objects() below and mirage/api/routers/config.py's
    # _sync_track_object_queries(). Auto-provisioned queries are kept in lockstep with
    # their owning camera's track list (recreated/deleted as that list changes) rather
    # than being independently editable, so the Queries page shows them read-only and
    # tags them with which camera's Track objects field they came from.
    source: str = "manual"


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
    # user needing to re-type it. Deliberately independent of `track`'s own
    # open-vocab bridge (split_track_objects/_sync_track_object_queries in
    # mirage/api/routers/config.py): this only ever affects closed-vocab detector
    # labels, since there's no open-vocab equivalent of "every possible free-text
    # query" to bypass to.
    track_all: bool = False
    filters: dict[str, ObjectFilterConfig] = Field(default_factory=dict)

    def filter_for(self, label: str) -> ObjectFilterConfig:
        return self.filters.get(label, ObjectFilterConfig())


def split_track_objects(track: list[str], known_labels: set[str]) -> tuple[list[str], list[str]]:
    """Splits a camera's objects.track list into (closed_vocab, open_vocab) terms by
    checking each word against `known_labels` (the routed detector's actual labelmap,
    e.g. COCO's 80 classes) case-insensitively. A word not in the labelmap -- e.g.
    "animals", or any species/object the closed-vocab model was never trained on -- is
    NOT a dead config value the way it used to be (see TODO_FIX_LIST.md item 4's
    original problem statement): it's routed to the open-vocabulary path instead, via
    an auto-provisioned OpenVocabQuery (mirage/api/routers/config.py's
    _sync_track_object_queries) checked directly against the whole motion-triggered
    frame (CameraConfig.openvocab_direct_frame). Pure function, no I/O -- callers pass
    in the labelmap already loaded from disk (mirage.detection.labelmap.load_labels).
    """
    known_lower = {label.lower() for label in known_labels}
    closed_vocab = [word for word in track if word.lower() in known_lower]
    open_vocab = [word for word in track if word.lower() not in known_lower]
    return closed_vocab, open_vocab


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
    detector: str = "default"  # which DetectorInstanceConfig this camera routes detection through
    # False (default): open-vocab queries (mirage.openvocab) only ever check crops of
    # objects the closed-vocab detector (`detector` above) already confirmed --
    # TODO_FIX_LIST.md item 4's original design. True: this camera's queries are checked
    # directly against the whole motion-triggered frame instead, independent of whatever
    # `detector`/`objects.track` would or wouldn't have detected -- for open-vocab items
    # genuinely outside the closed-vocab detector's label map (e.g. a specific object
    # type it was never trained on). See mirage/openvocab/dispatcher.py's two dispatch
    # paths for how this is actually used.
    openvocab_direct_frame: bool = False

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
    queries: list[OpenVocabQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_camera_detector_refs(self) -> "MirageConfig":
        for cam in self.cameras.values():
            if cam.detector not in self.detectors:
                raise ValueError(
                    f"camera {cam.name!r} references unknown detector {cam.detector!r}; "
                    f"known detectors: {sorted(self.detectors)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_query_camera_refs(self) -> "MirageConfig":
        for query in self.queries:
            unknown = [c for c in query.cameras if c not in self.cameras]
            if unknown:
                raise ValueError(
                    f"query {query.id!r} references unknown camera(s) {unknown}; known cameras: {sorted(self.cameras)}"
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
