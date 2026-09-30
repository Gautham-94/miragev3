// Mirrors mirage/api/schemas.py exactly -- keep these in sync with the backend's
// Pydantic response models field-for-field.

export interface Camera {
  name: string;
  enabled: boolean;
  width: number | null;
  height: number | null;
  fps: number;
  detector: string;
  record_enabled: boolean;
  track_objects: string[];
  track_all: boolean;
}

// Mirage V3 async species classification (mirage/species/, mirage/events/processor.py)
// -- 'not_applicable' for every label other than animal/bird; 'pending' until the
// species worker enriches the event, then 'complete'/'failed'/'skipped'.
export type SpeciesStatus = 'not_applicable' | 'pending' | 'complete' | 'failed' | 'skipped';

export interface Event {
  id: string;
  camera: string;
  label: string;
  sub_label: string | null;
  start_time: number;
  end_time: number | null;
  score: number;
  top_score: number;
  false_positive: boolean;
  zones: string[];
  has_clip: boolean;
  has_snapshot: boolean;
  species: string | null;
  species_status: SpeciesStatus;
  species_confidence: number | null;
  species_taxonomy: Record<string, unknown> | null;
}

export interface Recording {
  id: string;
  camera: string;
  path: string;
  start_time: number;
  end_time: number;
  duration: number;
  motion: number | null;
  objects: number | null;
  dBFS: number | null;
  segment_size_mb: number;
}

// 'rule' = a camera-level derived-condition alert (crowd count / dwell-time-loitering
// -- see mirage.tracking.rules.RulesEngine) rather than a per-object-label
// classification. Always outranks 'alert' (see mirage.events.review._SEVERITY_RANK).
export type ReviewSeverity = 'alert' | 'detection' | 'rule';

export interface ReviewSegment {
  id: string;
  camera: string;
  start_time: number;
  end_time: number | null;
  severity: ReviewSeverity;
  thumb_path: string | null;
  data: Record<string, unknown>;
}

export interface EventListParams {
  camera?: string;
  label?: string;
  after?: number;
  before?: number;
  include_false_positive?: boolean;
  species_status?: SpeciesStatus;
  limit?: number;
  offset?: number;
}

export interface RecordingListParams {
  camera?: string;
  after?: number;
  before?: number;
  limit?: number;
  offset?: number;
}

export interface ReviewListParams {
  camera?: string;
  severity?: ReviewSeverity;
  after?: number;
  before?: number;
  limit?: number;
  offset?: number;
}

export type ExecutionProvider = 'auto' | 'cpu' | 'coreml' | 'cuda';

// mirage.supervisor's own possible states (mirage/supervisor.py) -- "unknown" means the
// backend has no status file yet (mirage.supervisor was never run, or mirage.api was
// started before it ran once); the frontend treats that the same as a healthy idle state
// (no banner), since there's nothing actionable to show.
export type SystemState = 'unknown' | 'starting' | 'running' | 'stopping' | 'stopped' | 'crashed';

export interface SystemStatus {
  state: SystemState;
  pid: number | null;
  updated_at: number | null;
}

export interface SpeciesConfig {
  enabled: boolean;
}

export interface HwaccelConfig {
  enabled: boolean;
}

export interface Detector {
  name: string;
  device: string;
  model_width: number;
  model_height: number;
  execution_provider: ExecutionProvider;
  model_path: string;
  labelmap_path: string;
  enabled: boolean;
  cameras: string[];
  num_workers: number;
}

export interface DetectorWriteRequest {
  name: string;
  model_path: string;
  labelmap_path: string;
  device?: string;
  width?: number;
  height?: number;
  input_dtype?: 'int' | 'float' | 'float_denorm';
  pixel_format?: 'rgb' | 'bgr';
  layout?: 'nhwc' | 'nchw';
  execution_provider?: ExecutionProvider;
  enabled?: boolean;
  num_workers?: number;
}

export type RtspTransport = 'tcp' | 'udp';

export interface CameraWriteRequest {
  name: string;
  rtsp_url: string;
  // Optional cheap, low-resolution stream URL for the live-view grid tier (e.g. a
  // Hikvision-style Channels/102 substream) -- see
  // mirage.config.schema.CameraConfig.live_sub_url's own docstring. Left unset,
  // go2rtc transcodes the main stream down for the grid instead -- always works, just
  // costs an extra ffmpeg process per camera.
  live_sub_url?: string | null;
  detector: string;
  track_objects?: string[];
  track_all?: boolean;
  width?: number;
  height?: number;
  fps?: number;
  record_enabled?: boolean;
  retain_days?: number;
  segment_seconds?: number;
  alert_labels?: string[];
  detection_labels?: string[];
  enabled?: boolean;
  rtsp_transport?: RtspTransport;
  // See mirage.config.schema.RulesConfig's docstring -- both undefined/null (default)
  // disable their respective rule. crowd_threshold: alert once this many confirmed
  // "person" tracks are present at once. dwell_seconds: alert once ANY tracked object
  // has been continuously present for at least this long (covers both loitering and
  // queue-wait-time -- one rule, see RulesConfig.dwell_seconds for why).
  crowd_threshold?: number | null;
  dwell_seconds?: number | null;
  // See mirage.config.schema.ObjectFilterConfig's docstring -- min_score gates whether a
  // raw detection is tracked at all; threshold gates whether a tracked object's median
  // score is ever promoted from false_positive to true-positive (and therefore shown in
  // Events/Review). Keyed per-label (e.g. "animal"/"person"/"vehicle" -- see
  // ApiService.getDetectorLabels), NOT one shared value applied to every tracked label
  // uniformly. A label left out of this map keeps its existing filter untouched on the
  // backend (merge, not replace) -- undefined/omitted entirely keeps every label as-is.
  filters?: Record<string, LabelFilter>;
}

export interface LabelFilter {
  min_score: number;
  threshold: number;
}

export interface CameraConfigDetail {
  name: string;
  enabled: boolean;
  width: number;
  height: number;
  fps: number;
  detector: string;
  record_enabled: boolean;
  track_objects: string[];
  track_all: boolean;
  rtsp_url: string;
  rtsp_transport: RtspTransport;
  live_sub_url: string | null;
  retain_days: number;
  segment_seconds: number;
  alert_labels: string[];
  detection_labels: string[];
  crowd_threshold: number | null;
  dwell_seconds: number | null;
  // Keyed by every label the camera's assigned detector's labelmap declares, not just
  // whatever's in track_objects -- a label only tracked via track_all still gets its
  // own row. See CameraWriteRequest.filters.
  filters: Record<string, LabelFilter>;
}

export interface ConfigMutationResponse {
  ok: boolean;
  restart_required: boolean;
  camera: Camera;
}

export interface OnvifDevice {
  name: string | null;
  info: string | null;
  url: string;
}

export interface OnvifResolveResult {
  rtsp_url: string;
  profile_name: string | null;
  device_name: string | null;
}

export type LogCategory = 'motion' | 'detect' | 'species' | 'system';

export interface LogEntry {
  timestamp: number;
  category: LogCategory;
  message: string;
  camera: string | null;
}

export interface SystemCapabilities {
  cpu_count: number;
  total_memory_mb: number;
}
