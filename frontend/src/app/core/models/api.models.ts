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

export type ReviewSeverity = 'alert' | 'detection';

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

export interface Detector {
  name: string;
  device: string;
  model_width: number;
  model_height: number;
  execution_provider: ExecutionProvider;
  model_path: string;
  labelmap_path: string;
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
}

export type RtspTransport = 'tcp' | 'udp';

export interface CameraWriteRequest {
  name: string;
  rtsp_url: string;
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
  openvocab_direct_frame?: boolean;
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
  retain_days: number;
  segment_seconds: number;
  alert_labels: string[];
  detection_labels: string[];
  openvocab_direct_frame: boolean;
}

export interface ConfigMutationResponse {
  ok: boolean;
  restart_required: boolean;
  camera: Camera;
  // Which of the saved camera's Track objects words aren't in its detector's
  // vocabulary and are therefore being checked via open-vocabulary search instead
  // (see mirage/api/routers/config.py's _sync_track_object_queries).
  open_vocab_terms: string[];
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

export type QuerySource = 'manual' | 'track_objects';

export interface Query {
  id: string;
  text: string;
  cameras: string[];
  enabled: boolean;
  source: QuerySource;
}

export interface QueryWriteRequest {
  text: string;
  cameras?: string[];
  enabled?: boolean;
}

export interface QueryMatch {
  id: string;
  query_id: string;
  query_text: string;
  camera: string;
  object_id: string;
  matched_at: number;
  score: number;
  box: number[];
  has_thumb: boolean;
}
