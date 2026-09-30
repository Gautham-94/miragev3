import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, of } from 'rxjs';

import { environment } from '../../../environments/environment';
import {
  Camera,
  CameraConfigDetail,
  CameraWriteRequest,
  ConfigMutationResponse,
  Detector,
  DetectorWriteRequest,
  Event,
  ExecutionProvider,
  EventListParams,
  LogEntry,
  OnvifDevice,
  OnvifResolveResult,
  HwaccelConfig,
  Recording,
  RecordingListParams,
  ReviewListParams,
  ReviewSegment,
  SpeciesConfig,
  SystemCapabilities,
  SystemStatus,
} from '../models/api.models';

function toHttpParams(params: object): HttpParams {
  let httpParams = new HttpParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      httpParams = httpParams.set(key, String(value));
    }
  }
  return httpParams;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  listCameras(): Observable<Camera[]> {
    return this.http.get<Camera[]>(`${this.base}/api/cameras`);
  }

  getSystemStatus(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>(`${this.base}/api/system/status`);
  }

  restartPipeline(): Observable<{ ok: boolean }> {
    return this.http.post<{ ok: boolean }>(`${this.base}/api/system/restart`, {});
  }

  getCamera(name: string): Observable<Camera> {
    return this.http.get<Camera>(`${this.base}/api/cameras/${encodeURIComponent(name)}`);
  }

  listEvents(params: EventListParams = {}): Observable<Event[]> {
    return this.http.get<Event[]>(`${this.base}/api/events`, { params: toHttpParams(params) });
  }

  getEvent(id: string): Observable<Event> {
    return this.http.get<Event>(`${this.base}/api/events/${encodeURIComponent(id)}`);
  }

  eventSnapshotUrl(id: string): string {
    return `${this.base}/api/events/${encodeURIComponent(id)}/snapshot`;
  }

  eventClipUrl(id: string): string {
    return `${this.base}/api/events/${encodeURIComponent(id)}/clip`;
  }

  listRecordings(params: RecordingListParams = {}): Observable<Recording[]> {
    return this.http.get<Recording[]>(`${this.base}/api/recordings`, { params: toHttpParams(params) });
  }

  getRecording(id: string): Observable<Recording> {
    return this.http.get<Recording>(`${this.base}/api/recordings/${encodeURIComponent(id)}`);
  }

  recordingClipUrl(id: string): string {
    return `${this.base}/api/recordings/${encodeURIComponent(id)}/clip`;
  }

  listReviewSegments(params: ReviewListParams = {}): Observable<ReviewSegment[]> {
    return this.http.get<ReviewSegment[]>(`${this.base}/api/review`, { params: toHttpParams(params) });
  }

  getReviewSegment(id: string): Observable<ReviewSegment> {
    return this.http.get<ReviewSegment>(`${this.base}/api/review/${encodeURIComponent(id)}`);
  }

  reviewThumbnailUrl(id: string): string {
    return `${this.base}/api/review/${encodeURIComponent(id)}/thumbnail`;
  }

  reviewClipUrl(id: string): string {
    return `${this.base}/api/review/${encodeURIComponent(id)}/clip`;
  }

  // Reverse lookup, batched: for a page of sightings on the Detections page, resolves
  // each one's containing scene (or null) in one request instead of one per row -- see
  // mirage/api/routers/events.py's GET /scenes docstring.
  getScenesForEvents(ids: string[]): Observable<Record<string, ReviewSegment | null>> {
    if (ids.length === 0) return of({});
    return this.http.get<Record<string, ReviewSegment | null>>(`${this.base}/api/events/scenes`, {
      params: { ids: ids.join(',') },
    });
  }

  liveSnapshotUrl(camera: string): string {
    return `${this.base}/api/live/${encodeURIComponent(camera)}/snapshot.jpg`;
  }

  liveWebSocketUrl(camera: string): string {
    const wsBase = this.base.replace(/^http/, 'ws') || `ws://${window.location.host}`;
    return `${wsBase}/api/live/${encodeURIComponent(camera)}/ws`;
  }

  listDetectors(): Observable<Detector[]> {
    return this.http.get<Detector[]>(`${this.base}/api/config/detectors`);
  }

  createDetector(req: DetectorWriteRequest): Observable<Detector> {
    return this.http.post<Detector>(`${this.base}/api/config/detectors`, req);
  }

  updateDetector(name: string, req: DetectorWriteRequest): Observable<Detector> {
    return this.http.put<Detector>(`${this.base}/api/config/detectors/${encodeURIComponent(name)}`, req);
  }

  deleteDetector(name: string): Observable<void> {
    return this.http.delete<void>(`${this.base}/api/config/detectors/${encodeURIComponent(name)}`);
  }

  setDetectorEnabled(name: string, enabled: boolean): Observable<Detector> {
    return this.http.patch<Detector>(`${this.base}/api/config/detectors/${encodeURIComponent(name)}/enabled`, { enabled });
  }

  getSpeciesConfig(): Observable<SpeciesConfig> {
    return this.http.get<SpeciesConfig>(`${this.base}/api/config/species`);
  }

  setSpeciesEnabled(enabled: boolean): Observable<SpeciesConfig> {
    return this.http.patch<SpeciesConfig>(`${this.base}/api/config/species/enabled`, { enabled });
  }

  getHwaccelConfig(): Observable<HwaccelConfig> {
    return this.http.get<HwaccelConfig>(`${this.base}/api/config/hwaccel`);
  }

  setHwaccelEnabled(enabled: boolean): Observable<HwaccelConfig> {
    return this.http.patch<HwaccelConfig>(`${this.base}/api/config/hwaccel/enabled`, { enabled });
  }

  getLogs(params: { limit?: number; since?: number } = {}): Observable<LogEntry[]> {
    return this.http.get<LogEntry[]>(`${this.base}/api/system/logs`, { params: toHttpParams(params) });
  }

  getSystemCapabilities(): Observable<SystemCapabilities> {
    return this.http.get<SystemCapabilities>(`${this.base}/api/system/capabilities`);
  }

  setDetectorNumWorkers(name: string, numWorkers: number): Observable<Detector> {
    return this.http.patch<Detector>(`${this.base}/api/config/detectors/${encodeURIComponent(name)}/num-workers`, {
      num_workers: numWorkers,
    });
  }

  listExecutionProviders(): Observable<ExecutionProvider[]> {
    return this.http.get<ExecutionProvider[]>(`${this.base}/api/config/execution-providers`);
  }

  listDetectorBackends(): Observable<string[]> {
    return this.http.get<string[]>(`${this.base}/api/config/detector-backends`);
  }

  getDetectorLabels(name: string): Observable<string[]> {
    return this.http.get<string[]>(`${this.base}/api/config/detectors/${encodeURIComponent(name)}/labels`);
  }

  listCameraConfigs(): Observable<CameraConfigDetail[]> {
    return this.http.get<CameraConfigDetail[]>(`${this.base}/api/config/cameras`);
  }

  getCameraConfig(name: string): Observable<CameraConfigDetail> {
    return this.http.get<CameraConfigDetail>(`${this.base}/api/config/cameras/${encodeURIComponent(name)}`);
  }

  createCamera(req: CameraWriteRequest): Observable<ConfigMutationResponse> {
    return this.http.post<ConfigMutationResponse>(`${this.base}/api/config/cameras`, req);
  }

  updateCamera(name: string, req: CameraWriteRequest): Observable<ConfigMutationResponse> {
    return this.http.put<ConfigMutationResponse>(`${this.base}/api/config/cameras/${encodeURIComponent(name)}`, req);
  }

  deleteCamera(name: string): Observable<void> {
    return this.http.delete<void>(`${this.base}/api/config/cameras/${encodeURIComponent(name)}`);
  }

  scanOnvifDevices(): Observable<OnvifDevice[]> {
    return this.http.get<OnvifDevice[]>(`${this.base}/api/onvif/scan`);
  }

  resolveOnvifStream(ip: string, username: string, password: string, port = 80): Observable<OnvifResolveResult> {
    return this.http.get<OnvifResolveResult>(`${this.base}/api/onvif/resolve`, {
      params: toHttpParams({ ip, port, username, password }),
    });
  }

  eventsStreamUrl(): string {
    return `${this.base}/api/events/stream`;
  }
}
