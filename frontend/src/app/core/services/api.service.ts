import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

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
  OnvifDevice,
  OnvifResolveResult,
  Query,
  QueryMatch,
  QueryWriteRequest,
  Recording,
  RecordingListParams,
  ReviewListParams,
  ReviewSegment,
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

  listExecutionProviders(): Observable<ExecutionProvider[]> {
    return this.http.get<ExecutionProvider[]>(`${this.base}/api/config/execution-providers`);
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

  listQueries(): Observable<Query[]> {
    return this.http.get<Query[]>(`${this.base}/api/config/queries`);
  }

  createQuery(req: QueryWriteRequest): Observable<Query> {
    return this.http.post<Query>(`${this.base}/api/config/queries`, req);
  }

  updateQuery(id: string, req: QueryWriteRequest): Observable<Query> {
    return this.http.put<Query>(`${this.base}/api/config/queries/${encodeURIComponent(id)}`, req);
  }

  deleteQuery(id: string): Observable<void> {
    return this.http.delete<void>(`${this.base}/api/config/queries/${encodeURIComponent(id)}`);
  }

  listQueryMatches(params: { camera?: string; queryId?: string; limit?: number } = {}): Observable<QueryMatch[]> {
    return this.http.get<QueryMatch[]>(`${this.base}/api/query-matches`, {
      params: toHttpParams({ camera: params.camera, query_id: params.queryId, limit: params.limit }),
    });
  }

  queryMatchThumbnailUrl(id: string): string {
    return `${this.base}/api/query-matches/${encodeURIComponent(id)}/thumbnail`;
  }
}
