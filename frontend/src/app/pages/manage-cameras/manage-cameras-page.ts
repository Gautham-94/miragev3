import { Component, OnInit, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import { ApiService } from '../../core/services/api.service';
import { CameraConfigDetail } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';

@Component({
  selector: 'app-manage-cameras-page',
  standalone: true,
  imports: [RouterLink, Icon],
  templateUrl: './manage-cameras-page.html',
  styleUrl: './manage-cameras-page.scss',
})
export class ManageCamerasPage implements OnInit {
  protected readonly cameras = signal<CameraConfigDetail[]>([]);
  protected readonly loading = signal(true);
  protected readonly loadError = signal<string | null>(null);

  protected readonly pendingDelete = signal<string | null>(null);
  protected readonly deleting = signal(false);
  protected readonly deleteError = signal<string | null>(null);
  protected readonly justDeleted = signal(false);

  protected readonly togglingName = signal<string | null>(null);
  protected readonly toggleError = signal<string | null>(null);
  protected readonly justToggled = signal<{ name: string; enabled: boolean } | null>(null);

  constructor(private readonly api: ApiService, private readonly router: Router) {}

  ngOnInit(): void {
    this.load();
  }

  private load(): void {
    this.loading.set(true);
    this.api.listCameraConfigs().subscribe({
      next: (cameras) => {
        this.cameras.set(cameras);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(err?.error?.detail ?? 'Could not load cameras.');
        this.loading.set(false);
      },
    });
  }

  protected editCamera(name: string): void {
    this.router.navigate(['/add-camera'], { queryParams: { camera: name } });
  }

  protected toggleEnabled(camera: CameraConfigDetail): void {
    this.togglingName.set(camera.name);
    this.toggleError.set(null);
    const nextEnabled = !camera.enabled;

    this.api
      .updateCamera(camera.name, {
        name: camera.name,
        rtsp_url: camera.rtsp_url,
        detector: camera.detector,
        track_objects: camera.track_objects,
        width: camera.width,
        height: camera.height,
        fps: camera.fps,
        record_enabled: camera.record_enabled,
        retain_days: camera.retain_days,
        segment_seconds: camera.segment_seconds,
        alert_labels: camera.alert_labels,
        detection_labels: camera.detection_labels,
        enabled: nextEnabled,
        rtsp_transport: camera.rtsp_transport,
        min_score: camera.min_score,
        threshold: camera.threshold,
      })
      .subscribe({
        next: () => {
          this.togglingName.set(null);
          this.justToggled.set({ name: camera.name, enabled: nextEnabled });
          this.load();
        },
        error: (err) => {
          this.togglingName.set(null);
          this.toggleError.set(err?.error?.detail ?? `Could not ${nextEnabled ? 'start' : 'stop'} ${camera.name}.`);
        },
      });
  }

  protected confirmDelete(name: string): void {
    this.pendingDelete.set(name);
    this.deleteError.set(null);
  }

  protected cancelDelete(): void {
    this.pendingDelete.set(null);
  }

  protected deleteCamera(name: string): void {
    this.deleting.set(true);
    this.deleteError.set(null);
    this.api.deleteCamera(name).subscribe({
      next: () => {
        this.deleting.set(false);
        this.pendingDelete.set(null);
        this.justDeleted.set(true);
        this.load();
      },
      error: (err) => {
        this.deleting.set(false);
        this.deleteError.set(err?.error?.detail ?? `Could not delete ${name}.`);
      },
    });
  }
}
