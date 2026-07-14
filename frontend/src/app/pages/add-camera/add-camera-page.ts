import { Component, OnInit, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { ApiService } from '../../core/services/api.service';
import { Detector, OnvifDevice, RtspTransport } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';

type EntryMode = 'choose' | 'scan' | 'manual';
type Step = 'entry' | 'credentials' | 'details' | 'done' | 'loading';

@Component({
  selector: 'app-add-camera-page',
  standalone: true,
  imports: [FormsModule, Icon],
  templateUrl: './add-camera-page.html',
  styleUrl: './add-camera-page.scss',
})
export class AddCameraPage implements OnInit {
  // When editing an existing camera (reached from the Manage cameras page via
  // /add-camera?camera=<name>), the entry/scan/credentials steps are skipped entirely --
  // there's no re-discovery step, just the details form pre-filled from that camera's
  // current saved config, submitting via PUT instead of POST.
  protected readonly editingCameraName = signal<string | null>(null);
  protected readonly loadError = signal<string | null>(null);

  protected readonly step = signal<Step>('entry');
  protected readonly entryMode = signal<EntryMode>('choose');

  // Scan state
  protected readonly scanning = signal(false);
  protected readonly scanError = signal<string | null>(null);
  protected readonly devices = signal<OnvifDevice[]>([]);
  protected readonly selectedDevice = signal<OnvifDevice | null>(null);

  // Credentials step (scan path only)
  protected readonly onvifIp = signal('');
  protected readonly onvifPort = signal(80);
  protected readonly onvifUsername = signal('');
  protected readonly onvifPassword = signal('');
  protected readonly resolving = signal(false);
  protected readonly resolveError = signal<string | null>(null);

  // Details step (common to both entry modes) -- these MUST be signals, not plain
  // mutable class properties, so `canSubmitDetails` (a computed()) actually has a
  // reactive dependency to recompute from. This app runs zoneless change detection
  // (provideZonelessChangeDetection): a plain `[(ngModel)]`-bound property still updates
  // its own value correctly on user input, but a computed() built on top of PLAIN
  // properties never re-runs, since computed() only tracks signal reads -- confirmed
  // directly during development: the Save button stayed permanently disabled even with
  // valid values sitting in the bound plain properties, because canSubmitDetails() had
  // no signal to react to and simply never recomputed after its first (empty) read.
  protected readonly resolvedRtspUrl = signal('');
  protected readonly cameraName = signal('');
  protected readonly selectedDetector = signal('');
  protected readonly trackObjectsInput = signal('person');
  protected readonly width = signal(640);
  protected readonly height = signal(480);
  protected readonly fps = signal(5);
  protected readonly recordEnabled = signal(true);
  protected readonly retainDays = signal(7);
  // Matches Frigate's own real default (10s) -- see RecordConfig.segment_seconds's
  // docstring in mirage/config/schema.py.
  protected readonly segmentSeconds = signal(10);
  // TCP is the correct default (in-order, lossless) -- UDP is the escape hatch for
  // cameras whose RTSP-over-TCP implementation is unreliable (confirmed against a real
  // Hikvision camera: a single, otherwise-idle TCP RTSP session was reset by the camera
  // after ~2 seconds every time, while the same stream over UDP ran stable).
  protected readonly rtspTransport = signal<RtspTransport>('tcp');
  // False (default): open-vocab queries only ever check crops of objects the closed-
  // vocab detector already confirmed. True: this camera's queries are checked directly
  // against the whole motion-triggered frame instead -- for open-vocab items outside
  // that detector's label map entirely (see CameraConfig.openvocab_direct_frame).
  protected readonly openvocabDirectFrame = signal(false);

  protected readonly detectors = signal<Detector[]>([]);
  protected readonly saving = signal(false);
  protected readonly saveError = signal<string | null>(null);
  protected readonly savedCameraName = signal<string | null>(null);

  protected readonly canSubmitDetails = computed(() => {
    return (
      this.cameraName().trim().length > 0 &&
      this.resolvedRtspUrl().trim().length > 0 &&
      this.selectedDetector().length > 0
    );
  });

  constructor(
    private readonly api: ApiService,
    private readonly router: Router,
    private readonly route: ActivatedRoute,
  ) {}

  ngOnInit(): void {
    this.api.listDetectors().subscribe({
      next: (detectors) => {
        this.detectors.set(detectors);
        if (detectors.length > 0 && !this.selectedDetector()) this.selectedDetector.set(detectors[0].name);
      },
      error: () => this.detectors.set([]),
    });

    const editName = this.route.snapshot.queryParamMap.get('camera');
    if (editName) {
      this.editingCameraName.set(editName);
      this.step.set('loading');
      this.api.getCameraConfig(editName).subscribe({
        next: (cam) => {
          this.cameraName.set(cam.name);
          this.resolvedRtspUrl.set(cam.rtsp_url);
          this.selectedDetector.set(cam.detector);
          this.trackObjectsInput.set(cam.track_objects.join(', '));
          this.width.set(cam.width);
          this.height.set(cam.height);
          this.fps.set(cam.fps);
          this.recordEnabled.set(cam.record_enabled);
          this.retainDays.set(cam.retain_days);
          this.segmentSeconds.set(cam.segment_seconds);
          this.rtspTransport.set(cam.rtsp_transport);
          this.openvocabDirectFrame.set(cam.openvocab_direct_frame);
          this.step.set('details');
        },
        error: (err) => {
          this.loadError.set(err?.error?.detail ?? `Could not load camera "${editName}".`);
          this.step.set('entry');
        },
      });
    }
  }

  protected get isEditMode(): boolean {
    return this.editingCameraName() !== null;
  }

  // --- Entry step ---

  protected chooseScan(): void {
    this.entryMode.set('scan');
    this.runScan();
  }

  protected chooseManual(): void {
    this.entryMode.set('manual');
    this.step.set('details');
  }

  protected runScan(): void {
    this.scanning.set(true);
    this.scanError.set(null);
    this.devices.set([]);
    this.api.scanOnvifDevices().subscribe({
      next: (devices) => {
        this.devices.set(devices);
        this.scanning.set(false);
      },
      error: (err) => {
        this.scanError.set(err?.error?.detail ?? 'Scan failed. Is go2rtc running?');
        this.scanning.set(false);
      },
    });
  }

  protected selectDevice(device: OnvifDevice): void {
    this.selectedDevice.set(device);
    // go2rtc's discovered URL is onvif://user:pass@ip[:port] -- pull the IP (and port,
    // if the device advertised a non-default one, e.g. mock_cameras running several
    // devices on one host) out for the credentials step. The placeholder user/pass in
    // it are never real credentials, see mirage/api/routers/onvif.py.
    const match = device.url.match(/@([^/?:]+)(?::(\d+))?/);
    this.onvifIp.set(match ? match[1] : '');
    this.onvifPort.set(match?.[2] ? Number(match[2]) : 80);
    this.step.set('credentials');
  }

  protected backToDeviceList(): void {
    this.step.set('entry');
    this.resolveError.set(null);
  }

  // --- Credentials step (scan path) ---

  protected resolveStream(): void {
    if (!this.onvifIp() || !this.onvifUsername()) return;
    this.resolving.set(true);
    this.resolveError.set(null);
    this.api
      .resolveOnvifStream(this.onvifIp(), this.onvifUsername(), this.onvifPassword(), this.onvifPort())
      .subscribe({
        next: (result) => {
          this.resolvedRtspUrl.set(result.rtsp_url);
          // Prefer profile_name over device_name: profile_name identifies the specific
          // stream/channel (distinct per camera even on a multi-camera ONVIF host, e.g.
          // mock_cameras' front_door/backyard/garage all sharing one device_name of
          // "MockCameras MC-1000") while device_name only identifies the hardware model,
          // which can collide across multiple physically-different cameras and cause
          // each new save to silently overwrite the previous one (same derived name ==
          // same camera identity in the config API).
          this.cameraName.set(
            (result.profile_name ?? result.device_name ?? this.selectedDevice()?.name ?? 'camera')
              .toLowerCase()
              .replace(/[^a-z0-9]+/g, '_')
              .replace(/^_+|_+$/g, ''),
          );
          this.resolving.set(false);
          this.step.set('details');
        },
        error: (err) => {
          this.resolveError.set(err?.error?.detail ?? 'Could not resolve this camera’s stream. Check the credentials.');
          this.resolving.set(false);
        },
      });
  }

  // --- Details step ---

  protected backToEntry(): void {
    this.step.set('entry');
    this.entryMode.set('choose');
    this.devices.set([]);
    this.selectedDevice.set(null);
    this.saveError.set(null);
  }

  protected save(): void {
    if (!this.canSubmitDetails()) return;
    this.saving.set(true);
    this.saveError.set(null);

    const trackObjects = this.trackObjectsInput()
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const payload = {
      name: this.cameraName().trim(),
      rtsp_url: this.resolvedRtspUrl().trim(),
      detector: this.selectedDetector(),
      track_objects: trackObjects.length > 0 ? trackObjects : ['person'],
      width: this.width(),
      height: this.height(),
      fps: this.fps(),
      record_enabled: this.recordEnabled(),
      retain_days: this.retainDays(),
      segment_seconds: this.segmentSeconds(),
      rtsp_transport: this.rtspTransport(),
      openvocab_direct_frame: this.openvocabDirectFrame(),
    };

    const request = this.isEditMode
      ? this.api.updateCamera(this.editingCameraName()!, payload)
      : this.api.createCamera(payload);

    request.subscribe({
      next: (resp) => {
        this.saving.set(false);
        this.savedCameraName.set(resp.camera.name);
        this.step.set('done');
      },
      error: (err) => {
        this.saving.set(false);
        this.saveError.set(err?.error?.detail ?? 'Could not save this camera.');
      },
    });
  }

  protected goToLive(): void {
    this.router.navigate(['/live']);
  }

  protected goToManageCameras(): void {
    this.router.navigate(['/cameras']);
  }

  protected addAnother(): void {
    this.step.set('entry');
    this.entryMode.set('choose');
    this.devices.set([]);
    this.selectedDevice.set(null);
    this.onvifIp.set('');
    this.onvifPort.set(80);
    this.onvifUsername.set('');
    this.onvifPassword.set('');
    this.resolvedRtspUrl.set('');
    this.cameraName.set('');
    this.trackObjectsInput.set('person');
    this.rtspTransport.set('tcp');
    this.segmentSeconds.set(10);
    this.openvocabDirectFrame.set(false);
    this.savedCameraName.set(null);
  }
}
