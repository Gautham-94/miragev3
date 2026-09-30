import { Component, OnInit, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { ApiService } from '../../core/services/api.service';
import { Detector, LabelFilter, OnvifDevice, RtspTransport } from '../../core/models/api.models';
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
  // Optional -- a real, cheap low-resolution stream URL (e.g. a Hikvision-style
  // Channels/102 substream) go2rtc should pull directly for the live-view grid tier
  // instead of transcoding one itself from the main stream above. See
  // mirage.config.schema.CameraConfig.live_sub_url's own docstring for why this
  // matters at all -- left blank, the grid tile still works, just via a costlier
  // server-side ffmpeg transcode of the main stream.
  protected readonly liveSubUrl = signal('');
  protected readonly cameraName = signal('');
  protected readonly selectedDetector = signal('');
  protected readonly trackObjectsInput = signal('person');
  // Which tracked-object labels drive Review/Alerts severity for this camera (see
  // mirage.events.review.classify_severity: a label in this list -> Severity.alert; a
  // label in detectionLabelsInput below (if not already an alert label) -> the lower
  // Severity.detection; anything else never becomes a ReviewSegment at all). Defaults
  // mirror ReviewConfig's own schema default ("person, car") -- NOT auto-derived from
  // Track objects above, since a label can be tracked (shows in Events) without being
  // alert-worthy, and vice versa isn't allowed (an alert label not being tracked would
  // never fire, but that's the user's call to make, not this form's to enforce).
  protected readonly alertLabelsInput = signal('person, car');
  protected readonly detectionLabelsInput = signal('');
  // When true, this camera tracks EVERY label its routed detector knows (e.g. all 80
  // COCO classes) -- Track objects above is still saved but ignored by the tracking
  // gate while this is on (see ObjectsConfig.track_all's docstring), so re-disabling
  // it restores whatever was typed there without the user needing to retype it.
  protected readonly trackAll = signal(false);
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
  // Both empty string (disabled) by default -- see RulesConfig's own docstring.
  // crowdThreshold: alert once this many confirmed "person" tracks are present at
  // once. dwellSeconds: alert once ANY tracked object has been continuously present
  // for at least this long (covers both loitering and queue-wait-time).
  protected readonly crowdThreshold = signal<number | null>(null);
  protected readonly dwellSeconds = signal<number | null>(null);
  // See ObjectFilterConfig's own docstring -- min_score gates whether a raw detection is
  // tracked at all; threshold ("Confirm score" in the UI -- "threshold" alone reads
  // ambiguous next to Min score) gates whether a tracked object's median score is ever
  // promoted from false_positive to true-positive (and therefore shown in Events/
  // Review). Keyed per-label (e.g. "animal"/"person"/"vehicle" for the MegaDetector
  // plugins) -- a real bug this replaces: a single shared pair used to silently write
  // to whichever label happened to be Track objects[0], leaving every other label (
  // including ones only tracked via Track all) stuck on the 0.5/0.7 schema default with
  // no way to change them. availableLabels drives row order/presence; refreshed
  // whenever the selected detector changes (onDetectorChange) so it always matches that
  // detector's own labelmap.
  protected readonly availableLabels = signal<string[]>([]);
  protected readonly labelFilters = signal<Record<string, LabelFilter>>({});

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
        if (detectors.length > 0 && !this.selectedDetector()) {
          this.selectedDetector.set(detectors[0].name);
          this.loadLabelsForDetector(detectors[0].name);
        }
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
          this.liveSubUrl.set(cam.live_sub_url ?? '');
          this.selectedDetector.set(cam.detector);
          this.trackObjectsInput.set(cam.track_objects.join(', '));
          this.trackAll.set(cam.track_all);
          this.width.set(cam.width);
          this.height.set(cam.height);
          this.fps.set(cam.fps);
          this.recordEnabled.set(cam.record_enabled);
          this.retainDays.set(cam.retain_days);
          this.segmentSeconds.set(cam.segment_seconds);
          this.rtspTransport.set(cam.rtsp_transport);
          this.alertLabelsInput.set(cam.alert_labels.join(', '));
          this.detectionLabelsInput.set(cam.detection_labels.join(', '));
          this.crowdThreshold.set(cam.crowd_threshold);
          this.dwellSeconds.set(cam.dwell_seconds);
          // cam.filters already covers every label this camera's OWN detector declares
          // (see mirage.api.routers.config._camera_config_out), so this is authoritative
          // -- no separate getDetectorLabels call needed for the initial load.
          this.availableLabels.set(Object.keys(cam.filters));
          this.labelFilters.set(cam.filters);
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

  protected onDetectorChange(name: string): void {
    this.selectedDetector.set(name);
    this.loadLabelsForDetector(name);
  }

  // Fetches the newly-selected detector's real label set and refreshes the per-label
  // filter rows to match -- keeps any label that already has a row (e.g. switching
  // detectors and back), defaults a genuinely new label to ObjectFilterConfig's own
  // schema defaults (0.5/0.7). Never removes an existing entry from labelFilters (a
  // label no longer in availableLabels just stops being submitted -- see submit()),
  // so switching detectors and back loses nothing.
  private loadLabelsForDetector(name: string): void {
    this.api.getDetectorLabels(name).subscribe({
      next: (labels) => {
        this.availableLabels.set(labels);
        const current = this.labelFilters();
        const next = { ...current };
        for (const label of labels) {
          if (!(label in next)) next[label] = { min_score: 0.5, threshold: 0.7 };
        }
        this.labelFilters.set(next);
      },
      error: () => this.availableLabels.set([]),
    });
  }

  protected updateLabelFilter(label: string, field: 'min_score' | 'threshold', value: number): void {
    const current = this.labelFilters()[label] ?? { min_score: 0.5, threshold: 0.7 };
    this.labelFilters.set({ ...this.labelFilters(), [label]: { ...current, [field]: value } });
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
    const alertLabels = this.alertLabelsInput()
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const detectionLabels = this.detectionLabelsInput()
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const payload = {
      name: this.cameraName().trim(),
      rtsp_url: this.resolvedRtspUrl().trim(),
      live_sub_url: this.liveSubUrl().trim() || null,
      detector: this.selectedDetector(),
      track_objects: trackObjects.length > 0 ? trackObjects : ['person'],
      track_all: this.trackAll(),
      width: this.width(),
      height: this.height(),
      fps: this.fps(),
      record_enabled: this.recordEnabled(),
      retain_days: this.retainDays(),
      segment_seconds: this.segmentSeconds(),
      rtsp_transport: this.rtspTransport(),
      // Explicitly sent (not omitted) even when empty -- the backend treats a MISSING
      // field as "keep the schema default," but an omitted alertLabels here would
      // otherwise silently reset an already-customized alert list back to ["person",
      // "car"] on every single edit+save, which is exactly the bug this field fixes.
      alert_labels: alertLabels,
      detection_labels: detectionLabels,
      crowd_threshold: this.crowdThreshold(),
      dwell_seconds: this.dwellSeconds(),
      // Only labels currently shown as rows (availableLabels, this detector's own real
      // label set) are sent -- the backend merges (see
      // mirage.api.routers.config._build_camera_config), so a label from a PREVIOUSLY
      // selected detector that's no longer relevant here is simply left untouched on
      // the backend rather than resubmitted with a stale/default value.
      filters: Object.fromEntries(
        this.availableLabels().map((label) => [label, this.labelFilters()[label] ?? { min_score: 0.5, threshold: 0.7 }]),
      ),
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
    this.liveSubUrl.set('');
    this.cameraName.set('');
    this.trackObjectsInput.set('person');
    this.trackAll.set(false);
    this.alertLabelsInput.set('person, car');
    this.detectionLabelsInput.set('');
    this.rtspTransport.set('tcp');
    this.segmentSeconds.set(10);
    this.crowdThreshold.set(null);
    this.dwellSeconds.set(null);
    this.savedCameraName.set(null);
  }
}
