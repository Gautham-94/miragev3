import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router } from '@angular/router';
import { filter, switchMap } from 'rxjs';

import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { Camera } from '../../core/models/api.models';
import { CameraTile } from './camera-tile/camera-tile';
import { LiveActivation } from './live-activation';

// Cameras can now be added/removed/enabled/disabled live via the config API without a
// pipeline restart (see mirage/config_reload.py, TODO_FIX_LIST.md item 2/11) -- this
// page previously fetched the camera list exactly once on load and never again, so an
// already-open Live page kept showing a tile for a camera that had just been disabled
// on the Manage Cameras page, even though the backend had already correctly stopped
// tracking it. Polling (same visibleInterval pattern as Events/Review/Recordings)
// closes that gap; the `@for (... track camera.name)` loop in live-page.html already
// reconciles by name, so re-setting the full list on each tick does NOT tear down and
// recreate CameraTile instances (and their live MSE/WebRTC connection state) for
// cameras that are still present -- only tiles for genuinely added/removed cameras are
// created/destroyed.
const POLL_MS = 10000;

@Component({
  selector: 'app-live-page',
  standalone: true,
  imports: [CameraTile],
  templateUrl: './live-page.html',
  styleUrl: './live-page.scss',
})
export class LivePage implements OnInit {
  protected readonly cameras = signal<Camera[]>([]);
  protected readonly loading = signal(true);
  // Name of the single camera currently shown maximized, or null for the normal grid.
  // A camera name, not an index/reference, so it survives the `cameras` list being
  // replaced wholesale on every poll tick (see this class's own docstring above) --
  // an index would silently point at the wrong tile the moment the list re-orders.
  protected readonly maximizedCamera = signal<string | null>(null);

  // GET /api/cameras includes disabled cameras too (so Manage Cameras can still list/
  // re-enable them) -- a disabled camera has no CameraTracker/CameraCapture running at
  // all, so a Live tile for one would sit permanently stuck in a connecting/error state
  // rather than showing anything meaningful. Filtered here (not server-side) so the
  // page subtitle can still report the true configured count.
  protected readonly liveCameras = computed(() => this.cameras().filter((c) => c.enabled));

  constructor(
    private readonly api: ApiService,
    private readonly destroyRef: DestroyRef,
    private readonly router: Router,
    private readonly liveActivation: LiveActivation,
  ) {}

  ngOnInit(): void {
    // Set up exactly once (ngOnInit never runs again once LivePersistingRouteReuseStrategy
    // starts reusing this instance -- see its own docstring) and left running for the
    // app's lifetime: fires on every SUBSEQUENT navigation back to /live too, which is
    // exactly the "you're looking at this page again" signal CameraTile has no other way
    // to get. See LiveActivation's own docstring for why this matters at all.
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter((e) => e.urlAfterRedirects.startsWith('/live')),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(() => this.liveActivation.bump());

    visibleInterval(POLL_MS)
      .pipe(
        switchMap(() => this.api.listCameras()),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (cameras) => {
          this.cameras.set(cameras);
          this.loading.set(false);
          // The maximized camera was disabled/removed since the last poll -- fall back
          // to the grid rather than leaving the page stuck showing a stale/empty tile.
          const maximized = this.maximizedCamera();
          if (maximized && !cameras.some((c) => c.name === maximized && c.enabled)) {
            this.maximizedCamera.set(null);
          }
        },
        error: () => this.loading.set(false),
      });
  }

  protected toggleMaximized(cameraName: string): void {
    this.maximizedCamera.set(this.maximizedCamera() === cameraName ? null : cameraName);
  }
}
