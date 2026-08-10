import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { interval, startWith, switchMap } from 'rxjs';

import { ApiService } from '../core/services/api.service';
import { SystemState } from '../core/models/api.models';
import { Icon } from '../shared/icon/icon';

// Fast poll (not the usual 5-10s visibleInterval pace elsewhere in this app) -- this
// drives the restart button's own progress feedback, so the user watching it needs to
// see state changes promptly, not up to several seconds late. Deliberately NOT paused
// on tab-hidden (unlike visibleInterval-based pages): a restart the user just triggered
// should keep being tracked even if they alt-tab away and back, rather than resuming
// from a stale "restarting" state days later on a background tab.
const STATUS_POLL_MS = 1500;
// How long to keep showing "System ready" after a restart completes before the button
// quietly reverts to its normal idle state -- long enough to register as a deliberate
// confirmation, short enough to not linger and become visual clutter.
const READY_BANNER_MS = 4000;

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Icon],
  templateUrl: './shell.html',
  styleUrl: './shell.scss',
})
export class Shell implements OnInit {
  protected readonly navItems = [
    { path: '/review', label: 'Review', icon: 'grid' },
    { path: '/live', label: 'Live', icon: 'video' },
    { path: '/events', label: 'Events', icon: 'target' },
    { path: '/recordings', label: 'Recordings', icon: 'film' },
    { path: '/cameras', label: 'Cameras', icon: 'settings' },
    { path: '/detectors', label: 'Detectors', icon: 'cpu' },
    { path: '/config', label: 'Config', icon: 'settings' },
    { path: '/logs', label: 'Logs', icon: 'list' },
  ];

  protected readonly systemState = signal<SystemState>('unknown');
  // Set locally the instant the button is pressed, before the first poll can possibly
  // observe the supervisor's own "stopping"/"starting" transition (which happens on a
  // ~1s poll loop on the BACKEND too -- see mirage/supervisor.py's POLL_INTERVAL_SECONDS)
  // -- without this, pressing the button would show nothing happening for up to ~2.5s
  // combined worst-case latency (this page's own poll cadence + the supervisor's).
  protected readonly restartRequestedLocally = signal(false);
  protected readonly showReadyBanner = signal(false);
  private wasRestarting = false;
  private readyBannerTimeout: ReturnType<typeof setTimeout> | null = null;

  // "restarting" covers both the local optimistic flag (immediately after the button
  // press) and the real backend states that mean "the pipeline is not currently up" --
  // starting/stopping are both mid-restart from the user's point of view, one banner.
  protected readonly isRestarting = computed(
    () =>
      this.restartRequestedLocally() ||
      this.systemState() === 'starting' ||
      this.systemState() === 'stopping',
  );
  protected readonly isCrashed = computed(() => this.systemState() === 'crashed');

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngOnInit(): void {
    interval(STATUS_POLL_MS)
      .pipe(
        startWith(0),
        switchMap(() => this.api.getSystemStatus()),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (status) => this.applyStatus(status.state),
        // A network hiccup or the API being mid-restart itself shouldn't flip the
        // banner to something alarming -- just skip this tick and let the next poll
        // (1.5s later) recover naturally.
        error: () => {},
      });
  }

  private applyStatus(state: SystemState): void {
    const wasRestarting = this.wasRestarting;
    const isRestartingNow = state === 'starting' || state === 'stopping';

    this.systemState.set(state);

    if (isRestartingNow) {
      // The backend has confirmed the restart is genuinely underway -- safe to drop
      // the optimistic local flag now, the real polled state carries the signal.
      this.restartRequestedLocally.set(false);
    }

    if (wasRestarting && state === 'running') {
      this.restartRequestedLocally.set(false);
      this.showReadyBanner.set(true);
      if (this.readyBannerTimeout) clearTimeout(this.readyBannerTimeout);
      this.readyBannerTimeout = setTimeout(() => this.showReadyBanner.set(false), READY_BANNER_MS);
    }

    this.wasRestarting = isRestartingNow;
  }

  protected applyChanges(): void {
    if (this.isRestarting()) return;
    this.restartRequestedLocally.set(true);
    this.showReadyBanner.set(false);
    this.api.restartPipeline().subscribe({
      error: () => this.restartRequestedLocally.set(false),
    });
  }
}
