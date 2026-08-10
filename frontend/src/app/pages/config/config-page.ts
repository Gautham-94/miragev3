import { DecimalPipe } from '@angular/common';
import { Component, OnInit, computed, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { ApiService } from '../../core/services/api.service';
import { Detector, SystemCapabilities } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';

@Component({
  selector: 'app-config-page',
  standalone: true,
  imports: [Icon, RouterLink, DecimalPipe],
  templateUrl: './config-page.html',
  styleUrl: './config-page.scss',
})
export class ConfigPage implements OnInit {
  protected readonly detectors = signal<Detector[]>([]);
  protected readonly loading = signal(true);
  protected readonly loadError = signal<string | null>(null);
  // Detector name currently being toggled -- disables just that row's switch while
  // in flight, rather than a single page-wide "saving" flag, so other rows stay
  // interactive.
  protected readonly togglingName = signal<string | null>(null);
  protected readonly toggleError = signal<string | null>(null);

  // Detector name currently having its worker count changed -- same per-row-only
  // disable pattern as togglingName above.
  protected readonly changingWorkersName = signal<string | null>(null);
  protected readonly workersError = signal<string | null>(null);

  protected readonly capabilities = signal<SystemCapabilities | null>(null);

  // How many detector worker processes are already accounted for by OTHER enabled
  // detectors' num_workers -- used to warn if the total starts exceeding CPU core
  // count, since every worker process (this detector's or another's) competes for the
  // same physical cores/accelerator, not just this one detector's own headroom.
  protected readonly totalWorkersAcrossDetectors = computed(() =>
    this.detectors()
      .filter((d) => d.enabled)
      .reduce((sum, d) => sum + d.num_workers, 0),
  );

  protected readonly restarting = signal(false);
  protected readonly restartRequested = signal(false);

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.load();
    this.api.getSystemCapabilities().subscribe({
      next: (caps) => this.capabilities.set(caps),
      // Non-fatal: the worker-count controls still work without capability guidance,
      // they just won't show the "you have N cores" note.
      error: () => {},
    });
  }

  private load(): void {
    this.loading.set(true);
    this.api.listDetectors().subscribe({
      next: (detectors) => {
        this.detectors.set(detectors);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(err?.error?.detail ?? 'Could not load detectors.');
        this.loading.set(false);
      },
    });
  }

  protected toggleDetector(detector: Detector): void {
    this.togglingName.set(detector.name);
    this.toggleError.set(null);
    const nextEnabled = !detector.enabled;
    this.api.setDetectorEnabled(detector.name, nextEnabled).subscribe({
      next: (updated) => {
        this.togglingName.set(null);
        this.detectors.update((list) => list.map((d) => (d.name === updated.name ? updated : d)));
      },
      error: (err) => {
        this.togglingName.set(null);
        this.toggleError.set(err?.error?.detail ?? `Could not ${nextEnabled ? 'enable' : 'disable'} ${detector.name}.`);
      },
    });
  }

  protected changeNumWorkers(detector: Detector, delta: number): void {
    const next = Math.min(8, Math.max(1, detector.num_workers + delta));
    if (next === detector.num_workers) return;

    this.changingWorkersName.set(detector.name);
    this.workersError.set(null);
    this.api.setDetectorNumWorkers(detector.name, next).subscribe({
      next: (updated) => {
        this.changingWorkersName.set(null);
        this.detectors.update((list) => list.map((d) => (d.name === updated.name ? updated : d)));
      },
      error: (err) => {
        this.changingWorkersName.set(null);
        this.workersError.set(err?.error?.detail ?? `Could not change worker count for ${detector.name}.`);
      },
    });
  }

  protected requestRestart(): void {
    if (this.restarting()) return;
    this.restarting.set(true);
    this.restartRequested.set(false);
    this.api.restartPipeline().subscribe({
      next: () => {
        this.restarting.set(false);
        this.restartRequested.set(true);
      },
      error: () => {
        this.restarting.set(false);
      },
    });
  }
}
