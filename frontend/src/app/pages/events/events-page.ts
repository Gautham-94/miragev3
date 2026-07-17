import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { Event } from '../../core/models/api.models';
import { FilterOption, FilterSelect } from '../../shared/filter-select/filter-select';
import { Icon } from '../../shared/icon/icon';
import { Lightbox } from '../../shared/lightbox/lightbox';

const TIME_RANGE_OPTIONS: FilterOption[] = [
  { value: '', label: 'All time' },
  { value: '3600', label: 'Last hour' },
  { value: '86400', label: 'Last 24h' },
  { value: '604800', label: 'Last 7 days' },
];

const POLL_MS = 5000;

@Component({
  selector: 'app-events-page',
  standalone: true,
  imports: [FilterSelect, Icon, Lightbox],
  templateUrl: './events-page.html',
  styleUrl: './events-page.scss',
})
export class EventsPage implements OnInit {
  protected readonly events = signal<Event[]>([]);
  protected readonly loading = signal(true);
  protected readonly openEvent = signal<Event | null>(null);

  protected readonly cameraFilter = signal('');
  protected readonly labelFilter = signal('');
  protected readonly timeRangeFilter = signal('');
  protected readonly timeRangeOptions = TIME_RANGE_OPTIONS;

  protected readonly cameraOptions = computed<FilterOption[]>(() => {
    const cameras = Array.from(new Set(this.events().map((e) => e.camera))).sort();
    return [{ value: '', label: 'All cameras' }, ...cameras.map((c) => ({ value: c, label: c }))];
  });

  protected readonly labelOptions = computed<FilterOption[]>(() => {
    const labels = Array.from(new Set(this.events().map((e) => e.label))).sort();
    return [{ value: '', label: 'All labels' }, ...labels.map((l) => ({ value: l, label: l }))];
  });

  protected readonly filteredEvents = computed(() => {
    const camera = this.cameraFilter();
    const label = this.labelFilter();
    const rangeSeconds = Number(this.timeRangeFilter()) || null;
    const cutoff = rangeSeconds ? Date.now() / 1000 - rangeSeconds : null;
    return this.events().filter(
      (e) =>
        (!camera || e.camera === camera) &&
        (!label || e.label === label) &&
        (cutoff === null || e.start_time >= cutoff),
    );
  });

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngOnInit(): void {
    visibleInterval(POLL_MS)
      .pipe(
        switchMap(() => this.api.listEvents({ limit: 200 })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (events) => {
          this.events.set(events);
          this.loading.set(false);
        },
        error: () => this.loading.set(false),
      });
  }

  protected snapshotUrl(event: Event): string {
    return this.api.eventSnapshotUrl(event.id);
  }

  protected formatTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
  }

  protected openLightbox(event: Event): void {
    if (!event.has_snapshot) return;
    this.openEvent.set(event);
  }

  protected closeLightbox(): void {
    this.openEvent.set(null);
  }
}
