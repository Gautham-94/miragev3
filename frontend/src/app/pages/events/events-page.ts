import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { switchMap } from 'rxjs';

import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { Event, SpeciesStatus } from '../../core/models/api.models';
import { FilterOption, FilterSelect } from '../../shared/filter-select/filter-select';
import { Icon } from '../../shared/icon/icon';
import { Lightbox } from '../../shared/lightbox/lightbox';

const TIME_RANGE_OPTIONS: FilterOption[] = [
  { value: '', label: 'All time' },
  { value: '3600', label: 'Last hour' },
  { value: '86400', label: 'Last 24h' },
  { value: '604800', label: 'Last 7 days' },
];

// Every label the (optional, opt-in) species classifier ever enriches -- see
// mirage.events.processor.SPECIES_ENRICHABLE_LABELS. Every other label's
// species_status is always 'not_applicable' and never shown as a badge.
const SPECIES_ENRICHABLE_LABELS = new Set(['animal', 'bird']);

// A recording clip is unlikely to line up EXACTLY with an event's start/end (segment
// boundaries vs. detection boundaries), so the Recordings page is given a small
// padding window around the event to search within, not just the bare start/end.
const RECORDING_LOOKUP_PADDING_SECONDS = 30;

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
  protected readonly speciesFilter = signal('');
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

  // Distinct identified species names seen across currently-loaded events -- lets you
  // filter down to e.g. "Odocoileus virginianus" specifically, not just "animal".
  // Events with no species identified yet (pending/failed/skipped/not_applicable)
  // aren't part of this list; there's nothing meaningful to filter them by name-wise.
  protected readonly speciesOptions = computed<FilterOption[]>(() => {
    const species = Array.from(
      new Set(this.events().map((e) => e.species).filter((s): s is string => !!s)),
    ).sort();
    return [{ value: '', label: 'All species' }, ...species.map((s) => ({ value: s, label: s }))];
  });

  protected readonly filteredEvents = computed(() => {
    const camera = this.cameraFilter();
    const label = this.labelFilter();
    const species = this.speciesFilter();
    const rangeSeconds = Number(this.timeRangeFilter()) || null;
    const cutoff = rangeSeconds ? Date.now() / 1000 - rangeSeconds : null;
    return this.events().filter(
      (e) =>
        (!camera || e.camera === camera) &&
        (!label || e.label === label) &&
        (!species || e.species === species) &&
        (cutoff === null || e.start_time >= cutoff),
    );
  });

  constructor(
    private readonly api: ApiService,
    private readonly destroyRef: DestroyRef,
    private readonly router: Router,
  ) {}

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

  // Only animal/bird events ever get a species badge at all -- every other label's
  // species_status is permanently 'not_applicable' (see SPECIES_ENRICHABLE_LABELS),
  // which would be pure noise to show on every person/car/etc. card.
  protected showsSpeciesBadge(event: Event): boolean {
    return SPECIES_ENRICHABLE_LABELS.has(event.label);
  }

  protected speciesBadgeText(event: Event): string {
    switch (event.species_status) {
      case 'complete':
        return event.species ?? 'Unknown species';
      case 'pending':
        return 'Identifying…';
      case 'failed':
        return 'Species ID failed';
      case 'skipped':
        return 'Species not identified';
      default:
        return '';
    }
  }

  protected speciesConfidenceText(event: Event): string | null {
    if (event.species_status !== 'complete' || event.species_confidence === null) return null;
    return `${(event.species_confidence * 100).toFixed(0)}%`;
  }

  // Deep-links into the Recordings page, pre-filtered to this event's camera and a
  // time window around when it happened -- there is no direct event<->recording
  // foreign key (Events and Recordings are independent tables, see
  // mirage.db.models), so this is a best-effort "video from around this moment" link
  // rather than a guaranteed single-clip lookup. See RecordingsPage's query-param
  // handling for the other half of this link.
  protected viewVideo(event: Event): void {
    this.router.navigate(['/recordings'], {
      queryParams: {
        camera: event.camera,
        after: event.start_time - RECORDING_LOOKUP_PADDING_SECONDS,
        before: (event.end_time ?? event.start_time) + RECORDING_LOOKUP_PADDING_SECONDS,
      },
    });
  }
}
