import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { PagedList } from '../../core/paged-list';
import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { SseService } from '../../core/services/sse.service';
import { Event, SpeciesStatus } from '../../core/models/api.models';
import { FilterOption, FilterSelect } from '../../shared/filter-select/filter-select';
import { Icon } from '../../shared/icon/icon';
import { Lightbox } from '../../shared/lightbox/lightbox';
import { VideoLightbox } from '../../shared/video-lightbox/video-lightbox';

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

// Reconciliation-poll cadence: SSE (below) delivers new events live, so this is now
// just a periodic safety net for anything missed during a dropped SSE connection --
// slowed from the old 5s primary-polling cadence, not a latency-sensitive value anymore.
const RECONCILE_POLL_MS = 30000;
// Each card loads a real snapshot image on initial render -- keep this small; "Load
// more" fetches further pages on demand.
const PAGE_SIZE = 20;

@Component({
  selector: 'app-events-page',
  standalone: true,
  imports: [FilterSelect, Icon, Lightbox, VideoLightbox],
  templateUrl: './events-page.html',
  styleUrl: './events-page.scss',
})
export class EventsPage implements OnInit {
  private readonly paged = new PagedList<Event>(
    (limit, offset) => this.api.listEvents({ limit, offset }),
    (e) => e.id,
    PAGE_SIZE,
  );
  protected readonly events = this.paged.items;
  protected readonly hasMore = this.paged.hasMore;
  protected readonly loadingMore = this.paged.loadingMore;
  protected readonly loading = signal(true);
  protected readonly openEvent = signal<Event | null>(null);

  protected readonly playingClip = signal<Event | null>(null);
  protected readonly clipError = signal<string | null>(null);

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
    private readonly sse: SseService,
  ) {}

  ngOnInit(): void {
    this.api.listEvents({ limit: PAGE_SIZE }).subscribe({
      next: (events) => {
        this.paged.setFirstPage(events);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });

    this.sse
      .connect(this.api.eventsStreamUrl())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((msg) => {
        if (msg.type === 'event') this.paged.prependLive(msg.data as Event);
      });

    visibleInterval(RECONCILE_POLL_MS)
      .pipe(
        switchMap(() => this.api.listEvents({ limit: PAGE_SIZE })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({ next: (events) => this.paged.setFirstPage(events) });
  }

  protected loadMore(): void {
    this.paged.loadMore();
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

  protected clipUrl(event: Event): string {
    return this.api.eventClipUrl(event.id);
  }

  protected clipDownloadName(event: Event): string {
    const stamp = new Date(event.start_time * 1000).toISOString().replace(/[:.]/g, '-');
    return `${event.camera}_${stamp}.mp4`;
  }

  protected playClip(event: Event): void {
    this.clipError.set(null);
    this.playingClip.set(event);
  }

  protected closeClip(): void {
    this.playingClip.set(null);
  }

  protected onClipError(): void {
    // Same reasoning as ReviewPage.onClipError -- the clip endpoint can genuinely fail
    // (no recordings left to stitch -> 404, stitching itself failed -> 500), so surface
    // a clear message instead of a silently broken player.
    this.clipError.set('Could not load video for this event -- its recordings may have expired.');
    this.playingClip.set(null);
  }
}
