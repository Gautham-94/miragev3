import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { PagedList } from '../../core/paged-list';
import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { SseService } from '../../core/services/sse.service';
import { Event, ReviewSegment } from '../../core/models/api.models';
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
// mirage.events.processor.SPECIES_ENRICHABLE_LABELS. Duplicated from events-page.ts
// rather than shared -- this page is meant to stand alone (see its own docstring below).
const SPECIES_ENRICHABLE_LABELS = new Set(['animal', 'bird']);

// SSE delivers new sightings live; this is a periodic safety net for anything missed
// during a dropped SSE connection -- same reasoning as events-page.ts/review-page.ts.
const RECONCILE_POLL_MS = 30000;
const PAGE_SIZE = 20;

/**
 * Every distinct SIGHTING (one row per Event -- one per tracked object, no cross-object
 * suppression, see mirage/events/processor.py), as a flat, chronological list -- NOT
 * grouped by scene. A long continuous scene with a lion, then an elephant, then a
 * buffalo shows up as three separate rows here, each with its own boxed snapshot,
 * rather than one scene card hiding all three behind a click.
 *
 * Each row optionally links to the ReviewSegment ("scene") it belongs to via
 * ApiService.getScenesForEvents -- a batched reverse lookup (one request per page of
 * rows, not one per row) resolving mirage/api/routers/events.py's GET /scenes. Not
 * every sighting has a scene: ReviewSegmentMaintainer only opens one for labels in
 * review.alerts.labels/review.detections.labels (mirage/events/review.py), so a
 * sighting outside that filter gets no "Full scene" button at all. No separate
 * metadata panel/click for the scene -- its severity is shown directly on the row, and
 * the button plays its clip immediately (see playSceneClip).
 *
 * False positives are excluded by default -- same as ApiService.listEvents' own
 * default (include_false_positive: false).
 *
 * Deliberately does not touch review-page.ts/events-page.ts or their routes/backing
 * endpoints -- this remains purely additive.
 */
@Component({
  selector: 'app-detections-page',
  standalone: true,
  imports: [FilterSelect, Icon, Lightbox, VideoLightbox],
  templateUrl: './detections-page.html',
  styleUrl: './detections-page.scss',
})
export class DetectionsPage implements OnInit {
  private readonly paged = new PagedList<Event>(
    (limit, offset) => this.api.listEvents({ limit, offset }),
    (e) => e.id,
    PAGE_SIZE,
  );
  protected readonly sightings = this.paged.items;
  protected readonly hasMore = this.paged.hasMore;
  protected readonly loadingMore = this.paged.loadingMore;
  protected readonly loading = signal(true);

  // event id -> its scene, or null if it never qualified for one. Absent key = not
  // resolved yet (resolveScenes fills this in per page/live-push, batched).
  private readonly scenesByEventId = signal<Record<string, ReviewSegment | null>>({});

  protected readonly cameraFilter = signal('');
  protected readonly labelFilter = signal('');
  protected readonly speciesFilter = signal('');
  protected readonly timeRangeFilter = signal('');
  protected readonly timeRangeOptions = TIME_RANGE_OPTIONS;

  protected readonly cameraOptions = computed<FilterOption[]>(() => {
    const cameras = Array.from(new Set(this.sightings().map((e) => e.camera))).sort();
    return [{ value: '', label: 'All cameras' }, ...cameras.map((c) => ({ value: c, label: c }))];
  });

  protected readonly labelOptions = computed<FilterOption[]>(() => {
    const labels = Array.from(new Set(this.sightings().map((e) => e.label))).sort();
    return [{ value: '', label: 'All labels' }, ...labels.map((l) => ({ value: l, label: l }))];
  });

  protected readonly speciesOptions = computed<FilterOption[]>(() => {
    const species = Array.from(
      new Set(this.sightings().map((e) => e.species).filter((s): s is string => !!s)),
    ).sort();
    return [{ value: '', label: 'All species' }, ...species.map((s) => ({ value: s, label: s }))];
  });

  protected readonly filteredSightings = computed(() => {
    const camera = this.cameraFilter();
    const label = this.labelFilter();
    const species = this.speciesFilter();
    const rangeSeconds = Number(this.timeRangeFilter()) || null;
    const cutoff = rangeSeconds ? Date.now() / 1000 - rangeSeconds : null;
    return this.sightings().filter(
      (e) =>
        (!camera || e.camera === camera) &&
        (!label || e.label === label) &&
        (!species || e.species === species) &&
        (cutoff === null || e.start_time >= cutoff),
    );
  });

  protected readonly playingSceneClip = signal<ReviewSegment | null>(null);
  protected readonly sceneClipError = signal<string | null>(null);

  protected readonly playingClip = signal<Event | null>(null);
  protected readonly clipError = signal<string | null>(null);

  // Clicking a row shows the same boxed snapshot already used as its thumbnail, full
  // size -- same component/pattern as events-page.ts's own openLightbox/closeLightbox.
  protected readonly openSighting = signal<Event | null>(null);

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
        this.resolveScenes(events);
      },
      error: () => this.loading.set(false),
    });

    this.sse
      .connect(this.api.eventsStreamUrl())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((msg) => {
        if (msg.type === 'event') {
          const event = msg.data as Event;
          this.paged.prependLive(event);
          this.resolveScenes([event]);
        }
      });

    visibleInterval(RECONCILE_POLL_MS)
      .pipe(
        switchMap(() => this.api.listEvents({ limit: PAGE_SIZE })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (events) => {
          this.paged.setFirstPage(events);
          this.resolveScenes(events);
        },
      });
  }

  protected loadMore(): void {
    const before = this.sightings().length;
    this.paged.loadMore();
    // loadMore() is async internally (PagedList has no completion callback) -- resolve
    // scenes for the newly-appended slice once it lands. Cheap to poll a signal's own
    // computed length here rather than plumbing a callback through PagedList just for
    // this one caller.
    queueMicrotask(() => {
      const items = this.sightings();
      if (items.length > before) this.resolveScenes(items.slice(before));
    });
  }

  private resolveScenes(events: Event[]): void {
    const known = this.scenesByEventId();
    const unresolved = events.filter((e) => !(e.id in known)).map((e) => e.id);
    if (unresolved.length === 0) return;
    this.api.getScenesForEvents(unresolved).subscribe({
      next: (scenes) => this.scenesByEventId.update((current) => ({ ...current, ...scenes })),
      // Best-effort -- a sighting simply shows no "View scene" link if this fails.
      error: () => {},
    });
  }

  protected sceneFor(sighting: Event): ReviewSegment | null | undefined {
    return this.scenesByEventId()[sighting.id];
  }

  protected snapshotUrl(sighting: Event): string {
    return this.api.eventSnapshotUrl(sighting.id);
  }

  protected showsSpeciesBadge(sighting: Event): boolean {
    return SPECIES_ENRICHABLE_LABELS.has(sighting.label);
  }

  protected speciesBadgeText(sighting: Event): string {
    switch (sighting.species_status) {
      case 'complete':
        return sighting.species ?? 'Unknown species';
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

  protected clipUrl(sighting: Event): string {
    return this.api.eventClipUrl(sighting.id);
  }

  protected clipDownloadName(sighting: Event): string {
    const stamp = new Date(sighting.start_time * 1000).toISOString().replace(/[:.]/g, '-');
    return `${sighting.camera}_${stamp}.mp4`;
  }

  protected playClip(sighting: Event): void {
    this.clipError.set(null);
    this.playingClip.set(sighting);
  }

  protected closeClip(): void {
    this.playingClip.set(null);
  }

  protected onClipError(): void {
    this.clipError.set('Could not load video for this sighting -- its recordings may have expired.');
    this.playingClip.set(null);
  }

  protected openLightbox(sighting: Event): void {
    if (!sighting.has_snapshot) return;
    this.openSighting.set(sighting);
  }

  protected closeLightbox(): void {
    this.openSighting.set(null);
  }

  protected sceneClipUrl(scene: ReviewSegment): string {
    return this.api.reviewClipUrl(scene.id);
  }

  protected sceneClipDownloadName(scene: ReviewSegment): string {
    const stamp = new Date(scene.start_time * 1000).toISOString().replace(/[:.]/g, '-');
    return `${scene.camera}_scene_${stamp}.mp4`;
  }

  protected playSceneClip(scene: ReviewSegment): void {
    this.sceneClipError.set(null);
    this.playingSceneClip.set(scene);
  }

  protected closeSceneClip(): void {
    this.playingSceneClip.set(null);
  }

  protected onSceneClipError(): void {
    this.sceneClipError.set('Could not load video for this scene -- its recordings may have expired.');
    this.playingSceneClip.set(null);
  }

  protected formatTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
  }
}
