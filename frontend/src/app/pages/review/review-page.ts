import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { PagedList } from '../../core/paged-list';
import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { SseService } from '../../core/services/sse.service';
import { ReviewSegment, ReviewSeverity } from '../../core/models/api.models';
import { FilterOption, FilterSelect } from '../../shared/filter-select/filter-select';
import { Icon } from '../../shared/icon/icon';
import { VideoLightbox } from '../../shared/video-lightbox/video-lightbox';

const SEVERITY_OPTIONS: FilterOption[] = [
  { value: '', label: 'All severities' },
  { value: 'rule', label: 'Rule' },
  { value: 'alert', label: 'Alert' },
  { value: 'detection', label: 'Detection' },
];

// SSE (below) delivers new/updated segments live -- this is now just a periodic
// safety net for anything missed during a dropped SSE connection.
const RECONCILE_POLL_MS = 30000;
// Each card loads a real thumbnail image on initial render -- keep this small; "Load
// more" fetches further pages on demand.
const PAGE_SIZE = 20;

@Component({
  selector: 'app-review-page',
  standalone: true,
  imports: [FilterSelect, Icon, VideoLightbox],
  templateUrl: './review-page.html',
  styleUrl: './review-page.scss',
})
export class ReviewPage implements OnInit {
  private readonly paged = new PagedList<ReviewSegment>(
    (limit, offset) => this.api.listReviewSegments({ limit, offset }),
    (s) => s.id,
    PAGE_SIZE,
  );
  protected readonly segments = this.paged.items;
  protected readonly hasMore = this.paged.hasMore;
  protected readonly loadingMore = this.paged.loadingMore;
  protected readonly loading = signal(true);
  protected readonly selected = signal<ReviewSegment | null>(null);

  protected readonly playingClip = signal<ReviewSegment | null>(null);
  protected readonly clipError = signal<string | null>(null);

  protected readonly cameraFilter = signal('');
  protected readonly severityFilter = signal<string>('');

  protected readonly cameraOptions = computed<FilterOption[]>(() => {
    const cameras = Array.from(new Set(this.segments().map((s) => s.camera))).sort();
    return [{ value: '', label: 'All cameras' }, ...cameras.map((c) => ({ value: c, label: c }))];
  });

  protected readonly severityOptions = SEVERITY_OPTIONS;

  protected readonly filteredSegments = computed(() => {
    const camera = this.cameraFilter();
    const severity = this.severityFilter();
    return this.segments().filter(
      (s) => (!camera || s.camera === camera) && (!severity || s.severity === severity),
    );
  });

  constructor(
    private readonly api: ApiService,
    private readonly destroyRef: DestroyRef,
    private readonly sse: SseService,
  ) {}

  ngOnInit(): void {
    this.api.listReviewSegments({ limit: PAGE_SIZE }).subscribe({
      next: (segments) => {
        this.paged.setFirstPage(segments);
        this.loading.set(false);
        this.syncOpenDetail();
      },
      error: () => this.loading.set(false),
    });

    this.sse
      .connect(this.api.eventsStreamUrl())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((msg) => {
        if (msg.type === 'review_segment') {
          this.paged.prependLive(msg.data as ReviewSegment);
          this.syncOpenDetail();
        }
      });

    visibleInterval(RECONCILE_POLL_MS)
      .pipe(
        switchMap(() => this.api.listReviewSegments({ limit: PAGE_SIZE })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (segments) => {
          this.paged.setFirstPage(segments);
          this.syncOpenDetail();
        },
      });
  }

  protected loadMore(): void {
    this.paged.loadMore();
  }

  // Keep an open detail panel in sync with the freshest data (e.g. its end_time/
  // duration updating while ongoing) rather than showing a stale snapshot from
  // whenever it was first opened.
  private syncOpenDetail(): void {
    const openId = this.selected()?.id;
    if (openId) {
      this.selected.set(this.segments().find((s) => s.id === openId) ?? null);
    }
  }

  protected select(segment: ReviewSegment): void {
    this.selected.set(segment);
  }

  protected closeDetail(): void {
    this.selected.set(null);
  }

  protected thumbnailUrl(segment: ReviewSegment): string {
    return this.api.reviewThumbnailUrl(segment.id);
  }

  protected clipUrl(segment: ReviewSegment): string {
    return this.api.reviewClipUrl(segment.id);
  }

  protected clipDownloadName(segment: ReviewSegment): string {
    const stamp = new Date(segment.start_time * 1000).toISOString().replace(/[:.]/g, '-');
    return `${segment.camera}_${stamp}.mp4`;
  }

  protected playClip(segment: ReviewSegment): void {
    this.clipError.set(null);
    this.playingClip.set(segment);
  }

  protected closeClip(): void {
    this.playingClip.set(null);
  }

  protected onClipError(): void {
    // The clip endpoint can genuinely fail (segment still open -> 409, no recordings
    // left to stitch -> 404, stitching itself failed -> 500) -- the native <video>
    // element's own error event is the real signal here (its src request got the
    // failing status), so surface a clear message instead of a silently broken player.
    this.clipError.set('Could not load video for this alert -- it may still be recording, or its recordings may have expired.');
    this.playingClip.set(null);
  }

  protected objectsFor(segment: ReviewSegment): string[] {
    const objects = segment.data?.['objects'];
    return Array.isArray(objects) ? (objects as string[]) : [];
  }

  protected formatTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
  }

  protected formatDuration(segment: ReviewSegment): string {
    if (segment.end_time === null) return 'ongoing';
    const seconds = Math.round(segment.end_time - segment.start_time);
    return `${seconds}s`;
  }
}
