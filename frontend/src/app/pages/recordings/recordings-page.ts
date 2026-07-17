import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';

import { visibleInterval } from '../../core/rxjs/visible-interval';
import { ApiService } from '../../core/services/api.service';
import { Recording } from '../../core/models/api.models';
import { FilterOption, FilterSelect } from '../../shared/filter-select/filter-select';
import { Icon } from '../../shared/icon/icon';
import { VideoLightbox } from '../../shared/video-lightbox/video-lightbox';

interface RecordingGroup {
  dateLabel: string;
  recordings: Recording[];
}

const POLL_MS = 10000;

@Component({
  selector: 'app-recordings-page',
  standalone: true,
  imports: [FilterSelect, Icon, VideoLightbox],
  templateUrl: './recordings-page.html',
  styleUrl: './recordings-page.scss',
})
export class RecordingsPage implements OnInit {
  protected readonly recordings = signal<Recording[]>([]);
  protected readonly loading = signal(true);
  protected readonly hoveredId = signal<string | null>(null);
  protected readonly openRecording = signal<Recording | null>(null);

  protected readonly cameraFilter = signal('');

  protected readonly cameraOptions = computed<FilterOption[]>(() => {
    const cameras = Array.from(new Set(this.recordings().map((r) => r.camera))).sort();
    return [{ value: '', label: 'All cameras' }, ...cameras.map((c) => ({ value: c, label: c }))];
  });

  protected readonly groupedRecordings = computed<RecordingGroup[]>(() => {
    const camera = this.cameraFilter();
    const filtered = this.recordings().filter((r) => !camera || r.camera === camera);

    const groups = new Map<string, Recording[]>();
    for (const recording of filtered) {
      const dateLabel = new Date(recording.start_time * 1000).toLocaleDateString(undefined, {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
      const bucket = groups.get(dateLabel) ?? [];
      bucket.push(recording);
      groups.set(dateLabel, bucket);
    }
    return Array.from(groups.entries()).map(([dateLabel, recordings]) => ({ dateLabel, recordings }));
  });

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngOnInit(): void {
    visibleInterval(POLL_MS)
      .pipe(
        switchMap(() => this.api.listRecordings({ limit: 200 })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (recordings) => {
          this.recordings.set(recordings);
          this.loading.set(false);
        },
        error: () => this.loading.set(false),
      });
  }

  protected formatTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  }

  protected clipUrl(id: string): string {
    return this.api.recordingClipUrl(id);
  }

  protected onHover(id: string | null): void {
    this.hoveredId.set(id);
  }

  protected downloadName(recording: Recording): string {
    const stamp = new Date(recording.start_time * 1000).toISOString().replace(/[:.]/g, '-');
    return `${recording.camera}_${stamp}.mp4`;
  }

  // A <video> with just preload="metadata" and no autoplay doesn't reliably paint its
  // first frame in every browser (some WebKit builds show a blank black frame until you
  // actually seek) -- explicitly seeking to a tiny non-zero offset once metadata is
  // available forces the frame to decode and render as a real static poster image,
  // without ever starting playback.
  protected showPosterFrame(event: Event): void {
    const video = event.target as HTMLVideoElement;
    if (video.currentTime === 0) {
      video.currentTime = 0.1;
    }
  }

  protected openLightbox(recording: Recording): void {
    this.openRecording.set(recording);
  }

  protected closeLightbox(): void {
    this.openRecording.set(null);
  }
}
