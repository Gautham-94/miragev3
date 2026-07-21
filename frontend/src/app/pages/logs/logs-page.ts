import { Component, DestroyRef, OnInit, computed, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { interval, startWith, switchMap } from 'rxjs';

import { ApiService } from '../../core/services/api.service';
import { LogCategory, LogEntry } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';

const POLL_MS = 3000;
const MAX_ENTRIES = 500;

const CATEGORY_LABELS: Record<LogCategory, string> = {
  motion: 'Motion',
  detect: 'Detection',
  species: 'Species',
  system: 'System',
};

@Component({
  selector: 'app-logs-page',
  standalone: true,
  imports: [FormsModule, Icon],
  templateUrl: './logs-page.html',
  styleUrl: './logs-page.scss',
})
export class LogsPage implements OnInit {
  protected readonly entries = signal<LogEntry[]>([]);
  protected readonly loading = signal(true);
  protected readonly loadError = signal<string | null>(null);
  protected readonly categoryFilter = signal<LogCategory | 'all'>('all');
  protected readonly cameraFilter = signal<string | 'all'>('all');
  protected readonly paused = signal(false);

  protected readonly cameras = computed(() => {
    const set = new Set<string>();
    for (const e of this.entries()) {
      if (e.camera) set.add(e.camera);
    }
    return Array.from(set).sort();
  });

  protected readonly filteredEntries = computed(() => {
    const category = this.categoryFilter();
    const camera = this.cameraFilter();
    // Newest first for display -- entries() itself stays chronological (oldest-last-in
    // wins the poll-since dedup below), reversed only at render time.
    return this.entries()
      .filter((e) => (category === 'all' ? true : e.category === category))
      .filter((e) => (camera === 'all' ? true : e.camera === camera))
      .slice()
      .reverse();
  });

  private lastTimestamp = 0;

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngOnInit(): void {
    interval(POLL_MS)
      .pipe(
        startWith(0),
        switchMap(() => this.api.getLogs({ since: this.lastTimestamp || undefined })),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: (newEntries) => {
          this.loading.set(false);
          this.loadError.set(null);
          // Still fetched while paused (so `since` doesn't fall further behind and
          // cause a huge catch-up burst on resume) but not appended to the visible
          // feed until unpaused.
          if (newEntries.length === 0) return;
          this.lastTimestamp = newEntries[newEntries.length - 1].timestamp;
          if (this.paused()) return;
          this.entries.update((existing) => {
            const merged = [...existing, ...newEntries];
            return merged.length > MAX_ENTRIES ? merged.slice(merged.length - MAX_ENTRIES) : merged;
          });
        },
        error: (err) => {
          this.loading.set(false);
          this.loadError.set(err?.error?.detail ?? 'Could not load logs.');
        },
      });
  }

  protected categoryLabel(category: LogCategory): string {
    return CATEGORY_LABELS[category] ?? category;
  }

  protected togglePause(): void {
    this.paused.update((p) => !p);
  }

  protected formatTime(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString();
  }
}
