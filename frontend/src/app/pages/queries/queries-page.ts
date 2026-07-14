import { Component, OnInit, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../core/services/api.service';
import { CameraConfigDetail, Query, QueryMatch } from '../../core/models/api.models';
import { Icon } from '../../shared/icon/icon';
import { Lightbox } from '../../shared/lightbox/lightbox';

@Component({
  selector: 'app-queries-page',
  standalone: true,
  imports: [FormsModule, Icon, Lightbox],
  templateUrl: './queries-page.html',
  styleUrl: './queries-page.scss',
})
export class QueriesPage implements OnInit {
  protected readonly queries = signal<Query[]>([]);
  protected readonly cameras = signal<CameraConfigDetail[]>([]);
  protected readonly matches = signal<QueryMatch[]>([]);
  protected readonly loading = signal(true);
  protected readonly loadError = signal<string | null>(null);

  protected readonly showAddForm = signal(false);
  protected readonly editingQueryId = signal<string | null>(null);
  protected readonly text = signal('');
  protected readonly selectedCameras = signal<string[]>([]);
  protected readonly enabled = signal(true);

  protected readonly saving = signal(false);
  protected readonly saveError = signal<string | null>(null);
  protected readonly justAdded = signal(false);

  protected readonly pendingDelete = signal<string | null>(null);
  protected readonly deleting = signal(false);
  protected readonly deleteError = signal<string | null>(null);
  protected readonly justDeleted = signal(false);

  protected readonly openMatch = signal<QueryMatch | null>(null);

  protected readonly canSubmit = computed(() => this.text().trim().length > 0);

  constructor(private readonly api: ApiService) {}

  ngOnInit(): void {
    this.load();
    this.loadMatches();
  }

  private load(): void {
    this.loading.set(true);
    this.api.listQueries().subscribe({
      next: (queries) => {
        this.queries.set(queries);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(err?.error?.detail ?? 'Could not load queries.');
        this.loading.set(false);
      },
    });
    this.api.listCameraConfigs().subscribe({
      next: (cameras) => this.cameras.set(cameras),
      error: () => {},
    });
  }

  private loadMatches(): void {
    this.api.listQueryMatches({ limit: 50 }).subscribe({
      next: (matches) => this.matches.set(matches),
      error: () => {},
    });
  }

  protected queryTextFor(queryId: string): string {
    return this.queries().find((q) => q.id === queryId)?.text ?? '(deleted query)';
  }

  protected formatTime(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
  }

  protected thumbnailUrl(match: QueryMatch): string {
    return this.api.queryMatchThumbnailUrl(match.id);
  }

  protected openLightbox(match: QueryMatch): void {
    if (!match.has_thumb) return;
    this.openMatch.set(match);
  }

  protected closeLightbox(): void {
    this.openMatch.set(null);
  }

  protected openAddForm(): void {
    this.showAddForm.set(true);
    this.editingQueryId.set(null);
    this.justAdded.set(false);
    this.saveError.set(null);
  }

  protected editQuery(query: Query): void {
    this.editingQueryId.set(query.id);
    this.text.set(query.text);
    this.selectedCameras.set([...query.cameras]);
    this.enabled.set(query.enabled);
    this.showAddForm.set(true);
    this.justAdded.set(false);
    this.saveError.set(null);
  }

  protected cancelAddForm(): void {
    this.showAddForm.set(false);
    this.editingQueryId.set(null);
    this.text.set('');
    this.selectedCameras.set([]);
    this.enabled.set(true);
    this.saveError.set(null);
  }

  protected toggleCamera(name: string): void {
    const current = this.selectedCameras();
    this.selectedCameras.set(
      current.includes(name) ? current.filter((c) => c !== name) : [...current, name],
    );
  }

  protected save(): void {
    if (!this.canSubmit()) return;
    this.saving.set(true);
    this.saveError.set(null);

    const payload = { text: this.text().trim(), cameras: this.selectedCameras(), enabled: this.enabled() };
    const editingId = this.editingQueryId();
    const request = editingId ? this.api.updateQuery(editingId, payload) : this.api.createQuery(payload);

    request.subscribe({
      next: () => {
        this.saving.set(false);
        this.justAdded.set(true);
        this.cancelAddForm();
        this.showAddForm.set(false);
        this.load();
      },
      error: (err) => {
        this.saving.set(false);
        this.saveError.set(err?.error?.detail ?? `Could not ${editingId ? 'update' : 'add'} this query.`);
      },
    });
  }

  protected confirmDelete(id: string): void {
    this.pendingDelete.set(id);
    this.deleteError.set(null);
  }

  protected cancelDelete(): void {
    this.pendingDelete.set(null);
  }

  protected deleteQuery(id: string): void {
    this.deleting.set(true);
    this.deleteError.set(null);
    this.api.deleteQuery(id).subscribe({
      next: () => {
        this.deleting.set(false);
        this.pendingDelete.set(null);
        this.justDeleted.set(true);
        this.load();
      },
      error: (err) => {
        this.deleting.set(false);
        this.deleteError.set(err?.error?.detail ?? 'Could not delete this query.');
      },
    });
  }
}
