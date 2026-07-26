import { signal } from '@angular/core';
import { Observable } from 'rxjs';

/**
 * Accumulates offset-paged API results behind a signal, for list pages that both (a)
 * periodically re-poll just the first page for freshness and (b) let the user load
 * older rows on demand via a "Load more" button. Backed by the same limit/offset
 * pagination every list endpoint already exposes (see mirage/api/routers/*.py) --
 * this is purely a frontend accumulator, it doesn't change what the API supports.
 *
 * `id` is used to de-duplicate when a poll's fresh first page overlaps with rows
 * already appended via "load more" (e.g. a new row pushed everything else down one
 * slot since the last poll).
 */
export class PagedList<T> {
  readonly items = signal<T[]>([]);
  readonly loadingMore = signal(false);
  readonly hasMore = signal(true);

  constructor(
    private readonly fetchPage: (limit: number, offset: number) => Observable<T[]>,
    private readonly getId: (item: T) => string,
    private readonly pageSize: number,
  ) {}

  /** Replaces the first page (from a poll tick) without disturbing older loaded-more rows. */
  setFirstPage(freshFirstPage: T[]): void {
    const seen = new Set(freshFirstPage.map((item) => this.getId(item)));
    const rest = this.items()
      .slice(freshFirstPage.length)
      .filter((item) => !seen.has(this.getId(item)));
    this.items.set([...freshFirstPage, ...rest]);
    if (freshFirstPage.length < this.pageSize) {
      this.hasMore.set(false);
    }
  }

  loadMore(): void {
    if (this.loadingMore() || !this.hasMore()) return;
    this.loadingMore.set(true);
    this.fetchPage(this.pageSize, this.items().length).subscribe({
      next: (page) => {
        this.items.update((current) => [...current, ...page]);
        if (page.length < this.pageSize) this.hasMore.set(false);
        this.loadingMore.set(false);
      },
      error: () => this.loadingMore.set(false),
    });
  }

  reset(): void {
    this.items.set([]);
    this.hasMore.set(true);
  }

  /**
   * Prepends a live-pushed row (SSE) to the front of the list. If this id already
   * exists anywhere in the list (e.g. a ReviewSegment "update" notification for a
   * segment already loaded), it's replaced in place instead of duplicated, and its
   * position is left unchanged -- an update to an older segment shouldn't visually
   * jump it to the top. A genuinely new id is inserted at index 0.
   */
  prependLive(item: T): void {
    const id = this.getId(item);
    const current = this.items();
    const idx = current.findIndex((existing) => this.getId(existing) === id);
    if (idx === -1) {
      this.items.set([item, ...current]);
    } else {
      this.items.set([...current.slice(0, idx), item, ...current.slice(idx + 1)]);
    }
  }
}
