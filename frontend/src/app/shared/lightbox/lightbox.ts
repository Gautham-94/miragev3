import { Component, HostListener, input, output } from '@angular/core';

import { Icon } from '../icon/icon';

/**
 * Generic full-screen image viewer overlay -- shows a single image at full size with an
 * optional caption, closable via the X button, clicking the backdrop, or Escape.
 * Deliberately content-agnostic (just an image URL + alt + optional caption) so it can
 * be reused anywhere a thumbnail needs a "click to view full size" affordance, not just
 * on the Events page.
 */
@Component({
  selector: 'app-lightbox',
  standalone: true,
  imports: [Icon],
  templateUrl: './lightbox.html',
  styleUrl: './lightbox.scss',
})
export class Lightbox {
  readonly imageUrl = input.required<string>();
  readonly alt = input<string>('');
  readonly closed = output<void>();

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    this.close();
  }

  protected close(): void {
    this.closed.emit();
  }

  protected stopPropagation(event: Event): void {
    event.stopPropagation();
  }
}
