import { AfterViewInit, Component, ElementRef, HostListener, ViewChild, input, output } from '@angular/core';

import { Icon } from '../icon/icon';

/**
 * Full-screen in-page video player overlay -- plays a clip directly (via a real <video
 * controls> element) instead of forcing a new-tab open or a forced download. Closable
 * via the X button, clicking the backdrop, or Escape (all of which also pause playback,
 * so audio/video doesn't keep running behind a closed overlay). A separate download
 * affordance is passed in via downloadUrl/downloadName rather than assumed from
 * videoUrl, since the two can legitimately differ (e.g. a streamed/transcoded preview
 * URL vs. the original file for download).
 */
@Component({
  selector: 'app-video-lightbox',
  standalone: true,
  imports: [Icon],
  templateUrl: './video-lightbox.html',
  styleUrl: './video-lightbox.scss',
})
export class VideoLightbox implements AfterViewInit {
  readonly videoUrl = input.required<string>();
  readonly downloadUrl = input.required<string>();
  readonly downloadName = input<string>('recording.mp4');
  readonly closed = output<void>();
  readonly videoError = output<void>();

  @ViewChild('videoEl') private readonly videoElRef?: ElementRef<HTMLVideoElement>;

  ngAfterViewInit(): void {
    this.videoElRef?.nativeElement.play().catch(() => {
      // Autoplay can be blocked by the browser -- the visible <video controls> element
      // still lets the user press play manually, so this is not an error state.
    });
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    this.close();
  }

  protected close(): void {
    this.videoElRef?.nativeElement.pause();
    this.closed.emit();
  }

  protected stopPropagation(event: Event): void {
    event.stopPropagation();
  }

  protected onError(): void {
    this.videoError.emit();
  }
}
