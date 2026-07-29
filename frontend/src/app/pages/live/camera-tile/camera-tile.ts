import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  OnDestroy,
  ViewChild,
  input,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subscription, interval, startWith } from 'rxjs';

import { Camera } from '../../../core/models/api.models';
import { ApiService } from '../../../core/services/api.service';
import { Icon } from '../../../shared/icon/icon';
import { MsePlayer } from '../players/mse-player';
import { WebRtcPlayer } from '../players/webrtc-player';

const SNAPSHOT_POLL_MS = 1000;

type PlayerMode = 'connecting' | 'mse' | 'webrtc' | 'poster' | 'error';

/**
 * Tiered live player: prefer WebRTC (sub-second latency -- it's the real-time transport;
 * MSE buffers several frames of fmp4 before playback starts, typically adding 1-3s), fall
 * back to MSE if WebRTC isn't supported or fails to connect (e.g. UDP blocked by a
 * restrictive network), and fall back to a polled JPEG snapshot poster if neither
 * real-video tier connects (e.g. go2rtc unreachable). The poster is also what renders
 * during the "connecting" window before either player has started playing.
 */
@Component({
  selector: 'app-camera-tile',
  standalone: true,
  imports: [Icon],
  templateUrl: './camera-tile.html',
  styleUrl: './camera-tile.scss',
})
export class CameraTile implements AfterViewInit, OnDestroy {
  readonly camera = input.required<Camera>();

  @ViewChild('videoEl') private readonly videoElRef?: ElementRef<HTMLVideoElement>;

  protected readonly mode = signal<PlayerMode>('connecting');
  protected readonly snapshotUrl = signal<string | null>(null);

  private webrtc: WebRtcPlayer | null = null;
  private mse: MsePlayer | null = null;
  // Started/stopped as mode() transitions in/out of 'mse'/'webrtc' -- see
  // startSnapshotPolling/stopSnapshotPolling. The snapshot poster is only ever
  // rendered while mode() is 'connecting'/'poster'/'error' (see camera-tile.html's
  // own [class.visible] / @if gating: the <img class="poster"> is inside
  // `@if (mode() !== 'mse' && mode() !== 'webrtc')`) -- once a real video tier is
  // playing, polling a snapshot nobody sees was pure wasted HTTP+JPEG-decode work,
  // once per second, per tile, forever (OPTIMIZATION_OPPORTUNITIES.md item 5).
  private snapshotPollSub: Subscription | null = null;

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngAfterViewInit(): void {
    this.startSnapshotPolling();
    this.startVideoTiers();
  }

  private startSnapshotPolling(): void {
    if (this.snapshotPollSub) return; // already running
    this.snapshotPollSub = interval(SNAPSHOT_POLL_MS)
      .pipe(startWith(0), takeUntilDestroyed(this.destroyRef))
      .subscribe(() => this.refreshSnapshot());
  }

  private stopSnapshotPolling(): void {
    this.snapshotPollSub?.unsubscribe();
    this.snapshotPollSub = null;
  }

  // Every mode() transition goes through here so polling always stays in sync with
  // whether the poster is actually visible -- see snapshotPollSub's own docstring.
  private setMode(next: PlayerMode): void {
    this.mode.set(next);
    if (next === 'mse' || next === 'webrtc') {
      this.stopSnapshotPolling();
    } else {
      // 'connecting' / 'poster' / 'error' -- the poster is visible again (e.g. a
      // live stream just failed back to an error state), so polling must resume.
      this.startSnapshotPolling();
    }
  }

  private startVideoTiers(): void {
    const videoEl = this.videoElRef?.nativeElement;
    if (!videoEl) return;

    const wsUrl = this.api.liveWebSocketUrl(this.camera().name);

    if (WebRtcPlayer.isSupported()) {
      this.webrtc = new WebRtcPlayer(
        videoEl,
        wsUrl,
        () => this.onTierFailed('webrtc'),
        () => this.setMode('webrtc'),
      );
      this.webrtc.start();
    } else {
      this.tryMse(videoEl, wsUrl);
    }
  }

  private onTierFailed(failedTier: 'mse' | 'webrtc'): void {
    if (failedTier === 'webrtc') {
      this.webrtc?.destroy();
      this.webrtc = null;
      const videoEl = this.videoElRef?.nativeElement;
      if (videoEl) this.tryMse(videoEl, this.api.liveWebSocketUrl(this.camera().name));
    } else {
      this.mse?.destroy();
      this.mse = null;
      this.setMode('error');
    }
  }

  private tryMse(videoEl: HTMLVideoElement, wsUrl: string): void {
    if (!MsePlayer.isSupported()) {
      this.setMode('error');
      return;
    }
    this.mse = new MsePlayer(
      videoEl,
      wsUrl,
      () => this.onTierFailed('mse'),
      () => this.setMode('mse'),
    );
    this.mse.start();
  }

  private refreshSnapshot(): void {
    // Cache-bust with a timestamp query param so the browser always fetches a fresh
    // frame rather than serving a stale cached response for the same URL. Only runs
    // while snapshotPollSub is active -- see its own docstring for when that is.
    const url = `${this.api.liveSnapshotUrl(this.camera().name)}?t=${Date.now()}`;
    const img = new Image();
    img.onload = () => this.snapshotUrl.set(url);
    img.onerror = () => {
      if (this.mode() === 'connecting') this.setMode('error');
    };
    img.src = url;
  }

  ngOnDestroy(): void {
    this.mse?.destroy();
    this.webrtc?.destroy();
  }
}
