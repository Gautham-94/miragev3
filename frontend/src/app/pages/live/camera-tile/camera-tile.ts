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

// Reconnect backoff after BOTH video tiers have failed. A live stream drops for plenty
// of ordinary reasons -- go2rtc restarting (every pipeline restart does this), a camera
// blip, a laptop sleeping, or the browser tearing down a backgrounded tab's socket --
// and without a retry the tile stays stranded on the 1fps snapshot poller ("STILL")
// until the user happens to navigate away and back, which is what makes it look
// permanently stuck. Backoff so a genuinely-down backend isn't hammered once a second.
const RECONNECT_MIN_MS = 2000;
const RECONNECT_MAX_MS = 15000;

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

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelayMs = RECONNECT_MIN_MS;
  // Set once WebRTC has failed on this tile, so reconnect attempts go straight to MSE
  // instead of re-paying the (often multi-second) ICE timeout on every single retry.
  // WebRTC failing is a property of the deployment (no reachable ICE candidate), not a
  // transient, so re-probing it each time only delays recovery.
  private webrtcUnavailable = false;
  // Distinguishes "WebRTC can't work here at all" from "WebRTC was working and the
  // stream blipped" -- only the former should demote this tile to MSE for good.
  private webrtcEverPlayed = false;
  private destroyed = false;

  constructor(private readonly api: ApiService, private readonly destroyRef: DestroyRef) {}

  ngAfterViewInit(): void {
    this.startSnapshotPolling();
    this.startVideoTiers();
    // Coming back to a backgrounded tab is the single most common moment to discover the
    // socket died while it was hidden -- retry immediately rather than waiting out the
    // remaining backoff, so the tile recovers as soon as it's actually being looked at.
    document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  private readonly onVisibilityChange = (): void => {
    if (document.visibilityState !== 'visible' || this.destroyed) return;
    const mode = this.mode();
    if (mode === 'mse' || mode === 'webrtc') return; // already playing, nothing to do
    this.reconnectDelayMs = RECONNECT_MIN_MS;
    this.scheduleReconnect(0);
  };

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
      // A tier actually reached "playing" -- the next unrelated drop should retry
      // promptly rather than inheriting the backoff this recovery just climbed.
      this.reconnectDelayMs = RECONNECT_MIN_MS;
      this.stopSnapshotPolling();
    } else {
      // 'connecting' / 'poster' / 'error' -- the poster is visible again (e.g. a
      // live stream just failed back to an error state), so polling must resume.
      this.startSnapshotPolling();
    }
  }

  private startVideoTiers(): void {
    if (this.destroyed) return;
    const videoEl = this.videoElRef?.nativeElement;
    if (!videoEl) return;

    // Tear down anything left over from a previous attempt before starting a new one,
    // so a reconnect can't leave two players racing for the same <video> element.
    this.webrtc?.destroy();
    this.webrtc = null;
    this.mse?.destroy();
    this.mse = null;

    const wsUrl = this.api.liveWebSocketUrl(this.camera().name);

    if (WebRtcPlayer.isSupported() && !this.webrtcUnavailable) {
      this.webrtc = new WebRtcPlayer(
        videoEl,
        wsUrl,
        () => this.onTierFailed('webrtc'),
        () => {
          this.webrtcEverPlayed = true;
          this.setMode('webrtc');
        },
      );
      this.webrtc.start();
    } else {
      this.tryMse(videoEl, wsUrl);
    }
  }

  private onTierFailed(failedTier: 'mse' | 'webrtc'): void {
    if (this.destroyed) return;

    if (failedTier === 'webrtc') {
      this.webrtc?.destroy();
      this.webrtc = null;
      // Only give up on WebRTC permanently if it never once reached playback -- that
      // means the deployment has no reachable ICE candidate, and re-probing it would add
      // the full ICE timeout to every future reconnect. If it HAD been playing, this is
      // just a dropped stream, so keep it in the rotation for the next attempt.
      this.webrtcUnavailable = !this.webrtcEverPlayed;
      const videoEl = this.videoElRef?.nativeElement;
      if (videoEl) this.tryMse(videoEl, this.api.liveWebSocketUrl(this.camera().name));
    } else {
      this.mse?.destroy();
      this.mse = null;
      // Show the snapshot poster while we're down, then retry -- this used to stop here,
      // which stranded the tile on the poster permanently (see RECONNECT_MIN_MS above).
      this.setMode('error');
      this.scheduleReconnect(this.reconnectDelayMs);
      this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, RECONNECT_MAX_MS);
    }
  }

  private scheduleReconnect(delayMs: number): void {
    if (this.destroyed) return;
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.startVideoTiers();
    }, delayMs);
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
    this.destroyed = true;
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.mse?.destroy();
    this.webrtc?.destroy();
  }
}
