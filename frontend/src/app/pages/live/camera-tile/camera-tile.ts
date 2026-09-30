import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  OnDestroy,
  ViewChild,
  effect,
  input,
  output,
  signal,
  untracked,
} from '@angular/core';
import { Camera } from '../../../core/models/api.models';
import { ApiService } from '../../../core/services/api.service';
import { Icon } from '../../../shared/icon/icon';
import { LiveActivation } from '../live-activation';
import { LiveTierMemory } from '../live-tier-memory';
import { MsePlayer } from '../players/mse-player';
import { WebRtcPlayer } from '../players/webrtc-player';

// Snapshot poster backoff, same shape as RECONNECT_MIN/MAX_MS below: a camera that's
// genuinely down (no mock source, camera unplugged, etc.) doesn't need hammering once a
// second forever -- that's needless load on both the backend and the browser's shared
// per-origin connection pool (shared with everything else on the page, e.g. the Shell's
// own 1.5s system-status poll -- see shell.ts, and OPTIMIZATION_OPPORTUNITIES.md-style
// reasoning: several disconnected tiles at once WAS observed starving that poll of a
// connection slot long enough for its own switchMap to cancel it before it ever ran).
// Starts fast (a camera that just blipped should recover its poster promptly) and backs
// off the longer it stays down.
const SNAPSHOT_POLL_MIN_MS = 1000;
const SNAPSHOT_POLL_MAX_MS = 5000;
const SNAPSHOT_POLL_BACKOFF_FACTOR = 1.5;

// Reconnect backoff after BOTH video tiers have failed. A live stream drops for plenty
// of ordinary reasons -- go2rtc restarting (every pipeline restart does this), a camera
// blip, a laptop sleeping, or the browser tearing down a backgrounded tab's socket --
// and without a retry the tile stays stranded on the 1fps snapshot poller ("STILL")
// until the user happens to navigate away and back, which is what makes it look
// permanently stuck. Backoff so a genuinely-down backend isn't hammered once a second.
const RECONNECT_MIN_MS = 2000;
const RECONNECT_MAX_MS = 15000;

// Mirrors mirage.go2rtc.config.SUB_STREAM_SUFFIX (Python side) exactly -- keep both in
// sync if this ever changes. See streamKey()'s own docstring for what this selects.
const SUB_STREAM_SUFFIX = '_sub';

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
  // Whether THIS tile is the one currently shown maximized -- purely a styling/layout
  // concern (see camera-tile.scss's `.maximized` handling); the player tiers above are
  // completely orientation-agnostic, so maximizing never tears down or reconnects a
  // stream that's already playing.
  readonly maximized = input<boolean>(false);
  // Fired on every click of the maximize/close corner button, whether entering or
  // leaving the maximized state -- LivePage owns the single "which camera, if any, is
  // maximized" decision (see its own maximizedCamera signal), this tile just reports
  // "the user clicked my corner button."
  readonly toggleMaximize = output<void>();

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
  private snapshotPollTimer: ReturnType<typeof setTimeout> | null = null;
  private snapshotPollDelayMs = SNAPSHOT_POLL_MIN_MS;

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelayMs = RECONNECT_MIN_MS;
  // Distinguishes "WebRTC can't work here at all" from "WebRTC was working and the
  // stream blipped" -- only the former should demote this tile to MSE for good (see
  // LiveTierMemory, which is what actually remembers that verdict across this tile
  // being destroyed and recreated by a route navigation).
  private webrtcEverPlayed = false;
  // true only during the brief window where BOTH tiers are simultaneously attempting
  // to connect (LiveTierMemory has no verdict yet for this camera) -- see
  // startVideoTiers()/onTierWon()/onTierFailed() for how this changes their behavior.
  // Always false once a verdict exists, since then only one tier is ever started.
  private racing = false;
  private raceWebrtcFailed = false;
  private raceMseFailed = false;
  private destroyed = false;
  // Set by startVideoTiers() every time it (re)connects; compared against streamKey()
  // by the effect() below to tell "maximized() just changed, we're on the wrong go2rtc
  // stream now" apart from every OTHER reason startVideoTiers() might run (initial
  // connect, a dropped-connection retry) -- see streamKey()'s own docstring for why
  // maximizing needs a reconnect at all. Starts null specifically so effect()'s
  // mandatory first synchronous run (Angular runs a newly-created effect once
  // immediately) is a no-op instead of firing a redundant reconnect before
  // ngAfterViewInit's own initial startVideoTiers() call has even happened.
  private previousStreamKey: string | null = null;
  // Guards the SAME way previousStreamKey guards its own effect() below -- skip the
  // mandatory first synchronous run (Angular runs a newly-created effect once
  // immediately, which for this one could fire before ngAfterViewInit's own initial
  // startVideoTiers() call). See the activation effect() and refreshOnReactivation()'s
  // own docstrings for what this is actually for.
  private hasSeenActivation = false;

  constructor(
    private readonly api: ApiService,
    private readonly destroyRef: DestroyRef,
    private readonly tierMemory: LiveTierMemory,
    private readonly liveActivation: LiveActivation,
  ) {
    effect(() => {
      const key = this.streamKey();
      if (this.previousStreamKey !== null && this.previousStreamKey !== key) {
        this.startVideoTiers();
      }
    });

    effect(() => {
      // Must read ONLY liveActivation.activatedAt() here -- refreshOnReactivation() (and
      // everything it calls, e.g. startVideoTiers() -> setMode() -> this.mode.set(...))
      // has to run via untracked(). Otherwise this effect ALSO implicitly subscribes to
      // this.mode() through refreshOnReactivation() -- and since that chain eventually
      // changes mode() itself once the new connection lands, that change re-triggers THIS
      // SAME effect, which calls refreshOnReactivation() again, forever. Confirmed live
      // via temporary logging: this was the actual cause of a reported bug where every
      // tile went black and reconnected in lockstep every ~10s -- the effect was firing
      // repeatedly with an IDENTICAL activatedAt value (proving it was mode(), not a real
      // reactivation, driving each re-run), forcing a fresh WebRTC/MSE reconnect on every
      // tile once per reconnect's own completion time.
      this.liveActivation.activatedAt();
      if (this.hasSeenActivation) {
        untracked(() => this.refreshOnReactivation());
      } else {
        this.hasSeenActivation = true;
      }
    });
  }

  // The go2rtc stream to actually request: the low-res "<camera>_sub" tier normally
  // (see SUB_STREAM_SUFFIX and mirage.go2rtc.config.build_go2rtc_config's matching
  // backend-side registration), or the full-res main stream while THIS tile is the
  // maximized one. A live-view grid showing many cameras at once is bottlenecked by the
  // browser's decode capacity, not by anything this player code can optimize away --
  // Frigate's own docs land on the same fix for the same reason: don't decode full-res
  // video for a tile nobody's looking closely at.
  private streamKey(): string {
    return this.maximized() ? this.camera().name : `${this.camera().name}${SUB_STREAM_SUFFIX}`;
  }

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
    this.refreshOnReactivation();
  };

  // Shared by two triggers that both mean "this tile might be showing a stale picture
  // even though mode() still reads 'playing'": the browser tab regaining visibility
  // (onVisibilityChange above), and the Live route becoming active again after
  // LivePersistingRouteReuseStrategy detached it (the activation effect() in the
  // constructor -- see LiveActivation's own docstring for the full mechanism). Both
  // cases share the same root cause: browsers throttle/suspend decoding new video
  // frames for a <video> element that isn't actually being rendered (backgrounded tab,
  // or -- confirmed live, a real reported bug -- removed from the document entirely by
  // Angular's detach), so the WebRTC/MSE connection itself can stay genuinely healthy
  // (mode() never changes) while the visible picture is frozen on whatever frame was
  // decoded last. This used to just no-op whenever mode() already read 'mse'/'webrtc',
  // trusting that state -- which is exactly the bug: a still-'playing' tile is the ONE
  // case a stale frame can hide in.
  private refreshOnReactivation(): void {
    if (this.destroyed) return;
    const mode = this.mode();
    if (mode === 'mse' || mode === 'webrtc') {
      this.startVideoTiers();
      return;
    }
    this.reconnectDelayMs = RECONNECT_MIN_MS;
    this.scheduleReconnect(0);
  }

  private startSnapshotPolling(): void {
    if (this.snapshotPollTimer !== null) return; // already running
    this.snapshotPollDelayMs = SNAPSHOT_POLL_MIN_MS;
    this.refreshSnapshot(); // fire immediately, same as the old interval's startWith(0)
    this.scheduleNextSnapshotPoll();
  }

  private stopSnapshotPolling(): void {
    if (this.snapshotPollTimer !== null) {
      clearTimeout(this.snapshotPollTimer);
      this.snapshotPollTimer = null;
    }
  }

  private scheduleNextSnapshotPoll(): void {
    this.snapshotPollTimer = setTimeout(() => {
      this.refreshSnapshot();
      this.snapshotPollDelayMs = Math.min(
        this.snapshotPollDelayMs * SNAPSHOT_POLL_BACKOFF_FACTOR,
        SNAPSHOT_POLL_MAX_MS,
      );
      this.scheduleNextSnapshotPoll();
    }, this.snapshotPollDelayMs);
  }

  // Every mode() transition goes through here so polling always stays in sync with
  // whether the poster is actually visible -- see snapshotPollTimer's own docstring.
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

    // Tear down anything left over from a previous attempt before starting a new one.
    this.webrtc?.destroy();
    this.webrtc = null;
    this.mse?.destroy();
    this.mse = null;
    this.racing = false;

    const streamKey = this.streamKey();
    this.previousStreamKey = streamKey;
    const wsUrl = this.api.liveWebSocketUrl(streamKey);
    // Keyed by the base camera name, not streamKey() -- WebRTC's own reachability is a
    // property of the camera/network path, unaffected by which go2rtc stream variant
    // (main vs _sub) is being requested, so the verdict must be shared across both
    // rather than re-raced every time maximizing swaps the stream.
    const verdict = this.tierMemory.getVerdict(this.camera().name);
    const webrtcSupported = WebRtcPlayer.isSupported();

    if (!webrtcSupported || verdict === 'unavailable') {
      this.tryMse(videoEl, wsUrl);
      return;
    }
    if (verdict === 'available') {
      this.startWebRtc(videoEl, wsUrl);
      return;
    }

    // No verdict yet for this camera -- race both tiers instead of trying WebRTC
    // alone and only starting MSE after it fully fails. On a network where WebRTC's
    // ICE gathering is slow (or can't connect at all) but MSE would connect in a few
    // hundred ms, the old sequential fallback meant paying WebRTC's ENTIRE failure
    // timeline before MSE ever got a chance -- pure avoidable "connecting..." latency,
    // and it happened on every single first-time connection. Whichever tier's
    // onPlaying fires first wins (see onTierWon): the other is torn down immediately,
    // in that same synchronous callback, before it can deliver a late track/chunk that
    // would otherwise hijack an already-playing stream (srcObject always takes
    // precedence over src regardless of which was set first -- see
    // WebRtcPlayer.destroy()'s own docstring on this exact hazard). This only ever
    // runs once per camera per app lifetime: the moment WebRTC's own outcome is known
    // (see LiveTierMemory's own docstring for precisely what counts), every later
    // connect/reconnect skips straight to the winning tier alone, no more racing.
    this.racing = true;
    this.raceWebrtcFailed = false;
    this.raceMseFailed = false;
    this.startWebRtc(videoEl, wsUrl);
    this.tryMse(videoEl, wsUrl);
  }

  private startWebRtc(videoEl: HTMLVideoElement, wsUrl: string): void {
    // Captured by identity (not just checking `this.webrtc !== null`) so a callback
    // from an instance we've since destroyed and replaced -- deliberately, as a race
    // loser, or via a fresh reconnect -- is recognized as stale and ignored, even if
    // it fires late (e.g. destroy()'s own pc.close() can itself trigger a spurious
    // 'connectionstatechange' -> 'closed' -> onError on the instance being torn down).
    // Without this, tearing down the race LOSER could fire onTierFailed('webrtc') for
    // an instance that never genuinely failed, wrongly recording it as unavailable in
    // LiveTierMemory forever after.
    const player: WebRtcPlayer = new WebRtcPlayer(
      videoEl,
      wsUrl,
      () => {
        if (this.webrtc !== player) return;
        this.onTierFailed('webrtc');
      },
      () => {
        if (this.webrtc !== player) return;
        this.onTierWon('webrtc');
      },
    );
    this.webrtc = player;
    player.start();
  }

  // Shared by both the "one tier alone" and "racing" paths -- see the two callers.
  private onTierWon(tier: 'mse' | 'webrtc'): void {
    if (this.destroyed) return;
    if (this.racing) {
      this.racing = false;
      // Tear down the loser immediately, in this same synchronous tick -- see
      // startVideoTiers()'s own docstring on why that timing matters.
      if (tier === 'webrtc') {
        this.mse?.destroy();
        this.mse = null;
      } else {
        this.webrtc?.destroy();
        this.webrtc = null;
      }
    }
    if (tier === 'webrtc') {
      this.webrtcEverPlayed = true;
      this.tierMemory.markAvailable(this.camera().name);
    }
    // Deliberately NOT recording anything in LiveTierMemory when MSE wins -- MSE is
    // always the fallback/guaranteed tier, never something to "prefer" going forward;
    // WebRTC only ever gets skipped on a future connect once IT has its own proven
    // outcome (see markAvailable/markUnavailable's docstrings), not merely because it
    // happened to lose a timing race while still healthy.
    this.setMode(tier);
  }

  private onTierFailed(failedTier: 'mse' | 'webrtc'): void {
    if (this.destroyed) return;

    if (failedTier === 'webrtc') {
      this.webrtc?.destroy();
      this.webrtc = null;
      // Only give up on WebRTC permanently if it never once reached playback -- that
      // means the deployment has no reachable ICE candidate, and re-probing it would add
      // the full ICE timeout to every future reconnect. If it HAD been playing, this is
      // just a dropped stream, so keep it in the rotation for the next attempt. Recorded
      // in LiveTierMemory (not a local flag) so this verdict survives this tile being
      // torn down and recreated by navigating away from and back to Live.
      if (!this.webrtcEverPlayed) this.tierMemory.markUnavailable(this.camera().name);

      if (this.racing) {
        this.raceWebrtcFailed = true;
        if (!this.raceMseFailed) return; // MSE is still in the running -- give it its chance
        // both sides of the race have now failed -- fall through to the shared
        // give-up-and-retry path below, same as the non-racing MSE-failed case.
      } else {
        const videoEl = this.videoElRef?.nativeElement;
        if (videoEl) this.tryMse(videoEl, this.api.liveWebSocketUrl(this.streamKey()));
        return;
      }
    } else {
      this.mse?.destroy();
      this.mse = null;

      if (this.racing) {
        this.raceMseFailed = true;
        if (!this.raceWebrtcFailed) return; // WebRTC is still in the running -- give it its chance
        // both sides of the race have now failed -- fall through below.
      }
    }

    // Reached when: (a) not racing and MSE alone just failed (the ordinary
    // single-tier case), or (b) racing and BOTH tiers have now failed.
    this.racing = false;
    // Show the snapshot poster while we're down, then retry -- this used to stop here,
    // which stranded the tile on the poster permanently (see RECONNECT_MIN_MS above).
    this.setMode('error');
    this.scheduleReconnect(this.reconnectDelayMs);
    this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, RECONNECT_MAX_MS);
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
      if (this.racing) {
        // WebRTC alone is still in the running -- MSE simply isn't a candidate on
        // this browser, not a failure worth recording anywhere; let WebRTC decide
        // the outcome on its own, same as the "both failed" bookkeeping elsewhere.
        this.raceMseFailed = true;
        if (!this.raceWebrtcFailed) return;
        this.racing = false;
      }
      this.setMode('error');
      return;
    }
    // Identity-captured for the same staleness reason as startWebRtc() -- see its
    // own docstring.
    const player: MsePlayer = new MsePlayer(
      videoEl,
      wsUrl,
      () => {
        if (this.mse !== player) return;
        this.onTierFailed('mse');
      },
      () => {
        if (this.mse !== player) return;
        this.onTierWon('mse');
      },
    );
    this.mse = player;
    player.start();
  }

  private refreshSnapshot(): void {
    // Cache-bust with a timestamp query param so the browser always fetches a fresh
    // frame rather than serving a stale cached response for the same URL. Only runs
    // while snapshotPollTimer is active -- see its own docstring for when that is.
    const url = `${this.api.liveSnapshotUrl(this.streamKey())}?t=${Date.now()}`;
    const img = new Image();
    // Explicitly deprioritized (below the browser's default) so this poster poll -- up
    // to several tiles' worth, each polling independently -- yields its connection slot
    // to anything else competing for the same origin (e.g. the Shell's system-status
    // poll, see SNAPSHOT_POLL_MIN_MS's own docstring above) rather than starving it out
    // under Chrome's 6-connections-per-origin HTTP/1.1 cap. Set before `src` so it's in
    // effect from the moment the request is actually issued.
    img.fetchPriority = 'low';
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
    this.stopSnapshotPolling();
    this.mse?.destroy();
    this.webrtc?.destroy();
  }
}
