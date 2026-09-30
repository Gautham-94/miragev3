/**
 * MSE (Media Source Extensions) player: speaks go2rtc's WebSocket signaling protocol
 * directly (GET /api/ws?src=<name>, proxied through mirage's own API so the browser
 * never talks to go2rtc's port directly -- see mirage/api/routers/live.py).
 *
 * Protocol (verified against go2rtc's own reference client, www/video-rtc.js, and
 * Frigate's MsePlayer.tsx -- not guessed):
 *   1. Client sends {"type":"mse","value":"<comma-joined supported codec mime strings>"}
 *      once the MediaSource fires "sourceopen".
 *   2. Server replies {"type":"mse","value":"<chosen codec mime string>"} -- client
 *      creates a SourceBuffer for exactly that mime string.
 *   3. All further messages are BINARY (raw fmp4 fragments) appended directly via
 *      sourceBuffer.appendBuffer(); queued locally if the buffer is still updating.
 */

// How much buffered video to keep behind the current playback position. Live MSE never
// needs more than a few seconds of history -- go2rtc is a real-time source, not
// something the user seeks backward into -- but nothing here was ever trimming the
// SourceBuffer, so a long-running tile (a Live page left open for a while, which is
// exactly the normal way to use it) accumulated buffered fmp4 data forever. The
// resulting ever-growing memory footprint is a well-documented cause of decode-pipeline
// hitches/stutter on long-lived MSE sessions -- not the only possible cause of the
// stutter reported, but a real, fixable one this player was doing nothing about.
const MAX_BUFFERED_SECONDS = 20;
// Only bother checking/trimming this often -- sourceBuffer.remove() is itself an async
// operation that briefly sets updating=true (delaying the next pending append), so
// running it on every single appended chunk would add more jitter than it prevents.
const TRIM_CHECK_INTERVAL_MS = 5000;

const CANDIDATE_CODECS = [
  'avc1.640029',
  'avc1.64002A',
  'avc1.640033',
  'hvc1.1.6.L153.B0',
  'mp4a.40.2',
  'mp4a.40.5',
  'flac',
  'opus',
];

function supportedCodecsValue(): string {
  return CANDIDATE_CODECS.filter((codec) =>
    MediaSource.isTypeSupported(`video/mp4; codecs="${codec}"`),
  ).join(',');
}

export class MsePlayer {
  private ws: WebSocket | null = null;
  private mediaSource: MediaSource | null = null;
  private sourceBuffer: SourceBuffer | null = null;
  private readonly pendingChunks: ArrayBuffer[] = [];
  private objectUrl: string | null = null;
  private destroyed = false;
  private trimTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly videoEl: HTMLVideoElement,
    private readonly wsUrl: string,
    private readonly onError: () => void,
    private readonly onPlaying: () => void,
  ) {}

  static isSupported(): boolean {
    return typeof MediaSource !== 'undefined' && supportedCodecsValue().length > 0;
  }

  start(): void {
    this.mediaSource = new MediaSource();
    this.objectUrl = URL.createObjectURL(this.mediaSource);
    this.videoEl.src = this.objectUrl;

    this.mediaSource.addEventListener('sourceopen', () => this.onSourceOpen(), { once: true });
    this.videoEl.addEventListener('playing', () => this.onPlaying(), { once: true });
  }

  private onSourceOpen(): void {
    if (this.destroyed) return;

    this.ws = new WebSocket(this.wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.addEventListener('open', () => {
      this.ws?.send(JSON.stringify({ type: 'mse', value: supportedCodecsValue() }));
    });

    this.ws.addEventListener('message', (event) => this.onMessage(event));
    this.ws.addEventListener('error', () => this.onError());
    this.ws.addEventListener('close', () => {
      if (!this.destroyed) this.onError();
    });
  }

  private onMessage(event: MessageEvent): void {
    if (typeof event.data === 'string') {
      let msg: { type?: string; value?: string };
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'mse' && msg.value && this.mediaSource) {
        try {
          this.sourceBuffer = this.mediaSource.addSourceBuffer(msg.value);
          this.sourceBuffer.mode = 'segments';
          this.sourceBuffer.addEventListener('updateend', () => this.flushPending());
          // A decode error here is NOT the transient "buffer full" case appendChunk's own
          // catch shrugs off -- it permanently kills this SourceBuffer (and usually demotes
          // the whole MediaSource to 'ended'), so every future appendBuffer() will keep
          // throwing forever with nothing to recover into. Without this listener that
          // showed up as a silently-stuck black tile: the WebSocket stayed open (so
          // onError() was never reached via the close/error path below), frames kept
          // arriving and kept getting dropped, and mode() never left 'mse' to let the
          // poster mask the dead picture.
          this.sourceBuffer.addEventListener('error', () => this.onError());
          this.trimTimer = setInterval(() => this.trimBuffer(), TRIM_CHECK_INTERVAL_MS);
        } catch {
          this.onError();
        }
      }
      return;
    }

    // Binary fmp4 fragment.
    const chunk = event.data as ArrayBuffer;
    if (!this.sourceBuffer || this.sourceBuffer.updating) {
      this.pendingChunks.push(chunk);
      return;
    }
    this.appendChunk(chunk);
  }

  private flushPending(): void {
    if (!this.sourceBuffer || this.sourceBuffer.updating) return;
    const next = this.pendingChunks.shift();
    if (next) this.appendChunk(next);
  }

  private appendChunk(chunk: ArrayBuffer): void {
    try {
      this.sourceBuffer?.appendBuffer(chunk);
    } catch {
      // A QuotaExceededError while merely updating is transient (buffer momentarily
      // full) and safe to just drop -- the next keyframe resyncs it, as the comment here
      // used to claim unconditionally. But if the MediaSource itself is no longer 'open'
      // (closed/ended, e.g. after the decode error above, or the source erroring out from
      // under us), this SourceBuffer is permanently dead and dropping fragments forever
      // would just strand the tile silently -- treat that case as a real failure so
      // CameraTile actually reconnects instead of leaving a black frame up indefinitely.
      if (this.mediaSource?.readyState !== 'open') {
        this.onError();
      }
    }
  }

  // Removes buffered data older than MAX_BUFFERED_SECONDS behind the current playback
  // position, so a tile left open for a long time doesn't accumulate an ever-growing
  // SourceBuffer (see MAX_BUFFERED_SECONDS's own docstring for why that matters).
  // Skips this pass entirely if the buffer is already mid-operation (updating=true) or
  // has nothing old enough to trim yet -- both are just "try again next interval," not
  // errors. remove() completing fires the SAME 'updateend' event as appendBuffer, which
  // is what lets flushPending() release any chunk that arrived while this was running.
  private trimBuffer(): void {
    const sb = this.sourceBuffer;
    if (!sb || sb.updating || sb.buffered.length === 0) return;

    const bufferedStart = sb.buffered.start(0);
    const currentTime = this.videoEl.currentTime;
    const trimEnd = currentTime - MAX_BUFFERED_SECONDS;
    if (trimEnd <= bufferedStart) return; // nothing old enough yet

    try {
      sb.remove(bufferedStart, trimEnd);
    } catch {
      // Invalid state (e.g. mediaSource closing concurrently) -- next interval will
      // just try again, or destroy() will have torn everything down by then.
    }
  }

  destroy(): void {
    this.destroyed = true;
    this.ws?.close();
    this.ws = null;
    if (this.trimTimer !== null) {
      clearInterval(this.trimTimer);
      this.trimTimer = null;
    }
    // Mirrors WebRtcPlayer.destroy()'s videoEl.srcObject = null, for the same reason
    // (see that method's own docstring): revoking the blob URL below does NOT clear
    // videoEl.src, so without this the element is left pointing at a blob that no longer
    // resolves. The moment anything makes the browser re-touch that resource (a stall, a
    // route reactivation, even just this tab regaining focus), it shows up as a failed
    // net::ERR_FILE_NOT_FOUND request on a `blob:` URL and the tile renders a black frame
    // instead of falling through to CameraTile's poster -- opacity-hiding the <video> via
    // mode() doesn't detach the broken source sitting underneath it. Only touch videoEl if
    // it's still ours to touch: a later player's start() may already have overwritten
    // .src with its own fresh blob by the time this runs.
    if (this.videoEl.src === this.objectUrl) {
      this.videoEl.removeAttribute('src');
      this.videoEl.load();
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.mediaSource = null;
    this.sourceBuffer = null;
    this.pendingChunks.length = 0;
  }
}
