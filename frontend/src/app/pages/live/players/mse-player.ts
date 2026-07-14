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
      // Buffer full or in an invalid state -- drop this fragment rather than crash the
      // player; the next keyframe will resync playback.
    }
  }

  destroy(): void {
    this.destroyed = true;
    this.ws?.close();
    this.ws = null;
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.mediaSource = null;
    this.sourceBuffer = null;
    this.pendingChunks.length = 0;
  }
}
