/**
 * WebRTC player: speaks go2rtc's WebSocket signaling protocol directly (GET
 * /api/ws?src=<name>, proxied through mirage's own API -- see
 * mirage/api/routers/live.py).
 *
 * Protocol (verified against go2rtc's own reference client, www/video-rtc.js, and
 * Frigate's WebRTCPlayer.tsx -- not guessed):
 *   1. Client builds an RTCPeerConnection with recvonly video/audio transceivers.
 *   2. Client creates an offer, sets it as the local description, and sends
 *      {"type":"webrtc/offer","value":"<sdp>"} (the SDP string only, not the whole
 *      RTCSessionDescription object).
 *   3. Client relays each local ICE candidate as
 *      {"type":"webrtc/candidate","value":"<candidate string>"}.
 *   4. Server replies with {"type":"webrtc/answer","value":"<sdp>"} (applied via
 *      setRemoteDescription) and zero or more
 *      {"type":"webrtc/candidate","value":"<candidate string>"} messages (applied via
 *      addIceCandidate).
 */

const ICE_SERVERS: RTCIceServer[] = [
  { urls: ['stun:stun.cloudflare.com:3478', 'stun:stun.l.google.com:19302'] },
];

export class WebRtcPlayer {
  private ws: WebSocket | null = null;
  private pc: RTCPeerConnection | null = null;
  private destroyed = false;

  constructor(
    private readonly videoEl: HTMLVideoElement,
    private readonly wsUrl: string,
    private readonly onError: () => void,
    private readonly onPlaying: () => void,
  ) {}

  static isSupported(): boolean {
    return typeof RTCPeerConnection !== 'undefined';
  }

  async start(): Promise<void> {
    this.pc = new RTCPeerConnection({ bundlePolicy: 'max-bundle', iceServers: ICE_SERVERS });

    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.pc.addTransceiver('audio', { direction: 'recvonly' });

    this.pc.addEventListener('track', (event) => {
      if (this.videoEl.srcObject !== event.streams[0]) {
        this.videoEl.srcObject = event.streams[0];
      }
    });
    this.videoEl.addEventListener('playing', () => this.onPlaying(), { once: true });

    this.pc.addEventListener('connectionstatechange', () => {
      if (this.pc?.connectionState === 'failed' || this.pc?.connectionState === 'closed') {
        this.onError();
      }
    });

    this.pc.addEventListener('icecandidate', (event) => {
      if (event.candidate) {
        this.send({ type: 'webrtc/candidate', value: event.candidate.candidate });
      }
    });

    this.ws = new WebSocket(this.wsUrl);
    this.ws.addEventListener('open', async () => {
      if (!this.pc) return;
      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);
      this.send({ type: 'webrtc/offer', value: offer.sdp ?? '' });
    });
    this.ws.addEventListener('message', (event) => this.onMessage(event));
    this.ws.addEventListener('error', () => this.onError());
    this.ws.addEventListener('close', () => {
      if (!this.destroyed) this.onError();
    });
  }

  private async onMessage(event: MessageEvent): Promise<void> {
    if (typeof event.data !== 'string') return;
    let msg: { type?: string; value?: string };
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!this.pc) return;

    if (msg.type === 'webrtc/answer' && msg.value) {
      await this.pc.setRemoteDescription({ type: 'answer', sdp: msg.value });
    } else if (msg.type === 'webrtc/candidate' && msg.value) {
      try {
        await this.pc.addIceCandidate({ candidate: msg.value, sdpMid: '0' });
      } catch {
        // A stray/late candidate arriving after the connection settled is harmless.
      }
    }
  }

  private send(msg: { type: string; value: string }): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  destroy(): void {
    this.destroyed = true;
    this.ws?.close();
    this.ws = null;
    this.pc?.close();
    this.pc = null;
  }
}
