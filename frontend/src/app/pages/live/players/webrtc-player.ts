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

// go2rtc and the browser are both on the LAN (or the same machine) in this deployment --
// a public STUN server only matters for NAT traversal across the open internet, and
// including one here has been observed to make Chrome prioritize a srflx candidate pair
// through the public-facing IP over the perfectly good host candidate pair on the LAN,
// stalling ICE instead of just using the direct route. No STUN server needed for a
// same-network peer.
const ICE_SERVERS: RTCIceServer[] = [];

export class WebRtcPlayer {
  private ws: WebSocket | null = null;
  private pc: RTCPeerConnection | null = null;
  private destroyed = false;
  private readonly camDebugLabel: string;

  constructor(
    private readonly videoEl: HTMLVideoElement,
    private readonly wsUrl: string,
    private readonly onError: () => void,
    private readonly onPlaying: () => void,
  ) {
    this.camDebugLabel = wsUrl;
  }

  static isSupported(): boolean {
    return typeof RTCPeerConnection !== 'undefined';
  }

  async start(): Promise<void> {
    this.pc = new RTCPeerConnection({ bundlePolicy: 'max-bundle', iceServers: ICE_SERVERS });

    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.pc.addTransceiver('audio', { direction: 'recvonly' });

    this.pc.addEventListener('track', (event) => {
      console.log('[webrtc]', this.camDebugLabel, 'track event', event.track.kind);
      if (this.videoEl.srcObject !== event.streams[0]) {
        this.videoEl.srcObject = event.streams[0];
        // Some browsers don't reliably auto-start playback on a srcObject assigned from
        // a WebRTC track even with the autoplay attribute set -- kick it explicitly and
        // surface anything blocking it (e.g. an autoplay policy rejection).
        this.videoEl.play().catch((err) => {
          console.log('[webrtc]', this.camDebugLabel, 'video.play() rejected', err);
        });
      }
    });
    this.videoEl.addEventListener('playing', () => {
      console.log('[webrtc]', this.camDebugLabel, 'video element playing');
      this.onPlaying();
    }, { once: true });
    this.videoEl.addEventListener('loadedmetadata', () => {
      console.log('[webrtc]', this.camDebugLabel, 'loadedmetadata', this.videoEl.videoWidth, this.videoEl.videoHeight);
    }, { once: true });
    this.videoEl.addEventListener('canplay', () => {
      console.log('[webrtc]', this.camDebugLabel, 'canplay, paused=', this.videoEl.paused, 'readyState=', this.videoEl.readyState);
    }, { once: true });
    this.videoEl.addEventListener('error', () => {
      console.log('[webrtc]', this.camDebugLabel, 'video element error', this.videoEl.error);
    });

    this.pc.addEventListener('connectionstatechange', () => {
      console.log('[webrtc]', this.camDebugLabel, 'connectionState ->', this.pc?.connectionState);
      if (this.pc?.connectionState === 'failed' || this.pc?.connectionState === 'closed') {
        this.onError();
      }
    });
    this.pc.addEventListener('iceconnectionstatechange', () => {
      console.log('[webrtc]', this.camDebugLabel, 'iceConnectionState ->', this.pc?.iceConnectionState);
    });
    this.pc.addEventListener('icegatheringstatechange', () => {
      console.log('[webrtc]', this.camDebugLabel, 'iceGatheringState ->', this.pc?.iceGatheringState);
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
    this.ws.addEventListener('error', (e) => {
      console.log('[webrtc]', this.camDebugLabel, 'ws error', e);
      this.onError();
    });
    this.ws.addEventListener('close', (e) => {
      console.log('[webrtc]', this.camDebugLabel, 'ws close', e.code, e.reason);
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
      console.log('[webrtc]', this.camDebugLabel, 'got answer, setting remote description');
      await this.pc.setRemoteDescription({ type: 'answer', sdp: msg.value });
    } else if (msg.type === 'webrtc/candidate' && msg.value) {
      try {
        await this.pc.addIceCandidate({ candidate: msg.value, sdpMid: '0' });
      } catch (err) {
        console.log('[webrtc]', this.camDebugLabel, 'addIceCandidate failed', err);
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
    // Release the video element for the next tier. The `track` handler above assigns
    // videoEl.srcObject, and per the HTML spec srcObject takes precedence over the src
    // attribute -- so leaving it set here means CameraTile's WebRTC->MSE fallback
    // (onTierFailed -> tryMse -> MsePlayer.start()'s `videoEl.src = objectUrl`) is
    // silently ignored: the MediaSource never reaches "open", its sourceopen event never
    // fires, MsePlayer never opens its WebSocket, and -- because nothing errors either --
    // the tile sits in 'connecting' forever showing the snapshot poster ("STILL") instead
    // of falling through to a working MSE stream.
    this.videoEl.srcObject = null;
  }
}
