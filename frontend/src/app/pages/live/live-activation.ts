import { Injectable, signal } from '@angular/core';

/**
 * Tells every CameraTile "the Live route just became active again" -- driven by
 * LivePage's own router-event subscription (see its own comment for why that, and not
 * a component lifecycle hook, is the only available signal for this).
 *
 * Why this exists at all: LivePersistingRouteReuseStrategy (core/) keeps LivePage's
 * component tree alive across navigation instead of destroying it, specifically so
 * WebRTC/MSE connections don't reconnect from scratch every time you leave and come
 * back to Live. But "keep alive" in Angular's router means DETACHED, not merely
 * hidden -- the router's RouteReuseStrategy.store()/shouldDetach() path calls
 * ViewContainerRef.detach() internally, which REMOVES the view's DOM nodes from the
 * document entirely while away, not just CSS-hides them. Confirmed as the actual cause
 * of a real reported bug: a tile's `mode` signal stayed 'webrtc'/'mse' (the
 * RTCPeerConnection/WebSocket genuinely never dropped) the whole time, yet the visible
 * picture was frozen on whatever frame was on screen the moment the route was left --
 * browsers throttle/suspend decoding new frames for a <video> element that's been
 * removed from the document, since there's nothing to render them into. CameraTile has
 * no way to detect "I've been reattached" from its own lifecycle (ngOnInit/
 * ngAfterViewInit only ever run once, on true first creation -- that's the entire
 * point of the reuse strategy), so this service exists purely to give it that signal
 * from the one place that DOES see every return to Live: LivePage's own router
 * subscription, set up once and left running for the app's lifetime.
 */
@Injectable({ providedIn: 'root' })
export class LiveActivation {
  readonly activatedAt = signal(Date.now());

  bump(): void {
    this.activatedAt.set(Date.now());
  }
}
