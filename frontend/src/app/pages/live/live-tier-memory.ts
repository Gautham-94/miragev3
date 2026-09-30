import { Injectable } from '@angular/core';

export type WebRtcVerdict = 'unknown' | 'available' | 'unavailable';

/**
 * Remembers, per camera, WebRTC's own proven outcome -- across CameraTile instances,
 * and critically across Angular's own default route-reuse behavior (no
 * RouteReuseStrategy is registered, so navigating away from /live and back destroys
 * and recreates every CameraTile from scratch; without this, each one's own state
 * would reset to 'unknown' every single visit, so a deployment where WebRTC genuinely
 * can't connect (no reachable ICE candidate, e.g. UDP blocked) would re-race it, for
 * every tile, on every single visit to Live, instead of ever settling on the tier that
 * actually works).
 *
 * Three states, not a boolean, because CameraTile's connection strategy branches three
 * ways on this (see its own startVideoTiers()):
 *   - 'unknown'      -- never determined for this camera this app lifetime. Race both
 *                       WebRTC and MSE in parallel; whichever wins decides the verdict
 *                       from THIS point on (see markAvailable/markUnavailable's own
 *                       docstrings for exactly when each fires).
 *   - 'available'    -- WebRTC has proven it can play for this camera. Skip the race,
 *                       go straight to WebRTC alone (today's steady-state fast path).
 *   - 'unavailable'  -- WebRTC has proven it can NOT play for this camera (failed
 *                       without ever reaching playback). Skip the race, go straight to
 *                       MSE alone -- re-probing WebRTC would just re-pay its ICE/
 *                       connect timeout for a deployment that structurally can't use it.
 *
 * `providedIn: 'root'` -- one instance for the whole app's lifetime (a real page
 * reload, not a route navigation, is the only thing that should forget this), so this
 * genuinely persists across LivePage/CameraTile destroy-recreate cycles.
 */
@Injectable({ providedIn: 'root' })
export class LiveTierMemory {
  private readonly verdicts = new Map<string, WebRtcVerdict>();

  getVerdict(camera: string): WebRtcVerdict {
    return this.verdicts.get(camera) ?? 'unknown';
  }

  // Called only when WebRTC itself reaches playback -- whether it was tried alone
  // (verdict already 'available') or as one side of a race (it won, or the other side
  // failed first and this one went on to succeed regardless). Never called merely
  // because WebRTC "lost" a race to a faster MSE while still healthy/pending -- that's
  // not evidence WebRTC doesn't work here, just that it wasn't first this one time; see
  // CameraTile.onTierWon's own reasoning for why a race-loss alone leaves the verdict
  // untouched (an incorrect 'unavailable' here would wrongly skip WebRTC forever after).
  markAvailable(camera: string): void {
    this.verdicts.set(camera, 'available');
  }

  // Called only when WebRTC itself explicitly fails (its own onError fires) AND it
  // never once reached playback first (see CameraTile.webrtcEverPlayed) -- a dropped
  // stream that HAD been playing is not "unavailable," just momentarily down.
  markUnavailable(camera: string): void {
    this.verdicts.set(camera, 'unavailable');
  }
}
