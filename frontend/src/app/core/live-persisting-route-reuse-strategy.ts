import { ActivatedRouteSnapshot, DetachedRouteHandle, RouteReuseStrategy } from '@angular/router';

// Routes whose component instance (and its live DOM -- most importantly, the connected
// <video> elements a camera tile's WebRTC/MSE players are attached to) should survive
// navigating away, instead of Angular's default behavior of destroying and recreating it
// from scratch on every visit. 'live' is the one that actually matters: without this,
// every trip to another page and back tears down and fully reconnects every camera's
// WebRTC/MSE session (ICE renegotiation, a fresh MSE buffer, etc.) -- the same
// "why does it take a while to show all the streams again" complaint this whole session
// has been chasing, just from a different angle (LiveTierMemory fixed WHICH tier
// reconnects fastest; this fixes not needing to reconnect at all).
const REUSED_ROUTE_PATHS = new Set(['live']);

/**
 * Keeps exactly the routes in REUSED_ROUTE_PATHS alive (detached, not destroyed) when
 * navigating away, and reattaches the SAME component instance -- unchanged, still
 * connected -- when navigating back. Every other route keeps Angular's default
 * create-fresh-every-time behavior untouched.
 *
 * This is the Angular-native mechanism for exactly this (shouldDetach/store/
 * shouldAttach/retrieve -- see Angular's own RouteReuseStrategy docs), and is
 * deliberately preferred here over an app-level "hoist the live grid out of the router
 * entirely" approach: it needs no change to how LivePage or CameraTile are written, no
 * manual DOM reparenting, and no risk of the live grid's state (which tile is
 * maximized, scroll position, etc.) drifting out of sync with the route -- Angular's
 * router already treats it as this route's own component tree, we're just telling it
 * not to throw that tree away.
 */
export class LivePersistingRouteReuseStrategy implements RouteReuseStrategy {
  private readonly stored = new Map<string, DetachedRouteHandle>();

  private routeKey(route: ActivatedRouteSnapshot): string | null {
    const path = route.routeConfig?.path;
    return path !== undefined && REUSED_ROUTE_PATHS.has(path) ? path : null;
  }

  shouldDetach(route: ActivatedRouteSnapshot): boolean {
    return this.routeKey(route) !== null;
  }

  store(route: ActivatedRouteSnapshot, handle: DetachedRouteHandle | null): void {
    const key = this.routeKey(route);
    if (key === null) return;
    if (handle === null) {
      this.stored.delete(key);
    } else {
      this.stored.set(key, handle);
    }
  }

  shouldAttach(route: ActivatedRouteSnapshot): boolean {
    const key = this.routeKey(route);
    return key !== null && this.stored.has(key);
  }

  retrieve(route: ActivatedRouteSnapshot): DetachedRouteHandle | null {
    const key = this.routeKey(route);
    return key !== null ? (this.stored.get(key) ?? null) : null;
  }

  // Standard default: reuse the current component instance only when navigating to the
  // exact same route config (e.g. a query-param-only change), same as Angular's own
  // DefaultRouteReuseStrategy -- every other route's normal in-place navigation
  // behavior (not the detach/reattach path above) is unaffected by this class existing.
  shouldReuseRoute(future: ActivatedRouteSnapshot, curr: ActivatedRouteSnapshot): boolean {
    return future.routeConfig === curr.routeConfig;
  }
}
