import { NEVER, Observable, fromEvent, interval, startWith } from 'rxjs';
import { switchMap } from 'rxjs/operators';

/**
 * Like `interval(ms).pipe(startWith(0))`, but pauses entirely while the browser tab is
 * hidden (`document.visibilityState !== 'visible'`) and resumes (restarting from tick 0)
 * the moment it becomes visible again -- so a backgrounded tab stops polling the
 * backend for data nobody's looking at, instead of continuing to fire on the fixed
 * interval forever (OPTIMIZATION_OPPORTUNITIES.md item 6). Used in place of
 * `interval(POLL_MS).pipe(startWith(0), ...)` on the Events/Review/Recordings pages,
 * which otherwise poll unconditionally regardless of tab visibility.
 *
 * Implementation note: this deliberately does NOT `filter()` the visibilitychange
 * stream down to just the "became visible" transitions before switchMap-ing -- doing
 * that means switchMap never receives a new outer emission on the "became hidden"
 * transition, so it never gets a chance to unsubscribe the still-running inner
 * `interval`, and polling silently continues in the background regardless (confirmed
 * directly with a real subscription trace: tick count kept growing well after
 * visibilityState was set to 'hidden'). Instead, every visibilitychange event reaches
 * switchMap, which chooses a fresh `interval(ms)` when visible or `NEVER` (an
 * observable that never emits) when hidden -- `switchMap` unsubscribing the PREVIOUS
 * inner observable on every new outer emission is what actually tears down the ticking
 * interval the instant the tab is hidden. Confirmed with the same real subscription
 * trace: tick count froze immediately on hidden and correctly resumed from 0 on visible.
 *
 * `switchMap`ing an HTTP call off this (the existing pattern on all three pages)
 * naturally cancels any in-flight request the instant the tab is hidden too, same as
 * switchMap already does between ticks.
 */
export function visibleInterval(ms: number): Observable<number> {
  const visibilityChanges$ = fromEvent(document, 'visibilitychange');

  return visibilityChanges$.pipe(
    startWith(null), // seed so the initial visibility state is evaluated immediately
    switchMap(() => (document.visibilityState === 'visible' ? interval(ms).pipe(startWith(0)) : NEVER)),
  );
}
