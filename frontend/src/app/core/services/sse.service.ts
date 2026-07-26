import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

export interface SseMessage<T = unknown> {
  type: 'event' | 'review_segment' | 'query_match';
  data: T;
}

/**
 * One shared EventSource for the app's lifetime (per stream URL) -- native EventSource
 * auto-reconnects on drop/error, so no custom backoff logic is needed here. Pages that
 * want live updates subscribe to the returned Observable and filter by `type`; nothing
 * ever unsubscribes/closes the underlying connection (it's a singleton, torn down only
 * when the whole app/tab closes), same lifetime reasoning as ApiService's HttpClient.
 */
@Injectable({ providedIn: 'root' })
export class SseService {
  private readonly connections = new Map<string, EventSource>();
  private readonly messages$ = new Subject<SseMessage>();

  connect(url: string): Observable<SseMessage> {
    if (!this.connections.has(url)) {
      const eventSource = new EventSource(url);
      eventSource.onmessage = (ev: MessageEvent<string>) => {
        try {
          this.messages$.next(JSON.parse(ev.data) as SseMessage);
        } catch {
          // Malformed frame -- ignore rather than break the whole stream.
        }
      };
      this.connections.set(url, eventSource);
    }
    return this.messages$.asObservable();
  }
}
