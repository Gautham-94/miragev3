"""ZeroMQ pub/sub wrapper classes.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 11.2.

Two patterns:
  - An XSUB/XPUB proxy broker (ZmqProxy) so any number of publishers/subscribers can
    connect without direct point-to-point sockets.
  - Publisher/Subscriber wrapper classes that connect to that broker, using
    topic-prefixed string messages and ZeroMQ's built-in subscription-prefix filtering.

Per the spec: keep the detector-signal proxy on its own dedicated socket pair, separate
from general-purpose detection-result/config broadcast traffic, since it's on the hot path
for every single detection call.
"""

from __future__ import annotations

import threading
import time

import zmq

# ZMQ's well-known "slow joiner" problem: a PUB socket's connect() returns before the
# underlying transport handshake to the peer (here, the XSUB side of the proxy) has
# actually completed. A publish() call issued immediately after construction can be
# silently dropped with no error, since PUB sockets never buffer for a not-yet-connected
# subscriber. There is no portable "connection established" event to wait on instead, so
# a short fixed settle delay after connect() is the standard, pragmatic mitigation used
# broadly in ZMQ-based systems. This only costs anything once per Publisher's lifetime
# (paid at construction, not per publish() call), and in this codebase Publisher instances
# are long-lived (created once per process, not per message), so steady-state throughput
# is unaffected.
_PUB_CONNECT_SETTLE_SECONDS = 0.1


class ZmqProxy:
    """Bridges an XSUB bind (where publishers connect) and an XPUB bind (where
    subscribers connect) via zmq.proxy(), running the blocking proxy loop in its own
    thread so callers can just instantiate and forget.
    """

    def __init__(self, pub_addr: str, sub_addr: str, context: zmq.Context | None = None) -> None:
        self.context = context or zmq.Context.instance()
        self.pub_addr = pub_addr
        self.sub_addr = sub_addr
        self._frontend = self.context.socket(zmq.XSUB)
        self._frontend.bind(pub_addr)
        self._backend = self.context.socket(zmq.XPUB)
        self._backend.bind(sub_addr)
        self._thread = threading.Thread(target=self._run, daemon=True, name="zmq-proxy")
        self._thread.start()

    def _run(self) -> None:
        try:
            zmq.proxy(self._frontend, self._backend)
        except zmq.ZMQError:
            # Expected once close() tears down the sockets this proxy loop is blocked on
            # (zmq.proxy() has no clean "stop" API short of closing/terminating the
            # sockets/context it's using -- this is the standard way to end it).
            pass

    def close(self) -> None:
        self._frontend.close(linger=0)
        self._backend.close(linger=0)
        # Join so the proxy thread has actually exited before returning, preventing a
        # caller from tearing down the zmq.Context (ctx.term()) while this thread might
        # still be mid-way through handling the ZMQError from the socket closes above.
        self._thread.join(timeout=2)


class Publisher:
    def __init__(self, connect_addr: str, context: zmq.Context | None = None) -> None:
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(connect_addr)
        time.sleep(_PUB_CONNECT_SETTLE_SECONDS)  # avoid the PUB "slow joiner" message drop

    def publish(self, topic: str, payload: str = "") -> None:
        self.socket.send_string(f"{topic} {payload}")

    def close(self) -> None:
        self.socket.close(linger=0)


class Subscriber:
    def __init__(self, connect_addr: str, topic_filter: str, context: zmq.Context | None = None) -> None:
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(connect_addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)

    def wait_for_message(self, timeout: float) -> bool:
        """Blocks up to `timeout` seconds for a message matching this subscriber's topic
        filter. Returns True if one arrived, False on timeout. Drains only ONE message
        per call -- callers that need to drain all pending messages should call this in
        a loop with timeout=0 first (see drain_stale()).
        """
        events = self.socket.poll(timeout=int(timeout * 1000))
        if events == 0:
            return False
        self.socket.recv_string()
        return True

    def drain_stale(self) -> int:
        """Non-blocking: discards any already-pending messages (defensive cleanup before
        a fresh request, per spec section 3.2.2). Returns the number drained.
        """
        count = 0
        while self.socket.poll(timeout=0):
            self.socket.recv_string()
            count += 1
        return count

    def close(self) -> None:
        self.socket.close(linger=0)
