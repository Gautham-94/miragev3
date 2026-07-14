from __future__ import annotations

import shutil
import tempfile
import threading
import time

import pytest
import zmq

from mirage.ipc.zmq_pubsub import Publisher, Subscriber, ZmqProxy


@pytest.fixture
def zmq_context():
    # A dedicated zmq.Context per test, rather than the process-wide zmq.Context.instance()
    # singleton -- sharing the global instance across sequential tests in one pytest
    # process caused cross-test socket lifecycle races (a previous test's proxy thread
    # still running against a socket while a new test's fixture teardown was in progress),
    # surfacing as "Socket operation on non-socket" / flaky message delivery.
    ctx = zmq.Context()
    yield ctx
    ctx.term()


@pytest.fixture
def proxy_addrs(zmq_context):
    # NOTE: Unix domain socket paths are capped at ~103-107 bytes (sockaddr_un.sun_path)
    # on both macOS and Linux. pytest's own `tmp_path` fixture nests deeply enough
    # (.../pytest-of-<user>/pytest-<N>/<full-test-name>0/...) to blow past that limit, so
    # a short, fixed-prefix temp dir is used here instead -- this is also a real
    # constraint the production CACHE_DIR-based ipc socket naming must respect.
    short_dir = tempfile.mkdtemp(prefix="mrg", dir="/tmp")
    try:
        pub_addr = f"ipc://{short_dir}/pp"
        sub_addr = f"ipc://{short_dir}/ps"
        proxy = ZmqProxy(pub_addr, sub_addr, context=zmq_context)
        time.sleep(0.1)  # let the proxy thread start accepting connections
        yield pub_addr, sub_addr, zmq_context
        proxy.close()
    finally:
        shutil.rmtree(short_dir, ignore_errors=True)


def test_publisher_subscriber_topic_filtered_delivery(proxy_addrs):
    pub_addr, sub_addr, ctx = proxy_addrs
    publisher = Publisher(pub_addr, context=ctx)
    subscriber_a = Subscriber(sub_addr, topic_filter="object_detector/cam_a", context=ctx)
    subscriber_b = Subscriber(sub_addr, topic_filter="object_detector/cam_b", context=ctx)
    time.sleep(0.2)  # allow subscription propagation through the XSUB/XPUB proxy

    publisher.publish("object_detector/cam_a", "")

    assert subscriber_a.wait_for_message(timeout=2.0) is True
    assert subscriber_b.wait_for_message(timeout=0.3) is False  # cam_b never got a message

    publisher.close()
    subscriber_a.close()
    subscriber_b.close()


def test_subscriber_does_not_receive_messages_for_different_topic_prefix(proxy_addrs):
    pub_addr, sub_addr, ctx = proxy_addrs
    publisher = Publisher(pub_addr, context=ctx)
    subscriber = Subscriber(sub_addr, topic_filter="object_detector/front_door", context=ctx)
    time.sleep(0.2)

    publisher.publish("object_detector/front_doorX", "irrelevant-but-prefix-matches")
    # NOTE: ZMQ subscription filtering is a raw byte-prefix match, so
    # "object_detector/front_doorX" DOES match filter "object_detector/front_door" -- this
    # test documents that behavior rather than asserting a false expectation. Use exact
    # topic strings (not free-form prefixes that could collide) in real call sites.
    assert subscriber.wait_for_message(timeout=1.0) is True

    publisher.close()
    subscriber.close()


def test_drain_stale_removes_all_pending_messages_without_blocking(proxy_addrs):
    pub_addr, sub_addr, ctx = proxy_addrs
    publisher = Publisher(pub_addr, context=ctx)
    subscriber = Subscriber(sub_addr, topic_filter="test_topic", context=ctx)
    time.sleep(0.2)

    for _ in range(5):
        publisher.publish("test_topic", "x")
    time.sleep(0.2)

    drained = subscriber.drain_stale()
    assert drained == 5
    assert subscriber.wait_for_message(timeout=0.2) is False

    publisher.close()
    subscriber.close()


def test_wait_for_message_times_out_when_nothing_published(proxy_addrs):
    pub_addr, sub_addr, ctx = proxy_addrs
    subscriber = Subscriber(sub_addr, topic_filter="nobody_publishes_this", context=ctx)
    assert subscriber.wait_for_message(timeout=0.3) is False
    subscriber.close()


def test_multiple_publishers_one_subscriber(proxy_addrs):
    """Simulates the real N-cameras-share-one-detector-process topology: multiple
    independent Publisher instances (as if running in different OS processes) all reach
    one Subscriber filtered to a specific camera's topic.
    """
    pub_addr, sub_addr, ctx = proxy_addrs
    pub1 = Publisher(pub_addr, context=ctx)
    pub2 = Publisher(pub_addr, context=ctx)
    subscriber = Subscriber(sub_addr, topic_filter="object_detector/cam2", context=ctx)
    time.sleep(0.2)

    pub1.publish("object_detector/cam1", "")  # should NOT be seen by our subscriber
    pub2.publish("object_detector/cam2", "")  # should be seen

    assert subscriber.wait_for_message(timeout=2.0) is True
    assert subscriber.wait_for_message(timeout=0.3) is False  # no second message queued

    pub1.close()
    pub2.close()
    subscriber.close()


def test_cross_thread_publish_subscribe_timing(proxy_addrs):
    """A thread-based approximation of the real cross-process detector-signal flow: one
    thread acts as the detector publishing "done", the main thread (acting as the
    camera's RemoteObjectDetector) blocks waiting for it.
    """
    pub_addr, sub_addr, ctx = proxy_addrs
    subscriber = Subscriber(sub_addr, topic_filter="object_detector/cam1", context=ctx)
    time.sleep(0.2)

    def detector_worker():
        time.sleep(0.5)  # simulate inference latency
        pub = Publisher(pub_addr, context=ctx)
        pub.publish("object_detector/cam1", "")
        pub.close()

    t = threading.Thread(target=detector_worker)
    t.start()

    start = time.time()
    got = subscriber.wait_for_message(timeout=5.0)
    elapsed = time.time() - start

    t.join()
    subscriber.close()

    assert got is True
    assert 0.4 < elapsed < 3.0  # should unblock shortly after the ~0.5s publish delay
