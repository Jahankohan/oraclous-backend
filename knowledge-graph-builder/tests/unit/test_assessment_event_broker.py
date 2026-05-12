"""Unit tests for the in-process assessment event broker (TASK-081).

Exercises:

- publish / subscribe round-trip
- monotonic event_id per (graph_id, run_id)
- heartbeat throttling at 30s per module_run_id
- replay-since-cursor logic (in-buffer events with event_id > since)
- ring-buffer eviction beyond the configured size
- subscriber namespace separation by (graph_id, run_id)
- subscriber queue drop-oldest on overflow does not block publisher
- introspection helpers (subscriber_count, buffered_event_count)
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.assessment_event_broker import (
    DEFAULT_RING_BUFFER_SIZE,
    AssessmentEventBroker,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_broker(
    ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
    heartbeat_min_interval_s: float = 30.0,
) -> AssessmentEventBroker:
    return AssessmentEventBroker(
        ring_buffer_size=ring_buffer_size,
        heartbeat_min_interval_s=heartbeat_min_interval_s,
    )


async def _drain(subscription, count: int, timeout: float = 1.0):
    out = []
    for _ in range(count):
        event = await asyncio.wait_for(subscription.__anext__(), timeout=timeout)
        out.append(event)
    return out


# ── Publish + Subscribe ───────────────────────────────────────────────────────


class TestPublishSubscribe:
    """publish() + subscribe() round-trip."""

    @pytest.mark.asyncio
    async def test_publish_then_subscribe_replays_buffered_events(self):
        broker = _make_broker()
        broker.publish("g1", "r1", "finding.recorded", {"delta": 1})
        broker.publish("g1", "r1", "finding.recorded", {"delta": 2})

        sub = broker.subscribe("g1", "r1", since=0)
        events = await _drain(sub, 2)
        await sub.aclose()

        assert [e.payload["delta"] for e in events] == [1, 2]
        assert [e.event_id for e in events] == [1, 2]
        assert all(e.graph_id == "g1" and e.run_id == "r1" for e in events)

    @pytest.mark.asyncio
    async def test_subscribe_then_publish_streams_live(self):
        broker = _make_broker()
        sub = broker.subscribe("g1", "r1", since=0)

        # Pump the subscriber a tick so it has registered before publish.
        await asyncio.sleep(0)
        broker.publish(
            "g1", "r1", "module_run.status_changed", {"new_status": "running"}
        )

        ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        await sub.aclose()

        assert ev.type == "module_run.status_changed"
        assert ev.payload == {"new_status": "running"}

    @pytest.mark.asyncio
    async def test_event_id_is_monotonic_per_run(self):
        broker = _make_broker()
        for i in range(5):
            broker.publish("g1", "r1", "finding.recorded", {"i": i})

        sub = broker.subscribe("g1", "r1", since=0)
        events = await _drain(sub, 5)
        await sub.aclose()

        assert [e.event_id for e in events] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_event_id_namespace_per_graph_and_run(self):
        """event_id sequences are independent per (graph_id, run_id)."""
        broker = _make_broker()
        broker.publish("g1", "r1", "finding.recorded", {})
        broker.publish("g1", "r2", "finding.recorded", {})
        broker.publish("g2", "r1", "finding.recorded", {})

        for gid, rid in [("g1", "r1"), ("g1", "r2"), ("g2", "r1")]:
            sub = broker.subscribe(gid, rid, since=0)
            ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
            await sub.aclose()
            assert ev.event_id == 1  # each (gid,rid) starts at 1

    def test_publish_missing_graph_id_returns_none(self):
        broker = _make_broker()
        assert broker.publish("", "r1", "finding.recorded", {}) is None
        assert broker.publish("g1", "", "finding.recorded", {}) is None


# ── Heartbeat throttling ──────────────────────────────────────────────────────


class TestHeartbeatThrottle:
    """Heartbeat events are de-duped at the broker per module_run_id."""

    def test_heartbeat_throttled_within_window(self, monkeypatch):
        broker = _make_broker(heartbeat_min_interval_s=30.0)

        # Pin monotonic so the test is deterministic.
        now = [1000.0]
        monkeypatch.setattr(
            "app.services.assessment_event_broker.time.monotonic",
            lambda: now[0],
        )

        e1 = broker.publish(
            "g1", "r1", "module_run.heartbeat", {"module_run_id": "mr1"}
        )
        # 5s later — should be throttled
        now[0] = 1005.0
        e2 = broker.publish(
            "g1", "r1", "module_run.heartbeat", {"module_run_id": "mr1"}
        )
        # 31s after the first — should fire again
        now[0] = 1031.0
        e3 = broker.publish(
            "g1", "r1", "module_run.heartbeat", {"module_run_id": "mr1"}
        )

        assert e1 is not None
        assert e2 is None  # throttled
        assert e3 is not None

    def test_heartbeat_throttle_is_per_module_run_id(self, monkeypatch):
        broker = _make_broker(heartbeat_min_interval_s=30.0)

        now = [1000.0]
        monkeypatch.setattr(
            "app.services.assessment_event_broker.time.monotonic",
            lambda: now[0],
        )

        e_a = broker.publish(
            "g1", "r1", "module_run.heartbeat", {"module_run_id": "mr-A"}
        )
        # Different module_run within the same window — NOT throttled.
        e_b = broker.publish(
            "g1", "r1", "module_run.heartbeat", {"module_run_id": "mr-B"}
        )

        assert e_a is not None
        assert e_b is not None

    def test_heartbeat_throttle_does_not_affect_other_event_types(self, monkeypatch):
        broker = _make_broker(heartbeat_min_interval_s=30.0)

        now = [1000.0]
        monkeypatch.setattr(
            "app.services.assessment_event_broker.time.monotonic",
            lambda: now[0],
        )

        broker.publish("g1", "r1", "module_run.heartbeat", {"module_run_id": "mr1"})
        # Status change is not throttled even within the heartbeat window.
        ev = broker.publish(
            "g1",
            "r1",
            "module_run.status_changed",
            {"module_run_id": "mr1", "new_status": "running"},
        )
        assert ev is not None


# ── since-cursor replay ───────────────────────────────────────────────────────


class TestReplaySinceCursor:
    @pytest.mark.asyncio
    async def test_since_cursor_skips_replayed_events(self):
        broker = _make_broker()
        broker.publish("g1", "r1", "finding.recorded", {"i": 1})
        broker.publish("g1", "r1", "finding.recorded", {"i": 2})
        broker.publish("g1", "r1", "finding.recorded", {"i": 3})

        # Subscribe with since=2 — should only see event_id 3.
        sub = broker.subscribe("g1", "r1", since=2)
        ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        await sub.aclose()

        assert ev.event_id == 3
        assert ev.payload == {"i": 3}

    @pytest.mark.asyncio
    async def test_since_cursor_beyond_history_yields_only_new_events(self):
        broker = _make_broker()
        broker.publish("g1", "r1", "finding.recorded", {"i": 1})

        sub = broker.subscribe("g1", "r1", since=999)
        # No buffered events match since=999; publish a new one and confirm.
        await asyncio.sleep(0)
        broker.publish("g1", "r1", "finding.recorded", {"i": 2})

        ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        await sub.aclose()
        assert ev.payload == {"i": 2}

    @pytest.mark.asyncio
    async def test_since_zero_replays_full_buffer(self):
        broker = _make_broker()
        for i in range(5):
            broker.publish("g1", "r1", "finding.recorded", {"i": i})

        sub = broker.subscribe("g1", "r1", since=0)
        events = await _drain(sub, 5)
        await sub.aclose()
        assert [e.payload["i"] for e in events] == [0, 1, 2, 3, 4]


# ── Ring buffer eviction ──────────────────────────────────────────────────────


class TestRingBuffer:
    def test_ring_buffer_evicts_oldest_when_full(self):
        broker = _make_broker(ring_buffer_size=3)
        for i in range(5):
            broker.publish("g1", "r1", "finding.recorded", {"i": i})

        assert broker.buffered_event_count("g1", "r1") == 3

    @pytest.mark.asyncio
    async def test_evicted_events_unrecoverable_via_since(self):
        broker = _make_broker(ring_buffer_size=3)
        for i in range(5):
            broker.publish("g1", "r1", "finding.recorded", {"i": i})

        # Buffer now holds event_ids 3, 4, 5 (1 and 2 evicted).
        sub = broker.subscribe("g1", "r1", since=0)
        events = await _drain(sub, 3)
        await sub.aclose()
        assert [e.event_id for e in events] == [3, 4, 5]


# ── Subscriber lifecycle ──────────────────────────────────────────────────────


class TestSubscriberLifecycle:
    @pytest.mark.asyncio
    async def test_subscriber_count_tracks_active_subscriptions(self):
        broker = _make_broker()
        assert broker.subscriber_count("g1", "r1") == 0

        sub1 = broker.subscribe("g1", "r1", since=0)
        sub2 = broker.subscribe("g1", "r1", since=0)
        # Eagerly enter the generators so subscribers are registered.
        # Use asyncio.create_task to avoid blocking on the empty queue.
        # Instead, prime the generator with __anext__ via wait_for(timeout=0).
        # Simplest: publish one event to each and drain.
        broker.publish("g1", "r1", "finding.recorded", {})
        await asyncio.wait_for(sub1.__anext__(), timeout=1.0)
        await asyncio.wait_for(sub2.__anext__(), timeout=1.0)

        assert broker.subscriber_count("g1", "r1") == 2

        await sub1.aclose()
        await sub2.aclose()
        assert broker.subscriber_count("g1", "r1") == 0

    @pytest.mark.asyncio
    async def test_subscribers_isolated_by_run(self):
        broker = _make_broker()
        sub_a = broker.subscribe("g1", "r1", since=0)
        sub_b = broker.subscribe("g1", "r2", since=0)

        # Prime both subscribers.
        broker.publish("g1", "r1", "finding.recorded", {"target": "a"})
        ev = await asyncio.wait_for(sub_a.__anext__(), timeout=1.0)
        assert ev.payload == {"target": "a"}

        # sub_b should NOT receive the event for r1.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sub_b.__anext__(), timeout=0.1)

        await sub_a.aclose()
        await sub_b.aclose()


# ── Non-blocking publish ──────────────────────────────────────────────────────


class TestNonBlockingPublish:
    """Slow subscribers must not block fast publishers (fire-and-forget contract)."""

    def test_publish_is_sync_and_returns_quickly_even_when_no_consumer(self):
        broker = _make_broker()
        # Subscribe but never consume — events accumulate in the queue.
        # publish() must not block.
        sub = broker.subscribe("g1", "r1", since=0)
        # The generator hasn't been entered yet — that's the worst case
        # for the broker.
        for i in range(100):
            broker.publish("g1", "r1", "finding.recorded", {"i": i})
        # If we got here without blocking we're good.
        assert broker.buffered_event_count("g1", "r1") == 100
        # Drop the un-entered subscriber.
        del sub
