"""In-process pub/sub broker for assessment-run live events (STORY-026, TASK-081).

Architecture
------------

The broker is a lightweight, **in-process** publish/subscribe mechanism keyed
by `(graph_id, run_id)`. It backs the SSE `tail_run` endpoint: every
`AssessmentService` write method publishes an event after the Cypher commit;
SSE subscribers receive events via per-subscriber `asyncio.Queue`s.

Why in-process and not Redis Streams or Kafka right now:

- ADR-005 (Neo4j as L1 authority) is preserved — events are an **ephemeral
  relay** on top of authoritative `:Finding` / `:ModuleRun` writes. A lost
  event never loses data; the client can re-read via the REST endpoints.
- SPRINT-002 ships single-instance. When MILESTONE-002 introduces
  multi-replica deployments, swap the broker out for a Redis Streams or
  PostgreSQL `LISTEN/NOTIFY` adapter that satisfies the same interface (a
  small `publish(event)` + `subscribe(graph_id, run_id) -> AsyncIterator`
  contract). The service-layer and endpoint code does not change.
- Heartbeat throttling and the per-run ring buffer (cursor replay) live at
  this layer — outside the service so the service stays focused on Cypher.

Concurrency
-----------

- `publish()` is sync (cheap) and never blocks. Subscribers that fall behind
  drop the oldest events from their queue (bounded `Queue(maxsize=N)`); the
  publisher does **not** await any subscriber.
- The service-layer fire-and-forget contract is preserved by making
  `publish()` synchronous; the only async work is on the subscriber side
  (queue consumption).
- Heartbeat throttling: per `(graph_id, run_id, module_run_id)` the broker
  tracks `last_heartbeat_emitted_at`. Heartbeats arriving within
  `HEARTBEAT_MIN_INTERVAL_S` of the last emission are dropped at the
  broker, so subscribers see at most one heartbeat per module_run per
  window regardless of how often subagents heartbeat.
- Per-run ring buffer of the last N events (default 1000) supports
  `since=<event_id>` replay across reconnects. Each event carries a
  monotonic `event_id` so the cursor is just an integer.

Safety
------

- The broker performs **no** cross-tenant scope checks. The SSE endpoint
  validates `verify_graph_access` *before* it calls `subscribe()`. This is
  defense in depth: even if the endpoint were missing, the broker still
  enforces tenancy at the `(graph_id, run_id)` key — events are namespaced
  by graph_id, so cross-tenant `run_id` collisions cannot leak events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────

# Maximum events retained per (graph_id, run_id) for since-cursor replay.
# Tuned to comfortably hold a full Eurail-scale run (~600 findings + module
# transitions). Configurable per-instance via constructor.
DEFAULT_RING_BUFFER_SIZE = 1000

# Minimum seconds between heartbeat emissions per module_run_id. Subagents
# heartbeat aggressively; subscribers only need a liveness signal.
HEARTBEAT_MIN_INTERVAL_S = 30.0

# Per-subscriber queue bound. Slow subscribers lose oldest events but never
# block the publisher.
SUBSCRIBER_QUEUE_MAX = 4096


# ── Event types ───────────────────────────────────────────────────────────────

EventType = Literal[
    "module_run.status_changed",
    "module_run.heartbeat",
    "finding.recorded",
    "conflict.recorded",
    "unresolved_question.raised",
    "deliverable.persisted",
    "run.finalized",
]


@dataclass(frozen=True)
class AssessmentEvent:
    """A single event on the wire.

    `event_id` is monotonically increasing **per (graph_id, run_id)** within a
    process. A subscriber's `since` cursor is therefore an `int` that
    identifies "give me everything strictly greater than this".

    `payload` is a dict; the SSE endpoint JSON-encodes it.
    """

    event_id: int
    graph_id: str
    run_id: str
    type: EventType
    payload: dict[str, Any]
    emitted_at: float = field(default_factory=time.time)


# ── Subscriber handle ─────────────────────────────────────────────────────────


class _Subscriber:
    """A single in-process SSE subscriber.

    Each subscriber has its own bounded queue. The broker drops oldest
    events when a subscriber falls behind so a slow client cannot DoS others.
    """

    __slots__ = ("queue", "dropped")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AssessmentEvent] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_MAX
        )
        self.dropped: int = 0

    def push(self, event: AssessmentEvent) -> None:
        """Non-blocking enqueue. Drops oldest on overflow."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                _ = self.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover — defensive
                pass
            try:
                self.queue.put_nowait(event)
                self.dropped += 1
            except asyncio.QueueFull:  # pragma: no cover — defensive
                self.dropped += 1


# ── Broker ────────────────────────────────────────────────────────────────────


class AssessmentEventBroker:
    """In-process pub/sub keyed by `(graph_id, run_id)`.

    Construct one instance per process (`get_assessment_event_broker()` is the
    process-wide singleton accessor). Tests inject their own instance.
    """

    def __init__(
        self,
        ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
        heartbeat_min_interval_s: float = HEARTBEAT_MIN_INTERVAL_S,
    ) -> None:
        self._ring_buffer_size = ring_buffer_size
        self._heartbeat_min_interval_s = heartbeat_min_interval_s

        # (graph_id, run_id) → deque of recent events for cursor replay
        self._history: dict[tuple[str, str], deque[AssessmentEvent]] = {}

        # (graph_id, run_id) → set of subscribers
        self._subscribers: dict[tuple[str, str], set[_Subscriber]] = {}

        # (graph_id, run_id, module_run_id) → monotonic timestamp of the last
        # heartbeat emitted to subscribers. Used to throttle heartbeats.
        self._last_heartbeat: dict[tuple[str, str, str], float] = {}

        # Monotonic event-id counter per (graph_id, run_id)
        self._next_event_id: dict[tuple[str, str], int] = {}

    # ── Publish ──────────────────────────────────────────────────────────────

    def publish(
        self,
        graph_id: str,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> AssessmentEvent | None:
        """Publish an event for `(graph_id, run_id)`.

        Returns the persisted `AssessmentEvent`, or `None` when the event
        was throttled (currently only heartbeats throttle).

        Synchronous and non-blocking — fire-and-forget from the service
        layer is safe because per-subscriber queues are non-blocking
        (`put_nowait` with drop-oldest on overflow).
        """
        if not graph_id or not run_id:
            # Programmer error — log loudly and skip.
            logger.warning(
                "AssessmentEventBroker.publish: missing graph_id/run_id (type=%s)",
                event_type,
            )
            return None

        # Heartbeat throttling: collapse rapid heartbeats per module_run_id.
        if event_type == "module_run.heartbeat":
            module_run_id = payload.get("module_run_id")
            if module_run_id:
                key = (graph_id, run_id, str(module_run_id))
                now = time.monotonic()
                last = self._last_heartbeat.get(key)
                if last is not None and (now - last) < self._heartbeat_min_interval_s:
                    return None
                self._last_heartbeat[key] = now

        run_key = (graph_id, run_id)
        next_id = self._next_event_id.get(run_key, 0) + 1
        self._next_event_id[run_key] = next_id

        event = AssessmentEvent(
            event_id=next_id,
            graph_id=graph_id,
            run_id=run_id,
            type=event_type,
            payload=payload,
        )

        # Persist to ring buffer for since-cursor replay.
        history = self._history.get(run_key)
        if history is None:
            history = deque(maxlen=self._ring_buffer_size)
            self._history[run_key] = history
        history.append(event)

        # Fan-out to subscribers (non-blocking).
        for sub in list(self._subscribers.get(run_key, ())):
            sub.push(event)

        return event

    # ── Subscribe ────────────────────────────────────────────────────────────

    def subscribe(
        self,
        graph_id: str,
        run_id: str,
        since: int = 0,
    ) -> AsyncIterator[AssessmentEvent]:
        """Return an async iterator yielding events for `(graph_id, run_id)`.

        Registration is **synchronous** — by the time `subscribe()` returns,
        the subscriber is already in the fanout set, and any subsequent
        `publish()` will be delivered. This avoids the race where a caller
        creates the iterator and then publishes before entering the loop.

        Replay: every in-buffer event with `event_id > since` is enqueued
        before live events. Replay is one-shot at subscribe time.

        Cleanup: callers must `await aclose()` on the returned iterator
        (or exhaust it normally) to unregister. The wrapping endpoint uses
        a try/finally to guarantee this.
        """
        run_key = (graph_id, run_id)
        subscriber = _Subscriber()

        # Replay first — eagerly enqueue everything we still have buffered.
        history = self._history.get(run_key)
        if history is not None:
            for ev in history:
                if ev.event_id > since:
                    subscriber.push(ev)

        # Register synchronously so a publish() that races us is delivered.
        self._subscribers.setdefault(run_key, set()).add(subscriber)

        return self._iter(run_key, subscriber)

    async def _iter(
        self,
        run_key: tuple[str, str],
        subscriber: _Subscriber,
    ) -> AsyncIterator[AssessmentEvent]:
        try:
            while True:
                event = await subscriber.queue.get()
                yield event
        finally:
            subs = self._subscribers.get(run_key)
            if subs is not None:
                subs.discard(subscriber)
                if not subs:
                    self._subscribers.pop(run_key, None)

    # ── Introspection helpers (for tests + ops) ──────────────────────────────

    def subscriber_count(self, graph_id: str, run_id: str) -> int:
        return len(self._subscribers.get((graph_id, run_id), ()))

    def buffered_event_count(self, graph_id: str, run_id: str) -> int:
        hist = self._history.get((graph_id, run_id))
        return len(hist) if hist is not None else 0

    def clear(self) -> None:
        """Drop all state. Intended for tests only."""
        self._history.clear()
        self._subscribers.clear()
        self._last_heartbeat.clear()
        self._next_event_id.clear()


# ── Process-wide accessor ─────────────────────────────────────────────────────


_BROKER: AssessmentEventBroker | None = None


def get_assessment_event_broker() -> AssessmentEventBroker:
    """Return the process-wide broker singleton.

    The first caller constructs it with defaults. Tests that want a clean
    broker should call `set_assessment_event_broker(AssessmentEventBroker())`
    in a fixture or use the `clear()` helper.
    """
    global _BROKER
    if _BROKER is None:
        _BROKER = AssessmentEventBroker()
    return _BROKER


def set_assessment_event_broker(broker: AssessmentEventBroker | None) -> None:
    """Replace (or reset to `None`) the process-wide broker. Test-only."""
    global _BROKER
    _BROKER = broker
