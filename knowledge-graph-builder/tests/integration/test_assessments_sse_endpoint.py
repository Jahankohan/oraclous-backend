"""Integration tests for the SSE tail_run endpoint (TASK-081).

These tests exercise the **endpoint handler + StreamingResponse + broker**
wiring. They do NOT go through httpx.ASGITransport because that transport
buffers the entire ASGI response before returning (httpx upstream design;
`_transports/asgi.py` waits for `response_complete.set()`), which is
incompatible with infinite SSE streams.

Instead we call the endpoint function directly, get a `StreamingResponse`
back, and consume its body iterator with timeouts. This still exercises
the full handler: dependency injection, scope check, existence check,
subscription registration, frame formatting, and the broker fanout.

Authentication / scope checks are stubbed (mocked) — they have their own
unit-level coverage in `test_dependencies.py` and the read-endpoint
integration tests. What's unique to TASK-081 is the **streaming wire
behavior**, which is what these tests target.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.api.v1.endpoints.assessments_sse import tail_run_events
from app.services.assessment_event_broker import (
    AssessmentEventBroker,
    set_assessment_event_broker,
)
from app.services.assessment_service import AssessmentService

pytestmark = pytest.mark.asyncio


_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"sse-A-{_SESSION}"
_GID_B = f"sse-B-{_SESSION}"
_USER_A = f"user-A-{_SESSION}"
_USER_B = f"user-B-{_SESSION}"
_RUN_ID = f"run-sse-{_SESSION}"


# ── Test scaffolding ──────────────────────────────────────────────────────────


def _make_driver(run_exists_in_graph: str | None = _GID_A):
    """Mock AsyncDriver whose `_fetch_existing_run` returns a row only when
    the principal's graph_id matches `run_exists_in_graph`.
    """

    async def execute_query(query, params=None, **kwargs):
        params = params or {}
        if run_exists_in_graph and params.get("graph_id") == run_exists_in_graph:
            r = MagicMock()
            r.records = [{"template_id": "t", "subject_id": "s", "status": "running"}]
            return r
        r = MagicMock()
        r.records = []
        return r

    drv = MagicMock()
    drv.execute_query = AsyncMock(side_effect=execute_query)
    return drv


def _make_request(client_disconnected: bool = False):
    """Build a Starlette-compatible Request stub for the endpoint."""

    req = MagicMock()

    async def is_disconnected():
        return client_disconnected

    req.is_disconnected = is_disconnected
    return req


@pytest.fixture
def fresh_broker():
    """Pin a fresh broker into the process singleton for each test."""
    broker = AssessmentEventBroker(ring_buffer_size=100, heartbeat_min_interval_s=0.1)
    set_assessment_event_broker(broker)
    yield broker
    set_assessment_event_broker(None)


async def _call_endpoint(
    *,
    broker: AssessmentEventBroker,
    graph_id: str = _GID_A,
    user_id: str = _USER_A,
    run_id: str = _RUN_ID,
    since: int = 0,
    run_exists_in_graph: str | None = _GID_A,
    verify_raises: HTTPException | None = None,
) -> StreamingResponse:
    """Invoke the endpoint with all dependencies stubbed and return the
    StreamingResponse. Patches `verify_graph_access` in the endpoint module
    so we can simulate ReBAC denial without spinning up the auth stack.
    """
    from app.api.v1.endpoints import assessments_sse as mod

    original = mod.verify_graph_access
    if verify_raises is not None:

        async def _denied(*a, **kw):
            raise verify_raises

        mod.verify_graph_access = _denied
    else:
        mod.verify_graph_access = AsyncMock(return_value=graph_id)

    try:
        drv = _make_driver(run_exists_in_graph=run_exists_in_graph)
        svc = AssessmentService(drv, event_broker=broker)
        return await tail_run_events(
            request=_make_request(),
            run_id=run_id,
            since=since,
            user_id=user_id,
            graph_id=graph_id,
            svc=svc,
            broker=broker,
        )
    finally:
        mod.verify_graph_access = original


async def _read_frames(
    response: StreamingResponse,
    min_frames: int = 1,
    timeout: float = 1.0,
) -> list[dict]:
    """Consume the StreamingResponse body iterator until `min_frames` non-
    comment SSE frames arrive or `timeout` elapses.

    Each frame is parsed into `{id, event, data}` (data JSON-decoded).
    """
    out = ""
    iterator = response.body_iterator.__aiter__()
    deadline = asyncio.get_event_loop().time() + timeout

    while True:
        remaining = max(0.001, deadline - asyncio.get_event_loop().time())
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except (TimeoutError, StopAsyncIteration):
            break
        if isinstance(chunk, bytes | bytearray):
            chunk = chunk.decode("utf-8", errors="replace")
        out += chunk
        # Check how many event frames we have so far.
        frames = _parse_sse(out)
        non_comment = [f for f in frames if f.get("event")]
        if len(non_comment) >= min_frames:
            break

    # Close the iterator so the broker subscription unregisters.
    try:
        await response.body_iterator.aclose()
    except Exception:  # pragma: no cover
        pass

    return _parse_sse(out)


def _parse_sse(buf: str) -> list[dict]:
    """Parse SSE wire-format. Comment lines (`:` prefix) are kept as `{}` so
    the caller can distinguish them from event frames if needed."""
    frames = []
    for block in buf.split("\n\n"):
        if not block.strip():
            continue
        evt: dict = {}
        for line in block.splitlines():
            if not line:
                continue
            if line.startswith(":"):
                continue
            k, _, v = line.partition(":")
            v = v.lstrip()
            if k == "data":
                try:
                    evt["data"] = json.loads(v)
                except json.JSONDecodeError:
                    evt["data"] = v
            elif k in ("id", "event", "retry"):
                evt[k] = v
        if evt:
            frames.append(evt)
    return frames


# ── Happy path ────────────────────────────────────────────────────────────────


class TestSSEHappyPath:
    async def test_response_has_event_stream_content_type(self, fresh_broker):
        resp = await _call_endpoint(broker=fresh_broker)
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"
        # Production headers: no-cache + nginx buffering disabled.
        assert resp.headers.get("cache-control", "").startswith("no-cache")
        assert resp.headers.get("x-accel-buffering") == "no"
        # Drain so the broker subscription unregisters cleanly.
        await _read_frames(resp, min_frames=0, timeout=0.1)

    async def test_published_events_arrive_in_order(self, fresh_broker):
        resp = await _call_endpoint(broker=fresh_broker)

        async def _publish_later():
            await asyncio.sleep(0.05)
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "module_run.status_changed",
                {"module_run_id": "mr-1", "new_status": "running"},
            )
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": 5},
            )

        asyncio.create_task(_publish_later())
        frames = await _read_frames(resp, min_frames=2, timeout=2.0)
        event_frames = [f for f in frames if f.get("event")]

        assert len(event_frames) >= 2
        assert event_frames[0]["event"] == "module_run.status_changed"
        assert event_frames[1]["event"] == "finding.recorded"
        assert event_frames[0]["data"]["payload"]["new_status"] == "running"
        assert event_frames[1]["data"]["payload"]["finding_count_delta"] == 5
        # event_id is rendered as the `id:` SSE header too.
        assert event_frames[0]["id"] == "1"
        assert event_frames[1]["id"] == "2"


# ── since-cursor replay ───────────────────────────────────────────────────────


class TestSinceCursor:
    async def test_since_cursor_skips_replayed_events(self, fresh_broker):
        # Pre-publish 3 events before subscribing.
        for i in range(3):
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": i},
            )

        resp = await _call_endpoint(broker=fresh_broker, since=2)
        frames = await _read_frames(resp, min_frames=1, timeout=2.0)
        event_frames = [f for f in frames if f.get("event")]

        assert len(event_frames) == 1
        assert event_frames[0]["data"]["event_id"] == 3

    async def test_reconnect_with_cursor_only_gets_new_events(self, fresh_broker):
        # First connection — publish 2 events, capture the last id.
        resp1 = await _call_endpoint(broker=fresh_broker)

        async def _pub2():
            await asyncio.sleep(0.05)
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": 1},
            )
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": 2},
            )

        asyncio.create_task(_pub2())
        frames = await _read_frames(resp1, min_frames=2, timeout=2.0)
        event_frames = [f for f in frames if f.get("event")]
        assert len(event_frames) == 2
        last_id = max(int(f["id"]) for f in event_frames)
        assert last_id == 2

        # Second connection — since=last_id; publish a third event.
        resp2 = await _call_endpoint(broker=fresh_broker, since=last_id)

        async def _pub3():
            await asyncio.sleep(0.05)
            fresh_broker.publish(
                _GID_A,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": 99},
            )

        asyncio.create_task(_pub3())
        frames2 = await _read_frames(resp2, min_frames=1, timeout=2.0)
        event_frames2 = [f for f in frames2 if f.get("event")]
        assert len(event_frames2) == 1
        assert event_frames2[0]["data"]["event_id"] == 3
        assert event_frames2[0]["data"]["payload"]["finding_count_delta"] == 99


# ── Cross-tenant / scope ──────────────────────────────────────────────────────


class TestCrossTenant:
    async def test_tenant_b_subscribing_to_tenant_a_run_returns_404(self, fresh_broker):
        """Tenant B's principal subscribing to a run that lives in tenant A
        gets 404 — existence-masked, per ADR-018 §Tenancy and TASK-079's
        established policy."""
        with pytest.raises(HTTPException) as exc:
            await _call_endpoint(
                broker=fresh_broker,
                graph_id=_GID_B,
                user_id=_USER_B,
                run_exists_in_graph=_GID_A,  # run is in A only
            )
        assert exc.value.status_code == 404

    async def test_403_when_verify_graph_access_denies(self, fresh_broker):
        """ReBAC denial: HTTP 403 raised before the stream opens."""
        with pytest.raises(HTTPException) as exc:
            await _call_endpoint(
                broker=fresh_broker,
                verify_raises=HTTPException(status_code=403, detail="Access denied"),
            )
        assert exc.value.status_code == 403


# ── Cross-tenant fanout isolation at broker level ─────────────────────────────


class TestBrokerScoping:
    async def test_events_published_to_other_graph_dont_leak(self, fresh_broker):
        """Defense-in-depth: even if a subscriber survived a scope mistake,
        broker fanout by (graph_id, run_id) prevents cross-tenant leaks."""
        resp = await _call_endpoint(broker=fresh_broker)

        async def _pub_other_graph():
            await asyncio.sleep(0.05)
            # Same run_id, DIFFERENT graph — must NOT reach the subscriber on g_A.
            fresh_broker.publish(
                _GID_B,
                _RUN_ID,
                "finding.recorded",
                {"module_run_id": "mr-1", "finding_count_delta": 100},
            )

        asyncio.create_task(_pub_other_graph())
        frames = await _read_frames(resp, min_frames=1, timeout=0.5)
        event_frames = [f for f in frames if f.get("event")]
        assert event_frames == []
