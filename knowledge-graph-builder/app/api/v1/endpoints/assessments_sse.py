"""Server-Sent Events endpoint for live assessment-run progress (TASK-081).

`GET /api/v1/assessments/runs/{run_id}/events?since=<event_id>` streams the
state changes emitted by `AssessmentService` write paths during a live run.
Backs the orchestrator's `tail_run` (SPRINT-002 CLI + MILESTONE-002 UI).

Architecture
------------

- The in-process `AssessmentEventBroker` (TASK-081 §1) holds a per-(graph,
  run) ring buffer and a set of subscribers. The service-layer publishes on
  every successful Cypher commit; this endpoint subscribes and streams.
- Cross-tenant safety: the endpoint resolves the caller's `home_graph_id`,
  asserts `read` access via `verify_graph_access`, then asserts the run
  exists *in this tenant* before opening the stream. A run that lives in a
  different tenant returns 404 (existence-masked, matching the read
  endpoints' policy).
- `since` cursor is the largest `event_id` the client has already seen.
  The broker replays everything strictly greater than `since` from its
  ring buffer, then streams new events as they arrive.
- The connection stays open until the client disconnects. FastAPI /
  Starlette propagates the client's disconnect as an `asyncio.CancelledError`
  that unwinds the async generator and removes the subscriber.

Event shape on the wire
-----------------------

Each SSE event has:

    id: <event_id>
    event: <event_type>
    data: {"event_id": ..., "type": ..., "payload": {...}, "emitted_at": ...}

The `id` line lets EventSource-style clients auto-resume with the standard
`Last-Event-ID` header (Starlette respects it as `last_event_id` on
reconnect; clients that prefer the explicit `since=` query parameter can
use that instead). We surface both surfaces for clarity.

Out of scope
------------

- Backpressure heuristics beyond drop-oldest at the broker
- Redis Streams / multi-replica fan-out — that adapter ships alongside
  MILESTONE-002's horizontal-scale plan
- Authentication via query string — the existing JWT-in-`Authorization`
  header dependency stack handles auth (browsers can't set custom headers
  on EventSource, but the orchestrator is a CLI client that can)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    get_current_user,
    get_current_user_id,
    verify_graph_access,
)
from app.core.neo4j_client import neo4j_client
from app.services.assessment_event_broker import (
    AssessmentEvent,
    AssessmentEventBroker,
    get_assessment_event_broker,
)
from app.services.assessment_service import AssessmentService

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependencies (mirror assessments_reads.py for consistency) ────────────────


def _assessment_service() -> AssessmentService:
    """Build an `AssessmentService` bound to the live async Neo4j driver."""
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return AssessmentService(neo4j_client.async_driver)


def _principal_graph_id(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> str:
    """Resolve the caller's tenant `graph_id` from the JWT claim (TASK-081)."""
    graph_id = current_user.get("home_graph_id") or ""
    if not graph_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Principal has no home_graph_id claim. Assessment endpoints "
                "require a JWT bound to a tenant graph."
            ),
        )
    return str(graph_id)


def _event_broker() -> AssessmentEventBroker:
    """Inject the process-wide event broker (tests override via dependency)."""
    return get_assessment_event_broker()


# ── Wire format ───────────────────────────────────────────────────────────────


def _format_sse(event: AssessmentEvent) -> str:
    """Render an `AssessmentEvent` as an SSE wire frame.

    Format (one frame, terminated by `\\n\\n`):

        id: <event_id>
        event: <type>
        data: <json>
    """
    payload = {
        "event_id": event.event_id,
        "type": event.type,
        "payload": event.payload,
        "emitted_at": event.emitted_at,
    }
    return (
        f"id: {event.event_id}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


def _format_comment(message: str) -> str:
    """SSE comment line — used for the initial hello / keep-alive pings."""
    return f": {message}\n\n"


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get(
    "/assessments/runs/{run_id}/events",
    summary="SSE stream of live assessment-run state changes",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events stream",
        },
        400: {"description": "Missing home_graph_id"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def tail_run_events(
    request: Request,
    run_id: str = Path(..., max_length=128),
    since: int = Query(
        default=0,
        ge=0,
        description=(
            "Event-id cursor — replay every event with `event_id > since`, "
            "then stream new ones. Clients on reconnect should pass the "
            "largest `event_id` they've seen."
        ),
    ),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
    broker: AssessmentEventBroker = Depends(_event_broker),
) -> StreamingResponse:
    """SSE stream of live assessment-run state changes (TASK-081).

    **Auth**: caller's `home_graph_id` resolves the tenant scope.
    `verify_graph_access(graph_id, 'read')` runs before the stream opens.

    **Cross-tenant**: a `run_id` that does not exist in the caller's tenant
    returns 404 — existence is **not** distinguished from absence. This
    matches the read-endpoints' policy (TASK-079) and prevents `run_id`
    enumeration.

    **Reconnect semantics**: SSE clients reconnect with the last-seen
    `event_id` as `?since=`. The broker's ring buffer holds the last
    `DEFAULT_RING_BUFFER_SIZE` (1000) events per run — clients that fall
    behind further than that will miss the gap and must reconcile via the
    REST read endpoints.
    """
    # 1. Scope check — 403 before we open the stream.
    await verify_graph_access(graph_id, "read", user_id)

    # 2. Existence check — masks cross-tenant runs as 404 per ADR-018 §Tenancy.
    # We use the lightweight `_fetch_existing_run` helper (already scoped by
    # graph_id) rather than the heavier `get_run_detail` so the stream-open
    # latency stays low.
    run = await svc._fetch_existing_run(graph_id, run_id)  # noqa: SLF001
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AssessmentRun not found",
        )

    logger.info(
        "sse.tail_run: subscribe graph_id=%s run_id=%s user_id=%s since=%d",
        graph_id,
        run_id,
        user_id,
        since,
    )

    async def _generate() -> AsyncIterator[str]:
        """Produce SSE frames from the broker subscription."""
        # SSE "hello" — gives the client an immediate frame so proxies that
        # buffer the response don't sit on it. Also surfaces the cursor we
        # actually subscribed at, which is useful when `since` was clipped.
        yield _format_comment(f"tail_run open since={since}")

        subscription = broker.subscribe(graph_id, run_id, since=since)
        try:
            while True:
                # Race the subscription against a periodic keep-alive so
                # idle streams don't get reaped by intermediaries (typical
                # 60s idle timeout on cloud LBs). The keep-alive interval
                # is intentionally smaller than the heartbeat throttling
                # window — a stream may otherwise sit silent for >30s.
                try:
                    event = await asyncio.wait_for(
                        subscription.__anext__(), timeout=15.0
                    )
                except TimeoutError:
                    # Idle window — emit a comment-line keep-alive.
                    if await request.is_disconnected():
                        break
                    yield _format_comment("keep-alive")
                    continue
                except StopAsyncIteration:  # pragma: no cover — generator
                    break
                yield _format_sse(event)
        except asyncio.CancelledError:
            # Client disconnect: propagated as cancellation by Starlette.
            # The async generator's `aclose()` (called automatically when
            # we exit) drives the broker's `finally` block to unregister.
            raise
        finally:
            # Explicit close so the broker's `finally` block runs even when
            # we exit through the StopAsyncIteration path.
            await subscription.aclose()
            logger.info(
                "sse.tail_run: unsubscribe graph_id=%s run_id=%s user_id=%s",
                graph_id,
                run_id,
                user_id,
            )

    # `X-Accel-Buffering: no` disables nginx response buffering for this
    # connection so events are flushed to the client immediately.
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=headers,
    )
