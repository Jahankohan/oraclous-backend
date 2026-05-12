"""Public-facing agent endpoints callable with integration keys (STORY-022).

No JWT auth — authentication is via 'Authorization: Bearer <integration_key>'.
CORS enforcement is per-agent (checked against cors_origins list in :PublishedAgent).
Rate limiting is per-integration-key, in-memory token bucket.
"""

import json

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.services.agent_executor import AgentExecutor
from app.services.audit_service import log_public_call
from app.services.integration_key_service import IntegrationKeyService
from app.services.rate_limiter import rate_limiter

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────


class PublicChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class WebhookIngressRequest(BaseModel):
    source: str = ""
    event_type: str = ""
    message: str
    context: dict = {}
    session_id: str | None = None


# ── Auth + guards ──────────────────────────────────────────────────────────────


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _check_cors(origin: str | None, cors_origins: list[str]) -> None:
    """Raise 403 if origin is not in the allowlist (only when list is non-empty)."""
    if not cors_origins:
        return  # open — no restriction
    if not origin or origin not in cors_origins:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not in CORS allowlist for this agent",
        )


async def _resolve_published(slug: str, api_key: str | None) -> dict:
    """Validate integration key and return the :PublishedAgent dict."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing integration key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    driver = neo4j_client.async_driver
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    svc = IntegrationKeyService(driver)
    published = await svc.validate_key(slug, api_key)
    if not published:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked integration key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return published


def _check_rate_limit(published: dict) -> None:
    bucket_key = f"{published['agent_id']}:{published['key_last4']}"
    limit = published.get("rate_limit_rpm", 60)
    if not rate_limiter.check(bucket_key, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )


# ── Public chat endpoint ───────────────────────────────────────────────────────


@router.post(
    "/agents/{slug}/chat",
    summary="Call a published agent (sync or SSE streaming)",
)
async def public_chat(
    slug: str,
    body: PublicChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    api_key = _extract_bearer(authorization)
    published = await _resolve_published(slug, api_key)

    origin = request.headers.get("origin")
    _check_cors(origin, published.get("cors_origins") or [])
    _check_rate_limit(published)

    driver = neo4j_client.async_driver
    accept = request.headers.get("accept", "")

    if "text/event-stream" in accept:
        return await _sse_response(published, body, driver)
    return await _sync_response(published, body, driver)


async def _sync_response(
    published: dict, body: PublicChatRequest, driver
) -> JSONResponse:
    executor = await AgentExecutor.from_neo4j(
        driver, published["graph_id"], published["agent_id"]
    )
    result = await executor.run(body.message, body.session_id)

    await log_public_call(
        driver=driver,
        graph_id=published["graph_id"],
        agent_id=published["agent_id"],
        integration_key_last4=published["key_last4"],
        input_text=body.message,
        response_text=result.response,
    )
    return JSONResponse(
        {
            "response": result.response,
            "session_id": result.session_id,
            "provenance": result.provenance.model_dump() if result.provenance else None,
        }
    )


async def _sse_response(
    published: dict, body: PublicChatRequest, driver
) -> StreamingResponse:
    async def _stream():
        executor = await AgentExecutor.from_neo4j(
            driver, published["graph_id"], published["agent_id"]
        )
        full_response = ""
        provenance = None

        async for token, prov in executor.run_stream(body.message, body.session_id):
            if token is not None:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            else:
                provenance = prov

        await log_public_call(
            driver=driver,
            graph_id=published["graph_id"],
            agent_id=published["agent_id"],
            integration_key_last4=published["key_last4"],
            input_text=body.message,
            response_text=full_response,
        )

        prov_dict = (
            provenance.model_dump() if hasattr(provenance, "model_dump") else provenance
        )
        yield f"data: {json.dumps({'done': True, 'provenance': prov_dict, 'session_id': body.session_id})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Webhook ingress endpoint ───────────────────────────────────────────────────


@router.post(
    "/agents/{slug}/webhook",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Webhook ingress — accepts external events and runs the agent asynchronously",
)
async def webhook_ingress(
    slug: str,
    body: WebhookIngressRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    api_key = _extract_bearer(authorization)
    published = await _resolve_published(slug, api_key)
    _check_rate_limit(published)

    from app.tasks.webhook_tasks import run_webhook_agent_task

    run_webhook_agent_task.delay(
        graph_id=published["graph_id"],
        agent_id=published["agent_id"],
        slug=slug,
        message=body.message,
        session_id=body.session_id,
        context=body.context,
        egress_url=published.get("egress_url"),
        integration_key_last4=published["key_last4"],
    )
    return {"accepted": True, "slug": slug}
