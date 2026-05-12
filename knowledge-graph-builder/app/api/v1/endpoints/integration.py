"""Admin endpoints for publishing agents and managing integration keys (STORY-022)."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.integration_schemas import (
    PublishAgentRequest,
    PublishAgentResponse,
    PublishedAgentResponse,
    RotateKeyResponse,
)
from app.services.integration_key_service import IntegrationKeyService

router = APIRouter()
logger = get_logger(__name__)


def _integration_service() -> IntegrationKeyService:
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return IntegrationKeyService(neo4j_client.async_driver)


def _endpoint_url(slug: str) -> str:
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/public/agents/{slug}/chat"


@router.post(
    "/graphs/{graph_id}/agents/{agent_id}/publish",
    response_model=PublishAgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Publish an agent with a stable slug and integration key",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not found"},
        409: {"description": "Slug already taken"},
    },
)
async def publish_agent(
    graph_id: str,
    agent_id: str,
    data: PublishAgentRequest,
    user_id: str = Depends(get_current_user_id),
    svc: IntegrationKeyService = Depends(_integration_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    try:
        key, slug = await svc.publish_agent(
            agent_id=agent_id,
            graph_id=graph_id,
            org_id="",  # derived from graph ownership; placeholder for Phase 2 org extraction
            user_id=user_id,
            slug=data.slug,
            cors_origins=data.cors_origins,
            rate_limit_rpm=data.rate_limit_rpm,
            egress_url=data.egress_url,
        )
    except ValueError as exc:
        msg = str(exc)
        if "already taken" in msg:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    return PublishAgentResponse(
        slug=slug,
        endpoint_url=_endpoint_url(slug),
        integration_key=key,
        key_last4=key[-4:],
    )


@router.get(
    "/graphs/{graph_id}/agents/{agent_id}/publish",
    response_model=PublishedAgentResponse,
    summary="Get the published state of an agent",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not published or unpublished"},
    },
)
async def get_published_agent(
    graph_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: IntegrationKeyService = Depends(_integration_service),
):
    await verify_graph_access(graph_id, "read", user_id)
    published = await svc.get_published_by_agent(agent_id, graph_id)
    if not published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent is not published"
        )
    return PublishedAgentResponse(
        agent_id=published["agent_id"],
        slug=published["slug"],
        cors_origins=published.get("cors_origins") or [],
        rate_limit_rpm=published.get("rate_limit_rpm", 60),
        egress_url=published.get("egress_url"),
        key_last4=published["key_last4"],
        published_at=published["published_at"],
        unpublished_at=published.get("unpublished_at"),
    )


@router.delete(
    "/graphs/{graph_id}/agents/{agent_id}/publish",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Unpublish an agent (invalidates the integration key)",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not published"},
    },
)
async def unpublish_agent(
    graph_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: IntegrationKeyService = Depends(_integration_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    found = await svc.unpublish_agent(agent_id, graph_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent is not published"
        )


@router.post(
    "/graphs/{graph_id}/agents/{agent_id}/rotate-key",
    response_model=RotateKeyResponse,
    summary="Rotate the integration key (old key immediately invalidated)",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Agent not published"},
    },
)
async def rotate_integration_key(
    graph_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: IntegrationKeyService = Depends(_integration_service),
):
    await verify_graph_access(graph_id, "admin", user_id)
    try:
        new_key = await svc.rotate_key(agent_id, graph_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return RotateKeyResponse(integration_key=new_key, key_last4=new_key[-4:])
