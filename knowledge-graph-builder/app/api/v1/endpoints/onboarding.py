"""Fresh-user onboarding bootstrap.

One call that takes a brand-new authenticated user from "no workspace" to
"has a workspace and a sign-in code that names it." This collapses three
otherwise-separate steps (create graph → bind home_graph_id → re-mint token)
into a single endpoint so the friend's quickstart is exactly:

    1. POST /register/          (auth-service)
    2. POST /onboarding/bootstrap   (this endpoint)
    3. Use the returned access_token for everything else.

Idempotent: if the user already has a home_graph_id, returns the existing
workspace and a freshly-minted token (no second workspace gets created).
"""

import logging
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_current_user_id
from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.services.graph_node_service import GraphNodeService
from app.services.rebac_service import rebac_service

logger = logging.getLogger(__name__)
router = APIRouter()


class BootstrapRequest(BaseModel):
    workspace_name: str = Field(
        default="My Workspace",
        max_length=128,
        description="Human-readable name for the workspace. Defaults to 'My Workspace'.",
    )


class BootstrapResponse(BaseModel):
    graph_id: str
    workspace_name: str
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"
    already_initialized: bool = Field(
        default=False,
        description="True if the user already had a workspace; the existing one was returned.",
    )


async def _mint_user_token(user_id: str) -> dict[str, Any]:
    """Ask auth-service to issue a fresh token for an existing user.

    The new token will carry whatever home_graph_id is currently on the user
    record — which is why this must run *after* set_home_graph_id.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{settings.AUTH_SERVICE_URL}/internal/users/{user_id}/mint-token",
            headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
        )
    if response.status_code != 200:
        logger.error(
            "auth-service mint-token failed: %s %s",
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to mint workspace-bound token via auth-service",
        )
    return response.json()


async def _bind_home_graph(user_id: str, graph_id: str) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.put(
            f"{settings.AUTH_SERVICE_URL}/internal/users/{user_id}/home-graph",
            json={"home_graph_id": graph_id},
            headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
        )
    if response.status_code != 200:
        logger.error(
            "auth-service set-home-graph failed: %s %s",
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to bind home workspace via auth-service",
        )


@router.post(
    "/onboarding/bootstrap",
    response_model=BootstrapResponse,
    status_code=status.HTTP_200_OK,
    summary="One-call onboarding: create default workspace and re-mint sign-in code",
)
async def bootstrap(
    body: BootstrapRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
) -> BootstrapResponse:
    existing_home = current_user.get("home_graph_id")
    if existing_home:
        token_payload = await _mint_user_token(user_id)
        return BootstrapResponse(
            graph_id=existing_home,
            workspace_name=body.workspace_name,
            access_token=token_payload["access_token"],
            refresh_token=token_payload["refresh_token"],
            expires_in=token_payload["expires_in"],
            already_initialized=True,
        )

    if not neo4j_client.sync_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )

    graph_service = GraphNodeService(neo4j_client.sync_driver)
    graph_id = str(uuid4())
    graph_result = graph_service.create_graph(
        graph_id=graph_id,
        user_id=user_id,
        name=body.workspace_name,
        description="Default workspace created by onboarding bootstrap.",
    )

    if neo4j_client.async_driver:
        try:
            await rebac_service.register_new_graph(
                driver=neo4j_client.async_driver,
                user_id=user_id,
                graph_id=graph_id,
                name=body.workspace_name,
            )
        except Exception as rebac_exc:
            logger.warning(
                "ReBAC graph registration failed during onboarding (non-fatal): %s",
                rebac_exc,
            )

    await _bind_home_graph(user_id, graph_id)
    token_payload = await _mint_user_token(user_id)

    logger.info(
        "Onboarding bootstrap: user=%s graph=%s workspace_name=%r",
        user_id,
        graph_id,
        body.workspace_name,
    )

    return BootstrapResponse(
        graph_id=graph_result["graph_id"],
        workspace_name=graph_result["name"],
        access_token=token_payload["access_token"],
        refresh_token=token_payload["refresh_token"],
        expires_in=token_payload["expires_in"],
        already_initialized=False,
    )
