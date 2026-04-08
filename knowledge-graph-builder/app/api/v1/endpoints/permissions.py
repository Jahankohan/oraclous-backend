"""
Permission management endpoints — ORA-48 / ORA-52 ReBAC implementation.

Endpoints:
    GET    /graphs/{graphId}/members                 list members with active roles
    POST   /graphs/{graphId}/members                 grant a role to a user
    DELETE /graphs/{graphId}/members/{userId}        revoke a user's role
    GET    /graphs/{graphId}/subgraphs               list subgraph partitions
    POST   /graphs/{graphId}/subgraphs               create a subgraph partition

Architecture rules enforced:
    #4  — every Cypher query scoped to graph_id (enforced inside rebac_service)
    #5  — AsyncDriver used throughout (FastAPI endpoint)
    #8  — validate at the boundary only (request models handle this)
"""
from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.dependencies import get_current_user_id, verify_graph_access
from app.core.neo4j_client import neo4j_client
from app.schemas.permission_schemas import (
    GraphMemberResponse,
    GraphMembersResponse,
    RoleGrantRequest,
    SubGraphCreate,
    SubGraphListResponse,
    SubGraphResponse,
)
from app.services.rebac_service import rebac_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_VALID_ROLES = {"owner", "admin", "editor", "viewer", "restricted_viewer"}


def _require_driver():
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return neo4j_client.async_driver


# ── Members ────────────────────────────────────────────────────────────────

@router.get(
    "/graphs/{graphId}/members",
    response_model=GraphMembersResponse,
    summary="List graph members",
    description="Return all users with an active role on this graph.",
)
async def list_members(
    graphId: str = Path(..., description="Graph ID"),
    user_id: str = Depends(get_current_user_id),
):
    """Requires at least graph:read permission (enforced by verify_graph_access)."""
    await verify_graph_access(graphId, "read", user_id)
    driver = _require_driver()
    members = await rebac_service.list_graph_members(driver, graphId)
    return GraphMembersResponse(
        graph_id=graphId,
        members=[GraphMemberResponse(**m) for m in members],
    )


@router.post(
    "/graphs/{graphId}/members",
    response_model=GraphMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Grant a role to a user",
    description="Grant a named role to a user on this graph. Requires graph:manage_access (admin).",
)
async def grant_member_role(
    request: RoleGrantRequest,
    graphId: str = Path(..., description="Graph ID"),
    user_id: str = Depends(get_current_user_id),
):
    if request.role not in _VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role '{request.role}'. Must be one of: {sorted(_VALID_ROLES)}",
        )
    await verify_graph_access(graphId, "admin", user_id)
    driver = _require_driver()

    try:
        await rebac_service.grant_role(
            driver=driver,
            graph_id=graphId,
            target_user_id=request.user_id,
            role_name=request.role,
            granted_by=user_id,
            expires_at=request.expires_at,
            email=request.email,
        )
    except Exception as exc:
        logger.error(f"grant_role failed for graph {graphId}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to grant role",
        )

    return GraphMemberResponse(
        user_id=request.user_id,
        email=request.email,
        role=request.role,
        granted_at=None,
        expires_at=request.expires_at,
    )


@router.delete(
    "/graphs/{graphId}/members/{targetUserId}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a user's role",
    description="Soft-revoke a user's role on this graph. Requires admin access.",
)
async def revoke_member_role(
    graphId: str = Path(..., description="Graph ID"),
    targetUserId: str = Path(..., description="User ID to revoke"),
    role: str = "viewer",
    user_id: str = Depends(get_current_user_id),
):
    """
    role query param: the specific role to revoke (default: viewer).
    To revoke all roles, call this endpoint once per role held.
    """
    if role not in _VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role '{role}'. Must be one of: {sorted(_VALID_ROLES)}",
        )
    await verify_graph_access(graphId, "admin", user_id)
    driver = _require_driver()

    count = await rebac_service.revoke_role(
        driver=driver,
        graph_id=graphId,
        target_user_id=targetUserId,
        role_name=role,
    )
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active role assignment found",
        )


# ── SubGraphs ──────────────────────────────────────────────────────────────

@router.get(
    "/graphs/{graphId}/subgraphs",
    response_model=SubGraphListResponse,
    summary="List subgraph partitions",
    description="Return all named subgraph partitions within this graph.",
)
async def list_subgraphs(
    graphId: str = Path(..., description="Graph ID"),
    user_id: str = Depends(get_current_user_id),
):
    await verify_graph_access(graphId, "read", user_id)
    driver = _require_driver()
    subgraphs = await rebac_service.list_subgraphs(driver, graphId)
    return SubGraphListResponse(
        graph_id=graphId,
        subgraphs=[SubGraphResponse(**s) for s in subgraphs],
    )


@router.post(
    "/graphs/{graphId}/subgraphs",
    response_model=SubGraphResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a subgraph partition",
    description="Create a named partition within the graph for fine-grained access control. Requires admin.",
)
async def create_subgraph(
    request: SubGraphCreate,
    graphId: str = Path(..., description="Graph ID"),
    user_id: str = Depends(get_current_user_id),
):
    await verify_graph_access(graphId, "admin", user_id)
    driver = _require_driver()

    try:
        result = await rebac_service.create_subgraph(
            driver=driver,
            graph_id=graphId,
            name=request.name,
            description=request.description,
            created_by=user_id,
        )
    except Exception as exc:
        logger.error(f"create_subgraph failed for graph {graphId}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subgraph",
        )

    return SubGraphResponse(**result)
