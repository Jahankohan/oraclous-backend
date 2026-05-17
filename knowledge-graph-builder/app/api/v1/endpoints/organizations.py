"""Organization API endpoints (TASK-201).

Makes :Organization a first-class entity with CRUD over the PostgreSQL
``organizations`` table plus a synced Neo4j ``:Organization`` node.

Routes:
  POST   /organizations               — create an org (current user becomes owner)
  GET    /organizations               — list orgs the current user belongs to
  GET    /organizations/{org_id}      — get one org (any member)
  PATCH  /organizations/{org_id}      — update one org (owner-only)
  GET    /organizations/{org_id}/graphs — list the org's subgraphs (owner-only)
  POST   /organizations/{org_id}/members — register an org member (owner-only)
  GET    /organizations/{org_id}/members — list org members (owner-only)
  DELETE /organizations/{org_id}/members/{user_id} — remove a member (owner-only)
  POST   /organizations/{org_id}/members/{user_id}/subgraph-grants
                                          — grant a member subgraph access (owner-only)
  GET    /organizations/{org_id}/agents   — list the org's agents (owner-only)
  POST   /organizations/{org_id}/agents/{agent_id}/subgraph-grants
                                          — grant an agent subgraph access (owner-only)
  GET    /organizations/{org_id}/agents/{agent_id}/subgraph-grants
                                          — list an agent's subgraph grants (owner-only)
  DELETE /organizations/{org_id}/agents/{agent_id}/subgraph-grants/{graph_id}
                                          — revoke an agent's subgraph grant (owner-only)

Access control: the org list/read routes (GET /organizations and GET
/organizations/{org_id}) are member-readable — the caller must hold a
``:User-[:BELONGS_TO]->:Organization`` edge (TASK-208). Every mutating route
and the member/agent/subgraph-grant routes remain owner-only: the current user
must equal the org's ``owner_user_id``. The not-owner / not-member case returns
404 (not 403) so org existence is never leaked. The member registry
(TASK-203 Part 1) and agent registry (TASK-203 Part 2) are purely additive —
they do not modify existing per-graph member/agent endpoints or ReBAC code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database
from app.core.dependencies import get_neo4j_async_driver
from app.models.organization import Organization
from app.schemas.graph_schemas import GraphResponse
from app.schemas.org_agent_schemas import (
    AgentGrantResponse,
    AgentSubgraphGrantSpec,
    OrgAgentResponse,
)
from app.schemas.org_member_schemas import (
    OrgMemberCreate,
    OrgMemberResponse,
    SubgraphGrantSpec,
)
from app.schemas.organization_schemas import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)
from app.services import (
    org_agent_service,
    org_member_service,
    organization_service,
)

router = APIRouter()


def _neo4j_datetime_to_python(value: Any) -> datetime:
    """Coerce a Neo4j DateTime (or ISO string) into a Python datetime."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        return value.to_native()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _serialize_graph(graph: dict) -> GraphResponse:
    """Map a :Graph:__Platform__ node dict to the API GraphResponse."""
    return GraphResponse(
        id=UUID(graph["graph_id"]),
        name=graph["name"],
        description=graph.get("description", ""),
        user_id=UUID(graph["user_id"]),
        org_id=graph.get("org_id"),
        created_at=_neo4j_datetime_to_python(graph["created_at"]),
        updated_at=_neo4j_datetime_to_python(graph["updated_at"]),
        node_count=graph.get("node_count", 0),
        relationship_count=graph.get("relationship_count", 0),
        status=graph.get("status", "active"),
        schema_config={},
        federatable=graph.get("federatable", False),
        federation_group=graph.get("federation_group"),
    )


def _serialize(org: Organization, org_role: str | None = None) -> OrganizationResponse:
    """Map a SQL Organization row to the API response (ISO-8601 datetimes).

    ``org_role`` is the calling user's role on this org (owner|admin|member),
    or None when the caller's role is not resolved for the route.
    """
    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        description=org.description or "",
        owner_user_id=str(org.owner_user_id),
        settings=org.settings or {},
        status=org.status,
        created_at=org.created_at.isoformat() if org.created_at else "",
        updated_at=org.updated_at.isoformat() if org.updated_at else "",
        org_role=org_role,
    )


@router.post(
    "/organizations",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization(
    body: OrganizationCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrganizationResponse:
    """Create an organization. The current user becomes its owner."""
    org = await organization_service.create_organization(
        db,
        driver,
        name=body.name,
        description=body.description,
        settings=body.settings,
        owner_user_id=user_id,
    )
    return _serialize(org, org_role="owner")


@router.get(
    "/organizations",
    response_model=list[OrganizationResponse],
)
async def list_organizations(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[OrganizationResponse]:
    """List every organization the current user belongs to (any role).

    Each response carries ``org_role`` — the caller's role on that org.
    """
    orgs = await organization_service.list_user_organizations(db, driver, user_id)
    return [_serialize(o, org_role=role) for o, role in orgs]


@router.get(
    "/organizations/{org_id}",
    response_model=OrganizationResponse,
)
async def get_organization(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrganizationResponse:
    """Get a single organization. Readable by any member (owner|admin|member).

    404 if the org is missing OR the caller holds no BELONGS_TO edge to it —
    existence is never leaked to non-members. The response carries the
    caller's ``org_role``.
    """
    org = await organization_service.get_organization(db, org_id)
    role = await organization_service.get_user_org_role(driver, org_id, user_id)
    if org is None or role is None:
        # Never leak existence to non-members.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return _serialize(org, org_role=role)


@router.patch(
    "/organizations/{org_id}",
    response_model=OrganizationResponse,
)
async def update_organization(
    org_id: str,
    body: OrganizationUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrganizationResponse:
    """Update an organization. Owner-only — 404 if missing or not owned."""
    org = await organization_service.get_organization(db, org_id)
    if org is None or str(org.owner_user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    updated = await organization_service.update_organization(
        db,
        driver,
        org_id,
        name=body.name,
        description=body.description,
        settings=body.settings,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return _serialize(updated, org_role="owner")


@router.get(
    "/organizations/{org_id}/graphs",
    response_model=list[GraphResponse],
)
async def list_organization_graphs(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[GraphResponse]:
    """List the knowledge graphs (subgraphs) owned by an organization.

    Owner-only — 404 if the org is missing or not owned by the caller, so
    org existence is never leaked. Soft-deleted graphs are excluded (TASK-202).
    """
    org = await organization_service.get_organization(db, org_id)
    if org is None or str(org.owner_user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    graphs = await organization_service.list_org_graphs(driver, org_id)
    return [_serialize_graph(g) for g in graphs]


# ── Member registry (TASK-203 Part 1) ──────────────────────────────────────


async def _require_org_owner(
    db: AsyncSession, org_id: str, user_id: str
) -> Organization:
    """Return the org if *user_id* owns it, else raise 404.

    Mirrors the owner-only guard used by the org GET/PATCH/graphs routes:
    the not-owner and missing cases both yield 404 so existence is never
    leaked to non-owners.
    """
    org = await organization_service.get_organization(db, org_id)
    if org is None or str(org.owner_user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return org


@router.post(
    "/organizations/{org_id}/members",
    response_model=OrgMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_organization_member(
    org_id: str,
    body: OrgMemberCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrgMemberResponse:
    """Register a member at the organization. Owner-only.

    If ``subgraph_grants`` is supplied, the member is also granted the given
    ReBAC role on the named subgraphs (or all of them) in the same call.
    """
    await _require_org_owner(db, org_id, user_id)

    try:
        member = await org_member_service.add_member(
            driver,
            org_id=org_id,
            user_id=body.user_id,
            org_role=body.org_role,
            email=body.email,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    grants: list[dict] = []
    if body.subgraph_grants is not None:
        try:
            grants = await org_member_service.grant_member_subgraphs(
                driver,
                org_id=org_id,
                user_id=body.user_id,
                role=body.subgraph_grants.role,
                graph_ids=body.subgraph_grants.graph_ids,
                granted_by=user_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    return OrgMemberResponse(
        user_id=member["user_id"],
        email=member["email"],
        org_role=member["org_role"],
        since=member["since"],
        subgraph_grants=grants,
    )


@router.get(
    "/organizations/{org_id}/members",
    response_model=list[OrgMemberResponse],
)
async def list_organization_members(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[OrgMemberResponse]:
    """List the organization's members with their org_role and subgraph grants.

    Owner-only — 404 if the org is missing or not owned by the caller.
    """
    await _require_org_owner(db, org_id, user_id)

    members = await org_member_service.list_members(driver, org_id)
    return [
        OrgMemberResponse(
            user_id=m["user_id"],
            email=m["email"],
            org_role=m["org_role"],
            since=m["since"],
            subgraph_grants=m["subgraph_grants"],
        )
        for m in members
    ]


@router.delete(
    "/organizations/{org_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def remove_organization_member(
    org_id: str,
    user_id: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> None:
    """Remove a member from the organization. Owner-only.

    Returns 409 if it would remove the org's last owner; 204 otherwise.
    """
    await _require_org_owner(db, org_id, current_user_id)

    try:
        await org_member_service.remove_member(driver, org_id, user_id)
    except ValueError as exc:
        # "not a member" and "only owner" both surface here; treat the
        # last-owner refusal as a conflict, a missing membership as 404.
        message = str(exc)
        if "only owner" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=message
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=message
        ) from exc


@router.post(
    "/organizations/{org_id}/members/{user_id}/subgraph-grants",
    response_model=list[dict],
)
async def grant_member_subgraph_access(
    org_id: str,
    user_id: str,
    body: SubgraphGrantSpec,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[dict]:
    """Grant an existing member a ReBAC role on the org's subgraphs. Owner-only.

    Returns the list of ``{graph_id, role}`` actually granted.
    """
    await _require_org_owner(db, org_id, current_user_id)

    try:
        return await org_member_service.grant_member_subgraphs(
            driver,
            org_id=org_id,
            user_id=user_id,
            role=body.role,
            graph_ids=body.graph_ids,
            granted_by=current_user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


# ── Agent registry (TASK-203 Part 2) ────────────────────────────────────────


@router.get(
    "/organizations/{org_id}/agents",
    response_model=list[OrgAgentResponse],
)
async def list_organization_agents(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[OrgAgentResponse]:
    """List the agents that belong to the organization. Owner-only.

    404 if the org is missing or not owned by the caller — existence is masked.
    """
    await _require_org_owner(db, org_id, user_id)

    agents = await org_agent_service.list_org_agents(driver, org_id)
    return [
        OrgAgentResponse(
            agent_id=a["agent_id"],
            org_id=a["org_id"],
            graph_id=a["graph_id"],
            name=a["name"],
            description=a.get("description", ""),
            active=a.get("deactivated_at") is None,
        )
        for a in agents
    ]


@router.post(
    "/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
    response_model=list[AgentGrantResponse],
)
async def grant_agent_subgraph_access(
    org_id: str,
    agent_id: str,
    body: AgentSubgraphGrantSpec,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[AgentGrantResponse]:
    """Grant an org agent ``CAN_ACCESS`` on the org's subgraphs. Owner-only.

    Returns the list of ``{graph_id, level}`` actually granted. A bad level,
    an agent that does not belong to the org, or a graph the org does not own
    all surface as 400.
    """
    await _require_org_owner(db, org_id, user_id)

    try:
        granted = await org_agent_service.grant_agent_subgraphs(
            driver,
            org_id=org_id,
            agent_id=agent_id,
            level=body.level,
            graph_ids=body.graph_ids,
            granted_by=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return [AgentGrantResponse(**g) for g in granted]


@router.get(
    "/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
    response_model=list[AgentGrantResponse],
)
async def list_agent_subgraph_grants(
    org_id: str,
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[AgentGrantResponse]:
    """List an org agent's ``CAN_ACCESS`` subgraph grants. Owner-only.

    An agent that does not belong to the org surfaces as 400.
    """
    await _require_org_owner(db, org_id, user_id)

    try:
        grants = await org_agent_service.list_agent_grants(driver, org_id, agent_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return [AgentGrantResponse(**g) for g in grants]


@router.delete(
    "/organizations/{org_id}/agents/{agent_id}/subgraph-grants/{graph_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def revoke_agent_subgraph_grant(
    org_id: str,
    agent_id: str,
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> None:
    """Revoke an org agent's ``CAN_ACCESS`` grant on a subgraph. Owner-only.

    204 on success; 404 if there was no such grant. An agent that does not
    belong to the org surfaces as 400.
    """
    await _require_org_owner(db, org_id, user_id)

    try:
        deleted = await org_agent_service.revoke_agent_grant(
            driver, org_id, agent_id, graph_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such subgraph grant for this agent",
        )
