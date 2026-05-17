"""Organization API endpoints (TASK-201).

Makes :Organization a first-class entity with CRUD over the PostgreSQL
``organizations`` table plus a synced Neo4j ``:Organization`` node.

Routes:
  POST   /organizations               — create an org (current user becomes owner)
  GET    /organizations               — list orgs the current user belongs to
  GET    /organizations/{org_id}      — get one org (any member)
  PATCH  /organizations/{org_id}      — update one org (owner-only)
  GET    /organizations/{org_id}/graphs — list the org's subgraphs (any member)
  POST   /organizations/{org_id}/members — register an org member (owner-only)
  GET    /organizations/{org_id}/members — list org members (any member)
  DELETE /organizations/{org_id}/members/{user_id} — remove a member (owner-only)
  POST   /organizations/{org_id}/members/{user_id}/subgraph-grants
                                          — grant a member subgraph access (owner-only)
  GET    /organizations/{org_id}/agents   — list the org's agents (any member)
  POST   /organizations/{org_id}/agents/{agent_id}/subgraph-grants
                                          — grant an agent subgraph access (owner-only)
  GET    /organizations/{org_id}/agents/{agent_id}/subgraph-grants
                                          — list an agent's subgraph grants (owner-only)
  DELETE /organizations/{org_id}/agents/{agent_id}/subgraph-grants/{graph_id}
                                          — revoke an agent's subgraph grant (owner-only)

Access control: the org read routes — GET /organizations, GET
/organizations/{org_id} (TASK-208), and the three content listings GET
.../graphs, .../members, .../agents (TASK-209) — are member-readable: the
caller must hold a ``:User-[:BELONGS_TO]->:Organization`` edge. The graphs and
agents listings are additionally scoped for non-owner members to the subgraphs
they hold an active ReBAC ``HAS_ROLE`` on; the owner always sees everything.
Every mutating route and every subgraph-grant route remains owner-only: the
current user must equal the org's ``owner_user_id``. The not-owner / not-member
case returns 404 (not 403) so org existence is never leaked. The member registry
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
from app.core.config import settings
from app.core.dependencies import get_neo4j_async_driver
from app.core.logging import get_logger
from app.models.organization import Organization, OrgInvitation
from app.schemas.graph_schemas import GraphResponse
from app.schemas.org_agent_schemas import (
    AgentGrantResponse,
    AgentSubgraphGrantSpec,
    OrgAgentResponse,
)
from app.schemas.org_invitation_schemas import (
    OrgInvitationAcceptResponse,
    OrgInvitationCreate,
    OrgInvitationResponse,
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
    email_service,
    org_agent_service,
    org_invitation_service,
    org_member_service,
    organization_service,
)

router = APIRouter()
logger = get_logger(__name__)


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
    """List the knowledge graphs (subgraphs) of an organization.

    Member-readable (TASK-209): 404 if the org is missing or the caller holds
    no ``BELONGS_TO`` edge, so org existence is never leaked. The org owner
    sees every subgraph; a non-owner member sees only the subgraphs they hold
    an active ReBAC ``HAS_ROLE`` on. Soft-deleted graphs are excluded (TASK-202).
    """
    org = await organization_service.get_organization(db, org_id)
    org_role = await organization_service.get_user_org_role(driver, org_id, user_id)
    if org is None or org_role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    # Org owners and admins see every subgraph the org owns; a plain member
    # sees only the subgraphs they hold an active ReBAC role on.
    if org_role in ("owner", "admin"):
        graphs = await organization_service.list_org_graphs(driver, org_id)
    else:
        graphs = await organization_service.list_member_org_graphs(
            driver, org_id, user_id
        )
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


async def _require_org_member(
    db: AsyncSession, driver: AsyncDriver, org_id: str, user_id: str
) -> tuple[Organization, str]:
    """Return ``(org, org_role)`` if *user_id* belongs to the org, else 404.

    The member-readable counterpart of :func:`_require_org_owner` (TASK-209):
    used by the org content-listing GET routes. Missing org and not-a-member
    both yield 404 so existence is never leaked to non-members. ``org_role`` is
    the caller's ``BELONGS_TO`` role (``owner|admin|member``).
    """
    org = await organization_service.get_organization(db, org_id)
    org_role = await organization_service.get_user_org_role(driver, org_id, user_id)
    if org is None or org_role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return org, org_role


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

    Member-readable (TASK-209): any member sees the full org roster, read-only.
    404 if the org is missing or the caller holds no ``BELONGS_TO`` edge.
    """
    await _require_org_member(db, driver, org_id, user_id)

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


# ── Member invitations ──────────────────────────────────────────────────────


async def _require_org_admin_or_owner(
    db: AsyncSession, driver: AsyncDriver, org_id: str, user_id: str
) -> tuple[Organization, str]:
    """Return ``(org, org_role)`` if *user_id* is the org's owner or an admin,
    else raise 404.

    The invitation routes are owner-or-admin (not owner-only): the caller must
    hold a ``BELONGS_TO`` edge with org_role ``owner`` or ``admin``. Missing
    org and insufficient role both yield 404 so existence is never leaked.
    """
    org = await organization_service.get_organization(db, org_id)
    org_role = await organization_service.get_user_org_role(driver, org_id, user_id)
    if org is None or org_role not in ("owner", "admin"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return org, org_role


def _serialize_invitation(
    inv: OrgInvitation,
    *,
    invite_url: str | None = None,
    email_sent: bool | None = None,
) -> OrgInvitationResponse:
    """Map an OrgInvitation row to the API response (ISO-8601 datetimes).

    ``invite_url`` / ``email_sent`` are set only on the create response.
    """
    return OrgInvitationResponse(
        id=str(inv.id),
        org_id=str(inv.org_id),
        email=inv.email,
        org_role=inv.org_role,
        status=inv.status,
        invited_by_user_id=str(inv.invited_by_user_id),
        accepted_by_user_id=(
            str(inv.accepted_by_user_id) if inv.accepted_by_user_id else None
        ),
        created_at=inv.created_at.isoformat() if inv.created_at else "",
        expires_at=inv.expires_at.isoformat() if inv.expires_at else "",
        accepted_at=inv.accepted_at.isoformat() if inv.accepted_at else None,
        invite_url=invite_url,
        email_sent=email_sent,
    )


@router.post(
    "/organizations/{org_id}/invitations",
    response_model=OrgInvitationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization_invitation(
    org_id: str,
    body: OrgInvitationCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrgInvitationResponse:
    """Invite someone by email to join the organization. Owner or admin only.

    Creates a pending invitation and emails the invitee an accept link. If
    SMTP is not configured or the send fails, the invitation is still created
    (``email_sent: false``) — the link in ``invite_url`` is always usable.
    """
    org, _ = await _require_org_admin_or_owner(db, driver, org_id, user_id)

    try:
        invitation = await org_invitation_service.create_invitation(
            db,
            org_id=org_id,
            email=body.email,
            org_role=body.org_role,
            invited_by_user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    invite_url = f"{settings.INVITE_BASE_URL.rstrip('/')}/{invitation.token}"
    expiry_days = max(1, settings.INVITE_TTL_HOURS // 24)
    email_sent = False
    if email_service.is_configured():
        try:
            await email_service.send_email(
                to=invitation.email,
                subject=f'Invitation to join "{org.name}" on Oraclous',
                body_text=(
                    f'You have been invited to join the organization "{org.name}" '
                    f"on Oraclous as {invitation.org_role}.\n\n"
                    f"Accept the invitation:\n{invite_url}\n\n"
                    f"This link expires in {expiry_days} days."
                ),
                body_html=(
                    f"<p>You have been invited to join the organization "
                    f"<strong>{org.name}</strong> on Oraclous as "
                    f"<strong>{invitation.org_role}</strong>.</p>"
                    f'<p><a href="{invite_url}">Accept the invitation</a></p>'
                    f"<p>This link expires in {expiry_days} days.</p>"
                ),
            )
            email_sent = True
        except Exception as exc:  # noqa: BLE001 — send failure must not 500
            logger.warning(
                "invitation %s created but email send failed: %s",
                invitation.id,
                exc,
            )

    return _serialize_invitation(
        invitation, invite_url=invite_url, email_sent=email_sent
    )


@router.get(
    "/organizations/{org_id}/invitations",
    response_model=list[OrgInvitationResponse],
)
async def list_organization_invitations(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> list[OrgInvitationResponse]:
    """List the organization's invitations (all statuses). Owner or admin only."""
    await _require_org_admin_or_owner(db, driver, org_id, user_id)
    invitations = await org_invitation_service.list_invitations(db, org_id)
    return [_serialize_invitation(inv) for inv in invitations]


@router.delete(
    "/organizations/{org_id}/invitations/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def revoke_organization_invitation(
    org_id: str,
    invitation_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> None:
    """Revoke a pending invitation. Owner or admin only.

    204 on success; 404 if there is no pending invitation with that id in
    this org (already accepted/revoked/expired, or absent).
    """
    await _require_org_admin_or_owner(db, driver, org_id, user_id)
    revoked = await org_invitation_service.revoke_invitation(db, org_id, invitation_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending invitation with that id",
        )


@router.post(
    "/invitations/{token}/accept",
    response_model=OrgInvitationAcceptResponse,
)
async def accept_organization_invitation(
    token: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrgInvitationAcceptResponse:
    """Accept an invitation and join the organization.

    Any authenticated user may call this — the opaque token (delivered to the
    invited email) is the authorization. The caller is registered as an org
    member with the invited role. 400 if the token is unknown, already
    used/revoked, or expired.
    """
    try:
        result = await org_invitation_service.accept_invitation(
            db, driver, token=token, accepting_user_id=user_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return OrgInvitationAcceptResponse(**result)


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
    """List the agents that belong to the organization.

    Member-readable (TASK-209): 404 if the org is missing or the caller holds
    no ``BELONGS_TO`` edge — existence is masked. The org owner sees every
    agent; a non-owner member sees only agents that operate on a subgraph they
    can access (the agent's home graph, or a ``CAN_ACCESS``-granted graph, on
    which the caller holds an active ``HAS_ROLE``).
    """
    _, org_role = await _require_org_member(db, driver, org_id, user_id)

    # Org owners and admins see every agent; a plain member sees only agents
    # operating on a subgraph they can access.
    if org_role in ("owner", "admin"):
        agents = await org_agent_service.list_org_agents(driver, org_id)
    else:
        agents = await org_agent_service.list_member_org_agents(driver, org_id, user_id)
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
