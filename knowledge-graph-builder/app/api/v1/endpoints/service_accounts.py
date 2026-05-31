"""Service Account API endpoints.

Implements 9 endpoints from the ORA-81 spec:
  POST   /graphs/{graphId}/service-accounts        — create SA scoped to graph
  GET    /graphs/{graphId}/service-accounts        — list SAs (admin)
  GET    /service-accounts/{accountId}             — get SA metadata
  PATCH  /service-accounts/{accountId}             — update name/description
  DELETE /service-accounts/{accountId}             — revoke SA (soft delete)
  POST   /service-accounts/{accountId}/rotate-key  — rotate API key
  POST   /service-accounts/{accountId}/graph-grants           — add cross-graph grant
  GET    /service-accounts/{accountId}/graph-grants           — list grants
  DELETE /service-accounts/{accountId}/graph-grants/{graphId} — revoke grant

Security: every endpoint verifies the caller's tenant matches the SA's tenant.
Error responses never reveal existence of inaccessible graphs (always 403, not 404).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app.api.dependencies import (
    get_current_user,
    get_current_user_id,
    verify_graph_access,
)
from app.core.neo4j_client import neo4j_client
from app.schemas.service_account_schemas import (
    AddGraphGrantRequest,
    AuditLogListResponse,
    CreateServiceAccountRequest,
    GraphGrantResponse,
    SecurityAuditLogEntry,
    ServiceAccountCreatedResponse,
    ServiceAccountResponse,
    ServiceAccountRotatedResponse,
    UpdateServiceAccountRequest,
)
from app.services.service_account_service import service_account_service

router = APIRouter()


def _require_neo4j():
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return neo4j_client.async_driver


async def _resolve_tenant_id(current_user: dict, driver) -> str:
    """Resolve the caller's tenant_id (org ID).

    Service account JWTs always carry tenant_id — return it directly.
    Human user JWTs may omit tenant_id; fall back to a Neo4j lookup via the
    BELONGS_TO edge.  Raises HTTP 400 if the user has no org membership.
    """
    # SA principals always carry tenant_id in their JWT
    if current_user.get("principal_type") == "service_account":
        return current_user.get("tenant_id", "")

    # JWT already includes tenant_id (once auth-service propagates it)
    if current_user.get("tenant_id"):
        return current_user["tenant_id"]

    # Resolve from Neo4j: follow the user's BELONGS_TO → Organization edge
    user_id = current_user.get("id", "")
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (u:User:__Platform__ {user_id: $user_id, graph_id: "__system__"})
                  -[:BELONGS_TO]->(org:Organization {graph_id: "__system__"})
            RETURN org.org_id AS org_id
            LIMIT 1
            """,
            {"user_id": user_id},
        )
        record = await result.single()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "User has no organization membership — "
                "service account management requires an organization"
            ),
        )
    return record["org_id"]


# ── Graph-scoped SA management ─────────────────────────────────────────────


@router.post(
    "/graphs/{graphId}/service-accounts",
    response_model=ServiceAccountCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_service_account(
    graphId: str,
    body: CreateServiceAccountRequest,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Create a service account scoped to graphId. Caller must have writer|admin."""
    driver = _require_neo4j()
    await verify_graph_access(graphId, "write", user_id)
    tenant_id = await _resolve_tenant_id(current_user, driver)

    try:
        sa = await service_account_service.create_service_account(
            driver=driver,
            tenant_id=tenant_id,
            graph_id=graphId,
            created_by_user_id=user_id,
            name=body.name,
            description=body.description,
            level=body.level,
            expires_at=body.expires_at,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from None

    return ServiceAccountCreatedResponse(**sa)


@router.get(
    "/graphs/{graphId}/service-accounts",
    response_model=list[ServiceAccountResponse],
)
async def list_service_accounts(
    graphId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """List service accounts for graphId. Caller must have admin."""
    driver = _require_neo4j()
    await verify_graph_access(graphId, "admin", user_id)
    tenant_id = await _resolve_tenant_id(current_user, driver)

    rows = await service_account_service.list_service_accounts(
        driver, graphId, tenant_id
    )
    return [ServiceAccountResponse(**r) for r in rows]


# ── SA instance management ─────────────────────────────────────────────────


@router.get(
    "/service-accounts/{accountId}",
    response_model=ServiceAccountResponse,
)
async def get_service_account(
    accountId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Get SA metadata. Caller must own or admin the SA's home graph."""
    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        # Never reveal graph existence to unauthorized callers
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Verify caller can access the SA's home graph
    await verify_graph_access(sa["home_graph_id"], "read", user_id)
    return ServiceAccountResponse(**sa)


@router.patch(
    "/service-accounts/{accountId}",
    response_model=ServiceAccountResponse,
)
async def update_service_account(
    accountId: str,
    body: UpdateServiceAccountRequest,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Update name/description. Caller must have write access to the SA's home graph."""
    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    await verify_graph_access(sa["home_graph_id"], "write", user_id)

    updated = await service_account_service.update_service_account(
        driver, accountId, tenant_id, name=body.name, description=body.description
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )
    return ServiceAccountResponse(**updated)


@router.delete(
    "/service-accounts/{accountId}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def revoke_service_account(
    accountId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Revoke (soft-delete) a service account. Caller must admin the SA's home graph."""
    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    await verify_graph_access(sa["home_graph_id"], "admin", user_id)

    revoked = await service_account_service.revoke_service_account(
        driver, accountId, tenant_id, actor_user_id=user_id
    )
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )


@router.post(
    "/service-accounts/{accountId}/rotate-key",
    response_model=ServiceAccountRotatedResponse,
)
async def rotate_key(
    accountId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Rotate the API key — old key is invalidated immediately."""
    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    await verify_graph_access(sa["home_graph_id"], "write", user_id)

    try:
        result = await service_account_service.rotate_key(
            driver, accountId, tenant_id, user_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from None

    return ServiceAccountRotatedResponse(**result)


# ── Graph grants management ────────────────────────────────────────────────


@router.post(
    "/service-accounts/{accountId}/graph-grants",
    response_model=GraphGrantResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_graph_grant(
    accountId: str,
    body: AddGraphGrantRequest,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Grant cross-graph access. Caller must admin the TARGET graph.

    Service accounts CANNOT grant themselves additional access (no self-elevation).
    """
    # Service accounts cannot call grant endpoints
    if current_user.get("principal_type") == "service_account":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Service accounts cannot grant graph access",
        )

    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    # Verify SA exists in this tenant (403 if not — never 404)
    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Caller must admin the target graph
    await verify_graph_access(body.graph_id, "admin", user_id)

    try:
        grant = await service_account_service.add_graph_grant(
            driver=driver,
            sa_id=accountId,
            tenant_id=tenant_id,
            graph_id=body.graph_id,
            level=body.level,
            granted_by_user_id=user_id,
            expires_at=body.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from None

    return GraphGrantResponse(**grant)


@router.get(
    "/service-accounts/{accountId}/graph-grants",
    response_model=list[GraphGrantResponse],
)
async def list_graph_grants(
    accountId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """List all graph grants for a service account."""
    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    await verify_graph_access(sa["home_graph_id"], "read", user_id)

    grants = await service_account_service.list_graph_grants(
        driver, accountId, tenant_id
    )
    return [GraphGrantResponse(**g) for g in grants]


@router.delete(
    "/service-accounts/{accountId}/graph-grants/{graphId}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_graph_grant(
    accountId: str,
    graphId: str,
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Revoke a specific graph grant. Caller must admin the target graph."""
    # Service accounts cannot revoke grants
    if current_user.get("principal_type") == "service_account":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Service accounts cannot modify graph grants",
        )

    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    await verify_graph_access(graphId, "admin", user_id)

    deleted = await service_account_service.delete_graph_grant(
        driver, accountId, tenant_id, graphId
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )


# ── Audit log ──────────────────────────────────────────────────────────────


@router.get(
    "/service-accounts/{accountId}/audit-log",
    response_model=AuditLogListResponse,
)
async def get_sa_audit_log(
    accountId: str,
    limit: int = Query(default=50, ge=1, le=100),
    before: str | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
):
    """Return the security audit log for a service account (reverse-chronological).

    Auth rules (all failures → 403, never 404):
    1. service_account principals are denied unconditionally.
    2. tenant_id from JWT must match the SA's tenant_id.
    3. Caller must have admin on the SA's home_graph_id.

    Query params:
      limit — 1-100, default 50 (422 if >100 sent via non-Query path, Query enforces it)
      before — ISO-8601 cursor; only entries with timestamp < before are returned (422 if
               unparseable)
    """
    # Rule 1: service_account principals may not call this endpoint
    if current_user.get("principal_type") == "service_account":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    driver = _require_neo4j()
    tenant_id = await _resolve_tenant_id(current_user, driver)

    # Fetch SA (tenant-isolated) — 403 if not found
    sa = await service_account_service.get_service_account(driver, accountId, tenant_id)
    if not sa:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Rule 2: JWT tenant must match SA tenant (already enforced by tenant-isolated lookup,
    # but explicitly re-check for clarity)
    if sa.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    # Rule 3: caller must have admin on the SA's home graph
    await verify_graph_access(sa["home_graph_id"], "admin", user_id)

    # Parse optional before cursor
    before_dt: datetime | None = None
    if before is not None:
        try:
            before_dt = datetime.fromisoformat(before)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="before must be a valid ISO-8601 datetime string",
            )

    # Fetch limit+1 to determine has_more
    fetch_limit = limit + 1

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:SecurityAuditLog {sa_id: $sa_id, tenant_id: $tenant_id})
            WHERE $before IS NULL OR a.timestamp < datetime($before)
            WITH a
            ORDER BY a.timestamp DESC
            LIMIT $fetch_limit
            RETURN a.audit_log_id   AS audit_log_id,
                   a.event_type     AS event_type,
                   a.sa_id          AS sa_id,
                   a.actor_user_id  AS actor_user_id,
                   a.home_graph_id  AS home_graph_id,
                   a.tenant_id      AS tenant_id,
                   a.key_prefix     AS key_prefix,
                   toString(a.timestamp) AS timestamp
            """,
            {
                "sa_id": accountId,
                "tenant_id": tenant_id,
                "before": before,
                "fetch_limit": fetch_limit,
            },
        )
        rows = await result.data()

    has_more = len(rows) > limit
    items = rows[:limit]

    return AuditLogListResponse(
        items=[SecurityAuditLogEntry(**r) for r in items],
        total_count=len(items),
        has_more=has_more,
    )
