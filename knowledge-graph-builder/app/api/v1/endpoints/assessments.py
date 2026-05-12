"""Assessment substrate REST write endpoints (STORY-026, TASK-069).

Eight write endpoints that expose `AssessmentService` (TASK-068) over HTTP,
plus two heartbeat endpoints and the registry write endpoint
(`persist_registry_item`, per ADR-019).

Auth / scope:

-   Every endpoint authenticates via the existing `get_current_user` /
    `get_current_user_id` dependency stack (service-account JWT or user JWT).
-   `graph_id` is derived from the JWT principal's `home_graph_id` claim.
    The request body **never** supplies a `graph_id` that overrides the JWT
    claim — per ADR-010 §Scope Enforcer, an agent-supplied `graph_id` in the
    payload is ignored. The endpoint reaches into the Pydantic body and
    overwrites any nested `graph_id` (on `Subject`, `Finding`, etc.) with
    the principal-derived value before calling the service.
-   Catalog graph (`__assessments_catalog__`) writes are never accepted from
    callers; the service performs them internally for `:Source` dedup.
-   Registry `curated` writes require `admin` access to the `__registry__`
    catalog graph — checked via the same ReBAC primitive used everywhere else.

Per ADR-007 (MCP-First), these endpoints will be mirrored 1:1 by MCP wrappers
in SPRINT-002. The path shapes are deliberately resource-oriented (no verbs
except for the `:bulk` / `:finalize` / `:heartbeat` action suffixes that align
with Google AIP-136 and mirror the future MCP tool names).

Out of scope (per TASK-069):
- Read endpoints (SPRINT-002)
- MCP wrappers (SPRINT-002)
- SSE `tail_run` endpoint (SPRINT-002)
- Rate limiting (STORY-027, SPRINT-003)
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import (
    get_current_user,
    get_current_user_id,
    verify_graph_access,
)
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.assessment_schemas import (
    REGISTRY_CATALOG_GRAPH_ID,
    BulkResponse,
    CreateRunRequest,
    CreateRunResponse,
    FinalizeRunResponse,
    PersistDeliverableRequest,
    PersistFinalDocsRequest,
    RecordConflictRequest,
    RecordFindingBulkRequest,
    RecordUnresolvedQuestionRequest,
    RegistryItem,
    RegistryKind,
    UpdateModuleRunRequest,
)
from app.services.assessment_service import AssessmentService, RegistryOwnershipError

router = APIRouter()
logger = get_logger(__name__)


# ── Dependencies / helpers ────────────────────────────────────────────────────


def _assessment_service() -> AssessmentService:
    """Build an `AssessmentService` bound to the live async Neo4j driver.

    Exposed as a FastAPI dependency so integration tests can override it via
    `app.dependency_overrides[_assessment_service]` — see
    `tests/integration/test_assessments_endpoints.py`.
    """
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return AssessmentService(neo4j_client.async_driver)


def _principal_graph_id(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> str:
    """Resolve the caller's tenant `graph_id` from the JWT claim.

    Per ADR-010 §Scope Enforcer, the assessment write endpoints derive
    `graph_id` from the authenticated principal — never from the request
    body. Service-account JWTs carry `home_graph_id` in their claims (see
    `auth-service/app/core/jwt_handler.py`). User JWTs may also carry it
    once an explicit current-graph context has been bound.

    A principal without a usable `home_graph_id` is rejected with 400 so
    the caller gets a clear remediation: bind a graph context to the JWT.
    """
    graph_id = current_user.get("home_graph_id") or ""
    if not graph_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Principal has no home_graph_id claim. Assessment endpoints "
                "require a JWT bound to a tenant graph (service-account "
                "tokens always carry this; user tokens require an explicit "
                "graph context)."
            ),
        )
    return str(graph_id)


def _is_platform_admin(current_user: dict[str, Any]) -> bool:
    """Cheap principal-level platform-admin check.

    Used for the `curated` Registry writes per ADR-019 to short-circuit the
    Neo4j ACL lookup when the JWT itself does not declare an admin role.
    Returns True when the principal explicitly carries `platform_admin: true`
    or `role == 'platform-admin'`. Falls back to a ReBAC check on the
    `__registry__` catalog graph elsewhere in this module so this is purely
    an optimization, not a security boundary.
    """
    if current_user.get("platform_admin") is True:
        return True
    role = current_user.get("role") or ""
    return role.lower() in ("platform-admin", "platform_admin")


async def _verify_registry_curated_write(user_id: str) -> None:
    """Authorize a `curated` Registry write per ADR-019.

    `curated` items live in the `__registry__` catalog graph and are only
    writable by platform admins. We reuse the existing ReBAC primitive:
    admin-level access on `__registry__` IS the platform-admin role for
    this purpose, so we do not have to invent a new permission concept.
    """
    await verify_graph_access(REGISTRY_CATALOG_GRAPH_ID, "admin", user_id)


# ── /api/v1/assessments/runs ─────────────────────────────────────────────────


@router.post(
    "/assessments/runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an :AssessmentRun (plus all :ModuleRun rows in 'planned')",
    responses={
        400: {"description": "Invalid request (e.g. unknown template_slug)"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def create_run(
    body: CreateRunRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> CreateRunResponse:
    """Create a run + pre-create all :ModuleRun rows in `planned`.

    `graph_id` is taken from the JWT; any `graph_id` in `body.subject` is
    overwritten before the service call.
    """
    await verify_graph_access(graph_id, "write", user_id)

    # Force the body's nested graph_id to the JWT-derived value. The service
    # honors `body.subject.slug` for dedup, so this normalization is safe.
    body.subject.graph_id = graph_id

    try:
        return await svc.create_run(graph_id, body, created_by=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ── /api/v1/assessments/runs/{run_id}/module-runs/{module_run_id} ────────────


@router.patch(
    "/assessments/runs/{run_id}/module-runs/{module_run_id}",
    status_code=status.HTTP_200_OK,
    summary="Partial-update a :ModuleRun (status, heartbeat, evidence_count, …)",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        404: {"description": ":ModuleRun not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def update_module_run(
    run_id: str,
    module_run_id: str,
    body: UpdateModuleRunRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    await verify_graph_access(graph_id, "write", user_id)
    try:
        updated = await svc.update_module_run(graph_id, run_id, module_run_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="ModuleRun not found"
        )
    return {"updated": True, "module_run_id": module_run_id}


# ── /api/v1/assessments/runs/{run_id}/findings:bulk ──────────────────────────


@router.post(
    "/assessments/runs/{run_id}/findings:bulk",
    response_model=BulkResponse,
    status_code=status.HTTP_207_MULTI_STATUS,
    summary="Bulk-write :Finding rows under one :ModuleRun (per-record success/failure)",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def record_finding_bulk(
    run_id: str,
    body: RecordFindingBulkRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> BulkResponse:
    """Bulk-persist findings under a single :ModuleRun.

    All findings in `body.findings` must share the same `module_run_id` (the
    `:ModuleRun` whose evidence_count gets refreshed afterwards). The endpoint
    rejects mixed-parent batches with 400; per-finding failures inside a
    valid batch are surfaced via the per-record `BulkResponse` shape
    (HTTP 207).
    """
    await verify_graph_access(graph_id, "write", user_id)

    if not body.findings:
        return BulkResponse(total=0, succeeded=0, failed=0, results=[])

    # All findings must hang off the same ModuleRun. Mixed parents are an
    # easy-to-make caller mistake that would silently scatter the writes.
    parent_module_run_ids = {f.module_run_id for f in body.findings}
    if len(parent_module_run_ids) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "All findings in a single bulk request must share the same "
                f"module_run_id (got {sorted(parent_module_run_ids)!r})"
            ),
        )
    module_run_id = next(iter(parent_module_run_ids))

    # Normalize graph_id + run_id on every finding to the JWT-derived /
    # path-supplied values. ADR-010 §Scope Enforcer: body cannot escalate.
    for f in body.findings:
        f.graph_id = graph_id
        f.run_id = run_id

    try:
        return await svc.record_finding_bulk(
            graph_id, run_id, module_run_id, body.findings
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ── /api/v1/assessments/runs/{run_id}/conflicts ──────────────────────────────


@router.post(
    "/assessments/runs/{run_id}/conflicts",
    status_code=status.HTTP_201_CREATED,
    summary="Record a :Conflict and its [:INVOLVES] edges",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def record_conflict(
    run_id: str,
    body: RecordConflictRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    await verify_graph_access(graph_id, "write", user_id)

    # Normalize body to JWT/path-derived scope.
    body.conflict.graph_id = graph_id
    body.conflict.run_id = run_id

    try:
        created = await svc.record_conflict(graph_id, run_id, body.conflict)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"conflict_id": body.conflict.conflict_id, "created": created}


# ── /api/v1/assessments/runs/{run_id}/unresolved-questions ───────────────────


@router.post(
    "/assessments/runs/{run_id}/unresolved-questions",
    status_code=status.HTTP_201_CREATED,
    summary="Record an :UnresolvedQuestion + [:RAISED] edge from the :ModuleRun",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def record_unresolved_question(
    run_id: str,
    body: RecordUnresolvedQuestionRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    await verify_graph_access(graph_id, "write", user_id)

    body.question.graph_id = graph_id
    body.question.run_id = run_id

    try:
        created = await svc.record_unresolved_question(
            graph_id, run_id, body.question.module_run_id, body.question
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"question_id": body.question.question_id, "created": created}


# ── /api/v1/assessments/runs/{run_id}/deliverables ──────────────────────────


@router.post(
    "/assessments/runs/{run_id}/deliverables",
    status_code=status.HTTP_201_CREATED,
    summary="Persist a :Deliverable produced by a :ModuleRun (or the run itself)",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def persist_deliverable(
    run_id: str,
    body: PersistDeliverableRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    await verify_graph_access(graph_id, "write", user_id)

    body.deliverable.graph_id = graph_id
    body.deliverable.run_id = run_id

    try:
        created = await svc.persist_deliverable(graph_id, run_id, body.deliverable)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {
        "deliverable_id": body.deliverable.deliverable_id,
        "created": created,
    }


# ── /api/v1/assessments/runs/{run_id}/deliverables:bulk-final ────────────────


@router.post(
    "/assessments/runs/{run_id}/deliverables:bulk-final",
    response_model=BulkResponse,
    status_code=status.HTTP_207_MULTI_STATUS,
    summary="Bulk-persist the final 5-doc set (per-record success/failure)",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def persist_final_docs(
    run_id: str,
    body: PersistFinalDocsRequest,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> BulkResponse:
    await verify_graph_access(graph_id, "write", user_id)

    for d in body.deliverables:
        d.graph_id = graph_id
        d.run_id = run_id

    try:
        return await svc.persist_final_docs(graph_id, run_id, body.deliverables)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ── /api/v1/assessments/runs/{run_id}:finalize ───────────────────────────────


@router.post(
    "/assessments/runs/{run_id}:finalize",
    response_model=FinalizeRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the citation-coverage gate and finalize the run (finished | failed)",
    responses={
        400: {"description": "Invalid request or run not found"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def finalize_run(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> FinalizeRunResponse:
    await verify_graph_access(graph_id, "write", user_id)
    try:
        return await svc.finalize_run(graph_id, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ── Heartbeats (resumability, per STORY-026 §Acceptance Criteria) ───────────


@router.post(
    "/assessments/runs/{run_id}:heartbeat",
    status_code=status.HTTP_200_OK,
    summary="Orchestrator heartbeat — update :AssessmentRun.orchestrator_last_seen",
    responses={
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        404: {"description": ":AssessmentRun not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def heartbeat_run(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    """Bump the orchestrator's heartbeat on a run.

    Per STORY-026 §Acceptance Criteria, the Claude Code orchestrator pings
    this every 60 seconds. A run whose `orchestrator_last_seen` is older
    than 5 minutes is considered orphaned by the Celery reset agent.
    """
    await verify_graph_access(graph_id, "write", user_id)
    updated = await svc.heartbeat_run(graph_id, run_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    return {"updated": True, "run_id": run_id}


@router.post(
    "/assessments/runs/{run_id}/module-runs/{module_run_id}:heartbeat",
    status_code=status.HTTP_200_OK,
    summary="Subagent heartbeat — update :ModuleRun.last_heartbeat_at",
    responses={
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks write access to the tenant graph"},
        404: {"description": ":ModuleRun not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def heartbeat_module_run(
    run_id: str,
    module_run_id: str,
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    """Bump a :ModuleRun's `last_heartbeat_at` without altering status.

    Implemented as `update_module_run(last_heartbeat_at=now)` so we reuse a
    single Cypher write path. Per STORY-026 the stale-threshold is 5 minutes
    and the Celery reset agent watches this field.
    """
    await verify_graph_access(graph_id, "write", user_id)
    from datetime import datetime

    updated = await svc.update_module_run(
        graph_id,
        run_id,
        module_run_id,
        UpdateModuleRunRequest(last_heartbeat_at=datetime.now(UTC)),
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="ModuleRun not found"
        )
    return {"updated": True, "module_run_id": module_run_id}


# ── Registry (ADR-019) — write side only; reads land in SPRINT-002 ──────────


@router.post(
    "/assessments/registry/{kind}",
    status_code=status.HTTP_201_CREATED,
    summary="Persist a RegistryItem (skill/agent/tool/mcp-server) per ADR-019",
    responses={
        400: {"description": "Invalid request or visibility/graph mismatch"},
        401: {"description": "Missing or invalid JWT"},
        403: {
            "description": (
                "Caller lacks write access to the target graph, OR caller is "
                "not the owner of an existing public/yanked RegistryItem "
                "(ADR-019: public/yanked items are owner-only on update; "
                "curated writes require platform admin)."
            )
        },
        503: {"description": "Neo4j unavailable"},
    },
)
async def persist_registry_item(
    kind: RegistryKind,
    item: RegistryItem,
    current_user: dict[str, Any] = Depends(get_current_user),
    user_id: str = Depends(get_current_user_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> dict[str, Any]:
    """Write a RegistryItem under the visibility-appropriate graph.

    Per ADR-019:
    - `private` items go to the owner's tenant graph (JWT `home_graph_id`).
    - `public` items go to the catalog graph `__registry__`.
    - `curated` items go to `__registry__` AND require platform-admin.

    The endpoint binds `kind` and `owner_user_id` from the path / principal
    and overwrites the item body's `graph_id` to the visibility-resolved
    target before delegating to the service.
    """
    # Pin the kind from the path; pin the owner from the principal. Both
    # are scope-enforcement boundaries — body input is ignored.
    item.kind = kind
    item.owner_user_id = user_id

    # Resolve target graph from visibility per ADR-019.
    if item.visibility == "private":
        owner_graph_id = current_user.get("home_graph_id") or ""
        if not owner_graph_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Private RegistryItem requires a JWT bound to a tenant "
                    "graph (home_graph_id claim)."
                ),
            )
        item.graph_id = owner_graph_id
        await verify_graph_access(owner_graph_id, "write", user_id)
    elif item.visibility == "curated":
        # Only platform admins write curated items.
        await _verify_registry_curated_write(user_id)
        item.graph_id = REGISTRY_CATALOG_GRAPH_ID
    elif item.visibility in ("public", "yanked"):
        # Public writes land in __registry__. The service checks slug-namespace
        # collisions internally. For SPRINT-001 we require write access to the
        # catalog graph (matches the "publish" action's natural permission).
        await verify_graph_access(REGISTRY_CATALOG_GRAPH_ID, "write", user_id)
        item.graph_id = REGISTRY_CATALOG_GRAPH_ID
    else:  # defensive — Pydantic literal already constrains this
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported visibility: {item.visibility!r}",
        )

    try:
        created = await svc.persist_registry_item(
            item,
            owner_tenant_graph_id=item.graph_id
            if item.visibility == "private"
            else None,
        )
    except RegistryOwnershipError as exc:
        # Per TASK-073 Finding 1 (TASK-069): non-owner attempt to overwrite a
        # public/yanked RegistryItem is rejected with 403. The endpoint's
        # `verify_graph_access` gates "can write to the catalog at all" but
        # not "can write *this* item" — that's enforced in the service.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {
        "item_id": item.item_id,
        "kind": kind,
        "visibility": item.visibility,
        "graph_id": item.graph_id,
        "created": created,
    }
