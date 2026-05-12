"""Assessment substrate REST read endpoints (STORY-026, TASK-079).

Fourteen read endpoints that expose `AssessmentService` over HTTP, mirroring
the write-side patterns from `assessments.py` (TASK-069). Together with the
write side, this is the API surface STORY-026 §Frontend Monitoring Surface
specifies.

Auth / scope (same rules as the write endpoints):

-   Every endpoint authenticates via the existing
    `get_current_user` / `get_current_user_id` dependency stack.
-   `graph_id` is derived from the JWT principal's `home_graph_id` claim.
    Body / query `graph_id` is **never** honored — per ADR-010 §Scope
    Enforcer, agent-supplied scope is ignored.
-   The admin `/findings:search` endpoint is additionally gated by
    `verify_graph_access('__assessments_catalog__', 'admin', user_id)`.
-   Registry visibility (`private` vs `public` vs `curated` vs `yanked`) is
    enforced server-side per ADR-019.

Pagination uses an opaque base64url cursor — see
`app/api/v1/endpoints/_pagination.py` for the codec. Callers receive a
`next_cursor` string and echo it back to fetch the next page; they do not
crack it open.

Cross-tenant safety:
-   404 collapses with "not in your tenant" — the endpoint does not reveal
    whether a `run_id` exists in someone else's graph.
-   The service layer enforces `graph_id` on every Cypher statement.
-   Cross-graph hydration (`:Source`, `:Module`) is performed at the
    application layer per ADR-018; no Cypher MATCH spans graph_id partitions.

Out of scope (per TASK-079):
-   SSE `tail_run` endpoint (TASK-080)
-   MCP wrappers (TASK-081)
-   Rate limiting (STORY-027)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from app.api.dependencies import (
    get_current_user,
    get_current_user_id,
    verify_graph_access,
)
from app.api.v1.endpoints._pagination import (
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    FindingSearchResponse,
    ListConflictsResponse,
    ListDeliverablesResponse,
    ListFindingsResponse,
    ListModuleRunsResponse,
    ListRegistryItemsResponse,
    ListRunsResponse,
    ListTemplateModulesResponse,
    ListUnresolvedQuestionsResponse,
    PageMeta,
    RegistryItemContent,
    RegistryKind,
    RegistryVisibility,
    RunDetail,
    RunStatus,
    WaveStatusResponse,
)
from app.services.assessment_service import AssessmentService

router = APIRouter()
logger = get_logger(__name__)


# ── Dependencies / helpers ────────────────────────────────────────────────────


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
    """Resolve the caller's tenant `graph_id` from the JWT claim.

    Mirrors `assessments._principal_graph_id` (TASK-069). A principal without
    a usable `home_graph_id` is rejected with 400 — the read endpoints
    require a tenant context.
    """
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


def _decode_cursor_or_400(cursor: str | None) -> int:
    """Decode an opaque cursor; raise 400 on malformed input."""
    try:
        offset, _last_id = decode_cursor(cursor)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed cursor: {exc}",
        ) from exc
    return offset


def _next_cursor(has_more: bool, offset: int, page_size: int) -> str | None:
    """Compute the opaque next-page cursor; returns `None` when exhausted."""
    if not has_more:
        return None
    return encode_cursor(offset + page_size)


# =============================================================================
# Run-level read endpoints
# =============================================================================


@router.get(
    "/assessments/runs",
    response_model=ListRunsResponse,
    summary="List :AssessmentRun rows for the caller's tenant",
    responses={
        400: {"description": "Malformed cursor or missing home_graph_id"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_runs(
    status_filter: RunStatus | None = Query(default=None, alias="status"),
    subject: str | None = Query(
        default=None,
        max_length=128,
        description="Filter by :Subject.slug within the tenant graph.",
    ),
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Maximum page size; clamped to MAX_PAGE_SIZE.",
    ),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListRunsResponse:
    """Paginated list of assessment runs in the tenant.

    Sorted by `started_at DESC, run_id ASC`. Use `cursor` to fetch
    subsequent pages; clients must not parse the cursor string.
    """
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    rows, has_more = await svc.list_runs(
        graph_id,
        status=status_filter,
        subject_slug=subject,
        offset=offset,
        limit=page_size,
    )
    return ListRunsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/runs/{run_id}",
    response_model=RunDetail,
    summary="Run header + rollup counts",
    responses={
        400: {"description": "Missing home_graph_id"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def get_run(
    run_id: str = Path(..., max_length=128),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> RunDetail:
    await verify_graph_access(graph_id, "read", user_id)
    detail = await svc.get_run_detail(graph_id, run_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    return detail


@router.get(
    "/assessments/runs/{run_id}/waves/{wave}",
    response_model=WaveStatusResponse,
    summary="Per-wave done/failed/total + per-module status list",
    responses={
        400: {"description": "Malformed cursor or wave"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def get_wave_status(
    run_id: str = Path(..., max_length=128),
    wave: int = Path(..., ge=1),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> WaveStatusResponse:
    await verify_graph_access(graph_id, "read", user_id)
    waves = await svc.get_wave_status(graph_id, run_id, wave)
    if waves is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    return waves


@router.get(
    "/assessments/runs/{run_id}/module-runs",
    response_model=ListModuleRunsResponse,
    summary="All :ModuleRun rows for a run joined to :Module + :Agent",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_module_runs(
    run_id: str = Path(..., max_length=128),
    status_filter: RunStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListModuleRunsResponse:
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    result = await svc.list_module_runs(
        graph_id,
        run_id,
        status=status_filter,
        offset=offset,
        limit=page_size,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    rows, has_more = result
    return ListModuleRunsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/runs/{run_id}/findings",
    response_model=ListFindingsResponse,
    summary="Findings with :Source hydrated from the catalog by source_id",
    responses={
        400: {"description": "Malformed cursor or filter"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_findings(
    run_id: str = Path(..., max_length=128),
    module: str | None = Query(
        default=None, max_length=128, description="Filter by :Module.slug"
    ),
    dimension: str | None = Query(default=None, max_length=128),
    label: str | None = Query(default=None, max_length=64),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    source_type: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListFindingsResponse:
    """Findings table view with optional filters.

    `:Source` is hydrated at the application layer (ADR-018 §Tenancy): the
    per-tenant Cypher returns `source_id`, then a second Cypher fetches the
    catalog rows. Each query stays single-tenant from Neo4j's perspective.
    """
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    result = await svc.list_findings(
        graph_id,
        run_id,
        module_slug=module,
        dimension=dimension,
        label=label,
        min_confidence=min_confidence,
        source_type=source_type,
        offset=offset,
        limit=page_size,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    rows, has_more = result
    return ListFindingsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/runs/{run_id}/conflicts",
    response_model=ListConflictsResponse,
    summary="Conflicts with involved finding IDs",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_conflicts(
    run_id: str = Path(..., max_length=128),
    status_filter: str | None = Query(default=None, alias="status", max_length=64),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListConflictsResponse:
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    result = await svc.list_conflicts(
        graph_id,
        run_id,
        status=status_filter,
        offset=offset,
        limit=page_size,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    rows, has_more = result
    return ListConflictsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/runs/{run_id}/unresolved-questions",
    response_model=ListUnresolvedQuestionsResponse,
    summary="Unresolved questions raised by research modules",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_unresolved_questions(
    run_id: str = Path(..., max_length=128),
    status_filter: str | None = Query(default=None, alias="status", max_length=64),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListUnresolvedQuestionsResponse:
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    result = await svc.list_unresolved_questions(
        graph_id,
        run_id,
        status=status_filter,
        offset=offset,
        limit=page_size,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    rows, has_more = result
    return ListUnresolvedQuestionsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/runs/{run_id}/deliverables",
    response_model=ListDeliverablesResponse,
    summary="Deliverable metadata (no content)",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Run not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_deliverables(
    run_id: str = Path(..., max_length=128),
    kind: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListDeliverablesResponse:
    await verify_graph_access(graph_id, "read", user_id)
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    result = await svc.list_deliverables(
        graph_id,
        run_id,
        kind=kind,
        offset=offset,
        limit=page_size,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AssessmentRun not found"
        )
    rows, has_more = result
    return ListDeliverablesResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


# Accept header → Content-Type mapping for deliverable / registry content.
# `text/markdown` and `application/pdf` are common in the Eurail set;
# `text/html` is the final-html kind. `*/*` lets clients accept whatever we
# happen to have inline.
_MIME_TYPES = {
    "module-md": "text/markdown",
    "final-md": "text/markdown",
    "final-html": "text/html",
    "final-pdf": "application/pdf",
}


@router.get(
    "/assessments/runs/{run_id}/deliverables/{deliverable_id}/content",
    summary="Fetch a deliverable's content (inline payload or content_uri)",
    responses={
        200: {
            "content": {
                "text/markdown": {},
                "text/html": {},
                "application/pdf": {},
                "application/octet-stream": {},
            }
        },
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Deliverable not found in this tenant"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def get_deliverable_content(
    run_id: str = Path(..., max_length=128),
    deliverable_id: str = Path(..., max_length=128),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> Response:
    """Return the deliverable's content payload.

    SPRINT-001 storage: either `content_inline` (small markdown) or
    `content_uri` (filesystem placeholder). SPRINT-003 (TASK-082) introduces
    the Postgres `:Blob` CAS keyed by `sha256`; this endpoint will then
    transparently resolve `blob://sha256/...` URIs.

    For SPRINT-002 we return whichever shape is present:
    - If `content_inline` is non-empty, return it as the body with the kind's
      MIME type.
    - Else return a 200 with an empty body and the `Location` header set to
      the `content_uri` so callers can resolve it themselves.
    - 404 if the deliverable does not exist in the tenant.
    """
    await verify_graph_access(graph_id, "read", user_id)
    content = await svc.get_deliverable_content(graph_id, run_id, deliverable_id)
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deliverable not found",
        )

    mime = _MIME_TYPES.get(content["kind"] or "", "application/octet-stream")
    inline = content.get("content_inline") or ""
    if inline:
        return Response(content=inline, media_type=mime)
    # No inline payload — surface the URI in a Location header. SPRINT-003
    # CAS resolution will replace this branch with actual bytes.
    uri = content.get("content_uri") or ""
    headers: dict[str, str] = {}
    if uri:
        headers["Location"] = uri
    return Response(content="", media_type=mime, headers=headers)


# =============================================================================
# Template introspection
# =============================================================================


@router.get(
    "/assessments/templates/{template_slug}/modules",
    response_model=ListTemplateModulesResponse,
    summary="Template introspection — :Module rows for a template",
    responses={
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller lacks read access to the tenant graph"},
        404: {"description": "Template not found"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_template_modules(
    template_slug: str = Path(..., max_length=128),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListTemplateModulesResponse:
    """Module-shape preview for the UI's pre-run rendering.

    Template-layer entities live in the catalog graph (shared across
    tenants), but we still require the caller to have valid tenant-graph
    access so an unauthenticated browse of platform templates is not
    possible. The catalog itself is not tenant-specific so any authenticated
    tenant can introspect any template — that's a deliberate ADR-018 choice
    (templates are platform-published).
    """
    await verify_graph_access(graph_id, "read", user_id)
    result = await svc.list_template_modules(template_slug)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment template not found: slug={template_slug!r}",
        )
    meta, modules = result
    return ListTemplateModulesResponse(
        template_id=meta["template_id"],
        template_slug=meta["template_slug"],
        template_name=meta["template_name"],
        template_version=meta["template_version"],
        modules=modules,
    )


# =============================================================================
# Registry reads (ADR-019)
# =============================================================================


@router.get(
    "/assessments/registry/{kind}",
    response_model=ListRegistryItemsResponse,
    summary="List Registry items per ADR-019 visibility rules",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def list_registry_items(
    kind: RegistryKind = Path(...),
    owner: str | None = Query(
        default=None,
        max_length=128,
        description=(
            "Filter by `owner_user_id`. Note: a non-owner cannot see another "
            "user's private items even if they ask explicitly — ADR-019."
        ),
    ),
    visibility: RegistryVisibility | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> ListRegistryItemsResponse:
    """List Registry items per ADR-019 visibility rules.

    Default listing (no `visibility=`): `curated + public + caller-owned
    private`. Explicit `visibility=private` returns only the caller's own
    private items in their tenant graph. `visibility=public/curated/yanked`
    returns the catalog rows.

    Cross-tenant private leakage is structurally impossible — the service
    only queries the caller's tenant graph for private items and filters by
    `owner_user_id = caller`.
    """
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    rows, has_more = await svc.list_registry_items(
        caller_user_id=user_id,
        caller_graph_id=graph_id,
        kind=kind,
        owner_user_id=owner,
        visibility=visibility,
        offset=offset,
        limit=page_size,
    )
    return ListRegistryItemsResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )


@router.get(
    "/assessments/registry/{kind}/{slug}",
    response_model=RegistryItemContent,
    summary="Get a Registry item's metadata (latest version by default)",
    responses={
        401: {"description": "Missing or invalid JWT"},
        404: {"description": "Item not found OR private-owned-by-someone-else"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def get_registry_item(
    kind: RegistryKind = Path(...),
    slug: str = Path(..., max_length=128),
    version: str | None = Query(default=None, max_length=32),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> RegistryItemContent:
    """Resolve `<kind>/<slug>[@version]` per ADR-019 visibility.

    The response is shaped like the content endpoint (without the body) so
    callers can chain to `.../content` without re-rendering paths. The
    404/403 collapse is intentional per ADR-019 — a non-owner must not
    distinguish "another user's private item" from "no such slug".
    """
    item_row = await svc.get_registry_item(
        caller_user_id=user_id,
        caller_graph_id=graph_id,
        kind=kind,
        slug=slug,
        version=version,
    )
    if item_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Registry item not found: {kind}/{slug}",
        )
    content_type = "text/markdown" if kind in ("skill", "agent") else "application/json"
    return RegistryItemContent(
        item_id=item_row.item_id,
        kind=item_row.kind,
        slug=item_row.slug,
        version=item_row.version,
        content_type=content_type,
        content_inline=None,
        content_uri=item_row.content_uri,
        sha256=item_row.sha256,
    )


@router.get(
    "/assessments/registry/{kind}/{slug}/{version}/content",
    summary="Fetch a Registry item's content payload",
    responses={
        200: {
            "content": {
                "text/markdown": {},
                "application/json": {},
                "application/octet-stream": {},
            }
        },
        401: {"description": "Missing or invalid JWT"},
        404: {"description": "Item not found OR private-owned-by-someone-else"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def get_registry_item_content(
    kind: RegistryKind = Path(...),
    slug: str = Path(..., max_length=128),
    version: str = Path(..., max_length=32),
    user_id: str = Depends(get_current_user_id),
    graph_id: str = Depends(_principal_graph_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> Response:
    content = await svc.get_registry_item_content(
        caller_user_id=user_id,
        caller_graph_id=graph_id,
        kind=kind,
        slug=slug,
        version=version,
    )
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Registry item not found: {kind}/{slug}@{version}",
        )
    inline = content.content_inline or ""
    if inline:
        return Response(content=inline, media_type=content.content_type)
    headers: dict[str, str] = {}
    if content.content_uri:
        headers["Location"] = content.content_uri
    return Response(content="", media_type=content.content_type, headers=headers)


# =============================================================================
# Admin: cross-run findings search
# =============================================================================


@router.get(
    "/assessments/findings:search",
    response_model=FindingSearchResponse,
    summary="Cross-run findings search (admin-only)",
    responses={
        400: {"description": "Malformed cursor"},
        401: {"description": "Missing or invalid JWT"},
        403: {"description": "Caller is not a platform admin"},
        503: {"description": "Neo4j unavailable"},
    },
)
async def search_findings(
    source_url: str | None = Query(default=None, max_length=2048),
    dimension: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(default=None, max_length=4096),
    user_id: str = Depends(get_current_user_id),
    svc: AssessmentService = Depends(_assessment_service),
) -> FindingSearchResponse:
    """Admin-only cross-run findings search.

    Per TASK-079 contract: gated by `verify_graph_access('__assessments_catalog__',
    'admin', user_id)`. Returns 403 for non-admin callers. Each row carries
    its source `graph_id` so admins can trace origin.

    Note: this endpoint deliberately does NOT use `_principal_graph_id` — it
    is intentionally cross-tenant. The admin ACL on the catalog graph is the
    single auth boundary.
    """
    await verify_graph_access(ASSESSMENTS_CATALOG_GRAPH_ID, "admin", user_id)

    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    offset = _decode_cursor_or_400(cursor)

    rows, has_more = await svc.search_findings_admin(
        source_url=source_url,
        dimension=dimension,
        offset=offset,
        limit=page_size,
    )
    return FindingSearchResponse(
        items=rows,
        page=PageMeta(
            next_cursor=_next_cursor(has_more, offset, page_size),
            page_size=len(rows),
        ),
    )
