"""MCP tool wrappers for ``/api/v1/assessments/*`` REST endpoints (TASK-080).

Per ADR-007 every REST endpoint has a 1:1 MCP tool. The wrappers call the
service layer directly — they do not go through HTTP — so they share the
exact same boundary, auth plumbing, and Pydantic schemas as the REST
endpoints in:

- ``app/api/v1/endpoints/assessments.py`` (writes — TASK-069)
- ``app/api/v1/endpoints/assessments_reads.py`` (reads — TASK-079)

Naming convention: ``assessment.<verb>_<noun>``. Tool registration is
performed by ``app.mcp.server`` via the ``TOOLS`` list at the bottom.

Auth: every tool authenticates via the shared MCP bearer token
(``ORACLOUS_API_KEY``), resolves the principal through ``auth_service``,
and enforces ReBAC via ``verify_graph_access``. There is no parallel auth
path.

Error responses: every tool returns ``{"error": ..., "code": ...}`` for
auth / scope / validation failures. Successful tools return the same
Pydantic shape as the REST response model, serialized via
``model_dump(mode='json')`` so the MCP transport can encode it.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from app.api.v1.endpoints._pagination import (
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from app.mcp.tools._auth import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    AuthError,
    ScopeError,
    assert_catalog_admin,
    assert_graph_access,
    build_service,
    principal_graph_id,
    resolve_principal,
    tool_error,
)
from app.schemas.assessment_schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    BulkResponse,
    CreateRunRequest,
    CreateRunResponse,
    FinalizeRunResponse,
    PageMeta,
    PersistDeliverableRequest,
    PersistFinalDocsRequest,
    RecordConflictRequest,
    RecordFindingBulkRequest,
    RecordUnresolvedQuestionRequest,
    UpdateModuleRunRequest,
)

# ── Shared helpers ────────────────────────────────────────────────────────────


def _api_key() -> str | None:
    """Return the MCP server's bearer token (None when unset).

    We resolve at call-time, not at import-time, so tests can adjust
    ``ORACLOUS_API_KEY`` between cases.
    """
    return os.environ.get("ORACLOUS_API_KEY") or None


async def _principal_and_graph() -> tuple[dict[str, Any], str] | dict[str, Any]:
    """Resolve principal + tenant graph_id. Returns an error dict on failure.

    The return shape is "tuple on success, error-dict on failure" so each
    tool can do ``if isinstance(result, dict): return result`` — no
    exceptions cross the MCP boundary.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")
    try:
        gid = principal_graph_id(principal)
    except ScopeError as exc:
        return tool_error(str(exc), code="scope")
    return principal, gid


def _decode_cursor_or_error(cursor: str | None) -> int | dict[str, Any]:
    try:
        offset, _last_id = decode_cursor(cursor)
    except ValueError as exc:
        return tool_error(f"Malformed cursor: {exc}", code="bad_request")
    return offset


def _next_cursor(has_more: bool, offset: int, page_size: int) -> str | None:
    if not has_more:
        return None
    return encode_cursor(offset + page_size)


def _dump(model: Any) -> Any:
    """Convert a Pydantic model (or list of them) into JSON-friendly dicts."""
    if isinstance(model, list):
        return [_dump(m) for m in model]
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


# =============================================================================
# Writes (mirror app/api/v1/endpoints/assessments.py)
# =============================================================================


async def create_run(body: dict[str, Any]) -> dict[str, Any]:
    """Create an :AssessmentRun and all :ModuleRun rows in 'planned'.

    Mirrors ``POST /api/v1/assessments/runs`` (TASK-069). `body` must
    validate against ``CreateRunRequest``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = CreateRunRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")

    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    # Same ADR-010 normalization the REST endpoint applies.
    request.subject.graph_id = graph_id
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")

    try:
        result: CreateRunResponse = await svc.create_run(
            graph_id, request, created_by=str(principal["id"])
        )
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return _dump(result)


async def update_module_run(
    run_id: str, module_run_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Partial-update a :ModuleRun (status, heartbeat, evidence_count, …).

    Mirrors ``PATCH /api/v1/assessments/runs/{run_id}/module-runs/{module_run_id}``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        update = UpdateModuleRunRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        updated = await svc.update_module_run(graph_id, run_id, module_run_id, update)
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    if not updated:
        return tool_error("ModuleRun not found", code="not_found")
    return {"updated": True, "module_run_id": module_run_id}


async def record_finding_bulk(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Bulk-write :Finding rows under one :ModuleRun.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/findings:bulk``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = RecordFindingBulkRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    if not request.findings:
        return _dump(BulkResponse(total=0, succeeded=0, failed=0, results=[]))

    parent_module_run_ids = {f.module_run_id for f in request.findings}
    if len(parent_module_run_ids) != 1:
        return tool_error(
            (
                "All findings in a single bulk request must share the same "
                f"module_run_id (got {sorted(parent_module_run_ids)!r})"
            ),
            code="bad_request",
        )
    module_run_id = next(iter(parent_module_run_ids))

    # ADR-010 normalization.
    for f in request.findings:
        f.graph_id = graph_id
        f.run_id = run_id

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        bulk = await svc.record_finding_bulk(
            graph_id, run_id, module_run_id, request.findings
        )
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return _dump(bulk)


async def record_conflict(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Record a :Conflict and its [:INVOLVES] edges.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/conflicts``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = RecordConflictRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    request.conflict.graph_id = graph_id
    request.conflict.run_id = run_id

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        created = await svc.record_conflict(graph_id, run_id, request.conflict)
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return {"conflict_id": request.conflict.conflict_id, "created": created}


async def record_unresolved_question(
    run_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Record an :UnresolvedQuestion + [:RAISED] edge from the :ModuleRun.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/unresolved-questions``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = RecordUnresolvedQuestionRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    request.question.graph_id = graph_id
    request.question.run_id = run_id

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        created = await svc.record_unresolved_question(
            graph_id, run_id, request.question.module_run_id, request.question
        )
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return {"question_id": request.question.question_id, "created": created}


async def persist_deliverable(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Persist a :Deliverable produced by a :ModuleRun (or the run itself).

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/deliverables``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = PersistDeliverableRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    request.deliverable.graph_id = graph_id
    request.deliverable.run_id = run_id

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        created = await svc.persist_deliverable(graph_id, run_id, request.deliverable)
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return {
        "deliverable_id": request.deliverable.deliverable_id,
        "created": created,
    }


async def persist_final_docs(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Bulk-persist the final 5-doc set (per-record success/failure).

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/deliverables:bulk-final``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        request = PersistFinalDocsRequest.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")

    for d in request.deliverables:
        d.graph_id = graph_id
        d.run_id = run_id

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        bulk = await svc.persist_final_docs(graph_id, run_id, request.deliverables)
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return _dump(bulk)


async def finalize_run(run_id: str) -> dict[str, Any]:
    """Run the citation-coverage gate and finalize the run.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}:finalize``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        result: FinalizeRunResponse = await svc.finalize_run(graph_id, run_id)
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return _dump(result)


async def heartbeat_run(run_id: str) -> dict[str, Any]:
    """Orchestrator heartbeat — update :AssessmentRun.orchestrator_last_seen.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}:heartbeat``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    updated = await svc.heartbeat_run(graph_id, run_id)
    if not updated:
        return tool_error("AssessmentRun not found", code="not_found")
    return {"updated": True, "run_id": run_id}


async def heartbeat_module_run(run_id: str, module_run_id: str) -> dict[str, Any]:
    """Subagent heartbeat — update :ModuleRun.last_heartbeat_at.

    Mirrors ``POST /api/v1/assessments/runs/{run_id}/module-runs/{module_run_id}:heartbeat``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "write")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    updated = await svc.update_module_run(
        graph_id,
        run_id,
        module_run_id,
        UpdateModuleRunRequest(last_heartbeat_at=datetime.now(UTC)),
    )
    if not updated:
        return tool_error("ModuleRun not found", code="not_found")
    return {"updated": True, "module_run_id": module_run_id}


# =============================================================================
# Reads (mirror app/api/v1/endpoints/assessments_reads.py — first 9)
# =============================================================================


async def list_runs(
    status: str | None = None,
    subject: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Paginated list of :AssessmentRun rows for the tenant.

    Mirrors ``GET /api/v1/assessments/runs``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    rows, has_more = await svc.list_runs(
        graph_id,
        status=status,  # type: ignore[arg-type]
        subject_slug=subject,
        offset=offset,
        limit=page_size,
    )
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def get_run(run_id: str) -> dict[str, Any]:
    """Run header + rollup counts.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    detail = await svc.get_run_detail(graph_id, run_id)
    if detail is None:
        return tool_error("AssessmentRun not found", code="not_found")
    return _dump(detail)


async def get_wave_status(run_id: str, wave: int) -> dict[str, Any]:
    """Per-wave done/failed/total + per-module status list.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/waves/{wave}``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    waves = await svc.get_wave_status(graph_id, run_id, wave)
    if waves is None:
        return tool_error("AssessmentRun not found", code="not_found")
    return _dump(waves)


async def list_module_runs(
    run_id: str,
    status: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """All :ModuleRun rows for a run joined to :Module + :Agent.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/module-runs``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    result = await svc.list_module_runs(
        graph_id,
        run_id,
        status=status,  # type: ignore[arg-type]
        offset=offset,
        limit=page_size,
    )
    if result is None:
        return tool_error("AssessmentRun not found", code="not_found")
    rows, has_more = result
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def list_findings(
    run_id: str,
    module: str | None = None,
    dimension: str | None = None,
    label: str | None = None,
    min_confidence: float | None = None,
    source_type: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Findings table view with optional filters; :Source hydrated by id.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/findings``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
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
        return tool_error("AssessmentRun not found", code="not_found")
    rows, has_more = result
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def list_conflicts(
    run_id: str,
    status: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Conflicts with involved finding IDs.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/conflicts``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    result = await svc.list_conflicts(
        graph_id, run_id, status=status, offset=offset, limit=page_size
    )
    if result is None:
        return tool_error("AssessmentRun not found", code="not_found")
    rows, has_more = result
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def list_unresolved_questions(
    run_id: str,
    status: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Unresolved questions raised by research modules.

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/unresolved-questions``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    result = await svc.list_unresolved_questions(
        graph_id, run_id, status=status, offset=offset, limit=page_size
    )
    if result is None:
        return tool_error("AssessmentRun not found", code="not_found")
    rows, has_more = result
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def list_deliverables(
    run_id: str,
    kind: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Deliverable metadata (no content body).

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/deliverables``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    result = await svc.list_deliverables(
        graph_id, run_id, kind=kind, offset=offset, limit=page_size
    )
    if result is None:
        return tool_error("AssessmentRun not found", code="not_found")
    rows, has_more = result
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


async def get_deliverable_content(run_id: str, deliverable_id: str) -> dict[str, Any]:
    """Fetch a deliverable's content (inline payload or content_uri).

    Mirrors ``GET /api/v1/assessments/runs/{run_id}/deliverables/{deliverable_id}/content``.

    Note: REST returns raw bytes; the MCP wrapper returns a structured dict
    ``{kind, content_inline, content_uri}`` because MCP transports cannot
    return raw byte streams without an explicit content-type contract.
    Callers can resolve ``content_uri`` themselves.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    content = await svc.get_deliverable_content(graph_id, run_id, deliverable_id)
    if content is None:
        return tool_error("Deliverable not found", code="not_found")
    return {
        "kind": content.get("kind"),
        "content_inline": content.get("content_inline"),
        "content_uri": content.get("content_uri"),
    }


# =============================================================================
# Template introspection (read)
# =============================================================================


async def list_template_modules(template_slug: str) -> dict[str, Any]:
    """Template introspection — :Module rows for a template.

    Mirrors ``GET /api/v1/assessments/templates/{template_slug}/modules``.
    """
    pg = await _principal_and_graph()
    if isinstance(pg, dict):
        return pg
    principal, graph_id = pg
    try:
        await assert_graph_access(principal, graph_id, "read")
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    result = await svc.list_template_modules(template_slug)
    if result is None:
        return tool_error(
            f"Assessment template not found: slug={template_slug!r}",
            code="not_found",
        )
    meta, modules = result
    return {
        "template_id": meta["template_id"],
        "template_slug": meta["template_slug"],
        "template_name": meta["template_name"],
        "template_version": meta["template_version"],
        "modules": _dump(modules),
    }


# =============================================================================
# Admin: cross-run findings search
# =============================================================================


async def search_findings(
    source_url: str | None = None,
    dimension: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Cross-run findings search (admin-only).

    Mirrors ``GET /api/v1/assessments/findings:search``. Gated by admin
    access to the assessments catalog graph; deliberately does NOT scope
    to a single tenant — the admin ACL on ``__assessments_catalog__`` is
    the single auth boundary.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")
    try:
        await assert_catalog_admin(principal)
    except ScopeError as exc:
        return tool_error(str(exc), code="forbidden")
    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    rows, has_more = await svc.search_findings_admin(
        source_url=source_url,
        dimension=dimension,
        offset=offset,
        limit=page_size,
    )
    return {
        "items": _dump(rows),
        "page": _dump(
            PageMeta(
                next_cursor=_next_cursor(has_more, offset, page_size),
                page_size=len(rows),
            )
        ),
    }


# Sanity reference to silence "unused import" linters — the constant is
# part of the public attribution to ADR-018 documentation in tool docstrings.
_ASSESSMENTS_CATALOG = ASSESSMENTS_CATALOG_GRAPH_ID


# ── Tool registry ─────────────────────────────────────────────────────────────
#
# Each entry: (mcp tool name, callable). Names follow the ADR-007 convention
# ``assessment.<verb>_<noun>``. The list is ordered: writes first (mirroring
# `assessments.py`), then reads (mirroring `assessments_reads.py`), then the
# admin search. ``assessment.persist_registry_item`` lives in
# ``registry_tools`` because Registry is a separate concern even though the
# REST route lives under ``/api/v1/assessments/registry/{kind}``.

TOOLS: list[tuple[str, Any]] = [
    # Writes
    ("assessment.create_run", create_run),
    ("assessment.update_module_run", update_module_run),
    ("assessment.record_finding_bulk", record_finding_bulk),
    ("assessment.record_conflict", record_conflict),
    ("assessment.record_unresolved_question", record_unresolved_question),
    ("assessment.persist_deliverable", persist_deliverable),
    ("assessment.persist_final_docs", persist_final_docs),
    ("assessment.finalize_run", finalize_run),
    ("assessment.heartbeat_run", heartbeat_run),
    ("assessment.heartbeat_module_run", heartbeat_module_run),
    # Reads
    ("assessment.list_runs", list_runs),
    ("assessment.get_run", get_run),
    ("assessment.get_wave_status", get_wave_status),
    ("assessment.list_module_runs", list_module_runs),
    ("assessment.list_findings", list_findings),
    ("assessment.list_conflicts", list_conflicts),
    ("assessment.list_unresolved_questions", list_unresolved_questions),
    ("assessment.list_deliverables", list_deliverables),
    ("assessment.get_deliverable_content", get_deliverable_content),
    ("assessment.list_template_modules", list_template_modules),
    # Admin
    ("assessment.search_findings", search_findings),
]
