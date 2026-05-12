"""MCP tool wrappers for ``/api/v1/assessments/registry/*`` endpoints (TASK-080).

Per ADR-007 + ADR-019, every Registry REST endpoint has a 1:1 MCP tool.
The wrappers call the service layer directly — they do not go through
HTTP — so they share the same boundary, auth plumbing, and Pydantic
schemas as the REST endpoints in:

- ``app/api/v1/endpoints/assessments.py::persist_registry_item`` (write)
- ``app/api/v1/endpoints/assessments_reads.py`` (3 reads)

Naming convention: ``registry.<verb>_<noun>``. Tool registration is
performed by ``app.mcp.server`` via the ``TOOLS`` list at the bottom.

ADR-019 visibility enforcement (private / public / curated / yanked) is
implemented in the service layer; the wrappers replicate the REST
endpoint's visibility-resolution logic exactly so MCP callers get the
same 403/404 behavior. Curated writes require platform-admin (admin ACL
on ``__registry__``); public/yanked writes require write access to the
catalog graph; private writes require write access to the caller's
tenant graph.
"""

from __future__ import annotations

import os
from typing import Any

from app.api.v1.endpoints._pagination import (
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from app.mcp.tools._auth import (
    REGISTRY_CATALOG_GRAPH_ID,
    AuthError,
    ScopeError,
    assert_graph_access,
    assert_registry_curated_write,
    build_service,
    resolve_principal,
    tool_error,
)
from app.schemas.assessment_schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    PageMeta,
    RegistryItem,
)
from app.services.assessment_service import RegistryOwnershipError

# ── Shared helpers ────────────────────────────────────────────────────────────


def _api_key() -> str | None:
    """Return the MCP server's bearer token (None when unset)."""
    return os.environ.get("ORACLOUS_API_KEY") or None


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
    if isinstance(model, list):
        return [_dump(m) for m in model]
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


# =============================================================================
# Write — registry.persist_item  (mirrors POST /assessments/registry/{kind})
# =============================================================================


async def persist_item(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Persist a RegistryItem (skill/agent/tool/mcp-server) per ADR-019.

    Mirrors ``POST /api/v1/assessments/registry/{kind}``.

    ADR-019 visibility resolution is replicated here exactly: ``private``
    items land in the caller's tenant graph, ``public`` / ``yanked`` in
    ``__registry__``, ``curated`` in ``__registry__`` and admin-gated.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")

    if kind not in ("skill", "agent", "tool", "mcp-server"):
        return tool_error(
            f"Unsupported registry kind: {kind!r} "
            "(must be one of skill | agent | tool | mcp-server)",
            code="bad_request",
        )
    try:
        item = RegistryItem.model_validate(body)
    except Exception as exc:
        return tool_error(f"Invalid request body: {exc}", code="bad_request")

    user_id = str(principal["id"])
    # Pin path/principal boundaries — body input is ignored.
    item.kind = kind  # type: ignore[assignment]
    item.owner_user_id = user_id

    # Resolve target graph from visibility per ADR-019.
    if item.visibility == "private":
        owner_graph_id = principal.get("home_graph_id") or ""
        if not owner_graph_id:
            return tool_error(
                "Private RegistryItem requires a JWT bound to a tenant "
                "graph (home_graph_id claim).",
                code="bad_request",
            )
        item.graph_id = owner_graph_id
        try:
            await assert_graph_access(principal, owner_graph_id, "write")
        except ScopeError as exc:
            return tool_error(str(exc), code="forbidden")
    elif item.visibility == "curated":
        try:
            await assert_registry_curated_write(principal)
        except ScopeError as exc:
            return tool_error(str(exc), code="forbidden")
        item.graph_id = REGISTRY_CATALOG_GRAPH_ID
    elif item.visibility in ("public", "yanked"):
        try:
            await assert_graph_access(principal, REGISTRY_CATALOG_GRAPH_ID, "write")
        except ScopeError as exc:
            return tool_error(str(exc), code="forbidden")
        item.graph_id = REGISTRY_CATALOG_GRAPH_ID
    else:  # defensive — Pydantic literal already constrains this
        return tool_error(
            f"Unsupported visibility: {item.visibility!r}", code="bad_request"
        )

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    try:
        created = await svc.persist_registry_item(
            item,
            owner_tenant_graph_id=item.graph_id
            if item.visibility == "private"
            else None,
        )
    except RegistryOwnershipError as exc:
        return tool_error(str(exc), code="forbidden")
    except ValueError as exc:
        return tool_error(str(exc), code="bad_request")
    return {
        "item_id": item.item_id,
        "kind": kind,
        "visibility": item.visibility,
        "graph_id": item.graph_id,
        "created": created,
    }


# =============================================================================
# Reads — mirror the three GETs under /api/v1/assessments/registry/...
# =============================================================================


async def list_items(
    kind: str,
    owner: str | None = None,
    visibility: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> dict[str, Any]:
    """List Registry items per ADR-019 visibility rules.

    Mirrors ``GET /api/v1/assessments/registry/{kind}``.

    Default listing (no `visibility`): curated + public + caller-owned
    private. Explicit `visibility=private` returns only the caller's
    own private items in their tenant graph.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")

    if kind not in ("skill", "agent", "tool", "mcp-server"):
        return tool_error(f"Unsupported registry kind: {kind!r}", code="bad_request")
    if visibility is not None and visibility not in (
        "private",
        "public",
        "curated",
        "yanked",
    ):
        return tool_error(f"Unsupported visibility: {visibility!r}", code="bad_request")

    caller_user_id = str(principal["id"])
    caller_graph_id = principal.get("home_graph_id") or ""
    if not caller_graph_id:
        return tool_error("Principal has no home_graph_id claim.", code="scope")

    offset = _decode_cursor_or_error(cursor)
    if isinstance(offset, dict):
        return offset
    page_size = clamp_limit(limit, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    rows, has_more = await svc.list_registry_items(
        caller_user_id=caller_user_id,
        caller_graph_id=caller_graph_id,
        kind=kind,  # type: ignore[arg-type]
        owner_user_id=owner,
        visibility=visibility,  # type: ignore[arg-type]
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


async def get_item(kind: str, slug: str, version: str | None = None) -> dict[str, Any]:
    """Get a Registry item's metadata (latest version by default).

    Mirrors ``GET /api/v1/assessments/registry/{kind}/{slug}``. The 404
    deliberately collapses with "private-owned-by-someone-else" per
    ADR-019 — non-owners must not learn whether a private slug exists.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")

    if kind not in ("skill", "agent", "tool", "mcp-server"):
        return tool_error(f"Unsupported registry kind: {kind!r}", code="bad_request")
    caller_user_id = str(principal["id"])
    caller_graph_id = principal.get("home_graph_id") or ""
    if not caller_graph_id:
        return tool_error("Principal has no home_graph_id claim.", code="scope")

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    item_row = await svc.get_registry_item(
        caller_user_id=caller_user_id,
        caller_graph_id=caller_graph_id,
        kind=kind,  # type: ignore[arg-type]
        slug=slug,
        version=version,
    )
    if item_row is None:
        return tool_error(f"Registry item not found: {kind}/{slug}", code="not_found")
    content_type = "text/markdown" if kind in ("skill", "agent") else "application/json"
    return {
        "item_id": item_row.item_id,
        "kind": item_row.kind,
        "slug": item_row.slug,
        "version": item_row.version,
        "content_type": content_type,
        "content_inline": None,
        "content_uri": item_row.content_uri,
        "sha256": item_row.sha256,
    }


async def get_item_content(kind: str, slug: str, version: str) -> dict[str, Any]:
    """Fetch a Registry item's content payload.

    Mirrors ``GET /api/v1/assessments/registry/{kind}/{slug}/{version}/content``.
    """
    try:
        principal = await resolve_principal(_api_key())
    except AuthError as exc:
        return tool_error(str(exc), code="unauthenticated")

    if kind not in ("skill", "agent", "tool", "mcp-server"):
        return tool_error(f"Unsupported registry kind: {kind!r}", code="bad_request")
    caller_user_id = str(principal["id"])
    caller_graph_id = principal.get("home_graph_id") or ""
    if not caller_graph_id:
        return tool_error("Principal has no home_graph_id claim.", code="scope")

    try:
        svc = build_service()
    except ScopeError as exc:
        return tool_error(str(exc), code="unavailable")
    content = await svc.get_registry_item_content(
        caller_user_id=caller_user_id,
        caller_graph_id=caller_graph_id,
        kind=kind,  # type: ignore[arg-type]
        slug=slug,
        version=version,
    )
    if content is None:
        return tool_error(
            f"Registry item not found: {kind}/{slug}@{version}",
            code="not_found",
        )
    return _dump(content)


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS: list[tuple[str, Any]] = [
    ("registry.persist_item", persist_item),
    ("registry.list_items", list_items),
    ("registry.get_item", get_item),
    ("registry.get_item_content", get_item_content),
]
