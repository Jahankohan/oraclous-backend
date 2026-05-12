"""Shared auth / scope helpers for the MCP assessment + registry tools.

These helpers reuse the existing JWT plumbing (`auth_service.verify_token`)
and ReBAC primitive (`verify_graph_access`) so MCP tools authenticate
exactly like the REST endpoints. We do NOT invent a parallel auth path
per TASK-080 §Constraints.

The MCP server is configured with a single bearer token via the
``ORACLOUS_API_KEY`` env var (see ``app.mcp.server``). For every tool
call we:

1. Resolve the principal from that token via the existing auth-service.
2. Read ``home_graph_id`` from the principal claim (same rule the REST
   write endpoints apply).
3. Call ``verify_graph_access`` to enforce the per-tool ACL level.

If the token is missing or the principal has no usable ``home_graph_id``,
the tool returns an ``{"error": ...}`` dict — never a silent permit.
The MCP server-level ``_api_key()`` raises ``RuntimeError`` on missing
``ORACLOUS_API_KEY``, which we catch and surface as an error response so
the MCP client gets a clean per-tool failure rather than the whole
server going down.
"""

from __future__ import annotations

import contextvars
from typing import Any

from fastapi import HTTPException

from app.api.dependencies import _current_principal, verify_graph_access
from app.core.neo4j_client import neo4j_client
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    REGISTRY_CATALOG_GRAPH_ID,
)
from app.services.assessment_service import AssessmentService
from app.services.auth_service import auth_service

# Re-exported so registry_tools.py can use the constant without duplicating
# the import path.
__all__ = [
    "AuthError",
    "ScopeError",
    "ASSESSMENTS_CATALOG_GRAPH_ID",
    "REGISTRY_CATALOG_GRAPH_ID",
    "resolve_principal",
    "assert_graph_access",
    "assert_catalog_admin",
    "build_service",
    "tool_error",
]


class AuthError(Exception):
    """Raised when the MCP tool cannot establish a principal."""


class ScopeError(Exception):
    """Raised when the principal is authenticated but lacks the required scope."""


def tool_error(message: str, *, code: str = "error") -> dict[str, Any]:
    """Shape an error response consistent across every MCP tool.

    MCP clients receive ``{"error": ..., "code": ...}`` — never a silent
    permit, never a raised exception that crashes the transport.
    """
    return {"error": message, "code": code}


async def resolve_principal(api_key: str | None) -> dict[str, Any]:
    """Verify the MCP server's bearer token and return the principal dict.

    Raises:
        AuthError: if the token is missing or invalid.
    """
    if not api_key:
        raise AuthError(
            "ORACLOUS_API_KEY is not set; MCP tools require a bearer token."
        )
    try:
        principal = await auth_service.verify_token(api_key)
    except HTTPException as exc:
        raise AuthError(str(exc.detail)) from exc
    if not principal or "id" not in principal:
        raise AuthError("Auth service returned an empty / malformed principal.")
    return principal


def principal_graph_id(principal: dict[str, Any]) -> str:
    """Resolve the principal's tenant graph_id from the JWT claim.

    Mirrors ``assessments._principal_graph_id`` (TASK-069). Raises
    :class:`ScopeError` when the principal is not bound to a tenant graph.
    """
    graph_id = principal.get("home_graph_id") or ""
    if not graph_id:
        raise ScopeError(
            "Principal has no home_graph_id claim. Assessment MCP tools "
            "require a JWT bound to a tenant graph."
        )
    return str(graph_id)


# Token used inside the contextvar so that ``verify_graph_access`` can
# branch on principal_type. The shared infrastructure stores the principal
# in ``_current_principal`` (set by the REST ``get_current_user``); we set
# it explicitly here for the duration of one tool call.
def _bind_principal(principal: dict[str, Any]) -> contextvars.Token:
    return _current_principal.set(principal)


def _unbind_principal(token: contextvars.Token) -> None:
    _current_principal.reset(token)


async def assert_graph_access(
    principal: dict[str, Any], graph_id: str, required_level: str
) -> str:
    """Enforce ReBAC for the given graph + level using the shared primitive.

    Raises:
        ScopeError: if the principal lacks the required level.
    """
    token = _bind_principal(principal)
    try:
        return await verify_graph_access(graph_id, required_level, str(principal["id"]))
    except HTTPException as exc:
        raise ScopeError(str(exc.detail)) from exc
    finally:
        _unbind_principal(token)


async def assert_catalog_admin(principal: dict[str, Any]) -> None:
    """Enforce platform-admin via the same primitive used by REST.

    Used by the admin-only ``assessment.search_findings`` tool. Mirrors
    the REST `verify_graph_access(ASSESSMENTS_CATALOG_GRAPH_ID, 'admin', user_id)`
    gate at `assessments_reads.search_findings`.
    """
    await assert_graph_access(principal, ASSESSMENTS_CATALOG_GRAPH_ID, "admin")


async def assert_registry_curated_write(principal: dict[str, Any]) -> None:
    """Admin gate for `curated` Registry writes (ADR-019)."""
    await assert_graph_access(principal, REGISTRY_CATALOG_GRAPH_ID, "admin")


def build_service() -> AssessmentService:
    """Construct an ``AssessmentService`` bound to the live async driver.

    Mirrors the REST ``_assessment_service`` dependency. Raises
    :class:`ScopeError` (mapped to a 503-like response) when Neo4j isn't
    available, so tools fail-closed instead of hanging.
    """
    if not neo4j_client.async_driver:
        raise ScopeError("Neo4j connection not available; service unavailable.")
    return AssessmentService(neo4j_client.async_driver)
