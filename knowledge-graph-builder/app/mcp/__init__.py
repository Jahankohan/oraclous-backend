"""Generic MCP capability surface — TASK-229 / STORY-035 / ADR-023 + ADR-024.

A single MCP server (ADR-024 D7-R) that projects the existing REST surface into
namespaced, generic MCP tools. The projection is uniform *within* each I/O class
(ADR-023 D3): plain request/response, file upload, streaming/SSE, async job.

This package delivers the server, the per-I/O-class projection *mechanism*, and
the per-request auth wiring. Out of scope here, by design:

  * the curated tool set and typed per-tool schemas — TASK-230;
  * the exposure allowlist, mounting into the app, docker packaging — TASK-232.

`build_mcp_asgi_app()` returns the MCP ASGI app with `BearerTokenASGIMiddleware`
applied — every MCP request's `Authorization` bearer is sourced per-call and
re-validated by `auth_service.verify_token` via the dispatched REST request,
which is also the session TTL (the token's own `exp`; TASK-231 / ADR-023 D5).
"""

from app.mcp.exposure import (
    DangerousCapabilityError,
    assert_safe_registry,
    is_dangerous,
)
from app.mcp.server import build_mcp_asgi_app, build_mcp_server

__all__ = [
    "DangerousCapabilityError",
    "assert_safe_registry",
    "build_mcp_asgi_app",
    "build_mcp_server",
    "is_dangerous",
]
