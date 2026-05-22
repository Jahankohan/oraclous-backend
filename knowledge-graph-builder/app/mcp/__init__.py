"""Generic MCP capability surface — TASK-229 / STORY-035 / ADR-023 + ADR-024.

A single MCP server (ADR-024 D7-R) that projects the existing REST surface into
namespaced, generic MCP tools. The projection is uniform *within* each I/O class
(ADR-023 D3): plain request/response, file upload, streaming/SSE, async job.

This package delivers the server and the per-I/O-class projection *mechanism*.
Out of scope here, by design:

  * the curated tool set and typed per-tool schemas — TASK-230;
  * per-call principal re-validation / session TTL — TASK-231;
  * the exposure allowlist, mounting into the app, docker packaging — TASK-232.
"""

from app.mcp.server import build_mcp_server

__all__ = ["build_mcp_server"]
