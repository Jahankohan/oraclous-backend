"""The single Oraclous MCP server (ADR-024 D7-R).

One MCP server with namespaced tool families — not capability-grouped servers.
`build_mcp_server()` projects the capability registry into tools and returns a
configured `FastMCP`. `build_mcp_asgi_app()` returns that server's streamable
HTTP ASGI app wrapped in the per-request bearer-token middleware (TASK-231).

`stateless_http=True` is deliberate: the retired substrate's "stale session"
failure (a restarted server rejecting a client's old session id) cannot recur
when there is no server-side session to go stale. Because there is no
server-side session, the effective session TTL is the caller token's own `exp`
claim — re-checked on every tool call by `auth_service.verify_token` (see
`app/mcp/auth.py`); there is deliberately no separate session store. Mounting
this server into the FastAPI app and the docker packaging are TASK-232.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.core.logging import get_logger
from app.mcp.auth import BearerTokenASGIMiddleware
from app.mcp.projection import project
from app.mcp.registry import REGISTRY

logger = get_logger(__name__)

_INSTRUCTIONS = (
    "Oraclous generic capability surface. Every tool is a namespaced platform "
    "primitive (graph.*, ingest.*, ...) — compose them to build a workflow. "
    "Nothing here is use-case-specific; a use case is composed by the caller, "
    "never baked into a tool."
)


def build_mcp_server() -> FastMCP:
    """Build the Oraclous MCP server with all projected tools registered."""
    tools = []
    for spec in REGISTRY:
        tools.extend(project(spec))

    logger.info(
        "MCP server: projected %d tools from %d capabilities: %s",
        len(tools),
        len(REGISTRY),
        ", ".join(sorted(t.name for t in tools)),
    )
    return FastMCP(
        name="oraclous",
        instructions=_INSTRUCTIONS,
        tools=tools,
        stateless_http=True,
    )


def build_mcp_asgi_app(server: FastMCP | None = None):
    """Build the Oraclous MCP server's ASGI app with per-request auth wired in.

    Returns the FastMCP streamable-HTTP ASGI app wrapped in
    `BearerTokenASGIMiddleware`, which sources the `Authorization` bearer token
    from each real MCP request and binds it on the `context.py` contextvar that
    the projection's dispatch reads (TASK-231). A request with no bearer leaves
    the contextvar unbound, so the tool call fails closed.

    Pass an existing `server` to reuse one (e.g. in tests); otherwise a fresh
    server is built.
    """
    mcp_server = server if server is not None else build_mcp_server()
    return BearerTokenASGIMiddleware(mcp_server.streamable_http_app())
