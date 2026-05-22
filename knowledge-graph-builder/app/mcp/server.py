"""The single Oraclous MCP server (ADR-024 D7-R).

One MCP server with namespaced tool families — not capability-grouped servers.
`build_mcp_server()` projects the capability registry into tools and returns a
configured `FastMCP`.

`stateless_http=True` is deliberate: the retired substrate's "stale session"
failure (a restarted server rejecting a client's old session id) cannot recur
when there is no server-side session to go stale. Mounting this server into the
FastAPI app and the docker packaging are TASK-232.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.core.logging import get_logger
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
