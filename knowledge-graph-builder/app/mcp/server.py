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

from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from app.core.logging import get_logger
from app.mcp.auth import BearerTokenASGIMiddleware
from app.mcp.exposure import assert_safe_registry
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
    """Build the Oraclous MCP server with all projected tools registered.

    The registry is the exposure allowlist (ADR-023 D6) — a capability reaches
    MCP only when explicitly added as a `CapabilitySpec`. `assert_safe_registry`
    enforces D6's other half: it fails the build loudly if any registered spec
    is an administratively dangerous operation (data deletion, credential/key
    rotation, permission/grant or service-account management), so a dangerous
    capability can never be silently projected.
    """
    assert_safe_registry(REGISTRY)

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
        # The streamable-HTTP route is served at the server's *root*, not at
        # `/mcp`. The mount point is `/mcp` in `app.main` (TASK-232); leaving
        # FastMCP's default `/mcp` route would make the effective URL
        # `/mcp/mcp` — the exact doubled-prefix failure ADR-023 D4 forbids.
        # Root here + a `/mcp` mount there = a single, clean `/mcp` path.
        streamable_http_path="/",
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


class MCPMount:
    """A re-bindable ASGI mount for the MCP server (TASK-232 packaging).

    `app.main` mounts *this* at `/mcp`, then calls `run()` from its FastAPI
    `lifespan`. `run()` is an async context manager that:

      * builds a fresh `FastMCP` server (and so a fresh, single-use streamable-
        HTTP session manager — `StreamableHTTPSessionManager.run()` may be
        called only once per instance);
      * binds the auth-wrapped ASGI app of that server as the current handler;
      * runs that server's session-manager task group for the body of the
        `with`, then unbinds and tears it down.

    Why a re-bindable mount rather than a frozen pre-built app: the FastMCP
    session manager is single-use, so a process that enters its lifespan more
    than once (every MCP integration test does — it drives `app.main`'s
    lifespan repeatedly) would hit `RuntimeError` on the second `run()`.
    Rebuilding per lifespan entry gives each entry its own fresh session
    manager. A production process enters the lifespan exactly once, so it
    builds exactly one server — identical to a module singleton, but re-entrant
    for tests.

    Before `run()` has bound a handler, the mount answers 503 — the MCP surface
    is simply not live until the app lifespan has started it.
    """

    def __init__(self) -> None:
        self._handler = None

    async def __call__(self, scope, receive, send):
        handler = self._handler
        if handler is None:
            # The lifespan has not started the MCP server yet (or has stopped
            # it). Fail closed with a clear 503 rather than a confusing 500.
            if scope.get("type") == "http":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"MCP server is not running",
                    }
                )
                return
            # Non-HTTP scope with no handler — nothing to do.
            return
        await handler(scope, receive, send)

    @asynccontextmanager
    async def run(self):
        """Build a fresh MCP server, bind it, and run its session manager.

        Wire this into `app.main`'s `lifespan`:

            mcp_mount = MCPMount()
            app.mount("/mcp", mcp_mount)
            ...
            async with mcp_mount.run():
                yield

        The session-manager task group's cancel scope is entered and exited on
        the lifespan generator's own task — the constraint FastMCP documents.
        """
        server = build_mcp_server()
        self._handler = build_mcp_asgi_app(server)
        try:
            async with server.session_manager.run():
                logger.info("MCP server session manager started — /mcp is live")
                yield
        finally:
            self._handler = None
