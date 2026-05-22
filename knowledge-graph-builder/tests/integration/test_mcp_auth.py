"""Auth-wiring tests for the MCP surface — TASK-231 / STORY-035.

TASK-229 left `app/mcp/context.py` as a contextvar seam; the projection's
dispatch forwards whatever token is bound there. TASK-231 sources that token
from a *real* MCP request: `BearerTokenASGIMiddleware` (`app/mcp/auth.py`) reads
the `Authorization` header off each inbound MCP HTTP request and binds it for
the duration of that request.

These tests therefore drive a real HTTP request through the MCP server's ASGI
app (`build_mcp_asgi_app()`) over `httpx.ASGITransport` — not the direct
`mcp.call_tool` path used by `test_mcp_projection.py`. Only an HTTP request
exercises the header -> contextvar middleware, which is the property under test.

Coverage (ADR-023 D5 — re-validate the principal on every tool call):

  * a request with a valid `Authorization` header dispatches with that
    principal;
  * a request with NO header fails closed;
  * mid-session revocation — a token that resolved to a principal on an
    earlier call is rejected on a later call, and the later tool call is
    refused (re-validation is structural: every call re-runs `verify_token`);
  * cross-tenant — a principal calling a tool against a `graph_id` it has no
    ReBAC grant on is denied.

The external auth-service is bypassed the same way `test_mcp_projection.py`
does it: `app.api.dependencies.auth_service.verify_token` is patched. The real
projection, in-process dispatch, routing, `verify_graph_access` (real ReBAC
against Neo4j) and services all run.

Session TTL: `stateless_http=True` means there is no server-side session to go
stale (the retired substrate's TASK-091 failure). The effective session
lifetime is the caller token's own `exp`, enforced per call by `verify_token` —
the mid-session-revocation test exercises exactly that per-call freshness.

Run inside the kg-builder container:

    docker compose exec -T knowledge-graph-builder \\
        python -m pytest tests/integration/test_mcp_auth.py -q
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException, status

from app.mcp.server import build_mcp_asgi_app, build_mcp_server

# Two distinct principals — used by the cross-tenant test. The patched
# auth-service maps each token string to one of these.
_OWNER_ID = str(uuid.uuid4())
_OWNER: dict[str, Any] = {
    "id": _OWNER_ID,
    "principal_type": "user",
    "email": "mcp-auth-owner@example.com",
}
_OWNER_TOKEN = "mcp-auth-owner-token"  # noqa: S105 - test fixture, not a credential

_INTRUDER_ID = str(uuid.uuid4())
_INTRUDER: dict[str, Any] = {
    "id": _INTRUDER_ID,
    "principal_type": "user",
    "email": "mcp-auth-intruder@example.com",
}
_INTRUDER_TOKEN = "mcp-auth-intruder-token"  # noqa: S105 - test fixture

# A token that the patched auth-service treats as revoked / expired.
_REVOKED_TOKEN = "mcp-auth-revoked-token"  # noqa: S105 - test fixture

# DNS-rebinding protection on the MCP transport allows only localhost-family
# hosts; the ASGITransport never opens a socket, so this host is cosmetic but
# must satisfy that allowlist.
_BASE_URL = "http://localhost:9000"
# `build_mcp_asgi_app()` is tested here as a *standalone* ASGI app — not the
# `/mcp` mount in `app.main`. FastMCP serves its streamable-HTTP route at the
# server root (`streamable_http_path="/"`, set in `server.py` so the `/mcp`
# mount has no doubled prefix — ADR-023 D4), so the standalone app's route is
# `/`. The mounted surface is `/mcp`; that is exercised by test_mcp_exposure.py.
_MCP_PATH = "/"
_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _verify_token_router(token: str) -> dict[str, Any]:
    """Stand-in for the auth-service: map a token string to a principal.

    A valid token resolves to its principal; the revoked token raises the same
    401 `verify_token` raises when the real auth-service rejects a token. This
    is the per-call re-validation seam — every dispatched REST request calls
    through here, so a token that flips to revoked between calls is honored.
    """
    mapping = {_OWNER_TOKEN: _OWNER, _INTRUDER_TOKEN: _INTRUDER}
    if token in mapping:
        return mapping[token]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
    )


# --- fixtures ---------------------------------------------------------------


@asynccontextmanager
async def _mcp_client(*, verify_token: Any = None) -> AsyncIterator[httpx.AsyncClient]:
    """A real HTTP client bound to the MCP ASGI app, with auth patched.

    Yields an `httpx.AsyncClient` whose transport is the MCP server's ASGI app
    (middleware applied). Both the FastAPI app lifespan (so dispatch has a live
    backend) and the MCP streamable-HTTP session-manager lifespan are entered.

    A plain async context manager rather than a fixture: the MCP
    streamable-HTTP session manager runs an `anyio` task group whose cancel
    scope must be entered and exited in the *same* task. Entering this CM
    inside the test body keeps setup and teardown on the test's own task,
    which a module/function-scoped async fixture does not guarantee.

    `verify_token` overrides the auth-service stub for the revocation test;
    it defaults to `_verify_token_router`.
    """
    from app.main import app as fastapi_app

    side_effect = verify_token or _verify_token_router
    with patch("app.api.dependencies.auth_service") as mock_auth:
        mock_auth.verify_token = AsyncMock(side_effect=side_effect)

        server = build_mcp_server()
        asgi_app = build_mcp_asgi_app(server)
        # The session-manager task group lives on the inner Starlette app's
        # lifespan; it must be running before any request is handled.
        inner_app = server.streamable_http_app()

        async with fastapi_app.router.lifespan_context(fastapi_app):
            async with inner_app.router.lifespan_context(inner_app):
                transport = httpx.ASGITransport(app=asgi_app)
                async with httpx.AsyncClient(
                    transport=transport, base_url=_BASE_URL
                ) as client:
                    yield client


# --- helpers ----------------------------------------------------------------


def _parse_sse(text: str) -> dict[str, Any]:
    """Extract the JSON-RPC payload from an SSE-framed MCP response body."""
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :].strip())
    raise AssertionError(f"no SSE data line in MCP response: {text!r}")


async def _call_tool(
    client: httpx.AsyncClient,
    name: str,
    arguments: dict[str, Any],
    *,
    token: str | None,
) -> httpx.Response:
    """POST a JSON-RPC `tools/call` to the MCP ASGI app over real HTTP.

    When `token` is None no `Authorization` header is sent — the path that must
    fail closed. Returns the raw `httpx.Response`.
    """
    headers = dict(_MCP_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    return await client.post(_MCP_PATH, json=body, headers=headers)


def _tool_result(resp: httpx.Response) -> dict[str, Any]:
    """The JSON-RPC `result` object from a 200 MCP `tools/call` response."""
    assert resp.status_code == 200, f"MCP HTTP {resp.status_code}: {resp.text}"
    payload = _parse_sse(resp.text)
    assert "result" in payload, f"no JSON-RPC result: {payload}"
    return payload["result"]


def _tool_payload(resp: httpx.Response) -> Any:
    """The projected tool's JSON payload from a non-error `tools/call`."""
    result = _tool_result(resp)
    assert not result.get("isError"), f"tool reported an error: {result}"
    for block in result.get("content", []):
        if block.get("type") == "text":
            return json.loads(block["text"])
    raise AssertionError(f"no text content block in tool result: {result}")


def _is_tool_error(resp: httpx.Response) -> tuple[bool, str]:
    """Whether a 200 `tools/call` response is a rejection of any form, + text.

    A tool call can be refused two ways, both of which count here:
      * a FastMCP tool-level error — `result.isError` true (e.g. the
        projection's `_require_token()` raising for an unbound principal);
      * a structured REST error — the projection turns a REST 4xx into an
        `{"error": true, "http_status": ...}` JSON payload carried as a
        successful tool result (e.g. a 401 from `verify_token`, a 403 from
        `verify_graph_access`).
    """
    result = _tool_result(resp)
    text = ""
    structured_error = False
    for block in result.get("content", []):
        if block.get("type") == "text":
            raw = block.get("text", "")
            text += raw
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and parsed.get("error"):
                structured_error = True
    return bool(result.get("isError")) or structured_error, text


async def _create_graph(client: httpx.AsyncClient, name: str, *, token: str) -> str:
    """Create a graph through the MCP `graph.create` tool; return its id."""
    resp = await _call_tool(
        client,
        "graph.create",
        {"body": {"name": name, "description": "MCP auth-wiring test graph"}},
        token=token,
    )
    payload = _tool_payload(resp)
    assert not payload.get("error"), f"graph.create failed: {payload}"
    graph_id = payload.get("id")
    assert graph_id, f"graph.create returned no id: {payload}"
    return str(graph_id)


# --- tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_authorization_header_dispatches_with_principal():
    """A real MCP request carrying a valid `Authorization` header dispatches.

    Proves the header -> contextvar middleware: the projection's `_require_token`
    reads the contextvar `BearerTokenASGIMiddleware` bound from this request's
    header, so the tool dispatches as the owner principal and a graph the owner
    created reads back."""
    async with _mcp_client() as client:
        name = f"mcp-auth-valid-{uuid.uuid4().hex[:8]}"
        graph_id = await _create_graph(client, name, token=_OWNER_TOKEN)

        resp = await _call_tool(
            client, "graph.get", {"graph_id": graph_id}, token=_OWNER_TOKEN
        )
        payload = _tool_payload(resp)
        assert not payload.get("error"), f"graph.get failed: {payload}"
        assert str(payload.get("id")) == graph_id
        assert payload.get("name") == name


@pytest.mark.asyncio
async def test_missing_authorization_header_fails_closed():
    """An MCP request with NO `Authorization` header is refused.

    With no header the middleware binds None, so `_require_token()` raises and
    the tool call fails closed — the platform's deny-by-default rule. No env-var
    API key exists to fall back on (the retired TASK-091 static-key path)."""
    async with _mcp_client() as client:
        resp = await _call_tool(client, "graph.list", {}, token=None)
        is_error, text = _is_tool_error(resp)
        assert is_error, f"unauthenticated call was not refused: {text}"
        assert "principal" in text.lower() or "bearer" in text.lower(), text


@pytest.mark.asyncio
async def test_invalid_token_is_rejected():
    """A request whose bearer token the auth-service rejects is refused.

    A structurally-present but unrecognised token reaches `verify_token` via
    the dispatched REST request and is rejected there — an invalid credential
    never silently succeeds."""
    async with _mcp_client() as client:
        resp = await _call_tool(
            client,
            "graph.list",
            {},
            token="not-a-real-token",  # noqa: S106
        )
        is_error, text = _is_tool_error(resp)
        assert is_error, f"invalid token was not refused: {text}"


@pytest.mark.asyncio
async def test_mid_session_revocation_is_rejected_on_next_call():
    """A token valid on an earlier call is refused once revoked mid-session.

    Re-validation is structural: every tool call dispatches a fresh REST
    request that re-runs `verify_token`. Here the first call's token resolves
    to a principal; the auth-service is then flipped so the *same* token is
    rejected; the next tool call must be refused. `stateless_http=True` means
    there is no server-side session to keep a stale principal alive — the
    token's own validity is the session, checked per call."""

    # First MCP session: the revoked-token string is, for now, a valid
    # principal — the auth-service stub resolves it.
    def _initially_valid(token: str) -> dict[str, Any]:
        if token == _REVOKED_TOKEN:
            return _OWNER
        return _verify_token_router(token)

    async with _mcp_client(verify_token=_initially_valid) as client:
        first = await _call_tool(client, "graph.list", {}, token=_REVOKED_TOKEN)
        is_error, text = _is_tool_error(first)
        assert not is_error, f"first call should have succeeded: {text}"

        # Mid-session: the auth-service is flipped to reject that same token.
        # Re-patching `verify_token` in place leaves the live MCP session
        # untouched; the next tool call re-runs the now-rejecting stub.
        from app.api.dependencies import auth_service

        auth_service.verify_token = AsyncMock(side_effect=_verify_token_router)
        later = await _call_tool(client, "graph.list", {}, token=_REVOKED_TOKEN)
        is_error, text = _is_tool_error(later)
        assert is_error, f"revoked token was still accepted on a later call: {text}"


@pytest.mark.asyncio
async def test_cross_tenant_graph_access_is_denied():
    """A principal cannot reach a `graph_id` it has no ReBAC grant on.

    The owner creates a graph; an intruder principal — a different, valid
    token — calls a tool against that `graph_id`. `verify_graph_access` runs
    real ReBAC against Neo4j on the dispatched REST request and denies it. The
    MCP surface adds no tenancy bypass: `graph_id` scoping is enforced exactly
    where REST enforces it."""
    async with _mcp_client() as client:
        name = f"mcp-auth-tenant-{uuid.uuid4().hex[:8]}"
        graph_id = await _create_graph(client, name, token=_OWNER_TOKEN)

        resp = await _call_tool(
            client, "graph.get", {"graph_id": graph_id}, token=_INTRUDER_TOKEN
        )
        payload = _tool_payload(resp)
        # The projection surfaces a REST 4xx as a structured error object.
        assert payload.get("error"), (
            f"intruder reached another tenant's graph: {payload}"
        )
        assert payload.get("http_status") in (403, 404), payload
