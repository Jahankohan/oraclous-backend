"""Tests for the MCP exposure guard and packaging — TASK-232 / ADR-023 D6.

ADR-023 D6: a capability is exposed over MCP only when explicitly allowlisted,
and administratively dangerous operations — data deletion, credential/key
rotation, permission/grant management, service-account management — are
*excluded by default*.

The registry (`app/mcp/registry.py`) is the allowlist: the projection is
registry-driven, so adding a REST endpoint never auto-exposes it. The guard
(`app/mcp/exposure.py`) enforces the other half — `assert_safe_registry` fails
the build loudly if any registered spec is dangerous, so a dangerous capability
can never be silently projected even if a future edit adds one.

Coverage:

  * the guard rejects a dangerous spec — DELETE method, a `/permissions` path,
    a `/service-accounts` path, a key-rotation path;
  * the live registry passes the guard;
  * no DELETE-method / permission / service-account capability is in the live
    registry — proving D6 holds on the shipped curated set;
  * packaging — the MCP server is reachable at `/mcp` *through the main
    FastAPI app* with the app's own lifespan running (the mounted sub-app +
    session-manager wiring of ADR-024 D7-R).

The guard tests are pure (no Docker dependency). The packaging test drives a
real HTTP request through `app.main.app` over `httpx.ASGITransport`, so it
needs the live Dockerized stack (the app lifespan initializes Neo4j/Postgres).
All live in `integration/` next to the other MCP tests.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.mcp.exposure import (
    DangerousCapabilityError,
    assert_safe_registry,
    is_dangerous,
)
from app.mcp.registry import REGISTRY, CapabilitySpec, IOClass

# --- dangerous-spec fixtures ------------------------------------------------
# Each is one administratively dangerous capability the guard must reject.

_DELETE_SPEC = CapabilitySpec(
    name="graph.delete",
    io_class=IOClass.PLAIN,
    method="DELETE",
    path="/api/v1/graphs/{graph_id}",
    description="Delete a knowledge graph permanently.",
    path_params=("graph_id",),
)

_PERMISSIONS_SPEC = CapabilitySpec(
    name="graph.grant",
    io_class=IOClass.PLAIN,
    method="POST",
    path="/api/v1/graphs/{graph_id}/permissions",
    description="Grant a principal access to a graph.",
    path_params=("graph_id",),
)

_SERVICE_ACCOUNT_SPEC = CapabilitySpec(
    name="agent.provision",
    io_class=IOClass.PLAIN,
    method="POST",
    path="/api/v1/service-accounts",
    description="Provision an agent service account.",
)

_KEY_ROTATION_SPEC = CapabilitySpec(
    name="agent.cycle_secret",
    io_class=IOClass.PLAIN,
    method="POST",
    path="/api/v1/service-accounts/{account_id}/rotate",
    description="Rotate a service account's signing key.",
    path_params=("account_id",),
)


# --- the guard rejects every dangerous class --------------------------------


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        (_DELETE_SPEC, "DELETE method (data deletion)"),
        (_PERMISSIONS_SPEC, "/permissions path (grant management)"),
        (_SERVICE_ACCOUNT_SPEC, "/service-accounts path (account management)"),
        (_KEY_ROTATION_SPEC, "/rotate path (credential/key rotation)"),
    ],
)
def test_guard_flags_dangerous_spec(spec: CapabilitySpec, label: str):
    """`is_dangerous` flags each administratively dangerous capability."""
    assert is_dangerous(spec), f"guard failed to flag a dangerous spec: {label}"


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        (_DELETE_SPEC, "DELETE method"),
        (_PERMISSIONS_SPEC, "/permissions path"),
        (_SERVICE_ACCOUNT_SPEC, "/service-accounts path"),
        (_KEY_ROTATION_SPEC, "/rotate path"),
    ],
)
def test_assert_safe_registry_rejects_dangerous_spec(spec: CapabilitySpec, label: str):
    """A registry with one dangerous spec fails the build loudly (ADR-023 D6).

    `assert_safe_registry` raises `DangerousCapabilityError` rather than letting
    the dangerous capability be silently projected to an MCP tool."""
    contaminated = (*REGISTRY, spec)
    with pytest.raises(DangerousCapabilityError) as exc_info:
        assert_safe_registry(contaminated)
    # The error names the offending spec so the fix is obvious.
    assert spec.name in str(exc_info.value), str(exc_info.value)


def test_dangerous_status_path_is_caught():
    """A dangerous pattern on an async-job *status* path is also caught.

    The guard inspects the async-job status path/method too — a capability
    cannot smuggle a dangerous operation through its poll half."""
    async_dangerous = CapabilitySpec(
        name="ingest.text",
        io_class=IOClass.ASYNC_JOB,
        method="POST",
        path="/api/v1/graphs/{graph_id}/ingest",
        description="Submit a job.",
        path_params=("graph_id",),
        status_name="ingest.cancel",
        status_method="DELETE",
        status_path="/api/v1/graphs/{graph_id}/jobs/{job_id}",
        status_path_params=("graph_id", "job_id"),
    )
    assert is_dangerous(async_dangerous)


# --- the live shipped registry is safe --------------------------------------


def test_live_registry_passes_the_guard():
    """The current curated registry contains no dangerous capability.

    `assert_safe_registry(REGISTRY)` must not raise — this is exactly the call
    `build_mcp_server()` makes before projection."""
    assert_safe_registry(REGISTRY)  # raises if any spec is dangerous


def test_no_dangerous_capability_in_live_registry():
    """No DELETE-method, permission, or service-account capability is exposed.

    Asserts the D6-excluded classes are absent from the shipped curated set —
    the property ADR-023 D6 promises holds on what actually ships."""
    for spec in REGISTRY:
        assert spec.method.upper() != "DELETE", (
            f"{spec.name} exposes a DELETE method — excluded by ADR-023 D6"
        )
        path_lower = spec.path.lower()
        assert "/permissions" not in path_lower, (
            f"{spec.name} exposes a permissions endpoint — excluded by D6"
        )
        assert "/service-accounts" not in path_lower, (
            f"{spec.name} exposes a service-accounts endpoint — excluded by D6"
        )
        assert not is_dangerous(spec), (
            f"{spec.name} matches a dangerous pattern — excluded by D6"
        )


# --- packaging: the MCP server is mounted in the main app -------------------


@pytest.mark.asyncio
async def test_mcp_server_is_reachable_through_main_app():
    """The MCP server answers at `/mcp` *through the main FastAPI app*.

    Proves the ADR-024 D7-R packaging: the one MCP server is mounted into the
    existing `app.main.app` (no new docker service), and its streamable-HTTP
    session manager is run from the app's own `lifespan`. The test enters that
    lifespan and drives a real `tools/list` JSON-RPC request at `/mcp` over
    `httpx.ASGITransport` against `app.main.app` — not the standalone MCP ASGI
    app. A reachable, non-empty tool list means the mount and the session-
    manager wiring are both correct; if the lifespan wiring were wrong the
    request would 500 with "Task group is not initialized".

    `follow_redirects=True` mirrors a real MCP streamable-HTTP client: a
    Starlette `Mount` answers the bare `/mcp` with a 307 to `/mcp/`, which a
    conformant client follows. The single mount point is `/mcp` either way —
    there is no doubled `/mcp/mcp` prefix (ADR-023 D4)."""
    from app.main import app as fastapi_app

    principal = {
        "id": str(uuid.uuid4()),
        "principal_type": "user",
        "email": "mcp-packaging-test@example.com",
    }
    with patch("app.api.dependencies.auth_service") as mock_auth:
        mock_auth.verify_token = AsyncMock(return_value=principal)

        # Entering the main app's lifespan starts the MCP session manager
        # (TASK-232 wiring). A mounted sub-app's lifespan does not run on its
        # own — this is exactly the property under test.
        async with fastapi_app.router.lifespan_context(fastapi_app):
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost:9000",
                follow_redirects=True,
            ) as client:
                resp = await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    },
                    headers={
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                        "Authorization": "Bearer mcp-packaging-test-token",
                    },
                )

    assert resp.status_code == 200, f"/mcp HTTP {resp.status_code}: {resp.text}"
    # The streamable-HTTP transport frames the JSON-RPC reply as SSE.
    payload = None
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[len("data:") :].strip())
            break
    assert payload is not None, f"no SSE data line in /mcp response: {resp.text!r}"
    tools = payload.get("result", {}).get("tools", [])
    assert tools, f"/mcp tools/list returned no tools: {payload}"
    names = {t["name"] for t in tools}
    # A representative curated tool is present — the projection ran behind /mcp.
    assert "graph.create" in names, names
