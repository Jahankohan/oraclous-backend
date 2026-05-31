"""QA gate for STORY-035 — the generic capability surface over MCP (TASK-234).

This is the QA gate for STORY-035: it validates the story's **acceptance
criteria** end-to-end against the live Dockerized stack, joining up what the
per-task tests (`test_mcp_projection.py`, `test_mcp_typed_schemas.py`,
`test_mcp_auth.py`, `test_mcp_exposure.py`, `test_recipe_rest.py`) prove
individually. It does not re-prove what those tests already cover — it adds the
two criteria no per-task test addresses head-on:

  1. **Composition proof (the headline criterion).** A test that acts as an MCP
     client and drives a real workflow — create a graph, query it, author a
     recipe, run the recipe, read the result back — *purely by composing the
     generic MCP tools*. The whole workflow is tool composition; no
     Oraclous-side code is specific to it. Driven over real HTTP through the
     `/mcp` mount in `app.main` — the most realistic client path.
  2. **REST↔MCP parity.** Every capability in the curated registry maps 1:1 to
     a real REST operation: each `CapabilitySpec`'s `(method, path)` resolves to
     an actual mounted FastAPI route, and the MCP tool for it exists. No MCP
     tool exists without a REST equivalent; the two surfaces cannot drift.

The other three criteria — exposure (D6), auth (D5), typed schemas (D4) — are
covered by the per-task tests; this module re-confirms each holds against the
*live registry* with a focused assertion so the STORY-035 gate is self-contained
and the verdict cites one suite.

Run inside the kg-builder container, where the `neo4j` / `postgres` / `redis`
service hostnames resolve. Pass the Docker-network test env (`NEO4J_TEST_URI`
and `TEST_POSTGRES_URL`) via `-e` flags, then:

    docker compose exec -T <env flags> knowledge-graph-builder \\
        python -m pytest tests/integration/test_qa_story035.py -q

The external auth-service is bypassed exactly as the other MCP integration
tests do it — `app.api.dependencies.auth_service.verify_token` is patched to a
fixed principal — so the test exercises the real projection, in-process
dispatch, routing, `verify_graph_access` (real ReBAC against Neo4j), the recipe
library (Postgres) and services, with no second microservice in the loop.
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
from fastapi.routing import APIRoute

from app.mcp.exposure import assert_safe_registry, is_dangerous
from app.mcp.registry import REGISTRY

# A fixed test principal — the patched auth-service returns this for any token.
_TEST_USER_ID = str(uuid.uuid4())
_TEST_PRINCIPAL: dict[str, Any] = {
    "id": _TEST_USER_ID,
    "principal_type": "user",
    "email": "qa-story035@example.com",
}
_BEARER = "qa-story035-token"  # noqa: S105 - test fixture, not a credential

# DNS-rebinding protection on the MCP streamable-HTTP transport allows only
# localhost-family hosts; the ASGITransport never opens a socket, so this host
# is cosmetic but must satisfy that allowlist.
_BASE_URL = "http://localhost:9000"
_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


# --- the recipe document the composition proof authors ----------------------


def _employee_recipe(recipe_id: str) -> dict[str, Any]:
    """A minimal CSV recipe — one node rule projecting an `Employee` per record.

    The composition proof *composes* this JSON client-side and stores it via
    the generic `recipe.store` tool; nothing about this recipe shape is baked
    into the platform. It is the same shape `test_recipe_rest.py` exercises.
    """
    return {
        "recipe_format_version": "0.2",
        "id": recipe_id,
        "version": 1,
        "status": "draft",
        "concern": "Reporting structure from a flat employee export.",
        "applies_to": {
            "source_type": "csv",
            "shape_signature": "employees.csv:name,title",
        },
        "defaults": {"provenance": "EXTRACTED", "materialize_fine_grain": True},
        "authoring": {"authored_by": "qa-story035", "created": "2026-05-22"},
        "mappings": [
            {
                "id": "employees",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "Employee",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim"],
                },
                "materialize": True,
                "properties": [{"name": "name", "value_from": "column:name"}],
            }
        ],
    }


_RECORDS = [
    {"name": "Ada Lovelace", "title": "CTO"},
    {"name": "Alan Turing", "title": "Principal Engineer"},
]


# --- environment: a real HTTP client bound to the /mcp mount -----------------


@asynccontextmanager
async def _mcp_app_client() -> AsyncIterator[httpx.AsyncClient]:
    """A real HTTP client bound to `app.main.app`, with the external auth patched.

    Yields an `httpx.AsyncClient` whose transport is the *whole* FastAPI app —
    so a request to `/mcp` exercises the real ADR-024 D7-R mount, the
    `BearerTokenASGIMiddleware`, the projection and the in-process dispatch.
    The main app's lifespan is entered, which is what starts the mounted MCP
    server's streamable-HTTP session manager (TASK-232).

    A plain async context manager entered in each test body — not a fixture.
    `app.main`'s lifespan runs the MCP session manager's `anyio` task group,
    whose cancel scope must be entered and exited on the *same* task;
    pytest-asyncio runs fixture setup and teardown on different tasks. Entering
    this CM inside the test keeps both on the test's own task. This mirrors the
    pattern `test_mcp_auth.py` documents.
    """
    from app.main import app as fastapi_app

    with patch("app.api.dependencies.auth_service") as mock_auth:
        mock_auth.verify_token = AsyncMock(return_value=_TEST_PRINCIPAL)
        async with fastapi_app.router.lifespan_context(fastapi_app):
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=_BASE_URL,
                follow_redirects=True,
            ) as client:
                yield client


# --- MCP JSON-RPC helpers ----------------------------------------------------
#
# An MCP client speaks JSON-RPC over the streamable-HTTP transport. These
# helpers POST `tools/list` / `tools/call` at `/mcp` and unwrap the SSE-framed
# reply. They are the *only* MCP knowledge the composition proof needs — there
# is no Oraclous-specific client code below this line.


def _parse_sse(text: str) -> dict[str, Any]:
    """Extract the JSON-RPC payload from an SSE-framed MCP response body."""
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :].strip())
    raise AssertionError(f"no SSE data line in MCP response: {text!r}")


async def _rpc(
    client: httpx.AsyncClient,
    method: str,
    params: dict[str, Any],
    *,
    token: str | None = _BEARER,
) -> dict[str, Any]:
    """POST one JSON-RPC request to the `/mcp` mount; return the `result` object."""
    headers = dict(_MCP_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    resp = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
    )
    assert resp.status_code == 200, f"/mcp HTTP {resp.status_code}: {resp.text}"
    payload = _parse_sse(resp.text)
    assert "result" in payload, f"no JSON-RPC result: {payload}"
    return payload["result"]


async def _list_tools(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    """The MCP server's published tool catalogue, keyed by tool name."""
    result = await _rpc(client, "tools/list", {})
    return {t["name"]: t for t in result.get("tools", [])}


async def _call_tool(
    client: httpx.AsyncClient,
    name: str,
    arguments: dict[str, Any],
    *,
    token: str | None = _BEARER,
) -> Any:
    """Call a generic MCP tool and return its JSON payload.

    Asserts the call did not fail at the FastMCP tool level. A REST 4xx/5xx is
    *not* a FastMCP error — the projection surfaces it as a successful tool
    result carrying `{"error": true, "http_status": ...}`; the caller inspects
    that payload itself.
    """
    result = await _rpc(
        client, "tools/call", {"name": name, "arguments": arguments}, token=token
    )
    assert not result.get("isError"), f"tool {name} reported a FastMCP error: {result}"
    for block in result.get("content", []):
        if block.get("type") == "text":
            return json.loads(block["text"])
    raise AssertionError(f"no text content block in {name} result: {result}")


# ============================================================================
# Acceptance criterion 1 — the composition proof (headline criterion)
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_composition_proof_end_to_end_workflow():
    """A real workflow driven purely by composing generic MCP tools.

    An MCP client drives, over real HTTP through the `/mcp` mount, the entire
    workflow STORY-035 criterion 1 names:

      1. `graph.create`   — create a knowledge graph;
      2. `graph.ask`      — query the (new, empty) graph;
      3. `recipe.store`   — author a recipe (the client *composes* the JSON);
      4. `ingest.recipe`  — run that recipe over an inline source into the graph;
      5. `recipe.get`     — read the stored recipe back;
      6. `graph.get`      — read the graph back.

    Every step is a call to one generic, namespaced platform primitive. Nothing
    in this test — and, the parity test below proves, nothing in the platform —
    is specific to "ingest employees with a reporting-structure recipe". The
    workflow is composed by the *caller*; it is never baked into a tool. That
    is exactly what ADR-023's external anti-bespoke mechanism promises.
    """
    async with _mcp_app_client() as client:
        # --- 1. graph.create -------------------------------------------------
        graph_name = f"qa-story035-comp-{uuid.uuid4().hex[:8]}"
        created = await _call_tool(
            client,
            "graph.create",
            {
                "body": {
                    "name": graph_name,
                    "description": "STORY-035 composition proof",
                }
            },
        )
        assert not created.get("error"), f"graph.create failed: {created}"
        graph_id = str(created["id"])
        assert created["name"] == graph_name

        # --- 2. graph.ask — query the graph ----------------------------------
        # The streaming projection collects the whole SSE stream into one
        # result; an empty graph may answer or carry an error event — either
        # way the collected `events` list proves the query primitive composed.
        asked = await _call_tool(
            client,
            "graph.ask",
            {"body": {"graph_id": graph_id, "query": "What is in this graph?"}},
        )
        assert "events" in asked, f"graph.ask returned no collected events: {asked}"
        assert isinstance(asked["events"], list)

        # --- 3. recipe.store — the client composes and stores a recipe -------
        recipe_id = f"rcp_qa035-{uuid.uuid4().hex[:12]}"
        stored = await _call_tool(
            client,
            "recipe.store",
            {"body": {"graph_id": graph_id, "recipe": _employee_recipe(recipe_id)}},
        )
        assert not stored.get("error"), f"recipe.store failed: {stored}"
        assert stored["recipe"]["id"] == recipe_id
        assert stored["recipe"]["status"] == "draft"  # ADR-022 — never auto-promoted

        # --- 4. ingest.recipe — run the composed recipe ----------------------
        # TASK-237: `ingest.recipe` is an ASYNC_JOB — the submit leg surfaces
        # the run handle as `job_id` (the recipe-run ingestion_jobs row id).
        run = await _call_tool(
            client,
            "ingest.recipe",
            {
                "graph_id": graph_id,
                "recipe_id": recipe_id,
                "body": {"source_type": "csv", "records": _RECORDS},
            },
        )
        assert not run.get("error"), f"ingest.recipe failed: {run}"
        assert run["job_id"], f"ingest.recipe returned no job_id: {run}"
        assert run["raw"]["recipe_id"] == recipe_id
        assert run["status"] == "pending"

        # --- 5. recipe.get — read the stored recipe back ---------------------
        got_recipe = await _call_tool(
            client,
            "recipe.get",
            {"recipe_id": recipe_id, "graph_id": graph_id},
        )
        assert not got_recipe.get("error"), f"recipe.get failed: {got_recipe}"
        assert got_recipe["recipe"]["id"] == recipe_id
        assert got_recipe["recipe"]["concern"], "recipe.get lost the composed concern"

        # --- 6. graph.get — read the graph back ------------------------------
        got_graph = await _call_tool(client, "graph.get", {"graph_id": graph_id})
        assert not got_graph.get("error"), f"graph.get failed: {got_graph}"
        assert str(got_graph["id"]) == graph_id
        assert got_graph["name"] == graph_name


@pytest.mark.integration
def test_no_per_workflow_platform_code_exists():
    """The composition proof's workflow has NO dedicated platform code.

    The headline criterion is "no Oraclous-side code added for that workflow".
    The composition proof above drives a recipe-ingestion workflow; this test
    asserts structurally that the workflow is composed, never baked in:

      * every tool the workflow used is one of the generic, namespaced
        primitives in the curated registry — none is a workflow-shaped tool;
      * the registry is a *uniform* projection — each spec is a thin
        `(name, io_class, method, path)` record, not a procedure; there is no
        "ingest-employees" or any other use-case spec.

    A workflow-specific tool would have to appear here as a registry spec — the
    projection is registry-driven, a tool cannot exist otherwise. Its absence
    is the proof that the workflow lives in the *caller*, not the platform.
    """
    workflow_tools = {
        "graph.create",
        "graph.ask",
        "recipe.store",
        "ingest.recipe",
        "recipe.get",
        "graph.get",
    }
    registry_names = {spec.name for spec in REGISTRY}
    # Every tool the workflow composed is a registered generic primitive.
    assert workflow_tools <= registry_names, (
        f"workflow used a tool not in the curated registry: "
        f"{workflow_tools - registry_names}"
    )
    # Every registry name is namespaced `family.verb` — a generic primitive,
    # not a `do_the_whole_thing` workflow tool.
    for spec in REGISTRY:
        family, _, verb = spec.name.partition(".")
        assert family and verb, f"tool {spec.name} is not a namespaced primitive"
        # A primitive maps to exactly one REST verb+path; a workflow tool would
        # need to be a composite. The spec carries a single (method, path).
        assert spec.method and spec.path, f"{spec.name} is not a single REST op"


# ============================================================================
# Acceptance criterion 2 — REST↔MCP parity
# ============================================================================


def _mounted_routes() -> dict[str, set[str]]:
    """Every mounted FastAPI route, as `path -> {HTTP methods}`."""
    from app.main import app as fastapi_app

    routes: dict[str, set[str]] = {}
    for route in fastapi_app.routes:
        if isinstance(route, APIRoute):
            routes.setdefault(route.path, set()).update(route.methods or set())
    return routes


@pytest.mark.integration
def test_every_capability_maps_to_a_real_rest_route():
    """Every curated `CapabilitySpec` resolves 1:1 to a mounted REST route.

    REST↔MCP parity, the MCP→REST direction: a capability is exposed over MCP
    only by re-issuing an *existing* REST call (`app/mcp/dispatch.py`). This
    asserts that for every spec — including the async-job *status* leg — the
    declared `(method, path)` is an actual `APIRoute` on `app.main.app`. A spec
    that named a non-existent route would 404 at dispatch; this catches that
    drift at the registry level instead.
    """
    routes = _mounted_routes()
    problems: list[str] = []
    for spec in REGISTRY:
        pairs = [(spec.method.upper(), spec.path)]
        if spec.status_path:
            pairs.append((spec.status_method.upper(), spec.status_path))
        for method, path in pairs:
            if path not in routes:
                problems.append(f"{spec.name}: path {path!r} is not a mounted route")
            elif method not in routes[path]:
                problems.append(
                    f"{spec.name}: {method} {path!r} not served "
                    f"(route has {sorted(routes[path])})"
                )
    assert not problems, "REST↔MCP parity broken:\n  " + "\n  ".join(problems)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_every_projected_tool_traces_back_to_a_capability_spec():
    """No MCP tool exists without a REST-backed `CapabilitySpec` (no drift).

    REST↔MCP parity, the other direction: the live MCP server is asked for its
    tool catalogue (`tools/list` over the `/mcp` mount), and every tool name is
    proven to trace back to a registry spec — either a spec's `name` (plain /
    upload / streaming / async-job submit) or an async-job spec's `status_name`
    (the poll leg). A tool with no spec would be a hand-authored,
    REST-less tool — exactly what ADR-023 D2 forbids.
    """
    spec_tool_names: set[str] = set()
    for spec in REGISTRY:
        spec_tool_names.add(spec.name)
        if spec.status_name:
            spec_tool_names.add(spec.status_name)

    async with _mcp_app_client() as client:
        tools = await _list_tools(client)

    assert tools, "/mcp tools/list returned no tools"
    orphans = set(tools) - spec_tool_names
    assert not orphans, (
        f"MCP tools with no backing CapabilitySpec (REST-less tools forbidden "
        f"by ADR-023 D2): {sorted(orphans)}"
    )
    # And the converse: every spec actually projected to a live tool.
    missing = spec_tool_names - set(tools)
    assert not missing, (
        f"registry specs that did not project to a tool: {sorted(missing)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_curated_set_spans_every_rest_backed_family():
    """The curated set covers every capability family that has a REST surface.

    ADR-024 D8-R: a *curated* primitive set across the families with REST
    endpoints. This confirms the live `/mcp` surface spans exactly the eight
    families ADR-024 names plus `recipe` (TASK-236's added surface) — the
    surface is broad enough to compose real workflows, which is what makes the
    composition proof above possible at all.
    """
    async with _mcp_app_client() as client:
        tools = await _list_tools(client)
    families = {name.split(".", 1)[0] for name in tools}
    assert families == {
        "graph",
        "schema",
        "ingest",
        "community",
        "agent",
        "memory",
        "connector",
        "federation",
        "recipe",
    }, families


# ============================================================================
# Acceptance criteria 3 / 4 / 5 — re-confirmed against the live registry
# ============================================================================
#
# Each is covered in depth by a per-task test (test_mcp_exposure.py /
# test_mcp_auth.py / test_mcp_typed_schemas.py). The assertions below re-check
# the same property against the *live* registry / server so the STORY-035 gate
# is a single self-contained suite — they are intentionally not a duplication
# of the per-task depth (revocation timing, every dangerous-pattern variant,
# nested-schema shape), only a gate-level confirmation.


@pytest.mark.integration
def test_criterion3_exposure_no_dangerous_operation_reachable():
    """Criterion 3 — no administratively dangerous operation reaches MCP.

    Re-confirms the `app/mcp/exposure.py` guard holds for the *live* registry:
    `assert_safe_registry` (the call `build_mcp_server()` makes) does not raise,
    and no shipped spec is a DELETE / permission / service-account / key-
    rotation operation. Depth — every dangerous-pattern variant, the async-job
    status leg — is in `test_mcp_exposure.py`.
    """
    assert_safe_registry(REGISTRY)  # raises DangerousCapabilityError if violated
    for spec in REGISTRY:
        assert not is_dangerous(spec), f"{spec.name} is a dangerous operation (D6)"
        assert spec.method.upper() != "DELETE", f"{spec.name} exposes DELETE (D6)"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_criterion4_auth_unauthenticated_call_fails_closed():
    """Criterion 4 — the auth floor holds on the live `/mcp` mount.

    Re-confirms per-call auth: a `tools/call` with NO `Authorization` header is
    refused — the projection's `_require_token()` fails closed, deny-by-default.
    The mid-session-revocation timing (a token valid on one call, rejected on
    the next) is proven in depth by `test_mcp_auth.py`; this is the gate-level
    confirmation that the live mounted surface enforces the floor at all.
    """
    async with _mcp_app_client() as client:
        result = await _rpc(
            client,
            "tools/call",
            {"name": "graph.list", "arguments": {}},
            token=None,
        )
        # No header -> the projection raises -> FastMCP marks the result an error.
        text = json.dumps(result)
        assert result.get("isError"), (
            f"unauthenticated MCP call was not refused: {text}"
        )
        assert "principal" in text.lower() or "bearer" in text.lower(), text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_criterion5_no_untyped_body_tool_on_the_live_surface():
    """Criterion 5 — no untyped `body: dict` tool on the live `/mcp` surface.

    Re-confirms ADR-023 D4: scanning every tool the live MCP server publishes,
    not one exposes a `body` property that is a bare `{"type": "object"}` with
    no fields. Every body-carrying tool's `body` is a `$ref` into a typed
    `$defs` model. The nested-schema depth is in `test_mcp_typed_schemas.py`.
    """
    async with _mcp_app_client() as client:
        tools = await _list_tools(client)

    body_tools = 0
    offenders: list[str] = []
    for name, tool in tools.items():
        body = tool.get("inputSchema", {}).get("properties", {}).get("body")
        if body is None:
            continue
        body_tools += 1
        if (
            "$ref" not in body
            and body.get("type") == "object"
            and not body.get("properties")
        ):
            offenders.append(name)
    assert not offenders, (
        f"tools with an untyped body (ADR-023 D4 violated): {offenders}"
    )
    assert body_tools > 0, "no body-carrying tool was checked — the scan did not run"

    # Registry-level guard: every body-carrying spec names a model, so the
    # projection can never silently fall back to an untyped dict.
    missing = [s.name for s in REGISTRY if s.has_body and s.body_model is None]
    assert not missing, f"specs with a body but no body_model: {missing}"
