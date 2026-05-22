"""Integration test — the recipe pipeline REST surface (TASK-236 / STORY-035).

Validates that the minimal REST surface over the STORY-034 recipe pipeline
(`app/api/v1/endpoints/recipes.py`) works end-to-end against the live
Dockerized stack (Neo4j + Postgres + Redis), AND that the `recipe.*` /
`ingest.recipe` MCP tools — added to the curated registry — project and
dispatch through the same surface:

  * `GET  /api/v1/recipes`                          — list (PLAIN)
  * `GET  /api/v1/recipes/{recipe_id}`              — get   (PLAIN)
  * `POST /api/v1/recipes`                          — store (PLAIN, 201)
  * `POST /api/v1/graphs/{gid}/recipes/{rid}/run`   — run   (PLAIN, 202)

The recipe endpoints are tenant-scoped (`graph_id`); a graph is created through
the real `POST /graphs` endpoint, which grants the test principal ReBAC owner
access, so the real `verify_graph_access` checks pass.

Run inside the kg-builder container, where the `neo4j` / `postgres` / `redis`
service hostnames resolve. Pass the Docker-network test env via
`-e NEO4J_TEST_URI` and `-e TEST_POSTGRES_URL`, then:

    docker compose exec -T <env flags> knowledge-graph-builder \\
        python -m pytest tests/integration/test_recipe_rest.py -q

The external auth-service is bypassed exactly as `test_mcp_projection.py` does
it — `auth_service.verify_token` is patched to a fixed principal — so the test
exercises the real router, dispatch, `verify_graph_access` (real ReBAC against
Neo4j), the recipe library (Postgres), and the projection, with no second
microservice in the loop.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.mcp import build_mcp_server
from app.mcp.context import set_bearer_token

# A fixed test principal — the patched auth-service returns this for any token.
_TEST_USER_ID = str(uuid.uuid4())
_TEST_PRINCIPAL: dict[str, Any] = {
    "id": _TEST_USER_ID,
    "principal_type": "user",
    "email": "recipe-rest-test@example.com",
}
_BEARER = "recipe-rest-test-token"


# --- recipe document builder ------------------------------------------------


def _make_recipe(recipe_id: str) -> dict[str, Any]:
    """A minimal CSV recipe that validates against recipe.schema.json.

    One node rule projecting an `Employee` per record, keyed on the `name`
    column — the same shape `test_recipe_engine.py` exercises.
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
        "authoring": {"authored_by": "data-specialist", "created": "2026-05-22"},
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


# --- fixtures ---------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def env():
    """Initialise the app via its real lifespan and patch the external auth.

    Yields a `(rest_client, mcp_server)` pair sharing the one running app, so
    the REST surface and the projected MCP tools are tested against the same
    process and the same live Neo4j/Postgres.
    """
    from app.main import app

    auth_patch = patch("app.api.dependencies.auth_service")
    mock_auth = auth_patch.start()
    mock_auth.verify_token = AsyncMock(return_value=_TEST_PRINCIPAL)

    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {_BEARER}"},
            ) as client:
                yield client, build_mcp_server()
    finally:
        auth_patch.stop()


# --- MCP helpers ------------------------------------------------------------


def _unwrap(result: Any) -> Any:
    """Extract the JSON payload from a FastMCP `call_tool` result."""
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple):
        for part in result:
            if isinstance(part, dict):
                return part
        result = result[0]
    for block in result:
        text = getattr(block, "text", None)
        if text is not None:
            return json.loads(text)
    raise AssertionError(f"no JSON content block in tool result: {result!r}")


async def _mcp_call(mcp, name: str, args: dict[str, Any]) -> Any:
    """Bind the test principal and invoke a projected MCP tool."""
    set_bearer_token(_BEARER)
    return _unwrap(await mcp.call_tool(name, args))


async def _create_graph(client: AsyncClient, name: str) -> str:
    """Create a graph through the real REST endpoint; return its id.

    Creating a graph grants the creating principal ReBAC owner access, so the
    recipe endpoints' `verify_graph_access` checks pass for this principal.
    """
    resp = await client.post(
        "/api/v1/graphs",
        json={"name": name, "description": "recipe REST test graph"},
    )
    assert resp.status_code in (200, 201), f"graph create failed: {resp.text}"
    return str(resp.json()["id"])


# --- REST tests -------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_store_get_list_recipe(env):
    """POST /recipes stores a draft; GET round-trips it; GET /recipes lists it."""
    client, _ = env
    graph_id = await _create_graph(client, f"recipe-rest-{uuid.uuid4().hex[:8]}")
    recipe_id = f"rcp_rest-{uuid.uuid4().hex[:12]}"

    # store
    stored = await client.post(
        "/api/v1/recipes",
        json={"graph_id": graph_id, "recipe": _make_recipe(recipe_id)},
    )
    assert stored.status_code == 201, f"store failed: {stored.text}"
    stored_doc = stored.json()["recipe"]
    assert stored_doc["id"] == recipe_id
    assert stored_doc["version"] == 1
    assert stored_doc["status"] == "draft"  # ADR-022 — never auto-promoted

    # get
    got = await client.get(
        f"/api/v1/recipes/{recipe_id}", params={"graph_id": graph_id}
    )
    assert got.status_code == 200, f"get failed: {got.text}"
    assert got.json()["recipe"]["id"] == recipe_id

    # list
    listed = await client.get("/api/v1/recipes", params={"graph_id": graph_id})
    assert listed.status_code == 200, f"list failed: {listed.text}"
    body = listed.json()
    assert body["count"] >= 1
    assert any(r["id"] == recipe_id for r in body["recipes"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_store_rejects_invalid_recipe(env):
    """POST /recipes refuses a schema-invalid recipe with 400 — never coerces."""
    client, _ = env
    graph_id = await _create_graph(client, f"recipe-bad-{uuid.uuid4().hex[:8]}")

    bad = _make_recipe(f"rcp_bad-{uuid.uuid4().hex[:8]}")
    del bad["mappings"]  # `mappings` is required by recipe.schema.json

    resp = await client.post(
        "/api/v1/recipes", json={"graph_id": graph_id, "recipe": bad}
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_get_unknown_recipe_is_404(env):
    """GET an unknown recipe id is 404 — and a tenant cannot probe other tenants."""
    client, _ = env
    graph_id = await _create_graph(client, f"recipe-404-{uuid.uuid4().hex[:8]}")
    resp = await client.get(
        f"/api/v1/recipes/rcp_does-not-exist-{uuid.uuid4().hex[:8]}",
        params={"graph_id": graph_id},
    )
    assert resp.status_code == 404, f"expected 404, got {resp.status_code}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_recipe_enqueues_a_run(env):
    """POST .../recipes/{id}/run decomposes the inline source and enqueues a run.

    The run executes asynchronously in the Celery worker; the endpoint returns
    202 with a `run_id` (the Celery task id) — a recipe run is queueable
    end-to-end (the TASK-227 gap is closed by the Celery `include=` change).
    """
    client, _ = env
    graph_id = await _create_graph(client, f"recipe-run-{uuid.uuid4().hex[:8]}")
    recipe_id = f"rcp_run-{uuid.uuid4().hex[:12]}"

    stored = await client.post(
        "/api/v1/recipes",
        json={"graph_id": graph_id, "recipe": _make_recipe(recipe_id)},
    )
    assert stored.status_code == 201, stored.text

    run = await client.post(
        f"/api/v1/graphs/{graph_id}/recipes/{recipe_id}/run",
        json={"source_type": "csv", "records": _RECORDS},
    )
    assert run.status_code == 202, f"run failed: {run.text}"
    body = run.json()
    assert body["run_id"], f"run returned no run_id: {body}"
    assert body["recipe_id"] == recipe_id
    assert body["recipe_version"] == 1
    assert body["graph_id"] == graph_id
    assert body["status"] == "queued"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_unknown_source_type_is_400(env):
    """A run with a source_type no inline primitive supports is refused with 400."""
    client, _ = env
    graph_id = await _create_graph(client, f"recipe-srctype-{uuid.uuid4().hex[:8]}")
    recipe_id = f"rcp_srctype-{uuid.uuid4().hex[:12]}"
    stored = await client.post(
        "/api/v1/recipes",
        json={"graph_id": graph_id, "recipe": _make_recipe(recipe_id)},
    )
    assert stored.status_code == 201, stored.text

    resp = await client.post(
        f"/api/v1/graphs/{graph_id}/recipes/{recipe_id}/run",
        json={"source_type": "relational", "records": _RECORDS},
    )
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"


# --- MCP tests --------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_recipe_tools_project(env):
    """The `recipe.*` / `ingest.recipe` tools project, namespaced and typed."""
    _, mcp = env
    tools = await mcp.list_tools()
    by_name = {t.name: t for t in tools}
    for name in ("recipe.list", "recipe.get", "recipe.store", "ingest.recipe"):
        assert name in by_name, f"{name} not projected: {sorted(by_name)}"
        tool = by_name[name]
        assert tool.description, f"{name} has no description"
        assert tool.inputSchema.get("type") == "object", name
        assert "properties" in tool.inputSchema, name


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_recipe_store_get_list_dispatch(env):
    """The `recipe.store` / `recipe.get` / `recipe.list` tools dispatch through
    the real REST surface — store, read back, then list."""
    client, mcp = env
    graph_id = await _create_graph(client, f"recipe-mcp-{uuid.uuid4().hex[:8]}")
    recipe_id = f"rcp_mcp-{uuid.uuid4().hex[:12]}"

    stored = await _mcp_call(
        mcp,
        "recipe.store",
        {"body": {"graph_id": graph_id, "recipe": _make_recipe(recipe_id)}},
    )
    assert not stored.get("error"), f"recipe.store failed: {stored}"
    assert stored["recipe"]["id"] == recipe_id

    got = await _mcp_call(
        mcp, "recipe.get", {"recipe_id": recipe_id, "graph_id": graph_id}
    )
    assert not got.get("error"), f"recipe.get failed: {got}"
    assert got["recipe"]["id"] == recipe_id

    listed = await _mcp_call(mcp, "recipe.list", {"graph_id": graph_id})
    assert not listed.get("error"), f"recipe.list failed: {listed}"
    assert any(r["id"] == recipe_id for r in listed["recipes"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_ingest_recipe_dispatch(env):
    """The `ingest.recipe` tool dispatches a recipe run through the REST surface."""
    client, mcp = env
    graph_id = await _create_graph(client, f"recipe-mcprun-{uuid.uuid4().hex[:8]}")
    recipe_id = f"rcp_mcprun-{uuid.uuid4().hex[:12]}"

    stored = await _mcp_call(
        mcp,
        "recipe.store",
        {"body": {"graph_id": graph_id, "recipe": _make_recipe(recipe_id)}},
    )
    assert not stored.get("error"), f"recipe.store failed: {stored}"

    run = await _mcp_call(
        mcp,
        "ingest.recipe",
        {
            "graph_id": graph_id,
            "recipe_id": recipe_id,
            "body": {"source_type": "csv", "records": _RECORDS},
        },
    )
    assert not run.get("error"), f"ingest.recipe failed: {run}"
    assert run["run_id"], f"ingest.recipe returned no run_id: {run}"
    assert run["recipe_id"] == recipe_id
    assert run["status"] == "queued"
