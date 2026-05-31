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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
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


# --- environment ------------------------------------------------------------


@asynccontextmanager
async def env() -> AsyncIterator[tuple[AsyncClient, Any]]:
    """Initialise the app via its real lifespan and patch the external auth.

    Yields a `(rest_client, mcp_server)` pair sharing the one running app, so
    the REST surface and the projected MCP tools are tested against the same
    process and the same live Neo4j/Postgres.

    A plain async context manager entered in each test body — not a
    module-scoped fixture. `app.main`'s lifespan now runs the mounted MCP
    server's streamable-HTTP session manager (TASK-232), an `anyio` task group
    whose cancel scope must be entered and exited on the *same* task; a
    pytest fixture runs setup and teardown on different tasks. Entering this CM
    inside the test keeps both on the test's own task.
    """
    from app.main import app

    with patch("app.api.dependencies.auth_service") as mock_auth:
        mock_auth.verify_token = AsyncMock(return_value=_TEST_PRINCIPAL)
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": f"Bearer {_BEARER}"},
            ) as client:
                yield client, build_mcp_server()


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
async def test_rest_store_get_list_recipe():
    """POST /recipes stores a draft; GET round-trips it; GET /recipes lists it."""
    async with env() as (client, _):
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
async def test_rest_store_rejects_invalid_recipe():
    """POST /recipes refuses a schema-invalid recipe with 400 — never coerces."""
    async with env() as (client, _):
        graph_id = await _create_graph(client, f"recipe-bad-{uuid.uuid4().hex[:8]}")

        bad = _make_recipe(f"rcp_bad-{uuid.uuid4().hex[:8]}")
        del bad["mappings"]  # `mappings` is required by recipe.schema.json

        resp = await client.post(
            "/api/v1/recipes", json={"graph_id": graph_id, "recipe": bad}
        )
        assert resp.status_code == 400, (
            f"expected 400, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_get_unknown_recipe_is_404():
    """GET an unknown recipe id is 404 — and a tenant cannot probe other tenants."""
    async with env() as (client, _):
        graph_id = await _create_graph(client, f"recipe-404-{uuid.uuid4().hex[:8]}")
        resp = await client.get(
            f"/api/v1/recipes/rcp_does-not-exist-{uuid.uuid4().hex[:8]}",
            params={"graph_id": graph_id},
        )
        assert resp.status_code == 404, f"expected 404, got {resp.status_code}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_recipe_enqueues_a_run():
    """POST .../recipes/{id}/run decomposes the inline source and enqueues a run.

    The run executes asynchronously in the Celery worker; the endpoint returns
    202 with a `run_id`. Since TASK-237 the `run_id` is a first-class
    `ingestion_jobs` row id (`source_type="recipe"`, `status="pending"`), not a
    raw Celery task id — see `test_rest_run_recipe_is_pollable_job`.
    """
    async with env() as (client, _):
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
        # The run id is a UUID (the ingestion_jobs row id), not a Celery id.
        uuid.UUID(body["run_id"])
        assert body["recipe_id"] == recipe_id
        assert body["recipe_version"] == 1
        assert body["graph_id"] == graph_id
        assert body["status"] == "pending"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_recipe_is_pollable_job():
    """A recipe run is a first-class ingestion job — pollable, graph_id-scoped.

    TASK-237 (closes TASK-233 INFO-2): the run endpoint creates an
    `IngestionJob` row (`source_type="recipe"`) and returns its id as `run_id`.
    That row is then visible through the standard tenant-scoped jobs endpoint
    `GET /api/v1/graphs/{graph_id}/jobs/{job_id}` — no bespoke status route.
    """
    async with env() as (client, _):
        graph_id = await _create_graph(client, f"recipe-poll-{uuid.uuid4().hex[:8]}")
        recipe_id = f"rcp_poll-{uuid.uuid4().hex[:12]}"

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
        run_id = run.json()["run_id"]

        # The run is pollable through the standard graph_id-scoped jobs route.
        polled = await client.get(f"/api/v1/graphs/{graph_id}/jobs/{run_id}")
        assert polled.status_code == 200, f"poll failed: {polled.text}"
        job = polled.json()
        assert job["id"] == run_id
        assert str(job["graph_id"]) == graph_id
        assert job["source_type"] == "recipe"
        # The run is in a job lifecycle state — pending on enqueue, or a later
        # state if the worker has already picked it up.
        assert job["status"] in ("pending", "processing", "completed", "failed")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_recipe_cross_tenant_poll_denied():
    """A recipe-run job cannot be polled from another tenant's graph.

    The run handle lives inside the `graph_id` / ReBAC boundary (TASK-237):
    polling a recipe-run id under a graph the principal owns but which is not
    the run's graph returns 404 (the jobs query is `graph_id`-scoped) — the
    latent cross-tenant smell of a raw Celery id (TASK-233 INFO-2) is closed.
    """
    async with env() as (client, _):
        graph_a = await _create_graph(client, f"recipe-xtA-{uuid.uuid4().hex[:8]}")
        graph_b = await _create_graph(client, f"recipe-xtB-{uuid.uuid4().hex[:8]}")
        recipe_id = f"rcp_xt-{uuid.uuid4().hex[:12]}"

        stored = await client.post(
            "/api/v1/recipes",
            json={"graph_id": graph_a, "recipe": _make_recipe(recipe_id)},
        )
        assert stored.status_code == 201, stored.text

        run = await client.post(
            f"/api/v1/graphs/{graph_a}/recipes/{recipe_id}/run",
            json={"source_type": "csv", "records": _RECORDS},
        )
        assert run.status_code == 202, f"run failed: {run.text}"
        run_id = run.json()["run_id"]

        # Polling the run id under graph_b — a graph the principal owns but
        # which is NOT the run's graph — must not return the row.
        wrong = await client.get(f"/api/v1/graphs/{graph_b}/jobs/{run_id}")
        assert wrong.status_code == 404, (
            f"cross-tenant poll should be 404, got {wrong.status_code}: {wrong.text}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rest_run_unknown_source_type_is_400():
    """A run with a source_type no inline primitive supports is refused with 400."""
    async with env() as (client, _):
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
        assert resp.status_code == 400, (
            f"expected 400, got {resp.status_code}: {resp.text}"
        )


# --- MCP tests --------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_recipe_tools_project():
    """The `recipe.*` / `ingest.recipe` tools project, namespaced and typed.

    Since TASK-237 `ingest.recipe` is an ASYNC_JOB, so it projects a *pair* —
    the submit tool `ingest.recipe` and a distinct status tool
    `ingest.recipe_status` (NOT `ingest.job_status`, which `ingest.text`
    already owns — two specs cannot project the same tool name).
    """
    async with env() as (_, mcp):
        tools = await mcp.list_tools()
    by_name = {t.name: t for t in tools}
    for name in (
        "recipe.list",
        "recipe.get",
        "recipe.store",
        "ingest.recipe",
        "ingest.recipe_status",
    ):
        assert name in by_name, f"{name} not projected: {sorted(by_name)}"
        tool = by_name[name]
        assert tool.description, f"{name} has no description"
        assert tool.inputSchema.get("type") == "object", name
        assert "properties" in tool.inputSchema, name

    # The recipe status tool must not collide with the text-ingest one.
    assert "ingest.job_status" in by_name, "ingest.text status tool missing"
    assert by_name["ingest.recipe_status"].name != by_name["ingest.job_status"].name


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_recipe_store_get_list_dispatch():
    """The `recipe.store` / `recipe.get` / `recipe.list` tools dispatch through
    the real REST surface — store, read back, then list."""
    async with env() as (client, mcp):
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
async def test_mcp_ingest_recipe_dispatch():
    """The `ingest.recipe` ASYNC_JOB pair dispatches submit + status (TASK-237).

    The submit tool runs the recipe and surfaces the run handle as `job_id`
    (lifted from `RecipeRunResponse.run_id` via `job_id_field`). The paired
    `ingest.recipe_status` tool then polls that handle through the standard
    `/graphs/{graph_id}/jobs/{job_id}` route and returns the job row.
    """
    async with env() as (client, mcp):
        graph_id = await _create_graph(client, f"recipe-mcprun-{uuid.uuid4().hex[:8]}")
        recipe_id = f"rcp_mcprun-{uuid.uuid4().hex[:12]}"

        stored = await _mcp_call(
            mcp,
            "recipe.store",
            {"body": {"graph_id": graph_id, "recipe": _make_recipe(recipe_id)}},
        )
        assert not stored.get("error"), f"recipe.store failed: {stored}"

        # Submit leg — returns {job_id, status, raw}.
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
        job_id = run["job_id"]
        assert job_id, f"ingest.recipe returned no job_id: {run}"
        uuid.UUID(job_id)
        assert run["status"] == "pending"
        assert run["raw"]["recipe_id"] == recipe_id

        # Status leg — polls the job, graph_id-scoped.
        polled = await _mcp_call(
            mcp,
            "ingest.recipe_status",
            {"graph_id": graph_id, "job_id": job_id},
        )
        assert not polled.get("error"), f"ingest.recipe_status failed: {polled}"
        assert polled["id"] == job_id
        assert polled["source_type"] == "recipe"
        assert polled["status"] in ("pending", "processing", "completed", "failed")
