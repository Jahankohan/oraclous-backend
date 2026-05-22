"""Integration test for the MCP projection layer — TASK-229 / STORY-035.

Validates that the per-I/O-class REST -> MCP projection mechanism (ADR-023 D3,
one namespaced server per ADR-024 D7-R) produces working MCP tools for all four
I/O classes, dispatching through the live FastAPI app against the live
Dockerized stack (Neo4j + Postgres + Redis):

  * PLAIN      — graph.create / graph.get / graph.list
  * UPLOAD     — ingest.document
  * STREAMING  — graph.ask (the SSE stream is collected into one result)
  * ASYNC_JOB  — ingest.text (submit) + ingest.job_status (poll)

Run inside the kg-builder container, where the `neo4j` / `postgres` / `redis`
service hostnames resolve and the app `.env` config applies as-is:

    docker compose run --rm --no-deps knowledge-graph-builder \\
        python -m pytest tests/integration/test_mcp_projection.py -v

The external auth-service is bypassed exactly as the other API integration
tests do it — `auth_service.verify_token` is patched to a fixed principal — so
the test exercises the real projection, in-process dispatch, routing,
`verify_graph_access` (real ReBAC against Neo4j) and services, with no second
microservice in the loop.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.mcp import build_mcp_server
from app.mcp.context import reset_bearer_token, set_bearer_token

# A fixed test principal — the patched auth-service returns this for any token.
_TEST_USER_ID = str(uuid.uuid4())
_TEST_PRINCIPAL: dict[str, Any] = {
    "id": _TEST_USER_ID,
    "principal_type": "user",
    "email": "mcp-projection-test@example.com",
}
_BEARER = "mcp-projection-test-token"

# A structurally minimal PDF — enough to pass the upload endpoint's content
# checks; the actual extraction is a Celery job this test does not await.
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\n%%EOF\n"
)


# --- fixture ----------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def mcp_env():
    """Initialise the app via its real lifespan and patch the external auth."""
    from app.main import app

    auth_patch = patch("app.api.dependencies.auth_service")
    mock_auth = auth_patch.start()
    mock_auth.verify_token = AsyncMock(return_value=_TEST_PRINCIPAL)

    try:
        async with app.router.lifespan_context(app):
            yield build_mcp_server()
    finally:
        auth_patch.stop()


# --- helpers ----------------------------------------------------------------

# The bearer token is bound per-call: a contextvar set in the module fixture
# would not propagate into each test's own asyncio task. Binding it at the
# start of every call places it in the task that also runs the tool handler.


async def _call(mcp, name: str, args: dict[str, Any]) -> Any:
    """Bind the test principal and invoke a projected MCP tool."""
    set_bearer_token(_BEARER)
    return _unwrap(await mcp.call_tool(name, args))


def _unwrap(result: Any) -> Any:
    """Extract the JSON payload from a FastMCP `call_tool` result.

    `call_tool` returns either a structured dict or a sequence of content
    blocks; a projected tool always returns a JSON object, carried as the text
    of a single `TextContent` block.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple):
        # (content_blocks, structured_content)
        for part in result:
            if isinstance(part, dict):
                return part
        result = result[0]
    for block in result:
        text = getattr(block, "text", None)
        if text is not None:
            return json.loads(text)
    raise AssertionError(f"no JSON content block in tool result: {result!r}")


async def _create_graph(mcp, name: str) -> str:
    """Create a graph through the MCP plain tool; return its id."""
    res = await _call(
        mcp,
        "graph.create",
        {"body": {"name": name, "description": "MCP projection test graph"}},
    )
    assert not res.get("error"), f"graph.create failed: {res}"
    graph_id = res.get("id")
    assert graph_id, f"graph.create returned no id: {res}"
    return str(graph_id)


# --- tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_projects_all_four_classes(mcp_env):
    """The single server (ADR-024 D7-R) exposes the curated tool set, each
    namespaced and each carrying a published input schema (ADR-023 D4).

    The four I/O classes covered end-to-end by this module's dispatch tests are
    still present; the curated set (TASK-230) adds the rest of the families."""
    tools = await mcp_env.list_tools()
    names = {t.name for t in tools}
    # The four representative I/O-class tools still project.
    assert {
        "graph.create",  # PLAIN
        "graph.get",
        "graph.list",
        "ingest.document",  # UPLOAD
        "graph.ask",  # STREAMING
        "ingest.text",  # ASYNC_JOB submit
        "ingest.job_status",  # ASYNC_JOB poll
    } <= names, names
    # The curated set spans every family with REST endpoints (ADR-024 D8-R).
    families = {n.split(".", 1)[0] for n in names}
    assert families == {
        "graph",
        "schema",
        "ingest",
        "community",
        "agent",
        "memory",
        "connector",
        "federation",
    }, families
    for tool in tools:
        assert "." in tool.name, f"tool {tool.name} is not namespaced"
        assert tool.description, f"tool {tool.name} has no description"
        assert tool.inputSchema.get("type") == "object", tool.name
        # No tool publishes an empty/untyped schema container.
        assert "properties" in tool.inputSchema, tool.name


@pytest.mark.asyncio
async def test_plain_class_create_get_list(mcp_env):
    """PLAIN — request/response projects to a tool that round-trips through the
    real REST stack: create, then read back, then list."""
    name = f"mcp-plain-{uuid.uuid4().hex[:8]}"
    graph_id = await _create_graph(mcp_env, name)

    got = await _call(mcp_env, "graph.get", {"graph_id": graph_id})
    assert not got.get("error"), f"graph.get failed: {got}"
    assert str(got.get("id")) == graph_id
    assert got.get("name") == name

    listed = await _call(mcp_env, "graph.list", {})
    rows = listed if isinstance(listed, list) else listed.get("result", [])
    assert any(str(g.get("id")) == graph_id for g in rows), (
        f"created graph {graph_id} not in graph.list"
    )


@pytest.mark.asyncio
async def test_upload_class_ingests_document(mcp_env):
    """UPLOAD — base64 content in, multipart/form-data out; the endpoint accepts
    the file and returns an ingestion-job response."""
    graph_id = await _create_graph(mcp_env, f"mcp-upload-{uuid.uuid4().hex[:8]}")

    res = await _call(
        mcp_env,
        "ingest.document",
        {
            "graph_id": graph_id,
            "filename": "mcp-projection-test.pdf",
            "content_base64": base64.b64encode(_MINIMAL_PDF).decode("ascii"),
            "content_type": "application/pdf",
        },
    )
    assert not res.get("error"), f"ingest.document failed: {res}"
    # MultiModalJobResponse — a job was created for the uploaded file.
    assert res.get("job_id") or res.get("id"), f"no job id in upload result: {res}"
    assert res.get("filename") == "mcp-projection-test.pdf"


@pytest.mark.asyncio
async def test_streaming_class_collects_sse(mcp_env):
    """STREAMING — an SSE endpoint projects to a tool that collects the whole
    stream into one result. The collection mechanism is what is under test, so
    the assertion holds whether the answer succeeds or the stream carries an
    error event."""
    graph_id = await _create_graph(mcp_env, f"mcp-stream-{uuid.uuid4().hex[:8]}")

    res = await _call(
        mcp_env,
        "graph.ask",
        {"body": {"graph_id": graph_id, "query": "What is in this graph?"}},
    )
    # The streaming wrapper always returns a collected `events` list — never a
    # raw stream — regardless of whether the underlying answer succeeded.
    assert "events" in res, f"streaming result has no collected events: {res}"
    assert isinstance(res["events"], list)


@pytest.mark.asyncio
async def test_async_job_class_submit_and_status(mcp_env):
    """ASYNC_JOB — one capability projects to a submit tool and a status tool;
    submit returns a job id, the status tool polls it."""
    graph_id = await _create_graph(mcp_env, f"mcp-async-{uuid.uuid4().hex[:8]}")

    submitted = await _call(
        mcp_env,
        "ingest.text",
        {
            "graph_id": graph_id,
            "body": {
                "content": (
                    "Ada Lovelace worked with Charles Babbage on the "
                    "Analytical Engine. This is enough text to ingest."
                ),
                "source_type": "text",
            },
        },
    )
    assert not submitted.get("error"), f"ingest.text submit failed: {submitted}"
    job_id = submitted.get("job_id")
    assert job_id, f"submit returned no job id: {submitted}"

    status = await _call(
        mcp_env,
        "ingest.job_status",
        {"graph_id": graph_id, "job_id": str(job_id)},
    )
    assert not status.get("error"), f"ingest.job_status failed: {status}"
    assert "status" in status, f"job status record has no status field: {status}"


@pytest.mark.asyncio
async def test_missing_principal_fails_closed(mcp_env):
    """A tool call with no bound principal is refused — the projection fails
    closed (ADR-023 D5; the platform's deny-by-default rule)."""
    token = set_bearer_token(None)
    try:
        with pytest.raises(Exception) as exc_info:
            await mcp_env.call_tool("graph.list", {})
        assert "principal" in str(exc_info.value).lower()
    finally:
        reset_bearer_token(token)
        set_bearer_token(_BEARER)
