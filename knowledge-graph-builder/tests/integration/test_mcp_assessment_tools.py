"""Integration tests for TASK-080 MCP assessment + registry tools.

Exercises the wrappers through their actual entry points against a live
Neo4j (the ``neo4j_test_driver`` fixture from ``tests/conftest.py``).

Coverage:
- Auth boundary — missing JWT (no ORACLOUS_API_KEY) returns
  ``unauthenticated`` for every tool.
- Service-layer path — ``assessment.create_run`` then
  ``assessment.list_runs`` round-trips through the live driver, with the
  service constructed from the same test driver fixture.
- Cross-tenant isolation — a second tenant's principal cannot see tenant
  A's runs (the substrate's `graph_id` enforcement holds end-to-end).
- ADR-019 visibility on Registry — caller B cannot read caller A's
  private item via ``registry.get_item``.
- FastMCP transport smoke test (stdio: the MCP server module imports and
  the new tool table is reachable through ``mcp._tool_manager``;
  sse: the ASGI app responds to ``/health``).

The TASK-079 + TASK-082 integration tests already exercise the
end-to-end Cypher; this file's job is to prove the MCP wrappers preserve
auth + scope + payload shape.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)

os.environ.setdefault("ORACLOUS_API_KEY", "integ-test-key")

# Module imports below depend on env vars being set.
from app.core import neo4j_client as neo4j_client_mod
from app.mcp.tools import assessment_tools, registry_tools
from app.schemas.assessment_schemas import ASSESSMENTS_CATALOG_GRAPH_ID

_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"integ-mcp-A-{_SESSION}"
_GID_B = f"integ-mcp-B-{_SESSION}"
_USER_A = f"u-mcp-A-{_SESSION}"
_USER_B = f"u-mcp-B-{_SESSION}"
_TEMPLATE_ID = f"t-mcp-{_SESSION}"
_TEMPLATE_SLUG = f"assess-mcp-{_SESSION}"
_MODULE_ID = f"m-mcp-{_SESSION}"


def _principal_a() -> dict[str, Any]:
    return {
        "id": _USER_A,
        "principal_type": "service_account",
        "home_graph_id": _GID_A,
    }


def _principal_b() -> dict[str, Any]:
    return {
        "id": _USER_B,
        "principal_type": "service_account",
        "home_graph_id": _GID_B,
    }


# ── Seeds: catalog template + a Subject in tenant A ───────────────────────────


@pytest_asyncio.fixture
async def _seed_and_cleanup(neo4j_test_driver: AsyncDriver):
    # Wipe leftovers (idempotent)
    for gid in (_GID_A, _GID_B):
        await neo4j_test_driver.execute_query(
            "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
        )
    await neo4j_test_driver.execute_query(
        """
        MATCH (t:AssessmentTemplate {template_id: $tid})
        OPTIONAL MATCH (t)-[:HAS_MODULE]->(m:Module)
        DETACH DELETE t, m
        """,
        {"tid": _TEMPLATE_ID},
    )

    # Seed: catalog template + 1 module
    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET
            t.graph_id = $catalog,
            t.slug     = $slug,
            t.name     = 'MCP Test Template',
            t.version  = '0.0.1'
        MERGE (m:Module:__Platform__ {module_id: $mid})
        ON CREATE SET
            m.graph_id    = $catalog,
            m.template_id = $tid,
            m.slug        = $mslug,
            m.name        = 'MCP Test Module',
            m.wave        = 1,
            m.ordinal     = 0,
            m.kind        = 'research'
        MERGE (t)-[:HAS_MODULE]->(m)
        """,
        {
            "tid": _TEMPLATE_ID,
            "mid": _MODULE_ID,
            "slug": _TEMPLATE_SLUG,
            "mslug": f"m-slug-{_SESSION}",
            "catalog": ASSESSMENTS_CATALOG_GRAPH_ID,
        },
    )

    yield

    for gid in (_GID_A, _GID_B):
        await neo4j_test_driver.execute_query(
            "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
        )
    await neo4j_test_driver.execute_query(
        """
        MATCH (t:AssessmentTemplate {template_id: $tid})
        OPTIONAL MATCH (t)-[:HAS_MODULE]->(m:Module)
        DETACH DELETE t, m
        """,
        {"tid": _TEMPLATE_ID},
    )


@pytest_asyncio.fixture
async def wired_driver(
    neo4j_test_driver: AsyncDriver,
) -> AsyncGenerator[AsyncDriver, None]:
    """Point the global `neo4j_client.async_driver` at the test driver.

    The MCP tools build their service from this attribute, so we splice
    in the test driver for the duration of the test.
    """
    original = neo4j_client_mod.neo4j_client.async_driver
    neo4j_client_mod.neo4j_client.async_driver = neo4j_test_driver
    try:
        yield neo4j_test_driver
    finally:
        neo4j_client_mod.neo4j_client.async_driver = original


# ── Auth boundary — no JWT, every tool fails ──────────────────────────────────


class TestAuthBoundaryNoJwt:
    @pytest.mark.asyncio
    async def test_create_run_returns_unauthenticated_without_token(self, monkeypatch):
        monkeypatch.setenv("ORACLOUS_API_KEY", "")
        result = await assessment_tools.create_run(
            {
                "template_slug": _TEMPLATE_SLUG,
                "subject": {
                    "subject_id": "s1",
                    "graph_id": _GID_A,
                    "slug": "sj",
                    "name": "S",
                },
            }
        )
        assert result["code"] == "unauthenticated"

    @pytest.mark.asyncio
    async def test_registry_persist_item_returns_unauthenticated_without_token(
        self, monkeypatch
    ):
        monkeypatch.setenv("ORACLOUS_API_KEY", "")
        result = await registry_tools.persist_item(
            "skill",
            {
                "item_id": "x",
                "graph_id": _GID_A,
                "kind": "skill",
                "slug": "x",
                "version": "0.1.0",
                "visibility": "private",
                "owner_user_id": _USER_A,
                "name": "x",
            },
        )
        assert result["code"] == "unauthenticated"

    @pytest.mark.asyncio
    async def test_list_runs_unauthenticated_without_token(self, monkeypatch):
        monkeypatch.setenv("ORACLOUS_API_KEY", "")
        result = await assessment_tools.list_runs()
        assert result["code"] == "unauthenticated"


# ── End-to-end: create_run → list_runs against live Neo4j ─────────────────────


class TestCreateRunRoundTrip:
    @pytest.mark.asyncio
    async def test_create_run_then_list_runs_through_service(
        self, _seed_and_cleanup, wired_driver: AsyncDriver
    ):
        # Stub the auth + scope helpers so we exercise the service-layer path
        # without standing up auth-service.
        with (
            patch.object(
                assessment_tools,
                "resolve_principal",
                AsyncMock(return_value=_principal_a()),
            ),
            patch.object(
                assessment_tools,
                "assert_graph_access",
                AsyncMock(return_value=_GID_A),
            ),
        ):
            create = await assessment_tools.create_run(
                {
                    "template_slug": _TEMPLATE_SLUG,
                    "subject": {
                        "subject_id": f"sj-{_SESSION}",
                        "graph_id": _GID_A,
                        "slug": f"sj-slug-{_SESSION}",
                        "name": "Subject 1",
                    },
                }
            )
            assert "run_id" in create, create
            run_id = create["run_id"]
            assert create["status"] == "planned"
            assert create["already_existed"] is False

            listed = await assessment_tools.list_runs()
            assert any(r["run_id"] == run_id for r in listed["items"])

    @pytest.mark.asyncio
    async def test_tenant_b_cannot_see_tenant_a_run(
        self, _seed_and_cleanup, wired_driver: AsyncDriver
    ):
        # Create a run as tenant A
        with (
            patch.object(
                assessment_tools,
                "resolve_principal",
                AsyncMock(return_value=_principal_a()),
            ),
            patch.object(
                assessment_tools,
                "assert_graph_access",
                AsyncMock(return_value=_GID_A),
            ),
        ):
            create = await assessment_tools.create_run(
                {
                    "template_slug": _TEMPLATE_SLUG,
                    "subject": {
                        "subject_id": f"sj-x-{_SESSION}",
                        "graph_id": _GID_A,
                        "slug": f"sj-slug-x-{_SESSION}",
                        "name": "Subject X",
                    },
                }
            )
            run_id_a = create["run_id"]

        # As tenant B, get_run for run_id_a returns not_found
        with (
            patch.object(
                assessment_tools,
                "resolve_principal",
                AsyncMock(return_value=_principal_b()),
            ),
            patch.object(
                assessment_tools,
                "assert_graph_access",
                AsyncMock(return_value=_GID_B),
            ),
        ):
            tenant_b_view = await assessment_tools.get_run(run_id_a)
            assert tenant_b_view["code"] == "not_found"


# ── Registry visibility: ADR-019 enforced via the tool wrapper ────────────────


class TestRegistryAdr019:
    @pytest.mark.asyncio
    async def test_tenant_b_cannot_get_tenant_a_private_item(
        self, _seed_and_cleanup, wired_driver: AsyncDriver
    ):
        # Tenant A persists a private item.
        with (
            patch.object(
                registry_tools,
                "resolve_principal",
                AsyncMock(return_value=_principal_a()),
            ),
            patch.object(
                registry_tools,
                "assert_graph_access",
                AsyncMock(return_value=_GID_A),
            ),
        ):
            persisted = await registry_tools.persist_item(
                "skill",
                {
                    "item_id": f"ri-priv-{_SESSION}",
                    "graph_id": _GID_A,
                    "kind": "skill",
                    "slug": f"priv-{_SESSION}",
                    "version": "0.1.0",
                    "visibility": "private",
                    "owner_user_id": _USER_A,
                    "name": "Tenant A Private Skill",
                },
            )
            assert persisted["created"] is True, persisted

        # Tenant B asks for the same slug — must collapse to not_found
        # (ADR-019: a non-owner must not learn that a private slug exists).
        with (
            patch.object(
                registry_tools,
                "resolve_principal",
                AsyncMock(return_value=_principal_b()),
            ),
        ):
            tenant_b_view = await registry_tools.get_item("skill", f"priv-{_SESSION}")
            assert tenant_b_view["code"] == "not_found"


# ── FastMCP transport smoke test ──────────────────────────────────────────────


class TestFastMcpTransport:
    """Prove the tools are reachable through both supported transports."""

    @pytest.mark.asyncio
    async def test_stdio_path_exposes_assessment_tools(self):
        """The FastMCP `_tool_manager` table contains every new tool name."""
        # Server module is imported; tool registration runs at import-time.
        import app.mcp.server as srv

        tools = set(srv.mcp._tool_manager._tools.keys())
        expected = {name for name, _ in assessment_tools.TOOLS}
        expected |= {name for name, _ in registry_tools.TOOLS}
        missing = expected - tools
        assert not missing, f"Tools missing from FastMCP registry: {missing}"

    @pytest.mark.asyncio
    async def test_sse_app_responds_to_health(self):
        """The SSE ASGI app's /health endpoint still works after tool registration."""
        import httpx

        import app.mcp.server as srv

        app = srv.mcp.sse_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
