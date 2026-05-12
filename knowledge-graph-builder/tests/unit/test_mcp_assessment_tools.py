"""Unit tests for the TASK-080 assessment + registry MCP tools.

These tests run without live services. They mock ``auth_service.verify_token``,
``verify_graph_access``, and ``neo4j_client.async_driver`` so each tool's
*shape* — auth boundary, scope check, service call, response envelope — can
be asserted in isolation.

The matching integration tests (`tests/integration/test_mcp_assessment_tools.py`)
exercise the same tools through the full FastMCP transport (stdio + sse)
against a live Neo4j seeded with assessment fixtures.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ORACLOUS_API_KEY", "test-key-unit")

from app.mcp.tools import assessment_tools, registry_tools

# ── Shared mock principal ─────────────────────────────────────────────────────


def _principal(
    user_id: str = "user-001",
    home_graph_id: str = "graph-A",
    principal_type: str = "service_account",
) -> dict:
    return {
        "id": user_id,
        "home_graph_id": home_graph_id,
        "principal_type": principal_type,
    }


@pytest.fixture
def patched_principal():
    """Patch ``resolve_principal`` on both tool modules so every tool sees the same identity."""
    p = _principal()
    with (
        patch.object(assessment_tools, "resolve_principal", AsyncMock(return_value=p)),
        patch.object(registry_tools, "resolve_principal", AsyncMock(return_value=p)),
    ):
        yield p


@pytest.fixture
def patched_scope():
    """Patch ``assert_graph_access`` to succeed by default on both modules."""
    with (
        patch.object(
            assessment_tools,
            "assert_graph_access",
            AsyncMock(return_value="graph-A"),
        ),
        patch.object(
            registry_tools,
            "assert_graph_access",
            AsyncMock(return_value="graph-A"),
        ),
        patch.object(
            assessment_tools, "assert_catalog_admin", AsyncMock(return_value=None)
        ),
        patch.object(
            registry_tools,
            "assert_registry_curated_write",
            AsyncMock(return_value=None),
        ),
    ):
        yield


@pytest.fixture
def patched_service():
    """Patch ``build_service`` on both modules to return a configurable mock."""
    svc = MagicMock()
    with (
        patch.object(assessment_tools, "build_service", return_value=svc),
        patch.object(registry_tools, "build_service", return_value=svc),
    ):
        yield svc


# ── Auth boundary — missing JWT fails every tool ──────────────────────────────


_VALID_SUBJECT = {
    "subject_id": "subj-1",
    "graph_id": "graph-A",
    "slug": "subj-slug",
    "name": "Subject 1",
}


def _create_run_body(graph_id_override: str | None = None) -> dict:
    subj = dict(_VALID_SUBJECT)
    if graph_id_override is not None:
        subj["graph_id"] = graph_id_override
    return {"template_slug": "tmpl-1", "subject": subj}


class TestAuthBoundary:
    @pytest.mark.asyncio
    async def test_create_run_rejects_missing_token(self):
        from app.mcp.tools._auth import AuthError

        with patch.object(
            assessment_tools,
            "resolve_principal",
            AsyncMock(side_effect=AuthError("no token")),
        ):
            result = await assessment_tools.create_run(_create_run_body())
        assert result == {"error": "no token", "code": "unauthenticated"}

    @pytest.mark.asyncio
    async def test_list_runs_rejects_missing_token(self):
        from app.mcp.tools._auth import AuthError

        with patch.object(
            assessment_tools,
            "resolve_principal",
            AsyncMock(side_effect=AuthError("no token")),
        ):
            result = await assessment_tools.list_runs()
        assert result["code"] == "unauthenticated"

    @pytest.mark.asyncio
    async def test_registry_list_items_rejects_missing_token(self):
        from app.mcp.tools._auth import AuthError

        with patch.object(
            registry_tools,
            "resolve_principal",
            AsyncMock(side_effect=AuthError("no token")),
        ):
            result = await registry_tools.list_items("skill")
        assert result["code"] == "unauthenticated"

    @pytest.mark.asyncio
    async def test_search_findings_rejects_missing_token(self):
        from app.mcp.tools._auth import AuthError

        with patch.object(
            assessment_tools,
            "resolve_principal",
            AsyncMock(side_effect=AuthError("no token")),
        ):
            result = await assessment_tools.search_findings()
        assert result["code"] == "unauthenticated"


# ── Scope: principal without home_graph_id is rejected ────────────────────────


class TestScopeBoundary:
    @pytest.mark.asyncio
    async def test_create_run_rejects_missing_home_graph(self):
        no_gid = {"id": "u-1", "principal_type": "service_account"}
        with patch.object(
            assessment_tools, "resolve_principal", AsyncMock(return_value=no_gid)
        ):
            result = await assessment_tools.create_run(_create_run_body())
        assert result["code"] == "scope"

    @pytest.mark.asyncio
    async def test_registry_list_items_rejects_missing_home_graph(self):
        no_gid = {"id": "u-1", "principal_type": "service_account"}
        with patch.object(
            registry_tools, "resolve_principal", AsyncMock(return_value=no_gid)
        ):
            result = await registry_tools.list_items("skill")
        assert result["code"] == "scope"


# ── Forbidden — ReBAC denial bubbles up as forbidden ──────────────────────────


class TestForbidden:
    @pytest.mark.asyncio
    async def test_create_run_403_on_rebac_denial(self, patched_principal):
        from app.mcp.tools._auth import ScopeError

        with patch.object(
            assessment_tools,
            "assert_graph_access",
            AsyncMock(side_effect=ScopeError("Access denied")),
        ):
            result = await assessment_tools.create_run(_create_run_body())
        assert result == {"error": "Access denied", "code": "forbidden"}


# ── Happy-path service calls ──────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_create_run_calls_service_with_normalized_subject(
        self, patched_principal, patched_scope, patched_service
    ):
        from app.schemas.assessment_schemas import CreateRunResponse

        patched_service.create_run = AsyncMock(
            return_value=CreateRunResponse(
                run_id="run-001",
                template_id="t-1",
                subject_id="s-1",
                module_run_ids=["mr-1"],
                status="planned",
                already_existed=False,
            )
        )

        # Deliberately try to inject a foreign graph_id on the nested Subject
        # — the wrapper must overwrite it with the principal's graph_id
        # (ADR-010 Scope Enforcer).
        result = await assessment_tools.create_run(
            _create_run_body(graph_id_override="graph-OTHER")
        )

        assert result["run_id"] == "run-001"
        # Verify the service was called with the principal's graph_id
        called_args = patched_service.create_run.await_args
        graph_id_arg, request_arg = called_args.args[0], called_args.args[1]
        assert graph_id_arg == "graph-A"
        assert request_arg.subject.graph_id == "graph-A"  # ADR-010 normalization

    @pytest.mark.asyncio
    async def test_list_runs_returns_paged_response(
        self, patched_principal, patched_scope, patched_service
    ):
        from app.schemas.assessment_schemas import RunSummary

        patched_service.list_runs = AsyncMock(
            return_value=([], False),
        )
        result = await assessment_tools.list_runs(status=None, limit=10)
        assert result["items"] == []
        assert result["page"]["next_cursor"] is None
        assert result["page"]["page_size"] == 0

        # Non-empty + has_more=True → next_cursor must be set
        patched_service.list_runs = AsyncMock(
            return_value=(
                [
                    RunSummary(
                        run_id="r1",
                        template_id="t1",
                        subject_id="s1",
                        status="running",
                        module_run_total=0,
                        module_run_done=0,
                        module_run_failed=0,
                    )
                ],
                True,
            )
        )
        result = await assessment_tools.list_runs(limit=1)
        assert len(result["items"]) == 1
        assert result["page"]["next_cursor"] is not None

    @pytest.mark.asyncio
    async def test_get_run_returns_404_when_service_returns_none(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.get_run_detail = AsyncMock(return_value=None)
        result = await assessment_tools.get_run("missing")
        assert result == {
            "error": "AssessmentRun not found",
            "code": "not_found",
        }

    @pytest.mark.asyncio
    async def test_heartbeat_run_returns_404_on_unknown_run(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.heartbeat_run = AsyncMock(return_value=False)
        result = await assessment_tools.heartbeat_run("nope")
        assert result["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_heartbeat_module_run_calls_update_with_heartbeat_only(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.update_module_run = AsyncMock(return_value=True)
        result = await assessment_tools.heartbeat_module_run("r1", "mr1")
        assert result == {"updated": True, "module_run_id": "mr1"}
        # The wrapper sends an UpdateModuleRunRequest with only last_heartbeat_at set
        call = patched_service.update_module_run.await_args
        update_request = call.args[3]
        assert update_request.last_heartbeat_at is not None
        assert update_request.status is None

    @pytest.mark.asyncio
    async def test_record_finding_bulk_rejects_mixed_module_run_ids(
        self, patched_principal, patched_scope, patched_service
    ):
        body = {
            "findings": [
                {
                    "finding_id": "f1",
                    "graph_id": "graph-A",
                    "run_id": "r1",
                    "module_run_id": "mr1",
                    "claim": "claim 1",
                    "label": "DIRECT",
                    "confidence": 0.9,
                    "dimensions": ["d"],
                },
                {
                    "finding_id": "f2",
                    "graph_id": "graph-A",
                    "run_id": "r1",
                    "module_run_id": "mr2",  # different parent — illegal
                    "claim": "claim 2",
                    "label": "DIRECT",
                    "confidence": 0.9,
                    "dimensions": ["d"],
                },
            ]
        }
        result = await assessment_tools.record_finding_bulk("r1", body)
        assert result["code"] == "bad_request"
        assert "module_run_id" in result["error"]

    @pytest.mark.asyncio
    async def test_get_deliverable_content_404(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.get_deliverable_content = AsyncMock(return_value=None)
        result = await assessment_tools.get_deliverable_content("r", "d")
        assert result["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_search_findings_calls_admin_search(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.search_findings_admin = AsyncMock(return_value=([], False))
        result = await assessment_tools.search_findings(
            source_url="https://example.com"
        )
        assert result["items"] == []
        patched_service.search_findings_admin.assert_awaited_once()


# ── Registry tools ────────────────────────────────────────────────────────────


class TestRegistryWrites:
    @pytest.mark.asyncio
    async def test_persist_item_rejects_invalid_kind(self, patched_principal):
        result = await registry_tools.persist_item("bogus-kind", {})
        assert result["code"] == "bad_request"
        assert "kind" in result["error"]

    @pytest.mark.asyncio
    async def test_persist_item_private_lands_in_tenant_graph(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.persist_registry_item = AsyncMock(return_value=True)
        body = {
            "item_id": "ri-1",
            "graph_id": "graph-OTHER",  # will be overwritten
            "kind": "agent",  # will be pinned from path
            "slug": "my-skill",
            "version": "0.1.0",
            "visibility": "private",
            "owner_user_id": "WRONG",  # will be pinned from principal
            "name": "My Skill",
        }
        result = await registry_tools.persist_item("skill", body)
        assert result["created"] is True
        assert result["kind"] == "skill"
        assert result["graph_id"] == "graph-A"  # principal's home_graph_id
        # owner_user_id pinned
        called = patched_service.persist_registry_item.await_args
        sent_item = called.args[0]
        assert sent_item.owner_user_id == "user-001"
        assert sent_item.kind == "skill"

    @pytest.mark.asyncio
    async def test_persist_item_curated_requires_admin(self, patched_principal):
        """`curated` visibility must invoke the admin gate."""
        from app.mcp.tools._auth import ScopeError

        with (
            patch.object(
                registry_tools,
                "assert_registry_curated_write",
                AsyncMock(side_effect=ScopeError("admin required")),
            ),
        ):
            body = {
                "item_id": "ri-1",
                "graph_id": "__registry__",
                "kind": "skill",
                "slug": "x",
                "version": "0.1.0",
                "visibility": "curated",
                "owner_user_id": "platform",
                "name": "X",
            }
            result = await registry_tools.persist_item("skill", body)
        assert result["code"] == "forbidden"

    @pytest.mark.asyncio
    async def test_list_items_paged_response(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.list_registry_items = AsyncMock(return_value=([], False))
        result = await registry_tools.list_items("skill")
        assert result["items"] == []
        assert result["page"]["next_cursor"] is None

    @pytest.mark.asyncio
    async def test_get_item_404(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.get_registry_item = AsyncMock(return_value=None)
        result = await registry_tools.get_item("skill", "nope")
        assert result["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_item_content_404(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.get_registry_item_content = AsyncMock(return_value=None)
        result = await registry_tools.get_item_content("skill", "nope", "0.1.0")
        assert result["code"] == "not_found"


# ── Schema validation rejects malformed args before service is called ─────────


class TestSchemaValidation:
    @pytest.mark.asyncio
    async def test_create_run_rejects_missing_template_slug(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.create_run = AsyncMock()
        # Body missing `template_slug` fails Pydantic validation before
        # the service is ever touched.
        result = await assessment_tools.create_run({"subject": _VALID_SUBJECT})
        assert result["code"] == "bad_request"
        patched_service.create_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_module_run_rejects_garbage_body(
        self, patched_principal, patched_scope, patched_service
    ):
        patched_service.update_module_run = AsyncMock()
        # status='invalid' fails the Pydantic Literal["planned",...]
        result = await assessment_tools.update_module_run(
            "r1", "mr1", {"status": "invalid-status"}
        )
        assert result["code"] == "bad_request"
        patched_service.update_module_run.assert_not_awaited()
