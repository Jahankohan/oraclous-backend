"""Unit tests for the assessment-substrate REST read endpoints (TASK-079).

Mocks the auth pipeline and the `AssessmentService` to exercise the read
routes in isolation. Integration tests against real Neo4j live in
`tests/integration/test_assessments_read_endpoints.py`.

Covers:
- Endpoint dispatch: each route calls the matching `AssessmentService`
  method with the JWT-derived `graph_id` and the correct positional args.
- 404 propagation when the service returns `None`.
- 400 on malformed cursor.
- Cursor round-trip: the `next_cursor` in the response decodes to the next
  offset, and the second call returns the next page.
- 403 on `verify_graph_access` denial.
- Admin findings:search rejects non-admin callers with 403.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi import status as http_status
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_current_user, get_current_user_id
from app.api.v1.endpoints import assessments_reads as reads_mod
from app.api.v1.endpoints._pagination import decode_cursor
from app.api.v1.endpoints.assessments_reads import _assessment_service
from app.main import app
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    ConflictRow,
    DeliverableRow,
    FindingRow,
    FindingSearchRow,
    ModuleDefinition,
    ModuleRunRow,
    RegistryItemRow,
    RunDetail,
    RunSummary,
    Source,
    UnresolvedQuestionRow,
    WaveModuleStatus,
    WaveStatusResponse,
)

_TENANT = "graph-tenant-A"
_USER = "u-test-001"
_BASE = "/api/v1/api/v1/assessments"


def _principal(home_graph_id: str = _TENANT, **extra: Any) -> dict[str, Any]:
    d = {
        "id": _USER,
        "principal_type": "service_account",
        "home_graph_id": home_graph_id,
    }
    d.update(extra)
    return d


@pytest.fixture
def mock_svc() -> MagicMock:
    """Build a mocked `AssessmentService` with read methods stubbed."""
    svc = MagicMock()

    # list_runs → 1 row, no more pages
    svc.list_runs = AsyncMock(
        return_value=(
            [
                RunSummary(
                    run_id="run-1",
                    template_id="tpl-1",
                    template_slug="assess-v1",
                    subject_id="subj-1",
                    subject_slug="eurail",
                    subject_name="Eurail",
                    status="finished",
                    started_at=datetime(2026, 5, 1, tzinfo=UTC),
                    finished_at=datetime(2026, 5, 1, 1, tzinfo=UTC),
                    orchestrator_last_seen=datetime(2026, 5, 1, 1, tzinfo=UTC),
                    module_run_total=3,
                    module_run_done=3,
                    module_run_failed=0,
                )
            ],
            False,
        )
    )

    # get_run_detail → full rollup
    svc.get_run_detail = AsyncMock(
        return_value=RunDetail(
            run_id="run-1",
            graph_id=_TENANT,
            template_id="tpl-1",
            template_slug="assess-v1",
            subject_id="subj-1",
            subject_slug="eurail",
            subject_name="Eurail",
            status="finished",
            module_run_total=3,
            module_run_done=3,
            module_run_failed=0,
            finding_count=12,
            conflict_count=1,
            open_question_count=0,
            deliverable_count=5,
        )
    )

    # get_wave_status
    svc.get_wave_status = AsyncMock(
        return_value=WaveStatusResponse(
            run_id="run-1",
            wave=1,
            total=2,
            done=2,
            failed=0,
            running=0,
            planned=0,
            cancelled=0,
            modules=[
                WaveModuleStatus(
                    module_run_id="mr-1",
                    module_id="m-1",
                    module_slug="research-1",
                    module_name="R1",
                    module_kind="research",
                    status="finished",
                    evidence_count=5,
                )
            ],
        )
    )

    # list_module_runs
    svc.list_module_runs = AsyncMock(
        return_value=(
            [
                ModuleRunRow(
                    module_run_id="mr-1",
                    run_id="run-1",
                    module_id="m-1",
                    module_slug="research-1",
                    module_name="R1",
                    module_kind="research",
                    module_wave=1,
                    module_agent_id=None,
                    wave=1,
                    status="finished",
                    evidence_count=5,
                )
            ],
            False,
        )
    )

    # list_findings — with source hydrated
    svc.list_findings = AsyncMock(
        return_value=(
            [
                FindingRow(
                    finding_id="f-1",
                    run_id="run-1",
                    module_run_id="mr-1",
                    module_slug="research-1",
                    module_name="R1",
                    claim="Climate is warming",
                    label="DIRECT",
                    confidence=0.9,
                    dimensions=["climate"],
                    source_id="src-1",
                    source=Source(
                        source_id="src-1",
                        type="article",
                        url_normalized="https://example.com/a",
                        name="Example A",
                    ),
                )
            ],
            False,
        )
    )

    # list_conflicts
    svc.list_conflicts = AsyncMock(
        return_value=(
            [
                ConflictRow(
                    conflict_id="cf-1",
                    run_id="run-1",
                    topic="Topic",
                    summary="Summary",
                    status="open",
                    involved_finding_ids=["f-1", "f-2"],
                )
            ],
            False,
        )
    )

    # list_unresolved_questions
    svc.list_unresolved_questions = AsyncMock(
        return_value=(
            [
                UnresolvedQuestionRow(
                    question_id="q-1",
                    run_id="run-1",
                    module_run_id="mr-1",
                    module_slug="research-1",
                    text="Question?",
                    status="open",
                )
            ],
            False,
        )
    )

    # list_deliverables
    svc.list_deliverables = AsyncMock(
        return_value=(
            [
                DeliverableRow(
                    deliverable_id="d-1",
                    run_id="run-1",
                    module_run_id="mr-1",
                    kind="module-md",
                    filename="m1.md",
                    ordinal=0,
                    has_inline_content=True,
                )
            ],
            False,
        )
    )

    # get_deliverable_content
    svc.get_deliverable_content = AsyncMock(
        return_value={
            "kind": "module-md",
            "filename": "m1.md",
            "content_uri": None,
            "content_inline": "# Module markdown body",
            "sha256": None,
            "word_count": 3,
        }
    )

    # list_template_modules
    svc.list_template_modules = AsyncMock(
        return_value=(
            {
                "template_id": "tpl-1",
                "template_slug": "assess-v1",
                "template_name": "Assess v1",
                "template_version": "0.1.0",
            },
            [
                ModuleDefinition(
                    module_id="m-1",
                    template_id="tpl-1",
                    slug="research-1",
                    name="R1",
                    wave=1,
                    ordinal=0,
                    kind="research",
                )
            ],
        )
    )

    # Registry
    svc.list_registry_items = AsyncMock(
        return_value=(
            [
                RegistryItemRow(
                    item_id="ri-1",
                    graph_id=_TENANT,
                    kind="skill",
                    slug="my-skill",
                    version="0.1.0",
                    visibility="private",
                    owner_user_id=_USER,
                    name="My Skill",
                )
            ],
            False,
        )
    )
    svc.get_registry_item = AsyncMock(
        return_value=RegistryItemRow(
            item_id="ri-1",
            graph_id=_TENANT,
            kind="skill",
            slug="my-skill",
            version="0.1.0",
            visibility="private",
            owner_user_id=_USER,
            name="My Skill",
        )
    )
    from app.schemas.assessment_schemas import RegistryItemContent

    svc.get_registry_item_content = AsyncMock(
        return_value=RegistryItemContent(
            item_id="ri-1",
            kind="skill",
            slug="my-skill",
            version="0.1.0",
            content_type="text/markdown",
            content_inline="# Skill body",
            content_uri=None,
            sha256=None,
        )
    )

    # search_findings_admin
    svc.search_findings_admin = AsyncMock(
        return_value=(
            [
                FindingSearchRow(
                    finding_id="f-1",
                    graph_id="tenant-foo",
                    run_id="run-1",
                    module_run_id="mr-1",
                    claim="Cross-tenant claim",
                    label="DIRECT",
                    confidence=0.95,
                    dimensions=["x"],
                    source_id="src-1",
                    source=None,
                )
            ],
            False,
        )
    )

    return svc


@pytest_asyncio.fixture
async def client(mock_svc):
    """Async client with auth + service overridden."""
    app.dependency_overrides[get_current_user] = lambda: _principal()
    app.dependency_overrides[get_current_user_id] = lambda: _USER
    app.dependency_overrides[_assessment_service] = lambda: mock_svc

    original_verify = reads_mod.verify_graph_access
    reads_mod.verify_graph_access = AsyncMock(return_value=_TENANT)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    reads_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


# ── list_runs ────────────────────────────────────────────────────────────────


class TestListRuns:
    async def test_returns_rows_with_next_cursor_none(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["run_id"] == "run-1"
        assert body["page"]["next_cursor"] is None
        assert body["page"]["page_size"] == 1
        mock_svc.list_runs.assert_awaited_once()
        # Service was called with JWT-derived graph_id
        kwargs = mock_svc.list_runs.await_args.kwargs
        args = mock_svc.list_runs.await_args.args
        assert args[0] == _TENANT
        assert kwargs.get("offset") == 0

    async def test_with_status_filter_passes_through(self, client, mock_svc):
        await client.get(f"{_BASE}/runs?status=finished")
        kwargs = mock_svc.list_runs.await_args.kwargs
        assert kwargs.get("status") == "finished"

    async def test_with_subject_filter_passes_through(self, client, mock_svc):
        await client.get(f"{_BASE}/runs?subject=eurail")
        kwargs = mock_svc.list_runs.await_args.kwargs
        assert kwargs.get("subject_slug") == "eurail"

    async def test_malformed_cursor_returns_400(self, client):
        resp = await client.get(f"{_BASE}/runs?cursor=!!not-base64!!")
        assert resp.status_code == 400
        assert "cursor" in resp.json()["detail"].lower()

    async def test_cursor_round_trip_yields_correct_next_offset(self, client, mock_svc):
        # Mock 1-row page with has_more=True
        mock_svc.list_runs.return_value = (
            [
                RunSummary(
                    run_id="run-x",
                    template_id="tpl-1",
                    subject_id="subj-1",
                    status="planned",
                )
            ],
            True,
        )
        resp = await client.get(f"{_BASE}/runs?limit=1")
        body = resp.json()
        assert body["page"]["next_cursor"] is not None
        cursor = body["page"]["next_cursor"]

        # Decode the cursor and check the offset
        offset, _last_id = decode_cursor(cursor)
        assert offset == 1  # offset 0 + page_size 1

        # Second call with cursor → offset=1
        await client.get(f"{_BASE}/runs?limit=1&cursor={cursor}")
        kwargs = mock_svc.list_runs.await_args.kwargs
        assert kwargs.get("offset") == 1


# ── get_run ──────────────────────────────────────────────────────────────────


class TestGetRun:
    async def test_returns_detail(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-1"
        assert body["finding_count"] == 12
        graph_id, run_id = mock_svc.get_run_detail.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-1"

    async def test_404_when_not_in_tenant(self, client, mock_svc):
        mock_svc.get_run_detail.return_value = None
        resp = await client.get(f"{_BASE}/runs/run-missing")
        assert resp.status_code == 404


# ── get_wave_status ──────────────────────────────────────────────────────────


class TestGetWaveStatus:
    async def test_returns_wave(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/waves/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["wave"] == 1
        assert body["done"] == 2
        graph_id, run_id, wave = mock_svc.get_wave_status.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-1"
        assert wave == 1

    async def test_404_when_run_missing(self, client, mock_svc):
        mock_svc.get_wave_status.return_value = None
        resp = await client.get(f"{_BASE}/runs/run-x/waves/1")
        assert resp.status_code == 404

    async def test_wave_must_be_positive_int(self, client):
        resp = await client.get(f"{_BASE}/runs/run-1/waves/0")
        assert resp.status_code == 422


# ── list_module_runs ─────────────────────────────────────────────────────────


class TestListModuleRuns:
    async def test_returns_rows(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/module-runs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["module_run_id"] == "mr-1"

    async def test_status_filter_passes_through(self, client, mock_svc):
        await client.get(f"{_BASE}/runs/run-1/module-runs?status=running")
        kwargs = mock_svc.list_module_runs.await_args.kwargs
        assert kwargs.get("status") == "running"

    async def test_404_when_run_missing(self, client, mock_svc):
        mock_svc.list_module_runs.return_value = None
        resp = await client.get(f"{_BASE}/runs/run-x/module-runs")
        assert resp.status_code == 404


# ── list_findings ────────────────────────────────────────────────────────────


class TestListFindings:
    async def test_returns_rows_with_source_hydrated(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/findings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["source"]["source_id"] == "src-1"
        assert body["items"][0]["source"]["type"] == "article"

    async def test_all_filters_pass_through(self, client, mock_svc):
        await client.get(
            f"{_BASE}/runs/run-1/findings"
            "?module=research-1&dimension=climate&label=DIRECT"
            "&min_confidence=0.5&source_type=article"
        )
        kwargs = mock_svc.list_findings.await_args.kwargs
        assert kwargs["module_slug"] == "research-1"
        assert kwargs["dimension"] == "climate"
        assert kwargs["label"] == "DIRECT"
        assert kwargs["min_confidence"] == 0.5
        assert kwargs["source_type"] == "article"

    async def test_404_when_run_missing(self, client, mock_svc):
        mock_svc.list_findings.return_value = None
        resp = await client.get(f"{_BASE}/runs/run-x/findings")
        assert resp.status_code == 404

    async def test_min_confidence_outside_range_rejected(self, client):
        resp = await client.get(f"{_BASE}/runs/run-1/findings?min_confidence=1.5")
        assert resp.status_code == 422


# ── list_conflicts ───────────────────────────────────────────────────────────


class TestListConflicts:
    async def test_returns_rows(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/conflicts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["conflict_id"] == "cf-1"
        assert body["items"][0]["involved_finding_ids"] == ["f-1", "f-2"]

    async def test_status_filter_pass_through(self, client, mock_svc):
        await client.get(f"{_BASE}/runs/run-1/conflicts?status=resolved")
        kwargs = mock_svc.list_conflicts.await_args.kwargs
        assert kwargs.get("status") == "resolved"


# ── list_unresolved_questions ────────────────────────────────────────────────


class TestListUnresolvedQuestions:
    async def test_returns_rows(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/unresolved-questions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["question_id"] == "q-1"


# ── list_deliverables ────────────────────────────────────────────────────────


class TestListDeliverables:
    async def test_returns_rows(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/deliverables")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["deliverable_id"] == "d-1"
        assert body["items"][0]["has_inline_content"] is True


# ── get_deliverable_content ──────────────────────────────────────────────────


class TestGetDeliverableContent:
    async def test_returns_inline_markdown(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs/run-1/deliverables/d-1/content")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.text == "# Module markdown body"

    async def test_returns_location_header_when_only_uri(self, client, mock_svc):
        mock_svc.get_deliverable_content.return_value = {
            "kind": "final-pdf",
            "filename": "report.pdf",
            "content_uri": "file:///tmp/report.pdf",
            "content_inline": None,
            "sha256": "abc",
            "word_count": None,
        }
        resp = await client.get(f"{_BASE}/runs/run-1/deliverables/d-1/content")
        assert resp.status_code == 200
        assert resp.headers.get("location") == "file:///tmp/report.pdf"
        assert resp.headers["content-type"].startswith("application/pdf")

    async def test_404_when_missing(self, client, mock_svc):
        mock_svc.get_deliverable_content.return_value = None
        resp = await client.get(f"{_BASE}/runs/run-1/deliverables/d-missing/content")
        assert resp.status_code == 404


# ── list_template_modules ────────────────────────────────────────────────────


class TestListTemplateModules:
    async def test_returns_template_with_modules(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/templates/assess-v1/modules")
        assert resp.status_code == 200
        body = resp.json()
        assert body["template_slug"] == "assess-v1"
        assert len(body["modules"]) == 1
        assert body["modules"][0]["slug"] == "research-1"

    async def test_404_when_template_unknown(self, client, mock_svc):
        mock_svc.list_template_modules.return_value = None
        resp = await client.get(f"{_BASE}/templates/unknown/modules")
        assert resp.status_code == 404


# ── Registry ─────────────────────────────────────────────────────────────────


class TestRegistryReads:
    async def test_list_registry_items(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/registry/skill")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["item_id"] == "ri-1"
        kwargs = mock_svc.list_registry_items.await_args.kwargs
        assert kwargs["caller_user_id"] == _USER
        assert kwargs["caller_graph_id"] == _TENANT
        assert kwargs["kind"] == "skill"
        # No visibility filter unless explicitly asked
        assert kwargs.get("visibility") is None

    async def test_list_visibility_filter_pass_through(self, client, mock_svc):
        await client.get(f"{_BASE}/registry/skill?visibility=public")
        kwargs = mock_svc.list_registry_items.await_args.kwargs
        assert kwargs.get("visibility") == "public"

    async def test_list_owner_filter_pass_through(self, client, mock_svc):
        await client.get(f"{_BASE}/registry/skill?owner=alice")
        kwargs = mock_svc.list_registry_items.await_args.kwargs
        assert kwargs.get("owner_user_id") == "alice"

    async def test_get_item_returns_metadata(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/registry/skill/my-skill")
        assert resp.status_code == 200
        body = resp.json()
        assert body["item_id"] == "ri-1"
        assert body["version"] == "0.1.0"
        # Metadata endpoint never includes the inline body
        assert body["content_inline"] is None

    async def test_get_item_404_when_missing(self, client, mock_svc):
        mock_svc.get_registry_item.return_value = None
        resp = await client.get(f"{_BASE}/registry/skill/missing")
        assert resp.status_code == 404

    async def test_get_content_returns_inline_body(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/registry/skill/my-skill/0.1.0/content")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert resp.text == "# Skill body"

    async def test_get_content_404_when_missing(self, client, mock_svc):
        mock_svc.get_registry_item_content.return_value = None
        resp = await client.get(f"{_BASE}/registry/skill/missing/1.0.0/content")
        assert resp.status_code == 404


# ── Admin: findings:search ───────────────────────────────────────────────────


class TestAdminFindingsSearch:
    async def test_dispatches_when_admin(self, client, mock_svc):
        resp = await client.get(
            f"{_BASE}/findings:search?source_url=https://example.com/a"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["items"][0]["graph_id"] == "tenant-foo"
        kwargs = mock_svc.search_findings_admin.await_args.kwargs
        assert kwargs.get("source_url") == "https://example.com/a"

    async def test_403_when_admin_acl_denied(self, mock_svc):
        """The admin check is the single gate; denial → 403."""
        app.dependency_overrides[get_current_user] = lambda: _principal()
        app.dependency_overrides[get_current_user_id] = lambda: _USER
        app.dependency_overrides[_assessment_service] = lambda: mock_svc

        called: dict[str, Any] = {}

        async def _deny(graph_id: str, level: str, user_id: str):
            called["args"] = (graph_id, level, user_id)
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        original_verify = reads_mod.verify_graph_access
        reads_mod.verify_graph_access = _deny
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(
                    f"{_BASE}/findings:search?source_url=https://example.com/a"
                )
            assert resp.status_code == 403
            assert called["args"] == (
                ASSESSMENTS_CATALOG_GRAPH_ID,
                "admin",
                _USER,
            )
        finally:
            reads_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)


# ── Auth failures ────────────────────────────────────────────────────────────


class TestAuthFailures:
    async def test_missing_jwt_rejected(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.get(f"{_BASE}/runs")
        assert resp.status_code in (401, 403)

    async def test_403_propagates_from_verify_graph_access(self, mock_svc):
        app.dependency_overrides[get_current_user] = lambda: _principal()
        app.dependency_overrides[get_current_user_id] = lambda: _USER
        app.dependency_overrides[_assessment_service] = lambda: mock_svc

        async def _deny(*a, **kw):
            raise HTTPException(status_code=403, detail="Access denied")

        original_verify = reads_mod.verify_graph_access
        reads_mod.verify_graph_access = _deny
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(f"{_BASE}/runs/run-1")
            assert resp.status_code == 403
        finally:
            reads_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)

    async def test_400_when_missing_home_graph_id(self, mock_svc):
        """Principal without `home_graph_id` claim → 400."""
        app.dependency_overrides[get_current_user] = lambda: _principal(
            home_graph_id=""
        )
        app.dependency_overrides[get_current_user_id] = lambda: _USER
        app.dependency_overrides[_assessment_service] = lambda: mock_svc

        original_verify = reads_mod.verify_graph_access
        reads_mod.verify_graph_access = AsyncMock(return_value=_TENANT)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(f"{_BASE}/runs/run-1")
            assert resp.status_code == 400
            assert "home_graph_id" in resp.json()["detail"]
        finally:
            reads_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)


# ── Body-graph-id ignored ────────────────────────────────────────────────────


class TestScopeEnforcerReadSide:
    """Even on read endpoints, a query-string `graph_id` must not override the JWT.

    Read endpoints do not accept a `graph_id=` query parameter at all — the
    schema doesn't declare it. This test pins the behavior: if a caller tries
    to pass `graph_id=tenant-B`, FastAPI silently ignores it (it's not in the
    signature), and the service is called with the JWT-derived value.
    """

    async def test_graph_id_query_param_silently_ignored(self, client, mock_svc):
        resp = await client.get(f"{_BASE}/runs?graph_id=other-tenant")
        assert resp.status_code == 200
        args = mock_svc.list_runs.await_args.args
        assert args[0] == _TENANT  # JWT-derived, NOT 'other-tenant'
