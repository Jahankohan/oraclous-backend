"""Unit tests for the assessment-substrate REST endpoints (STORY-026, TASK-069).

Mocks the auth pipeline and the `AssessmentService` to exercise the routes in
isolation. Integration tests against real Neo4j live in
`tests/integration/test_assessments_endpoints.py`.

Covers (per TASK-069 DoD):
- Endpoint dispatch: each route calls the matching `AssessmentService` method
  with the JWT-derived `graph_id` and the correct positional arguments.
- ADR-010 §Scope Enforcer: a `graph_id` value supplied in the request body is
  ignored — the service is invoked with the JWT principal's `home_graph_id`.
- 401 on missing JWT, 403 on ReBAC denial.
- 207 Multi-Status on bulk responses.
- 404 on missing :ModuleRun / :AssessmentRun for the heartbeat + update paths.
- `persist_registry_item` routes by visibility: private → tenant graph;
  curated → __registry__ with platform-admin check; public → __registry__.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_current_user, get_current_user_id
from app.api.v1.endpoints import assessments as assessments_mod
from app.api.v1.endpoints.assessments import (
    _assessment_service,
    _principal_graph_id,
)
from app.main import app
from app.schemas.assessment_schemas import (
    REGISTRY_CATALOG_GRAPH_ID,
    BulkItemResult,
    BulkResponse,
    CreateRunResponse,
    FinalizeRunResponse,
)


# ── Constants ────────────────────────────────────────────────────────────────

_TENANT = "graph-tenant-A"
_OTHER_TENANT = "graph-tenant-B"
_USER = "u-test-001"
_BASE = "/api/v1/api/v1/assessments"


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _principal(
    home_graph_id: str = _TENANT,
    principal_type: str = "service_account",
    user_id: str = _USER,
    **extra: Any,
) -> dict[str, Any]:
    d = {
        "id": user_id,
        "principal_type": principal_type,
        "home_graph_id": home_graph_id,
    }
    d.update(extra)
    return d


@pytest.fixture
def mock_svc() -> AsyncMock:
    """Build a mocked :class:`AssessmentService` with every write method stubbed."""
    svc = MagicMock()
    svc.create_run = AsyncMock(
        return_value=CreateRunResponse(
            run_id="run-xyz",
            template_id="tpl-1",
            subject_id="subj-1",
            module_run_ids=["mr-1", "mr-2"],
            status="planned",
            already_existed=False,
        )
    )
    svc.update_module_run = AsyncMock(return_value=True)
    svc.record_finding_bulk = AsyncMock(
        return_value=BulkResponse(
            total=1,
            succeeded=1,
            failed=0,
            results=[BulkItemResult(id="f-1", success=True)],
        )
    )
    svc.record_conflict = AsyncMock(return_value=True)
    svc.record_unresolved_question = AsyncMock(return_value=True)
    svc.persist_deliverable = AsyncMock(return_value=True)
    svc.persist_final_docs = AsyncMock(
        return_value=BulkResponse(
            total=1,
            succeeded=1,
            failed=0,
            results=[BulkItemResult(id="d-1", success=True)],
        )
    )
    from datetime import datetime, timezone

    svc.finalize_run = AsyncMock(
        return_value=FinalizeRunResponse(
            run_id="run-xyz",
            passed=True,
            status="finished",
            finished_at=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
            direct_finding_count=10,
            inferred_finding_count=2,
            deliverable_count=5,
            unresolved_conflict_count=0,
            open_question_count=0,
            failure_reasons=[],
        )
    )
    svc.heartbeat_run = AsyncMock(return_value=True)
    svc.persist_registry_item = AsyncMock(return_value=True)
    return svc


@pytest_asyncio.fixture
async def client(mock_svc) -> AsyncClient:
    """An async client with auth + service dependencies overridden.

    The overrides install:
    - A service-account principal whose `home_graph_id` is the test tenant.
    - The mocked `AssessmentService` so we can assert on the service calls.
    - A no-op `verify_graph_access` so the ReBAC layer never hits Neo4j.
    """
    app.dependency_overrides[get_current_user] = lambda: _principal()
    app.dependency_overrides[get_current_user_id] = lambda: _USER
    app.dependency_overrides[_assessment_service] = lambda: mock_svc

    # Patch the verify_graph_access symbol in the endpoint module so the
    # endpoints' direct `await verify_graph_access(...)` calls become no-ops.
    original_verify = assessments_mod.verify_graph_access
    assessments_mod.verify_graph_access = AsyncMock(return_value=_TENANT)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    assessments_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


# ── /assessments/runs ────────────────────────────────────────────────────────


class TestCreateRunEndpoint:
    async def test_dispatches_to_service_with_jwt_graph_id(self, client, mock_svc):
        body = {
            "template_slug": "assess-v1",
            "subject": {
                "subject_id": "subj-1",
                # Caller-supplied graph_id that does NOT match the JWT —
                # ADR-010 says ignore it.
                "graph_id": _OTHER_TENANT,
                "slug": "eurail",
                "name": "Eurail",
            },
            "cli_flags": {},
        }
        resp = await client.post(f"{_BASE}/runs", json=body)
        assert resp.status_code == 201, resp.text
        assert resp.json()["run_id"] == "run-xyz"
        # Service was called with the JWT-derived graph_id, not the body's.
        mock_svc.create_run.assert_awaited_once()
        call = mock_svc.create_run.await_args
        assert call.args[0] == _TENANT
        request_arg = call.args[1]
        # Body's subject.graph_id should have been normalized to the JWT tenant.
        assert request_arg.subject.graph_id == _TENANT

    async def test_returns_400_when_service_rejects(self, client, mock_svc):
        mock_svc.create_run.side_effect = ValueError(
            "Assessment template not found: slug='nope'"
        )
        body = {
            "template_slug": "nope",
            "subject": {
                "subject_id": "subj-1",
                "graph_id": _TENANT,
                "slug": "eurail",
                "name": "Eurail",
            },
        }
        resp = await client.post(f"{_BASE}/runs", json=body)
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()


# ── /assessments/runs/{run_id}/module-runs/{module_run_id} ───────────────────


class TestUpdateModuleRunEndpoint:
    async def test_partial_update_dispatches(self, client, mock_svc):
        resp = await client.patch(
            f"{_BASE}/runs/run-xyz/module-runs/mr-1",
            json={"status": "running", "evidence_count": 5},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"updated": True, "module_run_id": "mr-1"}
        mock_svc.update_module_run.assert_awaited_once()
        call = mock_svc.update_module_run.await_args
        assert call.args[0] == _TENANT
        assert call.args[1] == "run-xyz"
        assert call.args[2] == "mr-1"
        update = call.args[3]
        assert update.status == "running"
        assert update.evidence_count == 5

    async def test_returns_404_when_module_run_missing(self, client, mock_svc):
        mock_svc.update_module_run.return_value = False
        resp = await client.patch(
            f"{_BASE}/runs/run-xyz/module-runs/missing",
            json={"status": "running"},
        )
        assert resp.status_code == 404


# ── /assessments/runs/{run_id}/findings:bulk ─────────────────────────────────


class TestRecordFindingBulkEndpoint:
    def _body(self, **overrides) -> dict:
        return {
            "findings": [
                {
                    "finding_id": "f-1",
                    "graph_id": overrides.get("graph_id", _OTHER_TENANT),  # ignored
                    "run_id": overrides.get("run_id", "run-xyz"),
                    "module_run_id": "mr-1",
                    "claim": "Climate is warming",
                    "label": "DIRECT",
                    "confidence": 0.9,
                }
            ]
        }

    async def test_returns_207_with_bulk_response_shape(self, client, mock_svc):
        resp = await client.post(
            f"{_BASE}/runs/run-xyz/findings:bulk", json=self._body()
        )
        assert resp.status_code == 207, resp.text
        body = resp.json()
        assert body["total"] == 1
        assert body["succeeded"] == 1
        assert body["results"][0]["id"] == "f-1"

    async def test_normalizes_body_graph_id_to_jwt(self, client, mock_svc):
        await client.post(f"{_BASE}/runs/run-xyz/findings:bulk", json=self._body())
        mock_svc.record_finding_bulk.assert_awaited_once()
        graph_id, run_id, module_run_id, findings = (
            mock_svc.record_finding_bulk.await_args.args
        )
        assert graph_id == _TENANT
        assert run_id == "run-xyz"
        assert module_run_id == "mr-1"
        # Each finding's graph_id+run_id should have been overwritten.
        assert findings[0].graph_id == _TENANT
        assert findings[0].run_id == "run-xyz"

    async def test_mixed_module_runs_rejected_400(self, client, mock_svc):
        body = {
            "findings": [
                {
                    "finding_id": "f-1",
                    "graph_id": _TENANT,
                    "run_id": "run-xyz",
                    "module_run_id": "mr-1",
                    "claim": "C1",
                },
                {
                    "finding_id": "f-2",
                    "graph_id": _TENANT,
                    "run_id": "run-xyz",
                    "module_run_id": "mr-2",  # different parent
                    "claim": "C2",
                },
            ]
        }
        resp = await client.post(f"{_BASE}/runs/run-xyz/findings:bulk", json=body)
        assert resp.status_code == 400
        assert "module_run_id" in resp.json()["detail"]

    async def test_empty_findings_returns_empty_bulk(self, client, mock_svc):
        resp = await client.post(
            f"{_BASE}/runs/run-xyz/findings:bulk", json={"findings": []}
        )
        assert resp.status_code == 207
        body = resp.json()
        assert body["total"] == 0
        assert body["results"] == []
        mock_svc.record_finding_bulk.assert_not_awaited()


# ── /assessments/runs/{run_id}/conflicts ─────────────────────────────────────


class TestRecordConflictEndpoint:
    async def test_dispatches(self, client, mock_svc):
        body = {
            "conflict": {
                "conflict_id": "cf-1",
                "graph_id": _OTHER_TENANT,  # ignored
                "run_id": "run-xyz",
                "topic": "T",
                "summary": "S",
                "status": "open",
                "involved_finding_ids": ["f-1", "f-2"],
            }
        }
        resp = await client.post(f"{_BASE}/runs/run-xyz/conflicts", json=body)
        assert resp.status_code == 201, resp.text
        assert resp.json() == {"conflict_id": "cf-1", "created": True}
        graph_id, run_id, conflict = mock_svc.record_conflict.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-xyz"
        assert conflict.graph_id == _TENANT  # normalized


# ── /assessments/runs/{run_id}/unresolved-questions ──────────────────────────


class TestRecordUnresolvedQuestionEndpoint:
    async def test_dispatches(self, client, mock_svc):
        body = {
            "question": {
                "question_id": "q-1",
                "graph_id": _OTHER_TENANT,
                "run_id": "run-xyz",
                "module_run_id": "mr-1",
                "text": "How does X affect Y?",
                "status": "open",
            }
        }
        resp = await client.post(
            f"{_BASE}/runs/run-xyz/unresolved-questions", json=body
        )
        assert resp.status_code == 201, resp.text
        assert resp.json() == {"question_id": "q-1", "created": True}
        graph_id, run_id, mr_id, question = (
            mock_svc.record_unresolved_question.await_args.args
        )
        assert graph_id == _TENANT
        assert run_id == "run-xyz"
        assert mr_id == "mr-1"
        assert question.graph_id == _TENANT


# ── /assessments/runs/{run_id}/deliverables ──────────────────────────────────


class TestPersistDeliverableEndpoint:
    async def test_dispatches(self, client, mock_svc):
        body = {
            "deliverable": {
                "deliverable_id": "d-1",
                "graph_id": _OTHER_TENANT,
                "run_id": "run-xyz",
                "module_run_id": "mr-1",
                "kind": "module-md",
                "filename": "module-1.md",
                "ordinal": 0,
            }
        }
        resp = await client.post(f"{_BASE}/runs/run-xyz/deliverables", json=body)
        assert resp.status_code == 201, resp.text
        graph_id, run_id, deliverable = mock_svc.persist_deliverable.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-xyz"
        assert deliverable.graph_id == _TENANT


# ── /assessments/runs/{run_id}/deliverables:bulk-final ──────────────────────


class TestPersistFinalDocsEndpoint:
    async def test_returns_207(self, client, mock_svc):
        body = {
            "deliverables": [
                {
                    "deliverable_id": "d-1",
                    "graph_id": _OTHER_TENANT,
                    "run_id": "run-xyz",
                    "kind": "final-md",
                    "filename": "01-intro.md",
                    "ordinal": 0,
                }
            ]
        }
        resp = await client.post(
            f"{_BASE}/runs/run-xyz/deliverables:bulk-final", json=body
        )
        assert resp.status_code == 207, resp.text
        body_resp = resp.json()
        assert body_resp["total"] == 1
        graph_id, run_id, deliverables = mock_svc.persist_final_docs.await_args.args
        assert graph_id == _TENANT
        assert deliverables[0].graph_id == _TENANT


# ── /assessments/runs/{run_id}:finalize ──────────────────────────────────────


class TestFinalizeRunEndpoint:
    async def test_dispatches(self, client, mock_svc):
        resp = await client.post(f"{_BASE}/runs/run-xyz:finalize")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["run_id"] == "run-xyz"
        assert body["passed"] is True
        graph_id, run_id = mock_svc.finalize_run.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-xyz"


# ── Heartbeats ────────────────────────────────────────────────────────────────


class TestHeartbeatEndpoints:
    async def test_run_heartbeat_dispatches(self, client, mock_svc):
        resp = await client.post(f"{_BASE}/runs/run-xyz:heartbeat")
        assert resp.status_code == 200, resp.text
        graph_id, run_id = mock_svc.heartbeat_run.await_args.args
        assert graph_id == _TENANT
        assert run_id == "run-xyz"

    async def test_run_heartbeat_404_when_missing(self, client, mock_svc):
        mock_svc.heartbeat_run.return_value = False
        resp = await client.post(f"{_BASE}/runs/unknown:heartbeat")
        assert resp.status_code == 404

    async def test_module_run_heartbeat_uses_update_module_run(
        self, client, mock_svc
    ):
        resp = await client.post(
            f"{_BASE}/runs/run-xyz/module-runs/mr-1:heartbeat"
        )
        assert resp.status_code == 200, resp.text
        # Implementation reuses update_module_run with last_heartbeat_at set.
        mock_svc.update_module_run.assert_awaited()
        call = mock_svc.update_module_run.await_args
        update = call.args[3]
        assert update.last_heartbeat_at is not None
        # Crucially, the heartbeat path does NOT touch status.
        assert update.status is None


# ── /assessments/registry/{kind} ─────────────────────────────────────────────


class TestPersistRegistryItemEndpoint:
    def _body(self, visibility: str = "private", **overrides) -> dict:
        body = {
            "item_id": "ri-test-1",
            "graph_id": overrides.get("graph_id", "ignored-by-server"),
            "kind": "skill",
            "slug": "my-skill",
            "version": "0.1.0",
            "visibility": visibility,
            "owner_user_id": "spoofed-user",  # ignored
            "name": "My Skill",
        }
        body.update(overrides)
        return body

    async def test_private_writes_to_tenant_graph(self, client, mock_svc):
        resp = await client.post(
            f"{_BASE}/registry/skill", json=self._body(visibility="private")
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["graph_id"] == _TENANT
        item_arg = mock_svc.persist_registry_item.await_args.args[0]
        assert item_arg.graph_id == _TENANT
        # owner_user_id pinned from principal, not from body
        assert item_arg.owner_user_id == _USER
        # kind pinned from path, not from body
        assert item_arg.kind == "skill"

    async def test_public_writes_to_registry_catalog(self, client, mock_svc):
        resp = await client.post(
            f"{_BASE}/registry/skill", json=self._body(visibility="public")
        )
        assert resp.status_code == 201, resp.text
        item_arg = mock_svc.persist_registry_item.await_args.args[0]
        assert item_arg.graph_id == REGISTRY_CATALOG_GRAPH_ID

    async def test_curated_calls_admin_verify(
        self, client, mock_svc, monkeypatch
    ):
        called = {}

        async def _spy_admin(user_id: str) -> None:
            called["user_id"] = user_id

        monkeypatch.setattr(
            assessments_mod, "_verify_registry_curated_write", _spy_admin
        )
        resp = await client.post(
            f"{_BASE}/registry/skill", json=self._body(visibility="curated")
        )
        assert resp.status_code == 201, resp.text
        assert called == {"user_id": _USER}
        item_arg = mock_svc.persist_registry_item.await_args.args[0]
        assert item_arg.graph_id == REGISTRY_CATALOG_GRAPH_ID

    async def test_non_owner_public_update_returns_403(self, client, mock_svc):
        """Per TASK-073 Finding 1 (TASK-069): RegistryOwnershipError → 403.

        The service rejects non-owner attempts to overwrite a public/yanked
        item; the endpoint must translate the typed exception to HTTP 403
        rather than letting it bubble to a 500.
        """
        from app.services.assessment_service import RegistryOwnershipError

        mock_svc.persist_registry_item.side_effect = RegistryOwnershipError(
            "RegistryItem 'ri-alice' is owned by another user"
        )
        resp = await client.post(
            f"{_BASE}/registry/skill",
            json=self._body(visibility="yanked", item_id="ri-alice"),
        )
        # The global HTTPException handler normalizes 403 bodies to the
        # structured KGB-4003 shape (see app/main.py http_exception_handler);
        # we assert on the status code, which is the security-critical signal.
        assert resp.status_code == 403, resp.text
        # The MERGE must NOT have been allowed to run beyond the typed
        # exception — the endpoint code translates RegistryOwnershipError
        # before the service is asked to write.
        mock_svc.persist_registry_item.assert_awaited_once()


# ── Auth failures ────────────────────────────────────────────────────────────


class TestAuthFailures:
    """When the auth dependencies aren't overridden, real verify_token runs.

    We test 401 by NOT overriding the auth dependency and sending no token —
    FastAPI's HTTPBearer dependency will reject with 403 (its default) or 401.
    We test 403 by leaving auth overridden but pointing
    `verify_graph_access` at a denial.
    """

    async def test_missing_jwt_rejected(self):
        # No dependency overrides at all — HTTPBearer rejects unauthenticated.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(f"{_BASE}/runs/run-xyz:finalize")
        # HTTPBearer returns 403 by default for missing creds in FastAPI;
        # accept either 401 or 403 as the canonical "no auth" signal.
        assert resp.status_code in (401, 403)

    async def test_wrong_tenant_jwt_gets_403(self, mock_svc):
        from fastapi import HTTPException, status

        app.dependency_overrides[get_current_user] = lambda: _principal()
        app.dependency_overrides[get_current_user_id] = lambda: _USER
        app.dependency_overrides[_assessment_service] = lambda: mock_svc

        async def _deny(*_a, **_kw):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        original_verify = assessments_mod.verify_graph_access
        assessments_mod.verify_graph_access = _deny
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(f"{_BASE}/runs/run-xyz:finalize")
            assert resp.status_code == 403
        finally:
            assessments_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)


# ── Principal graph_id resolution ────────────────────────────────────────────


class TestPrincipalGraphIdResolution:
    """Cover `_principal_graph_id`: principal without `home_graph_id` → 400."""

    async def test_missing_home_graph_id_returns_400(self, mock_svc):
        app.dependency_overrides[get_current_user] = lambda: _principal(
            home_graph_id=""
        )
        app.dependency_overrides[get_current_user_id] = lambda: _USER
        app.dependency_overrides[_assessment_service] = lambda: mock_svc

        original_verify = assessments_mod.verify_graph_access
        assessments_mod.verify_graph_access = AsyncMock(return_value=_TENANT)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(f"{_BASE}/runs/run-xyz:finalize")
            assert resp.status_code == 400
            assert "home_graph_id" in resp.json()["detail"]
        finally:
            assessments_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)
