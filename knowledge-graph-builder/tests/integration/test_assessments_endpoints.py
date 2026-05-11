"""Integration tests for the assessment-substrate REST endpoints (TASK-069).

Runs against the Docker Neo4j instance (the same `neo4j_test_driver` fixture
the rest of the integration suite uses). The auth pipeline is overridden so
each test can pin the JWT-derived principal directly.

Covers:
- Happy-path POST/PATCH for every endpoint, asserting Neo4j state.
- 207 Multi-Status on `findings:bulk` and `deliverables:bulk-final`.
- Idempotency: replaying `create_run` returns the same run; replaying
  `findings:bulk` produces no duplicate :Finding rows.
- Registry routing per ADR-019: private → tenant graph; public → __registry__;
  curated requires admin on __registry__ (rejected with 403 otherwise).
- Heartbeat endpoints touch the right field without altering status.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock

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

from app.api.dependencies import get_current_user, get_current_user_id
from app.api.v1.endpoints import assessments as assessments_mod
from app.api.v1.endpoints.assessments import _assessment_service
from app.main import app
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    REGISTRY_CATALOG_GRAPH_ID,
)
from app.services.assessment_service import AssessmentService

_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"integ-assess-rest-A-{_SESSION}"
_GID_OTHER = f"integ-assess-rest-B-{_SESSION}"
_USER = f"integ-rest-user-{_SESSION}"
_TEMPLATE_ID = f"t-rest-{_SESSION}"
_TEMPLATE_SLUG = f"assess-rest-{_SESSION}"

_BASE = "/api/v1/api/v1/assessments"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _seed_catalog_and_cleanup(neo4j_test_driver: AsyncDriver):
    """Seed a small test template + 3 modules; wipe tenant + catalog rows between tests."""

    async def _wipe():
        for gid in (_GID_A, _GID_OTHER):
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
        await neo4j_test_driver.execute_query(
            f"""
            MATCH (ri:RegistryItem) WHERE ri.item_id STARTS WITH 'ri-rest-{_SESSION}'
            DETACH DELETE ri
            """
        )
        await neo4j_test_driver.execute_query(
            f"""
            MATCH (s:Source) WHERE s.source_id STARTS WITH 'src-rest-{_SESSION}'
            DETACH DELETE s
            """
        )

    await _wipe()
    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET
            t.graph_id = $catalog,
            t.slug     = $slug,
            t.name     = 'REST Test Assessment',
            t.version  = '0.0.1'
        WITH t
        UNWIND $modules AS m
        MERGE (mod:Module:__Platform__ {module_id: m.module_id})
        ON CREATE SET
            mod.graph_id    = $catalog,
            mod.template_id = $tid,
            mod.slug        = m.slug,
            mod.name        = m.name,
            mod.wave        = m.wave,
            mod.ordinal     = m.ordinal,
            mod.kind        = m.kind
        MERGE (t)-[:HAS_MODULE]->(mod)
        """,
        {
            "tid": _TEMPLATE_ID,
            "slug": _TEMPLATE_SLUG,
            "catalog": ASSESSMENTS_CATALOG_GRAPH_ID,
            "modules": [
                {
                    "module_id": f"m-rest-{_SESSION}-r1",
                    "slug": "research-1",
                    "name": "R1",
                    "wave": 1,
                    "ordinal": 0,
                    "kind": "research",
                },
                {
                    "module_id": f"m-rest-{_SESSION}-r2",
                    "slug": "research-2",
                    "name": "R2",
                    "wave": 1,
                    "ordinal": 1,
                    "kind": "research",
                },
                {
                    "module_id": f"m-rest-{_SESSION}-a1",
                    "slug": "analysis-1",
                    "name": "A1",
                    "wave": 2,
                    "ordinal": 0,
                    "kind": "analysis",
                },
            ],
        },
    )

    yield

    await _wipe()


def _principal(home_graph_id: str = _GID_A) -> dict:
    return {
        "id": _USER,
        "principal_type": "service_account",
        "home_graph_id": home_graph_id,
    }


@pytest_asyncio.fixture
async def client(neo4j_test_driver: AsyncDriver) -> AsyncGenerator:
    """An async client with the assessment service wired to the test Neo4j driver."""
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_current_user] = lambda: _principal()
    app.dependency_overrides[get_current_user_id] = lambda: _USER
    app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
        neo4j_test_driver
    )

    original_verify = assessments_mod.verify_graph_access
    assessments_mod.verify_graph_access = AsyncMock(return_value=_GID_A)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    assessments_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _create_run(client, subject_slug: str = "eurail") -> dict:
    body = {
        "template_slug": _TEMPLATE_SLUG,
        "subject": {
            "subject_id": f"subj-rest-{_SESSION}-{subject_slug}",
            "graph_id": "ignored",  # Scope-Enforcer: body graph_id is ignored
            "slug": subject_slug,
            "name": subject_slug.title(),
        },
        "cli_flags": {"flag": "value"},
    }
    resp = await client.post(f"{_BASE}/runs", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── Happy paths ──────────────────────────────────────────────────────────────


class TestCreateRunEndpointIntegration:
    async def test_creates_run_and_module_runs(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client)
        assert data["status"] == "planned"
        assert len(data["module_run_ids"]) == 3
        assert data["already_existed"] is False

        # Run lives in _GID_A despite the body sending graph_id='ignored'
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.template_id AS template_id, r.status AS status
            """,
            {"gid": _GID_A, "rid": data["run_id"]},
        )
        assert len(result.records) == 1
        assert result.records[0]["template_id"] == _TEMPLATE_ID

    async def test_create_run_idempotent_replay_when_run_id_supplied(
        self, client
    ):
        run_id = f"run-rest-{_SESSION}-replay"
        body = {
            "run_id": run_id,
            "template_slug": _TEMPLATE_SLUG,
            "subject": {
                "subject_id": f"subj-rest-{_SESSION}-replay",
                "graph_id": _GID_A,
                "slug": "replay",
                "name": "Replay",
            },
        }
        first = await client.post(f"{_BASE}/runs", json=body)
        assert first.status_code == 201
        assert first.json()["already_existed"] is False

        second = await client.post(f"{_BASE}/runs", json=body)
        assert second.status_code == 201
        body2 = second.json()
        assert body2["already_existed"] is True
        assert body2["run_id"] == run_id


class TestUpdateModuleRunEndpointIntegration:
    async def test_status_transition_persists(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="update-mr")
        mr_id = data["module_run_ids"][0]
        resp = await client.patch(
            f"{_BASE}/runs/{data['run_id']}/module-runs/{mr_id}",
            json={"status": "running", "evidence_count": 7},
        )
        assert resp.status_code == 200, resp.text

        # Verify in Neo4j
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {graph_id: $gid, module_run_id: $mrid})
            RETURN mr.status AS status, mr.evidence_count AS cnt
            """,
            {"gid": _GID_A, "mrid": mr_id},
        )
        assert result.records[0]["status"] == "running"
        assert result.records[0]["cnt"] == 7

    async def test_missing_module_run_returns_404(self, client):
        data = await _create_run(client, subject_slug="missing-mr")
        resp = await client.patch(
            f"{_BASE}/runs/{data['run_id']}/module-runs/mr-does-not-exist",
            json={"status": "running"},
        )
        assert resp.status_code == 404


class TestRecordFindingBulkIntegration:
    async def test_bulk_207_with_per_record_results_and_idempotency(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="bulk-findings")
        run_id = data["run_id"]
        mr_id = data["module_run_ids"][0]

        findings = [
            {
                "finding_id": f"ev-rest-{_SESSION}-{i}",
                "graph_id": "ignored",
                "run_id": "ignored",  # both overwritten server-side
                "module_run_id": mr_id,
                "claim": f"claim {i}",
                "label": "DIRECT",
                "confidence": 0.8,
            }
            for i in range(3)
        ]
        resp = await client.post(
            f"{_BASE}/runs/{run_id}/findings:bulk", json={"findings": findings}
        )
        assert resp.status_code == 207, resp.text
        body = resp.json()
        assert body["total"] == 3
        assert body["succeeded"] == 3
        assert all(r["already_existed"] is False for r in body["results"])

        # Replay returns already_existed=True for each finding (idempotency)
        replay = await client.post(
            f"{_BASE}/runs/{run_id}/findings:bulk", json={"findings": findings}
        )
        assert replay.status_code == 207
        body2 = replay.json()
        assert body2["succeeded"] == 3
        assert all(r["already_existed"] is True for r in body2["results"])

        # Cypher count confirms no duplicate :Finding rows
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN count(f) AS cnt
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert result.records[0]["cnt"] == 3


class TestRecordConflictIntegration:
    async def test_persists_conflict(self, client, neo4j_test_driver: AsyncDriver):
        data = await _create_run(client, subject_slug="conflict-test")
        run_id = data["run_id"]
        body = {
            "conflict": {
                "conflict_id": f"cf-rest-{_SESSION}-1",
                "graph_id": "ignored",
                "run_id": run_id,
                "topic": "Battery costs",
                "summary": "Sources disagree",
                "status": "open",
                "involved_finding_ids": [],
            }
        }
        resp = await client.post(f"{_BASE}/runs/{run_id}/conflicts", json=body)
        assert resp.status_code == 201, resp.text
        assert resp.json()["created"] is True

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (c:Conflict:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN c.conflict_id AS id, c.status AS status, c.topic AS topic
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert len(result.records) == 1
        assert result.records[0]["status"] == "open"


class TestRecordUnresolvedQuestionIntegration:
    async def test_persists_question(self, client, neo4j_test_driver: AsyncDriver):
        data = await _create_run(client, subject_slug="q-test")
        run_id = data["run_id"]
        mr_id = data["module_run_ids"][0]
        body = {
            "question": {
                "question_id": f"q-rest-{_SESSION}-1",
                "graph_id": "ignored",
                "run_id": run_id,
                "module_run_id": mr_id,
                "text": "Why is X true?",
                "status": "open",
            }
        }
        resp = await client.post(
            f"{_BASE}/runs/{run_id}/unresolved-questions", json=body
        )
        assert resp.status_code == 201, resp.text

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (q:UnresolvedQuestion:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN q.question_id AS id, q.status AS status
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert len(result.records) == 1


class TestPersistDeliverableIntegration:
    async def test_persists_deliverable(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="deliv-test")
        run_id = data["run_id"]
        mr_id = data["module_run_ids"][0]
        body = {
            "deliverable": {
                "deliverable_id": f"d-rest-{_SESSION}-1",
                "graph_id": "ignored",
                "run_id": run_id,
                "module_run_id": mr_id,
                "kind": "module-md",
                "filename": "m1.md",
                "ordinal": 0,
                "content_uri": "file:///tmp/m1.md",
            }
        }
        resp = await client.post(
            f"{_BASE}/runs/{run_id}/deliverables", json=body
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["created"] is True

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (d:Deliverable:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN d.kind AS kind, d.filename AS filename
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert len(result.records) == 1
        assert result.records[0]["kind"] == "module-md"


class TestPersistFinalDocsIntegration:
    async def test_bulk_207_persists_each(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="final-docs")
        run_id = data["run_id"]
        body = {
            "deliverables": [
                {
                    "deliverable_id": f"d-rest-{_SESSION}-final-{i}",
                    "graph_id": "ignored",
                    "run_id": run_id,
                    "kind": "final-md",
                    "filename": f"0{i}-doc.md",
                    "ordinal": i,
                }
                for i in range(3)
            ]
        }
        resp = await client.post(
            f"{_BASE}/runs/{run_id}/deliverables:bulk-final", json=body
        )
        assert resp.status_code == 207, resp.text
        body_resp = resp.json()
        assert body_resp["total"] == 3
        assert body_resp["succeeded"] == 3

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (d:Deliverable:__Platform__ {graph_id: $gid, run_id: $rid, kind: 'final-md'})
            RETURN count(d) AS cnt
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert result.records[0]["cnt"] == 3


class TestFinalizeRunIntegration:
    async def test_finalize_passes_when_thresholds_met(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="finalize")
        run_id = data["run_id"]
        mr_id = data["module_run_ids"][0]
        # Add one direct finding + one deliverable
        await client.post(
            f"{_BASE}/runs/{run_id}/findings:bulk",
            json={
                "findings": [
                    {
                        "finding_id": f"ev-rest-{_SESSION}-fin",
                        "graph_id": "x",
                        "run_id": "x",
                        "module_run_id": mr_id,
                        "claim": "ok",
                        "label": "DIRECT",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        await client.post(
            f"{_BASE}/runs/{run_id}/deliverables",
            json={
                "deliverable": {
                    "deliverable_id": f"d-rest-{_SESSION}-fin",
                    "graph_id": "x",
                    "run_id": run_id,
                    "kind": "final-md",
                    "filename": "fin.md",
                    "ordinal": 0,
                }
            },
        )
        resp = await client.post(f"{_BASE}/runs/{run_id}:finalize")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["passed"] is True
        assert body["status"] == "finished"

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.status AS status
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert result.records[0]["status"] == "finished"

    async def test_finalize_fails_when_no_findings_or_deliverables(self, client):
        data = await _create_run(client, subject_slug="finalize-fail")
        run_id = data["run_id"]
        resp = await client.post(f"{_BASE}/runs/{run_id}:finalize")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["passed"] is False
        assert body["status"] == "failed"
        assert len(body["failure_reasons"]) >= 1


# ── Heartbeats ────────────────────────────────────────────────────────────────


class TestHeartbeatIntegration:
    async def test_run_heartbeat_updates_orchestrator_last_seen(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="hb-run")
        run_id = data["run_id"]

        # Get the original orchestrator_last_seen
        before = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.orchestrator_last_seen AS ts, r.status AS status
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        before_ts = before.records[0]["ts"]
        before_status = before.records[0]["status"]

        # Heartbeat
        resp = await client.post(f"{_BASE}/runs/{run_id}:heartbeat")
        assert resp.status_code == 200, resp.text

        after = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.orchestrator_last_seen AS ts, r.status AS status
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        # Heartbeat MUST NOT touch status; orchestrator_last_seen should be set.
        assert after.records[0]["ts"] is not None
        # The orchestrator_last_seen is updated; comparing equality of
        # neo4j.DateTime values via .to_native() guards against tz quirks.
        assert after.records[0]["status"] == before_status
        # And it should be >= the original (we tolerate same-millisecond
        # writes during fast test execution).
        assert after.records[0]["ts"] >= before_ts

    async def test_run_heartbeat_404_for_unknown_run(self, client):
        resp = await client.post(f"{_BASE}/runs/run-does-not-exist:heartbeat")
        assert resp.status_code == 404

    async def test_module_run_heartbeat_updates_last_heartbeat_at_only(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        data = await _create_run(client, subject_slug="hb-mr")
        run_id = data["run_id"]
        mr_id = data["module_run_ids"][0]
        resp = await client.post(
            f"{_BASE}/runs/{run_id}/module-runs/{mr_id}:heartbeat"
        )
        assert resp.status_code == 200, resp.text

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {graph_id: $gid, module_run_id: $mrid})
            RETURN mr.last_heartbeat_at AS ts, mr.status AS status
            """,
            {"gid": _GID_A, "mrid": mr_id},
        )
        assert result.records[0]["ts"] is not None
        # Status remains 'planned' — heartbeat must not flip it.
        assert result.records[0]["status"] == "planned"


# ── Registry (ADR-019) ───────────────────────────────────────────────────────


class TestPersistRegistryItemIntegration:
    async def test_private_writes_to_tenant_graph(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        body = {
            "item_id": f"ri-rest-{_SESSION}-priv",
            "graph_id": "ignored",
            "kind": "skill",
            "slug": "my-private-skill",
            "version": "0.1.0",
            "visibility": "private",
            "owner_user_id": "spoofed",  # ignored
            "name": "My Private Skill",
        }
        resp = await client.post(f"{_BASE}/registry/skill", json=body)
        assert resp.status_code == 201, resp.text
        assert resp.json()["graph_id"] == _GID_A
        assert resp.json()["created"] is True

        # Verify it landed in the tenant graph
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $id})
            RETURN ri.graph_id AS gid, ri.visibility AS vis,
                   ri.owner_user_id AS owner
            """,
            {"id": body["item_id"]},
        )
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec["gid"] == _GID_A
        assert rec["vis"] == "private"
        assert rec["owner"] == _USER  # pinned from JWT, not body

    async def test_public_writes_to_registry_catalog(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        body = {
            "item_id": f"ri-rest-{_SESSION}-pub",
            "graph_id": "ignored",
            "kind": "skill",
            "slug": "my-public-skill",
            "version": "0.1.0",
            "visibility": "public",
            "owner_user_id": "spoofed",
            "name": "My Public Skill",
        }
        resp = await client.post(f"{_BASE}/registry/skill", json=body)
        assert resp.status_code == 201, resp.text
        assert resp.json()["graph_id"] == REGISTRY_CATALOG_GRAPH_ID

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $id})
            RETURN ri.graph_id AS gid, ri.visibility AS vis
            """,
            {"id": body["item_id"]},
        )
        assert result.records[0]["gid"] == REGISTRY_CATALOG_GRAPH_ID
        assert result.records[0]["vis"] == "public"

    async def test_curated_rejected_for_non_admin(self, client):
        # verify_graph_access is currently mocked to allow tenant writes.
        # Pin it to deny the __registry__-admin check so the curated path
        # falls into 403.
        from fastapi import HTTPException, status

        called: dict = {"args": None}

        async def _selective_deny(graph_id: str, level: str, user_id: str):
            called["args"] = (graph_id, level, user_id)
            if graph_id == REGISTRY_CATALOG_GRAPH_ID and level == "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
                )
            return graph_id

        original = assessments_mod.verify_graph_access
        assessments_mod.verify_graph_access = _selective_deny
        try:
            body = {
                "item_id": f"ri-rest-{_SESSION}-cur",
                "graph_id": "ignored",
                "kind": "skill",
                "slug": "should-not-land",
                "version": "0.1.0",
                "visibility": "curated",
                "owner_user_id": "x",
                "name": "Curated",
            }
            resp = await client.post(f"{_BASE}/registry/skill", json=body)
            assert resp.status_code == 403
            # The endpoint actually called the admin check
            assert called["args"] == (
                REGISTRY_CATALOG_GRAPH_ID,
                "admin",
                _USER,
            )
        finally:
            assessments_mod.verify_graph_access = original

    async def test_curated_succeeds_for_admin(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        # All verify_graph_access calls already return _GID_A (allow),
        # so the curated path is permitted.
        body = {
            "item_id": f"ri-rest-{_SESSION}-cur-ok",
            "graph_id": "ignored",
            "kind": "skill",
            "slug": "curated-ok",
            "version": "1.0.0",
            "visibility": "curated",
            "owner_user_id": "x",
            "name": "Curated OK",
        }
        resp = await client.post(f"{_BASE}/registry/skill", json=body)
        assert resp.status_code == 201, resp.text
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $id})
            RETURN ri.graph_id AS gid, ri.visibility AS vis
            """,
            {"id": body["item_id"]},
        )
        assert result.records[0]["gid"] == REGISTRY_CATALOG_GRAPH_ID
        assert result.records[0]["vis"] == "curated"

    async def test_non_owner_cannot_hijack_public_item_via_endpoint(
        self, client, neo4j_test_driver: AsyncDriver
    ):
        """Per TASK-073 Finding 1 (TASK-069): Alice publishes a public item;
        Bob (different user, same __registry__ write permission) attempts to
        yank/rewrite it via POST /assessments/registry/skill. The endpoint
        must return 403 and Alice's row must be untouched.
        """
        from httpx import ASGITransport, AsyncClient

        item_id = f"ri-rest-{_SESSION}-hijack"

        # Step 1: Alice publishes a public item via her own client session.
        # (The default `client` fixture's principal IS Alice — _USER.)
        alice_body = {
            "item_id": item_id,
            "graph_id": "ignored",
            "kind": "skill",
            "slug": "alice-eurail",
            "version": "1.0.0",
            "visibility": "public",
            "owner_user_id": "spoofed",  # endpoint pins to _USER (Alice)
            "name": "Eurail Report",
            "description": "Alice's flagship skill",
        }
        resp = await client.post(f"{_BASE}/registry/skill", json=alice_body)
        assert resp.status_code == 201, resp.text

        # Step 2: Bob (different user_id) attempts to overwrite. We swap in a
        # second client wired to "bob"'s principal but keep the same Neo4j
        # backend so the MERGE collision is real.
        bob_user = f"bob-{_SESSION}"

        original_user_dep = app.dependency_overrides.get(get_current_user)
        original_id_dep = app.dependency_overrides.get(get_current_user_id)
        app.dependency_overrides[get_current_user] = lambda: {
            "id": bob_user,
            "principal_type": "service_account",
            "home_graph_id": _GID_A,
        }
        app.dependency_overrides[get_current_user_id] = lambda: bob_user

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as bob_client:
                bob_body = {
                    "item_id": item_id,  # collision with Alice's item
                    "graph_id": "ignored",
                    "kind": "skill",
                    "slug": "alice-eurail",
                    "version": "1.0.0",
                    "visibility": "yanked",
                    "owner_user_id": "ignored",
                    "name": "Eurail Report (deprecated, do not use)",
                    "description": "Bob's hijack",
                }
                hijack_resp = await bob_client.post(
                    f"{_BASE}/registry/skill", json=bob_body
                )
                assert hijack_resp.status_code == 403, hijack_resp.text
        finally:
            if original_user_dep is not None:
                app.dependency_overrides[get_current_user] = original_user_dep
            if original_id_dep is not None:
                app.dependency_overrides[get_current_user_id] = original_id_dep

        # Step 3: Alice's row must be untouched.
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $id})
            RETURN ri.name AS name, ri.visibility AS vis,
                   ri.owner_user_id AS owner, ri.description AS description
            """,
            {"id": item_id},
        )
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec["name"] == "Eurail Report"
        assert rec["vis"] == "public"
        assert rec["owner"] == _USER  # still Alice
        assert rec["description"] == "Alice's flagship skill"
