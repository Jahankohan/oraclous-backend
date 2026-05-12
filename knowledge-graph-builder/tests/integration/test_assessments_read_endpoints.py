"""Integration tests for the assessment-substrate REST read endpoints (TASK-079).

Runs against the Docker Neo4j instance (`neo4j_test_driver` fixture). Builds
a small seeded fixture: template + 3 modules in the catalog graph; one run
with module-runs, findings (with sources), conflicts, deliverables, and
unresolved questions in tenant A.

Covers:
- Happy-path reads for every endpoint.
- 404 propagation for unknown run / template / deliverable.
- Cursor pagination round-trip — issue a small page, follow `next_cursor`,
  drain to end.
- Cross-tenant isolation: tenant B's JWT against a tenant-A run returns 404
  (not 200, not 403 — we do not leak existence).
- Registry visibility per ADR-019: tenant B cannot list / read tenant A's
  private items even with an `owner=alice` query.
- Admin findings:search returns rows when admin ACL passes; denied when not.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
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
from app.api.v1.endpoints import assessments_reads as reads_mod
from app.api.v1.endpoints._pagination import decode_cursor
from app.api.v1.endpoints.assessments_reads import _assessment_service
from app.main import app
from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    REGISTRY_CATALOG_GRAPH_ID,
)
from app.services.assessment_service import AssessmentService

_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"integ-read-A-{_SESSION}"
_GID_B = f"integ-read-B-{_SESSION}"
_USER_A = f"user-A-{_SESSION}"
_USER_B = f"user-B-{_SESSION}"
_TEMPLATE_ID = f"t-read-{_SESSION}"
_TEMPLATE_SLUG = f"assess-read-{_SESSION}"
_RUN_ID = f"run-read-{_SESSION}"
_SUBJECT_ID = f"subj-read-{_SESSION}"
_MR_W1_A = f"mr-w1a-{_SESSION}"
_MR_W1_B = f"mr-w1b-{_SESSION}"
_MR_W2 = f"mr-w2-{_SESSION}"
_M_W1_A = f"m-w1a-{_SESSION}"
_M_W1_B = f"m-w1b-{_SESSION}"
_M_W2 = f"m-w2-{_SESSION}"
_FINDINGS = [f"f-{_SESSION}-{i}" for i in range(5)]
_CONFLICT_ID = f"cf-{_SESSION}-1"
_QUESTION_ID = f"q-{_SESSION}-1"
_DELIVERABLE_ID = f"d-{_SESSION}-1"
_SOURCE_ID = f"src-{_SESSION}-1"
_REG_PRIVATE_A = f"ri-priv-A-{_SESSION}"
_REG_PRIVATE_B = f"ri-priv-B-{_SESSION}"
_REG_PUBLIC = f"ri-pub-{_SESSION}"
_REG_CURATED = f"ri-cur-{_SESSION}"

_BASE = "/api/v1/api/v1/assessments"


# ── Fixtures ──────────────────────────────────────────────────────────────────


async def _wipe(driver: AsyncDriver):
    for gid in (_GID_A, _GID_B):
        await driver.execute_query(
            "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
        )
    await driver.execute_query(
        """
        MATCH (t:AssessmentTemplate {template_id: $tid})
        OPTIONAL MATCH (t)-[:HAS_MODULE]->(m:Module)
        DETACH DELETE t, m
        """,
        {"tid": _TEMPLATE_ID},
    )
    await driver.execute_query(
        f"MATCH (s:Source) WHERE s.source_id STARTS WITH 'src-{_SESSION}' DETACH DELETE s"
    )
    await driver.execute_query(
        f"MATCH (ri:RegistryItem) WHERE ri.item_id STARTS WITH 'ri-' AND ri.item_id ENDS WITH '-{_SESSION}' DETACH DELETE ri"
    )


@pytest_asyncio.fixture(autouse=True)
async def _seed_and_cleanup(neo4j_test_driver: AsyncDriver):
    await _wipe(neo4j_test_driver)

    # 1. Catalog: template + 3 modules
    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET
            t.graph_id = $catalog,
            t.slug     = $slug,
            t.name     = 'Read Test Template',
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
                    "module_id": _M_W1_A,
                    "slug": f"research-1-{_SESSION}",
                    "name": "R1A",
                    "wave": 1,
                    "ordinal": 0,
                    "kind": "research",
                },
                {
                    "module_id": _M_W1_B,
                    "slug": f"research-2-{_SESSION}",
                    "name": "R1B",
                    "wave": 1,
                    "ordinal": 1,
                    "kind": "research",
                },
                {
                    "module_id": _M_W2,
                    "slug": f"analysis-1-{_SESSION}",
                    "name": "A1",
                    "wave": 2,
                    "ordinal": 0,
                    "kind": "analysis",
                },
            ],
        },
    )

    # 2. Tenant A: subject, run, 3 module-runs (1 wave-1 done, 1 wave-1
    # failed, 1 wave-2 planned), 5 findings (4 attached to mr-w1a, 1 to
    # mr-w1b which is failed), 1 conflict, 1 question, 1 deliverable.
    started = datetime(2026, 5, 1, tzinfo=UTC).isoformat()
    await neo4j_test_driver.execute_query(
        """
        MERGE (subj:Subject:__Platform__ {graph_id: $gid, slug: 'eurail'})
        ON CREATE SET
            subj.subject_id = $subj_id,
            subj.name = 'Eurail'
        MERGE (r:AssessmentRun:__Platform__ {run_id: $run_id})
        ON CREATE SET
            r.graph_id = $gid,
            r.template_id = $tid,
            r.subject_id = $subj_id,
            r.status = 'running',
            r.started_at = datetime($started),
            r.orchestrator_last_seen = datetime($started),
            r.cli_flags = '{"flag":"value"}'
        MERGE (mr1a:ModuleRun:__Platform__ {module_run_id: $mr_w1_a})
        ON CREATE SET
            mr1a.graph_id = $gid,
            mr1a.run_id = $run_id,
            mr1a.module_id = $m_w1_a,
            mr1a.wave = 1,
            mr1a.status = 'finished',
            mr1a.evidence_count = 4
        MERGE (r)-[:HAS_MODULE_RUN]->(mr1a)
        MERGE (mr1b:ModuleRun:__Platform__ {module_run_id: $mr_w1_b})
        ON CREATE SET
            mr1b.graph_id = $gid,
            mr1b.run_id = $run_id,
            mr1b.module_id = $m_w1_b,
            mr1b.wave = 1,
            mr1b.status = 'failed',
            mr1b.evidence_count = 1,
            mr1b.failure_reason = 'simulated'
        MERGE (r)-[:HAS_MODULE_RUN]->(mr1b)
        MERGE (mr2:ModuleRun:__Platform__ {module_run_id: $mr_w2})
        ON CREATE SET
            mr2.graph_id = $gid,
            mr2.run_id = $run_id,
            mr2.module_id = $m_w2,
            mr2.wave = 2,
            mr2.status = 'planned',
            mr2.evidence_count = 0
        MERGE (r)-[:HAS_MODULE_RUN]->(mr2)
        """,
        {
            "gid": _GID_A,
            "run_id": _RUN_ID,
            "tid": _TEMPLATE_ID,
            "subj_id": _SUBJECT_ID,
            "mr_w1_a": _MR_W1_A,
            "mr_w1_b": _MR_W1_B,
            "mr_w2": _MR_W2,
            "m_w1_a": _M_W1_A,
            "m_w1_b": _M_W1_B,
            "m_w2": _M_W2,
            "started": started,
        },
    )

    # 3. Source in catalog
    await neo4j_test_driver.execute_query(
        """
        MERGE (s:Source:__Platform__ {source_id: $sid})
        ON CREATE SET
            s.graph_id = $catalog,
            s.type = 'article',
            s.url_normalized = $url,
            s.name = 'Example Source'
        """,
        {
            "sid": _SOURCE_ID,
            "catalog": ASSESSMENTS_CATALOG_GRAPH_ID,
            "url": f"https://example.com/{_SESSION}",
        },
    )

    # 4. Findings under mr-w1a (good) and mr-w1b (failed)
    findings_payload = []
    for i, fid in enumerate(_FINDINGS[:4]):
        findings_payload.append(
            {
                "fid": fid,
                "mr": _MR_W1_A,
                "claim": f"Claim {i}",
                "label": "DIRECT",
                "confidence": 0.7 + i * 0.05,
                "dim": ["climate"] if i % 2 == 0 else ["tech"],
                "source_id": _SOURCE_ID if i == 0 else None,
            }
        )
    findings_payload.append(
        {
            "fid": _FINDINGS[4],
            "mr": _MR_W1_B,
            "claim": "Failed finding",
            "label": "INFERRED",
            "confidence": 0.4,
            "dim": [],
            "source_id": None,
        }
    )
    await neo4j_test_driver.execute_query(
        """
        UNWIND $findings AS row
        MERGE (f:Finding:__Platform__ {finding_id: row.fid})
        ON CREATE SET
            f.graph_id = $gid,
            f.run_id = $run_id,
            f.module_run_id = row.mr,
            f.claim = row.claim,
            f.label = row.label,
            f.confidence = row.confidence,
            f.dimensions = row.dim,
            f.source_id = row.source_id
        WITH f, row
        MATCH (mr:ModuleRun:__Platform__ {graph_id: $gid, module_run_id: row.mr})
        MERGE (mr)-[:PRODUCED]->(f)
        WITH f, row
        WHERE row.source_id IS NOT NULL
        MATCH (s:Source:__Platform__ {source_id: row.source_id})
        MERGE (f)-[:CITES {quote: 'q', locator: 'p1'}]->(s)
        """,
        {
            "gid": _GID_A,
            "run_id": _RUN_ID,
            "findings": findings_payload,
        },
    )

    # 5. Conflict + question + deliverable
    await neo4j_test_driver.execute_query(
        """
        MERGE (c:Conflict:__Platform__ {conflict_id: $cid})
        ON CREATE SET
            c.graph_id = $gid,
            c.run_id = $run_id,
            c.topic = 'Topic',
            c.summary = 'Summary',
            c.status = 'open'
        WITH c
        MATCH (f:Finding:__Platform__ {graph_id: $gid, finding_id: $f0})
        MERGE (c)-[:INVOLVES]->(f)
        WITH c
        MATCH (f:Finding:__Platform__ {graph_id: $gid, finding_id: $f1})
        MERGE (c)-[:INVOLVES]->(f)
        """,
        {
            "gid": _GID_A,
            "run_id": _RUN_ID,
            "cid": _CONFLICT_ID,
            "f0": _FINDINGS[0],
            "f1": _FINDINGS[1],
        },
    )
    await neo4j_test_driver.execute_query(
        """
        MERGE (q:UnresolvedQuestion:__Platform__ {question_id: $qid})
        ON CREATE SET
            q.graph_id = $gid,
            q.run_id = $run_id,
            q.module_run_id = $mr,
            q.text = 'Why?',
            q.status = 'open'
        WITH q
        MATCH (mr:ModuleRun:__Platform__ {graph_id: $gid, module_run_id: $mr})
        MERGE (mr)-[:RAISED]->(q)
        """,
        {
            "gid": _GID_A,
            "run_id": _RUN_ID,
            "qid": _QUESTION_ID,
            "mr": _MR_W1_A,
        },
    )
    await neo4j_test_driver.execute_query(
        """
        MERGE (d:Deliverable:__Platform__ {deliverable_id: $did})
        ON CREATE SET
            d.graph_id = $gid,
            d.run_id = $run_id,
            d.module_run_id = $mr,
            d.kind = 'module-md',
            d.filename = 'm1.md',
            d.ordinal = 0,
            d.content_inline = '# Module 1 markdown body'
        WITH d
        MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $run_id})
        MERGE (r)-[:HAS_DELIVERABLE]->(d)
        """,
        {
            "gid": _GID_A,
            "run_id": _RUN_ID,
            "did": _DELIVERABLE_ID,
            "mr": _MR_W1_A,
        },
    )

    # 6. Registry items: tenant-A-private, tenant-B-private, public, curated
    await neo4j_test_driver.execute_query(
        """
        MERGE (a:RegistryItem:__Platform__ {item_id: $ria})
        ON CREATE SET
            a.graph_id = $gid_a,
            a.kind = 'skill',
            a.slug = 'a-private',
            a.version = '0.1.0',
            a.visibility = 'private',
            a.owner_user_id = $user_a,
            a.name = 'A Private'
        MERGE (b:RegistryItem:__Platform__ {item_id: $rib})
        ON CREATE SET
            b.graph_id = $gid_b,
            b.kind = 'skill',
            b.slug = 'b-private',
            b.version = '0.1.0',
            b.visibility = 'private',
            b.owner_user_id = $user_b,
            b.name = 'B Private'
        MERGE (p:RegistryItem:__Platform__ {item_id: $rip})
        ON CREATE SET
            p.graph_id = $catalog,
            p.kind = 'skill',
            p.slug = 'public-skill',
            p.version = '0.1.0',
            p.visibility = 'public',
            p.owner_user_id = $user_a,
            p.name = 'Public Skill',
            p.content_inline = '# Public skill body'
        MERGE (c:RegistryItem:__Platform__ {item_id: $ric})
        ON CREATE SET
            c.graph_id = $catalog,
            c.kind = 'skill',
            c.slug = 'curated-skill',
            c.version = '0.1.0',
            c.visibility = 'curated',
            c.owner_user_id = 'platform',
            c.name = 'Curated Skill'
        """,
        {
            "ria": _REG_PRIVATE_A,
            "rib": _REG_PRIVATE_B,
            "rip": _REG_PUBLIC,
            "ric": _REG_CURATED,
            "gid_a": _GID_A,
            "gid_b": _GID_B,
            "user_a": _USER_A,
            "user_b": _USER_B,
            "catalog": REGISTRY_CATALOG_GRAPH_ID,
        },
    )

    yield

    await _wipe(neo4j_test_driver)


def _principal(home_graph_id: str = _GID_A, user_id: str = _USER_A) -> dict:
    return {
        "id": user_id,
        "principal_type": "service_account",
        "home_graph_id": home_graph_id,
    }


@pytest_asyncio.fixture
async def client_a(neo4j_test_driver: AsyncDriver) -> AsyncGenerator:
    """An async client with tenant A's principal."""
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_current_user] = lambda: _principal(_GID_A, _USER_A)
    app.dependency_overrides[get_current_user_id] = lambda: _USER_A
    app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
        neo4j_test_driver
    )

    original_verify = reads_mod.verify_graph_access
    reads_mod.verify_graph_access = AsyncMock(return_value=_GID_A)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    reads_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


@pytest_asyncio.fixture
async def client_b(neo4j_test_driver: AsyncDriver) -> AsyncGenerator:
    """An async client with tenant B's principal — used for cross-tenant tests."""
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_current_user] = lambda: _principal(_GID_B, _USER_B)
    app.dependency_overrides[get_current_user_id] = lambda: _USER_B
    app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
        neo4j_test_driver
    )

    original_verify = reads_mod.verify_graph_access
    reads_mod.verify_graph_access = AsyncMock(return_value=_GID_B)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    reads_mod.verify_graph_access = original_verify
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)
    app.dependency_overrides.pop(_assessment_service, None)


# ── Happy-path reads ─────────────────────────────────────────────────────────


class TestRunListAndDetail:
    async def test_list_runs_returns_seeded_run(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert any(r["run_id"] == _RUN_ID for r in body["items"])

    async def test_filter_by_status(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs?status=running")
        body = resp.json()
        assert any(r["run_id"] == _RUN_ID for r in body["items"])

    async def test_filter_by_subject_slug(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs?subject=eurail")
        body = resp.json()
        assert any(r["run_id"] == _RUN_ID for r in body["items"])
        assert body["items"][0]["subject_slug"] == "eurail"

    async def test_get_run_detail_includes_aggregates(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == _RUN_ID
        assert body["module_run_total"] == 3
        assert body["module_run_done"] == 1
        assert body["module_run_failed"] == 1
        # 4 findings on the non-failed mr; the failed mr's finding is excluded
        assert body["finding_count"] == 4
        assert body["conflict_count"] == 1
        assert body["open_question_count"] == 1
        assert body["deliverable_count"] == 1

    async def test_get_run_returns_404_for_unknown_run(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/run-unknown-{_SESSION}")
        assert resp.status_code == 404


class TestWaveStatus:
    async def test_wave_1_counts_correctly(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/waves/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["wave"] == 1
        assert body["total"] == 2
        assert body["done"] == 1
        assert body["failed"] == 1
        # Catalog hydration: module slugs filled in
        slugs = {m["module_slug"] for m in body["modules"]}
        assert any(s and s.startswith("research-") for s in slugs)

    async def test_unknown_run_returns_404(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/missing/waves/1")
        assert resp.status_code == 404


class TestModuleRuns:
    async def test_list_returns_all_three(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/module-runs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 3
        # Module slug hydrated from catalog
        assert all(item["module_slug"] is not None for item in body["items"])

    async def test_filter_by_status(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/module-runs?status=failed")
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["module_run_id"] == _MR_W1_B


class TestFindings:
    async def test_list_returns_all_with_source_hydrated(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/findings?limit=20")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 5
        # The one with source_id should have its catalog row hydrated
        cited = [f for f in body["items"] if f["source_id"] == _SOURCE_ID]
        assert len(cited) == 1
        assert cited[0]["source"] is not None
        assert cited[0]["source"]["url_normalized"] == f"https://example.com/{_SESSION}"
        assert cited[0]["source"]["type"] == "article"
        # source_quote/source_locator come from the [:CITES] edge
        assert cited[0]["source_quote"] == "q"

    async def test_filter_by_label(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/findings?label=INFERRED")
        body = resp.json()
        assert all(f["label"] == "INFERRED" for f in body["items"])

    async def test_filter_by_min_confidence(self, client_a):
        resp = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/findings?min_confidence=0.75"
        )
        body = resp.json()
        assert all(f["confidence"] >= 0.75 for f in body["items"])

    async def test_filter_by_dimension(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/findings?dimension=climate")
        body = resp.json()
        assert all("climate" in f["dimensions"] for f in body["items"])

    async def test_filter_by_source_type(self, client_a):
        # source_type post-filter — only 1 finding has a source attached
        resp = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/findings?source_type=article"
        )
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["source"]["type"] == "article"


class TestConflicts:
    async def test_returns_with_involved_finding_ids(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/conflicts")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["conflict_id"] == _CONFLICT_ID
        assert set(item["involved_finding_ids"]) == {_FINDINGS[0], _FINDINGS[1]}


class TestUnresolvedQuestions:
    async def test_returns_open_question(self, client_a):
        resp = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/unresolved-questions?status=open"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["question_id"] == _QUESTION_ID


class TestDeliverables:
    async def test_list_returns_seeded(self, client_a):
        resp = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/deliverables")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["has_inline_content"] is True

    async def test_get_content_returns_inline_markdown(self, client_a):
        resp = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/deliverables/{_DELIVERABLE_ID}/content"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "Module 1 markdown body" in resp.text


class TestTemplateModules:
    async def test_returns_template_with_modules(self, client_a):
        resp = await client_a.get(f"{_BASE}/templates/{_TEMPLATE_SLUG}/modules")
        assert resp.status_code == 200
        body = resp.json()
        assert body["template_id"] == _TEMPLATE_ID
        # All 3 modules; sorted by wave then ordinal
        assert len(body["modules"]) == 3
        assert body["modules"][0]["wave"] == 1


# ── Cursor pagination round-trip ─────────────────────────────────────────────


class TestCursorRoundTrip:
    async def test_findings_paginate_with_cursor(self, client_a):
        # Page size 2 over 5 findings → 3 pages
        page_1 = await client_a.get(f"{_BASE}/runs/{_RUN_ID}/findings?limit=2")
        body_1 = page_1.json()
        assert len(body_1["items"]) == 2
        assert body_1["page"]["next_cursor"] is not None

        cursor_1 = body_1["page"]["next_cursor"]
        offset_1, _ = decode_cursor(cursor_1)
        assert offset_1 == 2

        page_2 = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/findings?limit=2&cursor={cursor_1}"
        )
        body_2 = page_2.json()
        assert len(body_2["items"]) == 2
        # No overlap between pages
        ids_1 = {f["finding_id"] for f in body_1["items"]}
        ids_2 = {f["finding_id"] for f in body_2["items"]}
        assert ids_1.isdisjoint(ids_2)

        cursor_2 = body_2["page"]["next_cursor"]
        page_3 = await client_a.get(
            f"{_BASE}/runs/{_RUN_ID}/findings?limit=2&cursor={cursor_2}"
        )
        body_3 = page_3.json()
        # Last page: 1 row, no next cursor
        assert len(body_3["items"]) == 1
        assert body_3["page"]["next_cursor"] is None


# ── Cross-tenant isolation ───────────────────────────────────────────────────


class TestCrossTenantIsolation:
    async def test_tenant_b_cannot_see_tenant_a_runs(self, client_b):
        # Tenant B is in graph_id _GID_B; the seeded run lives in _GID_A.
        # The list query is graph_id-scoped server-side, so B sees no
        # tenant-A rows.
        resp = await client_b.get(f"{_BASE}/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert not any(r["run_id"] == _RUN_ID for r in body["items"])

    async def test_tenant_b_gets_404_for_tenant_a_run_detail(self, client_b):
        resp = await client_b.get(f"{_BASE}/runs/{_RUN_ID}")
        assert resp.status_code == 404

    async def test_tenant_b_gets_404_for_tenant_a_findings(self, client_b):
        resp = await client_b.get(f"{_BASE}/runs/{_RUN_ID}/findings")
        assert resp.status_code == 404

    async def test_tenant_b_gets_404_for_tenant_a_deliverable_content(self, client_b):
        resp = await client_b.get(
            f"{_BASE}/runs/{_RUN_ID}/deliverables/{_DELIVERABLE_ID}/content"
        )
        assert resp.status_code == 404


# ── Registry visibility (ADR-019) ────────────────────────────────────────────


class TestRegistryVisibility:
    async def test_tenant_a_sees_own_private_plus_public_and_curated(self, client_a):
        resp = await client_a.get(f"{_BASE}/registry/skill")
        assert resp.status_code == 200
        body = resp.json()
        ids = {item["item_id"] for item in body["items"]}
        # A sees A's private, the public, and the curated; NOT B's private
        assert _REG_PRIVATE_A in ids
        assert _REG_PUBLIC in ids
        assert _REG_CURATED in ids
        assert _REG_PRIVATE_B not in ids

    async def test_tenant_b_cannot_see_tenant_a_private(self, client_b):
        resp = await client_b.get(f"{_BASE}/registry/skill")
        body = resp.json()
        ids = {item["item_id"] for item in body["items"]}
        assert _REG_PRIVATE_A not in ids
        # B sees public, curated, and B's own private
        assert _REG_PUBLIC in ids
        assert _REG_CURATED in ids
        assert _REG_PRIVATE_B in ids

    async def test_owner_query_does_not_leak_other_users_private(self, client_b):
        # B explicitly asks for items owned by user_A — must NOT return A's private
        resp = await client_b.get(
            f"{_BASE}/registry/skill?owner={_USER_A}&visibility=private"
        )
        body = resp.json()
        ids = {item["item_id"] for item in body["items"]}
        assert _REG_PRIVATE_A not in ids

    async def test_public_visibility_explicit_filter(self, client_b):
        resp = await client_b.get(f"{_BASE}/registry/skill?visibility=public")
        body = resp.json()
        ids = {item["item_id"] for item in body["items"]}
        assert _REG_PUBLIC in ids
        # B's private is NOT included when explicitly filtering for public
        assert _REG_PRIVATE_B not in ids

    async def test_get_public_skill_returns_content(self, client_a):
        resp = await client_a.get(f"{_BASE}/registry/skill/public-skill/0.1.0/content")
        assert resp.status_code == 200
        assert "Public skill body" in resp.text

    async def test_tenant_b_cannot_get_tenant_a_private_content(self, client_b):
        # B asks for A's private slug — 404 (not 403; we collapse to avoid leak)
        resp = await client_b.get(f"{_BASE}/registry/skill/a-private/0.1.0/content")
        assert resp.status_code == 404


# ── Admin: cross-run findings:search ─────────────────────────────────────────


class TestAdminFindingsSearch:
    async def test_admin_can_see_cross_tenant_rows(
        self, neo4j_test_driver: AsyncDriver
    ):
        """Admin ACL on `__assessments_catalog__` allows cross-tenant search."""
        from httpx import ASGITransport, AsyncClient

        app.dependency_overrides[get_current_user] = lambda: _principal(_GID_A, _USER_A)
        app.dependency_overrides[get_current_user_id] = lambda: _USER_A
        app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
            neo4j_test_driver
        )
        original_verify = reads_mod.verify_graph_access
        reads_mod.verify_graph_access = AsyncMock(return_value=_GID_A)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(
                    f"{_BASE}/findings:search?source_url=https://example.com/{_SESSION}"
                )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            # Should find at least the one seeded finding cited from this source
            assert any(item["source_id"] == _SOURCE_ID for item in body["items"])
        finally:
            reads_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)

    async def test_non_admin_rejected_with_403(self, neo4j_test_driver: AsyncDriver):
        """`verify_graph_access('__assessments_catalog__', 'admin', user)` denial → 403."""
        from fastapi import HTTPException
        from fastapi import status as http_status
        from httpx import ASGITransport, AsyncClient

        app.dependency_overrides[get_current_user] = lambda: _principal(_GID_A, _USER_A)
        app.dependency_overrides[get_current_user_id] = lambda: _USER_A
        app.dependency_overrides[_assessment_service] = lambda: AssessmentService(
            neo4j_test_driver
        )

        called: dict = {}

        async def _selective_deny(graph_id: str, level: str, user_id: str):
            called["args"] = (graph_id, level, user_id)
            if graph_id == ASSESSMENTS_CATALOG_GRAPH_ID and level == "admin":
                raise HTTPException(
                    status_code=http_status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )
            return graph_id

        original_verify = reads_mod.verify_graph_access
        reads_mod.verify_graph_access = _selective_deny
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(
                    f"{_BASE}/findings:search?source_url=https://example.com/{_SESSION}"
                )
            assert resp.status_code == 403
            assert called["args"] == (
                ASSESSMENTS_CATALOG_GRAPH_ID,
                "admin",
                _USER_A,
            )
        finally:
            reads_mod.verify_graph_access = original_verify
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_current_user_id, None)
            app.dependency_overrides.pop(_assessment_service, None)
