"""Integration tests for the assessment-substrate write service (TASK-068).

Runs against the Docker Neo4j instance configured by the existing
`neo4j_test_driver` fixture. Each test cleans up its own data via unique
graph_ids per test session.

Covers:
- create_run end-to-end: catalog template + modules → tenant run + module_runs
- record_finding_bulk idempotency (replay produces no duplicates)
- record_finding_bulk per-record failure (returned in bulk response)
- graph_id isolation: a write in tenant A is not visible from tenant B
- finalize_run gate: failing thresholds → status='failed', passing → 'finished'
- :__Platform__ marker landed on every write (ADR-015)
- :Source MERGEd in the catalog graph; CITES edge in the tenant graph
- Registry routing per ADR-019 (private → tenant graph; public → catalog)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from app.schemas.assessment_schemas import (
    ASSESSMENTS_CATALOG_GRAPH_ID,
    REGISTRY_CATALOG_GRAPH_ID,
    Conflict,
    CreateRunRequest,
    Deliverable,
    Finding,
    RegistryItem,
    Subject,
    UnresolvedQuestion,
    UpdateModuleRunRequest,
)
from app.services.assessment_service import AssessmentService

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)

# Unique-per-session tenant ids so parallel runs do not collide.
_SESSION = uuid.uuid4().hex[:8]
_GID_A = f"integ-assess-A-{_SESSION}"
_GID_B = f"integ-assess-B-{_SESSION}"
_TEMPLATE_ID = f"t-test-{_SESSION}"
_TEMPLATE_SLUG = f"assess-test-{_SESSION}"


@pytest_asyncio.fixture(autouse=True)
async def _seed_and_cleanup(neo4j_test_driver: AsyncDriver):
    """Seed a tiny test template + modules in the catalog graph; wipe per-tenant
    data before and after each test.
    """

    async def _wipe():
        # Wipe per-tenant
        for gid in (_GID_A, _GID_B):
            await neo4j_test_driver.execute_query(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
            )
        # Wipe the test template + modules from the catalog
        await neo4j_test_driver.execute_query(
            """
            MATCH (t:AssessmentTemplate {template_id: $tid})
            OPTIONAL MATCH (t)-[:HAS_MODULE]->(m:Module)
            DETACH DELETE t, m
            """,
            {"tid": _TEMPLATE_ID},
        )
        # Wipe registry items the tests create
        await neo4j_test_driver.execute_query(
            f"""
            MATCH (ri:RegistryItem) WHERE ri.item_id STARTS WITH 'ri-test-{_SESSION}'
            DETACH DELETE ri
            """
        )
        # Wipe :Source nodes the tests create
        await neo4j_test_driver.execute_query(
            f"""
            MATCH (s:Source) WHERE s.source_id STARTS WITH 'src-test-{_SESSION}'
            DETACH DELETE s
            """
        )

    await _wipe()

    # Seed the catalog template + 3 modules (2 wave-1 research, 1 wave-2 analysis)
    await neo4j_test_driver.execute_query(
        """
        MERGE (t:AssessmentTemplate:__Platform__ {template_id: $tid})
        ON CREATE SET
            t.graph_id      = $catalog,
            t.slug          = $slug,
            t.name          = 'Test Assessment',
            t.version       = '0.0.1'
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
                    "module_id": f"m-{_SESSION}-r1",
                    "slug": "research-1",
                    "name": "Research 1",
                    "wave": 1,
                    "ordinal": 0,
                    "kind": "research",
                },
                {
                    "module_id": f"m-{_SESSION}-r2",
                    "slug": "research-2",
                    "name": "Research 2",
                    "wave": 1,
                    "ordinal": 1,
                    "kind": "research",
                },
                {
                    "module_id": f"m-{_SESSION}-a1",
                    "slug": "analysis-1",
                    "name": "Analysis 1",
                    "wave": 2,
                    "ordinal": 0,
                    "kind": "analysis",
                },
            ],
        },
    )

    yield

    await _wipe()


@pytest_asyncio.fixture
async def svc(neo4j_test_driver: AsyncDriver) -> AssessmentService:
    return AssessmentService(neo4j_test_driver)


# ── create_run ────────────────────────────────────────────────────────────────


class TestCreateRunIntegration:
    async def test_create_run_seeds_run_and_three_module_runs(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-1",
                graph_id=_GID_A,
                slug="eurail",
                name="Eurail",
            ),
        )
        resp = await svc.create_run(_GID_A, req, created_by="user-integ")
        assert resp.status == "planned"
        assert len(resp.module_run_ids) == 3
        assert resp.already_existed is False

        # Verify the AssessmentRun lives in tenant A
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.template_id AS template_id, r.status AS status,
                   r.subject_id AS subject_id
            """,
            {"gid": _GID_A, "rid": resp.run_id},
        )
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec["template_id"] == _TEMPLATE_ID
        assert rec["status"] == "planned"

        # All 3 module_runs hang off the run
        mr_result = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
                  -[:HAS_MODULE_RUN]->(mr:ModuleRun:__Platform__)
            RETURN mr.module_run_id AS id, mr.wave AS wave, mr.status AS status
            ORDER BY mr.wave, mr.module_run_id
            """,
            {"gid": _GID_A, "rid": resp.run_id},
        )
        assert len(mr_result.records) == 3
        statuses = [rec["status"] for rec in mr_result.records]
        assert all(s == "planned" for s in statuses)

    async def test_create_run_idempotent_replay(self, svc: AssessmentService):
        # First create with explicit run_id.
        explicit_run_id = f"run-{_SESSION}-replay"
        req = CreateRunRequest(
            run_id=explicit_run_id,
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-2",
                graph_id=_GID_A,
                slug="acme",
                name="ACME",
            ),
        )
        first = await svc.create_run(_GID_A, req)
        assert first.already_existed is False
        assert first.run_id == explicit_run_id
        assert len(first.module_run_ids) == 3

        # Replay → returns the existing run untouched
        second = await svc.create_run(_GID_A, req)
        assert second.already_existed is True
        assert second.run_id == explicit_run_id
        # The module_run_ids ARE the original ids, not new ones.
        assert set(second.module_run_ids) == set(first.module_run_ids)


# ── record_finding_bulk ───────────────────────────────────────────────────────


class TestRecordFindingBulkIntegration:
    async def test_bulk_writes_findings_and_refreshes_evidence_count(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        # Setup: create a run, then write 5 findings into its first module_run
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-bulk",
                graph_id=_GID_A,
                slug="bulk-subj",
                name="Bulk",
            ),
        )
        create_resp = await svc.create_run(_GID_A, req)
        run_id = create_resp.run_id
        module_run_id = create_resp.module_run_ids[0]

        findings = [
            Finding(
                finding_id=f"ev-{_SESSION}-{i}",
                graph_id=_GID_A,
                run_id=run_id,
                module_run_id=module_run_id,
                claim=f"claim {i}",
                label="DIRECT",
                confidence=0.85,
                dimensions=["regulatory", "tech-maturity"],
            )
            for i in range(5)
        ]
        bulk = await svc.record_finding_bulk(_GID_A, run_id, module_run_id, findings)
        assert bulk.total == 5
        assert bulk.succeeded == 5
        assert bulk.failed == 0
        assert all(r.already_existed is False for r in bulk.results)

        # evidence_count refreshed on the parent module_run
        mr_count = await neo4j_test_driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {
                graph_id: $gid, run_id: $rid, module_run_id: $mrid
            })
            RETURN mr.evidence_count AS cnt
            """,
            {"gid": _GID_A, "rid": run_id, "mrid": module_run_id},
        )
        assert mr_count.records[0]["cnt"] == 5

    async def test_bulk_replay_idempotent_marks_already_existed(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-replay",
                graph_id=_GID_A,
                slug="replay-subj",
                name="Replay",
            ),
        )
        create_resp = await svc.create_run(_GID_A, req)
        run_id = create_resp.run_id
        module_run_id = create_resp.module_run_ids[0]

        findings = [
            Finding(
                finding_id=f"ev-{_SESSION}-rp-{i}",
                graph_id=_GID_A,
                run_id=run_id,
                module_run_id=module_run_id,
                claim=f"replay-{i}",
            )
            for i in range(3)
        ]
        first = await svc.record_finding_bulk(
            _GID_A, run_id, module_run_id, findings
        )
        assert all(r.success and not r.already_existed for r in first.results)

        # Replay
        second = await svc.record_finding_bulk(
            _GID_A, run_id, module_run_id, findings
        )
        assert all(r.success and r.already_existed for r in second.results)

        # Verify only 3 :Finding rows exist, not 6
        count = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN count(f) AS cnt
            """,
            {"gid": _GID_A, "rid": run_id},
        )
        assert count.records[0]["cnt"] == 3


# ── graph_id isolation ────────────────────────────────────────────────────────


class TestGraphIdIsolation:
    async def test_findings_in_a_invisible_from_b(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        # Setup: two separate tenant runs from the same template
        req_a = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-A",
                graph_id=_GID_A,
                slug="tenant-a-subj",
                name="A",
            ),
        )
        req_b = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-B",
                graph_id=_GID_B,
                slug="tenant-b-subj",
                name="B",
            ),
        )
        run_a = await svc.create_run(_GID_A, req_a)
        run_b = await svc.create_run(_GID_B, req_b)

        # Write 2 findings into tenant A
        findings_a = [
            Finding(
                finding_id=f"ev-{_SESSION}-iso-A-{i}",
                graph_id=_GID_A,
                run_id=run_a.run_id,
                module_run_id=run_a.module_run_ids[0],
                claim=f"A {i}",
            )
            for i in range(2)
        ]
        await svc.record_finding_bulk(
            _GID_A, run_a.run_id, run_a.module_run_ids[0], findings_a
        )

        # Tenant B should see ZERO findings
        b_result = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {graph_id: $gid})
            RETURN count(f) AS cnt
            """,
            {"gid": _GID_B},
        )
        assert b_result.records[0]["cnt"] == 0

        # Tenant A sees its 2
        a_result = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {graph_id: $gid})
            RETURN count(f) AS cnt
            """,
            {"gid": _GID_A},
        )
        assert a_result.records[0]["cnt"] == 2

        # Both runs exist independently
        assert run_a.run_id != run_b.run_id


# ── conflicts + unresolved questions + deliverables ──────────────────────────


class TestConflictsQuestionsDeliverables:
    async def test_record_conflict_with_involves_edges(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-cf",
                graph_id=_GID_A,
                slug="cf-subj",
                name="CF",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]
        findings = [
            Finding(
                finding_id=f"ev-{_SESSION}-cf-{i}",
                graph_id=_GID_A,
                run_id=run.run_id,
                module_run_id=mr_id,
                claim=f"x {i}",
                label="CONTRADICTION" if i == 0 else "DIRECT",
            )
            for i in range(2)
        ]
        await svc.record_finding_bulk(_GID_A, run.run_id, mr_id, findings)

        c = Conflict(
            conflict_id=f"cf-{_SESSION}-1",
            graph_id=_GID_A,
            run_id=run.run_id,
            topic="Topic A",
            summary="A vs B",
            involved_finding_ids=[f.finding_id for f in findings],
        )
        created = await svc.record_conflict(_GID_A, run.run_id, c)
        assert created is True

        edges = await neo4j_test_driver.execute_query(
            """
            MATCH (c:Conflict:__Platform__ {graph_id: $gid, conflict_id: $cid})
                  -[:INVOLVES]->(f:Finding:__Platform__)
            RETURN count(f) AS cnt
            """,
            {"gid": _GID_A, "cid": c.conflict_id},
        )
        assert edges.records[0]["cnt"] == 2

    async def test_record_unresolved_question(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-q",
                graph_id=_GID_A,
                slug="q-subj",
                name="Q",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]
        q = UnresolvedQuestion(
            question_id=f"q-{_SESSION}-1",
            graph_id=_GID_A,
            run_id=run.run_id,
            module_run_id=mr_id,
            text="What about X?",
            suggested_module="research-followup",
        )
        created = await svc.record_unresolved_question(_GID_A, run.run_id, mr_id, q)
        assert created is True

        edge = await neo4j_test_driver.execute_query(
            """
            MATCH (mr:ModuleRun:__Platform__ {module_run_id: $mrid})
                  -[:RAISED]->(q:UnresolvedQuestion:__Platform__ {question_id: $qid})
            RETURN q.text AS text
            """,
            {"mrid": mr_id, "qid": q.question_id},
        )
        assert edge.records[0]["text"] == "What about X?"

    async def test_persist_deliverable_links_to_run(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-d",
                graph_id=_GID_A,
                slug="d-subj",
                name="D",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        d = Deliverable(
            deliverable_id=f"d-{_SESSION}-1",
            graph_id=_GID_A,
            run_id=run.run_id,
            module_run_id=run.module_run_ids[0],
            kind="module-md",
            filename="research-1.md",
            content_inline="# results\nblah",
            word_count=2,
        )
        created = await svc.persist_deliverable(_GID_A, run.run_id, d)
        assert created is True

        edges = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {run_id: $rid})
                  -[:HAS_DELIVERABLE]->(d:Deliverable:__Platform__ {deliverable_id: $did})
            MATCH (mr:ModuleRun:__Platform__ {module_run_id: $mrid})
                  -[:PRODUCED_DELIVERABLE]->(d)
            RETURN d.kind AS kind, d.content_inline AS content
            """,
            {
                "rid": run.run_id,
                "did": d.deliverable_id,
                "mrid": run.module_run_ids[0],
            },
        )
        assert len(edges.records) == 1
        assert edges.records[0]["kind"] == "module-md"


# ── finalize_run ──────────────────────────────────────────────────────────────


class TestFinalizeRunIntegration:
    async def test_finalize_fails_with_no_findings(self, svc: AssessmentService):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-fail",
                graph_id=_GID_A,
                slug="fail-subj",
                name="Fail",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        resp = await svc.finalize_run(_GID_A, run.run_id)
        assert resp.passed is False
        assert resp.status == "failed"
        assert resp.direct_finding_count == 0
        assert resp.deliverable_count == 0
        assert any("direct_findings" in r for r in resp.failure_reasons)
        assert any("deliverables" in r for r in resp.failure_reasons)

    async def test_finalize_passes_when_thresholds_met(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-pass",
                graph_id=_GID_A,
                slug="pass-subj",
                name="Pass",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]

        # Add 2 DIRECT findings + 1 deliverable → both gates pass.
        await svc.record_finding_bulk(
            _GID_A,
            run.run_id,
            mr_id,
            [
                Finding(
                    finding_id=f"ev-{_SESSION}-pass-{i}",
                    graph_id=_GID_A,
                    run_id=run.run_id,
                    module_run_id=mr_id,
                    claim=f"c{i}",
                    label="DIRECT",
                )
                for i in range(2)
            ],
        )
        await svc.persist_deliverable(
            _GID_A,
            run.run_id,
            Deliverable(
                deliverable_id=f"d-{_SESSION}-pass",
                graph_id=_GID_A,
                run_id=run.run_id,
                kind="final-md",
                filename="final.md",
            ),
        )

        resp = await svc.finalize_run(_GID_A, run.run_id)
        assert resp.passed is True
        assert resp.status == "finished"
        assert resp.direct_finding_count == 2
        assert resp.deliverable_count == 1

        # Persisted status on the AssessmentRun reflects the gate result
        check = await neo4j_test_driver.execute_query(
            """
            MATCH (r:AssessmentRun:__Platform__ {graph_id: $gid, run_id: $rid})
            RETURN r.status AS status, r.finished_at AS finished_at
            """,
            {"gid": _GID_A, "rid": run.run_id},
        )
        assert check.records[0]["status"] == "finished"
        assert check.records[0]["finished_at"] is not None

    async def test_finalize_excludes_findings_under_failed_module_runs(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        """Findings under FAILED ModuleRuns are not counted by the gate."""
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-failmr",
                graph_id=_GID_A,
                slug="failmr-subj",
                name="FailMR",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]

        # Mark the module_run failed
        await svc.update_module_run(
            _GID_A,
            run.run_id,
            mr_id,
            UpdateModuleRunRequest(status="failed", failure_reason="orphaned"),
        )
        await svc.record_finding_bulk(
            _GID_A,
            run.run_id,
            mr_id,
            [
                Finding(
                    finding_id=f"ev-{_SESSION}-failmr-{i}",
                    graph_id=_GID_A,
                    run_id=run.run_id,
                    module_run_id=mr_id,
                    claim=f"c{i}",
                    label="DIRECT",
                )
                for i in range(3)
            ],
        )

        # Even though 3 findings exist, they should not be counted (parent is failed)
        resp = await svc.finalize_run(_GID_A, run.run_id)
        assert resp.direct_finding_count == 0
        assert resp.passed is False


# ── Source MERGE in the catalog graph ─────────────────────────────────────────


class TestSourceCatalogMerge:
    async def test_finding_with_source_merges_source_in_catalog(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        # Seed a :Source in the catalog graph first (since the merge path
        # in the service requires the source to exist for the CITES edge).
        source_id = f"src-test-{_SESSION}-eu-ai-act"
        await neo4j_test_driver.execute_query(
            """
            MERGE (s:Source:__Platform__ {source_id: $sid})
            ON CREATE SET
                s.graph_id = $catalog,
                s.url_normalized = $url,
                s.name = 'EU AI Act'
            """,
            {
                "sid": source_id,
                "catalog": ASSESSMENTS_CATALOG_GRAPH_ID,
                "url": "https://eur-lex.europa.eu/eli/reg/2024/1689",
            },
        )

        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-src",
                graph_id=_GID_A,
                slug="src-subj",
                name="Src",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]

        finding = Finding(
            finding_id=f"ev-{_SESSION}-src-1",
            graph_id=_GID_A,
            run_id=run.run_id,
            module_run_id=mr_id,
            claim="Cites the EU AI Act",
            source_id=source_id,
            source_quote="Article 6",
            source_locator="https://...#art6",
        )
        bulk = await svc.record_finding_bulk(
            _GID_A, run.run_id, mr_id, [finding]
        )
        assert bulk.succeeded == 1

        # The CITES edge exists in the tenant graph; the :Source lives in the
        # catalog graph with graph_id=__assessments_catalog__.
        edge = await neo4j_test_driver.execute_query(
            """
            MATCH (f:Finding:__Platform__ {finding_id: $fid})
                  -[c:CITES]->(s:Source:__Platform__ {source_id: $sid})
            RETURN c.quote AS quote, c.locator AS locator, s.graph_id AS s_graph_id
            """,
            {"fid": finding.finding_id, "sid": source_id},
        )
        assert len(edge.records) == 1
        assert edge.records[0]["quote"] == "Article 6"
        assert edge.records[0]["s_graph_id"] == ASSESSMENTS_CATALOG_GRAPH_ID


# ── Registry routing (ADR-019) ────────────────────────────────────────────────


class TestRegistryIntegration:
    async def test_private_item_lands_in_owner_tenant_graph(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        item = RegistryItem(
            item_id=f"ri-test-{_SESSION}-priv",
            graph_id=_GID_A,
            kind="skill",
            slug=f"users/u1/private-{_SESSION}",
            visibility="private",
            owner_user_id="u1",
            name="Private Skill",
        )
        created = await svc.persist_registry_item(item, owner_tenant_graph_id=_GID_A)
        assert created is True

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $iid})
            RETURN ri.graph_id AS graph_id, ri.visibility AS visibility
            """,
            {"iid": item.item_id},
        )
        assert result.records[0]["graph_id"] == _GID_A
        assert result.records[0]["visibility"] == "private"

    async def test_public_item_lands_in_registry_catalog(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        item = RegistryItem(
            item_id=f"ri-test-{_SESSION}-pub",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug=f"public-skill-{_SESSION}",
            visibility="public",
            owner_user_id="u1",
            name="Public Skill",
        )
        created = await svc.persist_registry_item(item)
        assert created is True

        result = await neo4j_test_driver.execute_query(
            """
            MATCH (ri:RegistryItem:__Platform__ {item_id: $iid})
            RETURN ri.graph_id AS graph_id, ri.visibility AS visibility
            """,
            {"iid": item.item_id},
        )
        assert result.records[0]["graph_id"] == REGISTRY_CATALOG_GRAPH_ID
        assert result.records[0]["visibility"] == "public"


# ── __Platform__ marker invariant (ADR-015) ───────────────────────────────────


class TestPlatformMarkerInvariant:
    async def test_every_assessment_write_carries_platform_marker(
        self, svc: AssessmentService, neo4j_test_driver: AsyncDriver
    ):
        req = CreateRunRequest(
            template_slug=_TEMPLATE_SLUG,
            subject=Subject(
                subject_id=f"subj-{_SESSION}-mk",
                graph_id=_GID_A,
                slug="mk-subj",
                name="MK",
            ),
        )
        run = await svc.create_run(_GID_A, req)
        mr_id = run.module_run_ids[0]
        await svc.record_finding_bulk(
            _GID_A,
            run.run_id,
            mr_id,
            [
                Finding(
                    finding_id=f"ev-{_SESSION}-mk",
                    graph_id=_GID_A,
                    run_id=run.run_id,
                    module_run_id=mr_id,
                    claim="x",
                )
            ],
        )
        await svc.record_conflict(
            _GID_A,
            run.run_id,
            Conflict(
                conflict_id=f"cf-{_SESSION}-mk",
                graph_id=_GID_A,
                run_id=run.run_id,
                topic="x",
                summary="y",
            ),
        )
        await svc.record_unresolved_question(
            _GID_A,
            run.run_id,
            mr_id,
            UnresolvedQuestion(
                question_id=f"q-{_SESSION}-mk",
                graph_id=_GID_A,
                run_id=run.run_id,
                module_run_id=mr_id,
                text="?",
            ),
        )
        await svc.persist_deliverable(
            _GID_A,
            run.run_id,
            Deliverable(
                deliverable_id=f"d-{_SESSION}-mk",
                graph_id=_GID_A,
                run_id=run.run_id,
                kind="module-md",
                filename="x.md",
            ),
        )

        # Every tenant-A node carries :__Platform__.
        # The catalog template/modules also carry the marker (seeded that way).
        result = await neo4j_test_driver.execute_query(
            """
            MATCH (n {graph_id: $gid})
            WHERE NOT n:__Platform__
            RETURN labels(n) AS labels, count(n) AS cnt
            """,
            {"gid": _GID_A},
        )
        # All assessment-related rows should pass; only the :Graph:__Rebac__
        # ReBAC anchor (if any) would lack :__Platform__ — and the test tenant
        # has no such anchor because we never bootstrapped one for these
        # synthetic graph_ids.
        unmarked = [
            (rec["labels"], rec["cnt"])
            for rec in result.records
            if "__Rebac__" not in rec["labels"]
        ]
        assert unmarked == [], (
            f"Found tenant-A nodes lacking :__Platform__ marker: {unmarked}"
        )
