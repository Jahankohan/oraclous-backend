"""Unit tests for the assessment-substrate write service (STORY-026, TASK-068).

Mocks the async Neo4j driver and asserts the service:

- Always passes `graph_id` in every Cypher params dict (multi-tenant rule)
- Builds parameterized queries (no f-string interpolation of user input)
- Carries the `:__Platform__` marker on every platform-managed write (ADR-015)
- Returns idempotency-aware responses (already_existed flags)
- Falls back gracefully on bulk per-record failures
- Routes RegistryItem writes per ADR-019 (private → tenant graph;
  curated/public → __registry__)

Integration tests against a real Neo4j live in
`tests/integration/test_assessment_service.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

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

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_driver(record_sequences: list[list[dict]] | None = None):
    """Build a mock AsyncDriver whose execute_query returns the next pre-recorded
    list of records on each call.
    """
    driver = AsyncMock()
    record_sequences = record_sequences or [[]]
    results: list[MagicMock] = []
    for recs in record_sequences:
        result = MagicMock()
        result.records = recs
        results.append(result)
    driver.execute_query = AsyncMock(side_effect=results)
    return driver


def _rec(**kwargs) -> dict:
    """A dict-style record that supports key access like Neo4j's Record."""
    return kwargs


# ── Pydantic schema validation ────────────────────────────────────────────────


class TestSchemas:
    def test_finding_confidence_bounds(self):
        with pytest.raises(Exception):
            Finding(
                finding_id="ev-1",
                graph_id="g1",
                run_id="r1",
                module_run_id="mr1",
                claim="x",
                confidence=1.5,  # > 1.0
            )

    def test_finding_default_label_is_direct(self):
        f = Finding(
            finding_id="ev-1",
            graph_id="g1",
            run_id="r1",
            module_run_id="mr1",
            claim="x",
        )
        assert f.label == "DIRECT"
        assert f.confidence == 0.0

    def test_module_wave_must_be_positive(self):
        # Module is imported lazily here to avoid clutter in the import block.
        from app.schemas.assessment_schemas import Module

        with pytest.raises(Exception):
            Module(
                module_id="m-1",
                template_id="t-1",
                slug="x",
                name="x",
                wave=0,
                ordinal=0,
                kind="research",
            )

    def test_registry_item_yank_consistency(self):
        # yanked_at on a private item is rejected by the validator.
        with pytest.raises(Exception):
            RegistryItem(
                item_id="ri-1",
                graph_id="g-owner-1",
                kind="skill",
                slug="users/u1/x",
                visibility="private",
                owner_user_id="u1",
                name="x",
                yanked_at=datetime.now(UTC),
            )


# ── AssessmentService — graph_id enforcement ──────────────────────────────────


class TestAssessmentServiceGraphIdEnforcement:
    async def test_create_run_rejects_empty_graph_id(self):
        svc = AssessmentService(_make_driver())
        req = CreateRunRequest(
            template_slug="t",
            subject=Subject(subject_id="s1", graph_id="g1", slug="s", name="S"),
        )
        with pytest.raises(ValueError, match="graph_id is required"):
            await svc.create_run("", req)

    async def test_update_module_run_rejects_missing_ids(self):
        svc = AssessmentService(_make_driver())
        with pytest.raises(ValueError):
            await svc.update_module_run("g1", "", "mr1", UpdateModuleRunRequest())

    async def test_finalize_run_rejects_missing_ids(self):
        svc = AssessmentService(_make_driver())
        with pytest.raises(ValueError):
            await svc.finalize_run("g1", "")


# ── create_run ────────────────────────────────────────────────────────────────


class TestCreateRun:
    async def test_create_run_persists_template_and_module_runs(self):
        template_recs = [_rec(template_id="t-001", slug="assess-v1")]
        module_recs = [
            _rec(module_id="m-1", slug="m1", wave=1, ordinal=0, kind="research"),
            _rec(module_id="m-2", slug="m2", wave=1, ordinal=1, kind="research"),
            _rec(module_id="m-3", slug="m3", wave=2, ordinal=0, kind="analysis"),
        ]
        # call order:
        # 1) _fetch_template_by_slug
        # 2) _fetch_modules_for_template
        # 3) _fetch_existing_run -> empty
        # 4) _merge_subject_tenant
        # 5) AssessmentRun MERGE  (returns a record per TASK-073 Finding 1 fix:
        #     the post-MERGE WHERE filter must see the row to confirm same-tenant)
        # 6,7,8) ModuleRun MERGE × 3
        driver = _make_driver(
            record_sequences=[
                template_recs,  # template lookup
                module_recs,  # modules
                [],  # no existing run
                [_rec(subject_id="subj-1")],  # subject merge
                [_rec(id="run-id-stub")],  # run MERGE: returns the new run's id
                [],
                [],
                [],  # 3 module-run MERGEs
            ]
        )
        svc = AssessmentService(driver)
        req = CreateRunRequest(
            template_slug="assess-v1",
            subject=Subject(
                subject_id="subj-suggested",
                graph_id="tenant-A",
                slug="eurail",
                name="Eurail",
            ),
        )
        resp = await svc.create_run("tenant-A", req, created_by="user-1")
        assert resp.template_id == "t-001"
        assert resp.subject_id == "subj-1"
        assert len(resp.module_run_ids) == 3
        assert resp.status == "planned"
        assert resp.already_existed is False

        # Every Cypher call carried params with graph_id when expected.
        # Calls 4+5+6-8 (subject + run + 3 module-runs) must include graph_id.
        for call in driver.execute_query.call_args_list[3:]:
            params = call[0][1]
            assert params.get("graph_id") == "tenant-A"

    async def test_create_run_is_idempotent_on_replay(self):
        # _fetch_template_by_slug → 1
        # _fetch_modules_for_template → 1
        # _fetch_existing_run → returns the existing run
        # _fetch_module_run_ids → returns 2 ids
        driver = _make_driver(
            record_sequences=[
                [_rec(template_id="t-1", slug="assess-v1")],
                [_rec(module_id="m-1", slug="m1", wave=1, ordinal=0, kind="research")],
                [_rec(template_id="t-1", subject_id="subj-1", status="running")],
                [_rec(module_run_id="mr-a"), _rec(module_run_id="mr-b")],
            ]
        )
        svc = AssessmentService(driver)
        req = CreateRunRequest(
            run_id="run-explicit",
            template_slug="assess-v1",
            subject=Subject(
                subject_id="subj-1", graph_id="tenant-A", slug="eurail", name="Eurail"
            ),
        )
        resp = await svc.create_run("tenant-A", req)
        assert resp.run_id == "run-explicit"
        assert resp.already_existed is True
        assert resp.status == "running"
        assert resp.module_run_ids == ["mr-a", "mr-b"]
        # No MERGE on AssessmentRun (only template/module fetches + idempotency probe + module-run id fetch)
        assert driver.execute_query.call_count == 4

    async def test_create_run_raises_when_template_missing(self):
        driver = _make_driver(record_sequences=[[]])  # template fetch returns nothing
        svc = AssessmentService(driver)
        with pytest.raises(ValueError, match="template not found"):
            await svc.create_run(
                "tenant-A",
                CreateRunRequest(
                    template_slug="nonexistent",
                    subject=Subject(
                        subject_id="s1", graph_id="tenant-A", slug="s", name="S"
                    ),
                ),
            )

    async def test_create_run_raises_when_template_has_no_modules(self):
        driver = _make_driver(
            record_sequences=[
                [_rec(template_id="t-1", slug="assess-v1")],
                [],  # no modules
            ]
        )
        svc = AssessmentService(driver)
        with pytest.raises(ValueError, match="has no :Module rows"):
            await svc.create_run(
                "tenant-A",
                CreateRunRequest(
                    template_slug="assess-v1",
                    subject=Subject(
                        subject_id="s1", graph_id="tenant-A", slug="s", name="S"
                    ),
                ),
            )


# ── update_module_run ─────────────────────────────────────────────────────────


class TestUpdateModuleRun:
    async def test_update_with_no_fields_only_verifies_existence(self):
        driver = _make_driver(record_sequences=[[_rec(id="mr-1")]])
        svc = AssessmentService(driver)
        ok = await svc.update_module_run(
            "tenant-A", "run-1", "mr-1", UpdateModuleRunRequest()
        )
        assert ok is True
        params = driver.execute_query.call_args[0][1]
        assert params == {
            "graph_id": "tenant-A",
            "run_id": "run-1",
            "module_run_id": "mr-1",
        }

    async def test_update_status_to_running(self):
        driver = _make_driver(record_sequences=[[_rec(id="mr-1")]])
        svc = AssessmentService(driver)
        now = datetime.now(UTC)
        ok = await svc.update_module_run(
            "tenant-A",
            "run-1",
            "mr-1",
            UpdateModuleRunRequest(status="running", started_at=now),
        )
        assert ok is True
        cypher, params = driver.execute_query.call_args[0]
        assert "mr.status = $status" in cypher
        assert "mr.started_at = datetime($started_at)" in cypher
        assert params["status"] == "running"
        assert params["started_at"] == now.isoformat()
        assert params["graph_id"] == "tenant-A"

    async def test_heartbeat_run_updates_orchestrator_last_seen(self):
        driver = _make_driver(record_sequences=[[_rec(id="run-1")]])
        svc = AssessmentService(driver)
        ok = await svc.heartbeat_run("tenant-A", "run-1")
        assert ok is True
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == "tenant-A"
        assert params["run_id"] == "run-1"
        assert "now" in params


# ── record_finding_bulk ───────────────────────────────────────────────────────


class TestRecordFindingBulk:
    async def test_bulk_per_record_succeeds_and_fails_independently(self):
        # Order of execute_query calls:
        # 1. Parent ModuleRun verification → 1 record
        # 2..4. Each finding MERGE returns created=True (3 findings)
        # 5. evidence_count refresh
        driver = _make_driver(
            record_sequences=[
                [_rec(id="mr-1")],
                [_rec(created=True)],
                [_rec(created=True)],
                [_rec(created=False)],  # already_existed = True for idx 2
                [],  # refresh
            ]
        )
        svc = AssessmentService(driver)
        findings = [
            Finding(
                finding_id=f"ev-{i}",
                graph_id="tenant-A",
                run_id="run-1",
                module_run_id="mr-1",
                claim=f"claim-{i}",
                label="DIRECT",
                confidence=0.9,
            )
            for i in range(3)
        ]
        resp = await svc.record_finding_bulk("tenant-A", "run-1", "mr-1", findings)
        assert resp.total == 3
        assert resp.succeeded == 3
        assert resp.failed == 0
        assert resp.results[2].already_existed is True

    async def test_bulk_with_failing_record_keeps_others_succeeding(self):
        # Parent OK; finding #1 OK; finding #2 raises; finding #3 OK; refresh.
        async def side_effect(cypher, params, *args, **kwargs):
            # Parent check
            if "RETURN mr.module_run_id AS id" in cypher and "LIMIT 1" in cypher:
                m = MagicMock()
                m.records = [_rec(id="mr-1")]
                return m
            if params.get("finding_id") == "ev-2-broken":
                raise RuntimeError("simulated write failure")
            if "MERGE (f:Finding" in cypher:
                m = MagicMock()
                m.records = [_rec(created=True)]
                return m
            # refresh
            m = MagicMock()
            m.records = []
            return m

        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=side_effect)
        svc = AssessmentService(driver)
        findings = [
            Finding(
                finding_id="ev-1",
                graph_id="tenant-A",
                run_id="run-1",
                module_run_id="mr-1",
                claim="ok",
            ),
            Finding(
                finding_id="ev-2-broken",
                graph_id="tenant-A",
                run_id="run-1",
                module_run_id="mr-1",
                claim="broken",
            ),
            Finding(
                finding_id="ev-3",
                graph_id="tenant-A",
                run_id="run-1",
                module_run_id="mr-1",
                claim="ok",
            ),
        ]
        resp = await svc.record_finding_bulk("tenant-A", "run-1", "mr-1", findings)
        assert resp.total == 3
        assert resp.succeeded == 2
        assert resp.failed == 1
        # ordered results
        assert resp.results[0].success is True
        assert resp.results[1].success is False
        assert resp.results[1].error == "simulated write failure"
        assert resp.results[2].success is True

    async def test_bulk_raises_when_parent_module_run_missing(self):
        driver = _make_driver(record_sequences=[[]])  # parent not found
        svc = AssessmentService(driver)
        findings = [
            Finding(
                finding_id="ev-1",
                graph_id="tenant-A",
                run_id="run-1",
                module_run_id="mr-missing",
                claim="x",
            )
        ]
        with pytest.raises(ValueError, match="ModuleRun not found"):
            await svc.record_finding_bulk("tenant-A", "run-1", "mr-missing", findings)


# ── record_conflict ───────────────────────────────────────────────────────────


class TestRecordConflict:
    async def test_record_conflict_writes_involves_edges(self):
        # Per TASK-073 Finding 1 fix, the call order now includes a pre-MERGE
        # probe to detect cross-tenant id collisions.
        # 1) cross-tenant probe → empty (no existing :Conflict)
        # 2) Conflict MERGE → created=True
        # 3 + 4) one [:INVOLVES] edge per involved_finding_id
        driver = _make_driver(record_sequences=[[], [_rec(created=True)], [], []])
        svc = AssessmentService(driver)
        conflict = Conflict(
            conflict_id="cf-1",
            graph_id="tenant-A",
            run_id="run-1",
            topic="x",
            summary="y",
            involved_finding_ids=["ev-1", "ev-2"],
        )
        created = await svc.record_conflict("tenant-A", "run-1", conflict)
        assert created is True
        # probe + MERGE on Conflict + 2 INVOLVES edges = 4 calls
        assert driver.execute_query.call_count == 4
        # The MERGE + edge writes carry the graph_id parameter
        for call in driver.execute_query.call_args_list[1:]:
            assert call[0][1]["graph_id"] == "tenant-A"

    async def test_record_conflict_cross_tenant_collision_rejected(self):
        """Pre-MERGE probe detects a foreign-tenant :Conflict with the same id."""
        # 1) probe returns a foreign-tenant row
        driver = _make_driver(record_sequences=[[_rec(graph_id="OTHER-tenant")]])
        svc = AssessmentService(driver)
        conflict = Conflict(
            conflict_id="cf-x",
            graph_id="tenant-A",
            run_id="run-1",
            topic="x",
            summary="y",
        )
        with pytest.raises(RuntimeError, match="different tenant"):
            await svc.record_conflict("tenant-A", "run-1", conflict)
        # MERGE never ran — only the probe
        assert driver.execute_query.call_count == 1

    async def test_record_conflict_rejects_run_id_mismatch(self):
        svc = AssessmentService(_make_driver())
        bad = Conflict(
            conflict_id="cf-1",
            graph_id="tenant-A",
            run_id="OTHER-run",
            topic="x",
            summary="y",
        )
        with pytest.raises(ValueError, match="does not match"):
            await svc.record_conflict("tenant-A", "run-1", bad)


# ── record_unresolved_question ────────────────────────────────────────────────


class TestRecordUnresolvedQuestion:
    async def test_question_writes_raised_edge(self):
        # 1) cross-tenant probe → empty
        # 2) Question MERGE + RAISED edge
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]])
        svc = AssessmentService(driver)
        q = UnresolvedQuestion(
            question_id="q-1",
            graph_id="tenant-A",
            run_id="run-1",
            module_run_id="mr-1",
            text="What is X?",
        )
        created = await svc.record_unresolved_question("tenant-A", "run-1", "mr-1", q)
        assert created is True
        cypher, params = driver.execute_query.call_args[0]
        assert "MERGE (q:UnresolvedQuestion:__Platform__" in cypher
        assert "MERGE (mr)-[:RAISED]->(q)" in cypher
        assert params["graph_id"] == "tenant-A"

    async def test_question_cross_tenant_collision_rejected(self):
        """Pre-MERGE probe detects foreign-tenant :UnresolvedQuestion."""
        driver = _make_driver(record_sequences=[[_rec(graph_id="OTHER-tenant")]])
        svc = AssessmentService(driver)
        q = UnresolvedQuestion(
            question_id="q-x",
            graph_id="tenant-A",
            run_id="run-1",
            module_run_id="mr-1",
            text="?",
        )
        with pytest.raises(RuntimeError, match="different tenant"):
            await svc.record_unresolved_question("tenant-A", "run-1", "mr-1", q)
        assert driver.execute_query.call_count == 1


# ── persist_deliverable / persist_final_docs ──────────────────────────────────


class TestPersistDeliverable:
    async def test_deliverable_with_module_run_writes_three_calls(self):
        # Per TASK-073 Finding 1 fix:
        # 1) cross-tenant probe → empty
        # 2) Deliverable MERGE
        # 3) [:PRODUCED_DELIVERABLE] wiring (because module_run_id is set)
        driver = _make_driver(record_sequences=[[], [_rec(created=True)], []])
        svc = AssessmentService(driver)
        d = Deliverable(
            deliverable_id="d-1",
            graph_id="tenant-A",
            run_id="run-1",
            module_run_id="mr-1",
            kind="module-md",
            filename="m1.md",
        )
        created = await svc.persist_deliverable("tenant-A", "run-1", d)
        assert created is True
        assert driver.execute_query.call_count == 3

    async def test_deliverable_without_module_run_writes_two_calls(self):
        # 1) probe → empty; 2) Deliverable MERGE
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]])
        svc = AssessmentService(driver)
        d = Deliverable(
            deliverable_id="d-final",
            graph_id="tenant-A",
            run_id="run-1",
            kind="final-html",
            filename="final.html",
        )
        created = await svc.persist_deliverable("tenant-A", "run-1", d)
        assert created is True
        assert driver.execute_query.call_count == 2

    async def test_deliverable_cross_tenant_collision_rejected(self):
        """Pre-MERGE probe detects foreign-tenant :Deliverable."""
        driver = _make_driver(record_sequences=[[_rec(graph_id="OTHER-tenant")]])
        svc = AssessmentService(driver)
        d = Deliverable(
            deliverable_id="d-x",
            graph_id="tenant-A",
            run_id="run-1",
            kind="module-md",
            filename="x.md",
        )
        with pytest.raises(RuntimeError, match="different tenant"):
            await svc.persist_deliverable("tenant-A", "run-1", d)
        assert driver.execute_query.call_count == 1

    async def test_persist_final_docs_returns_bulk_response(self):
        # 5 final docs: each gets a probe + a MERGE call (no module_run_id).
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]] * 5)
        svc = AssessmentService(driver)
        deliverables = [
            Deliverable(
                deliverable_id=f"d-{i}",
                graph_id="tenant-A",
                run_id="run-1",
                kind="final-html",
                filename=f"f{i}.html",
                ordinal=i,
            )
            for i in range(5)
        ]
        resp = await svc.persist_final_docs("tenant-A", "run-1", deliverables)
        assert resp.total == 5
        assert resp.succeeded == 5
        assert resp.failed == 0


# ── finalize_run ──────────────────────────────────────────────────────────────


class TestFinalizeRun:
    async def test_passing_run_marks_finished(self):
        # counts query → 1 record with passing thresholds
        # then update query (returns no records, doesn't matter)
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        direct_count=5,
                        inferred_count=2,
                        deliverable_count=3,
                        unresolved_conflict_count=0,
                        open_question_count=1,
                    )
                ],
                [],
            ]
        )
        svc = AssessmentService(driver)
        resp = await svc.finalize_run("tenant-A", "run-1")
        assert resp.passed is True
        assert resp.status == "finished"
        assert resp.direct_finding_count == 5
        assert resp.failure_reasons == []
        # The update query was called with status='finished'
        _, params = driver.execute_query.call_args_list[1][0]
        assert params["status"] == "finished"
        assert params["graph_id"] == "tenant-A"

    async def test_failing_run_marks_failed_with_reasons(self):
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        direct_count=0,  # zero direct findings — fails gate
                        inferred_count=2,
                        deliverable_count=0,  # zero deliverables — fails gate
                        unresolved_conflict_count=1,
                        open_question_count=4,
                    )
                ],
                [],
            ]
        )
        svc = AssessmentService(driver)
        resp = await svc.finalize_run("tenant-A", "run-1")
        assert resp.passed is False
        assert resp.status == "failed"
        assert len(resp.failure_reasons) == 2
        assert any("direct_findings" in r for r in resp.failure_reasons)
        assert any("deliverables" in r for r in resp.failure_reasons)

    async def test_run_not_found_raises(self):
        driver = _make_driver(record_sequences=[[]])
        svc = AssessmentService(driver)
        with pytest.raises(ValueError, match="not found"):
            await svc.finalize_run("tenant-A", "missing-run")


# ── Registry routing (ADR-019) ────────────────────────────────────────────────


class TestRegistryRouting:
    async def test_private_routes_to_owner_tenant_graph(self):
        # Per TASK-073 Finding 1 fix: ownership/tenancy probe runs first.
        # 1) probe → empty; 2) RegistryItem MERGE
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]])
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-1",
            graph_id="tenant-owner",
            kind="skill",
            slug="users/u1/my-skill",
            visibility="private",
            owner_user_id="u1",
            name="my skill",
        )
        created = await svc.persist_registry_item(
            item, owner_tenant_graph_id="tenant-owner"
        )
        assert created is True
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == "tenant-owner"
        assert params["visibility"] == "private"

    async def test_public_routes_to_registry_catalog(self):
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]])
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-2",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="my-skill",
            visibility="public",
            owner_user_id="u1",
            name="my skill",
        )
        await svc.persist_registry_item(item)
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == REGISTRY_CATALOG_GRAPH_ID

    async def test_curated_routes_to_registry_catalog(self):
        driver = _make_driver(record_sequences=[[], [_rec(created=True)]])
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-3",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="eurail-report",
            version="1.0.0",
            visibility="curated",
            owner_user_id="oraclous-admin",
            name="Eurail Report",
        )
        await svc.persist_registry_item(item)
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == REGISTRY_CATALOG_GRAPH_ID

    async def test_public_non_owner_update_raises_ownership_error(self):
        """Per TASK-073 Finding 1 (TASK-069): non-owner overwrite is rejected."""
        from app.services.assessment_service import RegistryOwnershipError

        # 1) probe returns an existing public item owned by 'alice'
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        graph_id=REGISTRY_CATALOG_GRAPH_ID,
                        owner_user_id="alice",
                        visibility="public",
                    )
                ]
            ]
        )
        svc = AssessmentService(driver)
        # 'bob' (different owner) tries to update Alice's public item
        item = RegistryItem(
            item_id="ri-alice",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="alice-skill",
            visibility="yanked",
            owner_user_id="bob",
            name="Alice's skill (hijacked)",
        )
        with pytest.raises(RegistryOwnershipError, match="owned by another user"):
            await svc.persist_registry_item(item)
        # MERGE never ran — only the probe
        assert driver.execute_query.call_count == 1

    async def test_public_owner_update_succeeds(self):
        """Owner CAN update their own public item."""
        # 1) probe returns Alice's existing public item
        # 2) MERGE
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        graph_id=REGISTRY_CATALOG_GRAPH_ID,
                        owner_user_id="alice",
                        visibility="public",
                    )
                ],
                [_rec(created=False)],
            ]
        )
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-alice",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="alice-skill",
            visibility="public",
            owner_user_id="alice",  # same as existing
            name="Alice's skill v2",
        )
        created = await svc.persist_registry_item(item)
        assert created is False  # already existed
        assert driver.execute_query.call_count == 2

    async def test_curated_admin_can_update_others_item(self):
        """Curated writes are admin-gated at the endpoint and exempt from
        the owner check (admins may update items owned by other users)."""
        # 1) probe returns an existing curated item owned by 'eve'
        # 2) MERGE
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        graph_id=REGISTRY_CATALOG_GRAPH_ID,
                        owner_user_id="eve",
                        visibility="curated",
                    )
                ],
                [_rec(created=False)],
            ]
        )
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-curated",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="eurail-report",
            visibility="curated",
            owner_user_id="oraclous-admin",
            name="Eurail Report (curated update)",
        )
        # Should NOT raise — curated → no owner check
        created = await svc.persist_registry_item(item)
        assert created is False
        assert driver.execute_query.call_count == 2

    async def test_public_cross_tenant_collision_rejected(self):
        """Existing RegistryItem in a different graph than the target raises."""
        # 1) probe returns existing item in tenant graph 'tenant-X' but we're
        # writing public (target = __registry__)
        driver = _make_driver(
            record_sequences=[
                [
                    _rec(
                        graph_id="tenant-X",
                        owner_user_id="u1",
                        visibility="private",
                    )
                ],
            ]
        )
        svc = AssessmentService(driver)
        item = RegistryItem(
            item_id="ri-collision",
            graph_id=REGISTRY_CATALOG_GRAPH_ID,
            kind="skill",
            slug="something",
            visibility="public",
            owner_user_id="u1",
            name="x",
        )
        with pytest.raises(RuntimeError, match="different graph"):
            await svc.persist_registry_item(item)
        assert driver.execute_query.call_count == 1

    async def test_private_without_owner_tenant_id_raises(self):
        svc = AssessmentService(_make_driver())
        item = RegistryItem(
            item_id="ri-4",
            graph_id="tenant-x",
            kind="skill",
            slug="users/u1/x",
            visibility="private",
            owner_user_id="u1",
            name="x",
        )
        with pytest.raises(ValueError, match="owner_tenant_graph_id is required"):
            await svc.persist_registry_item(item)

    async def test_graph_id_mismatch_raises(self):
        svc = AssessmentService(_make_driver())
        # private item with graph_id NOT matching owner_tenant_graph_id arg
        item = RegistryItem(
            item_id="ri-5",
            graph_id="tenant-A",
            kind="skill",
            slug="users/u1/x",
            visibility="private",
            owner_user_id="u1",
            name="x",
        )
        with pytest.raises(ValueError, match="does not match"):
            await svc.persist_registry_item(item, owner_tenant_graph_id="tenant-B")


# ── Catalog-graph constant ────────────────────────────────────────────────────


class TestCatalogGraphConstants:
    def test_assessment_catalog_constant(self):
        assert ASSESSMENTS_CATALOG_GRAPH_ID == "__assessments_catalog__"

    def test_registry_catalog_constant(self):
        assert REGISTRY_CATALOG_GRAPH_ID == "__registry__"
