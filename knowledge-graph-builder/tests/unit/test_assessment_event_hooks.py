"""Unit tests for AssessmentService → broker publish hooks (TASK-081).

These tests verify the **wiring**: every write method publishes the right
event with the right payload to the injected broker. They do not exercise
the broker internals (covered by `test_assessment_event_broker.py`) nor
the Neo4j writes (covered by `test_assessment_service.py`).

Pattern: stub the AsyncDriver with pre-recorded records, give the service
a real `AssessmentEventBroker`, and assert via the broker's history.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.assessment_schemas import (
    Conflict,
    Deliverable,
    Finding,
    UnresolvedQuestion,
    UpdateModuleRunRequest,
)
from app.services.assessment_event_broker import AssessmentEventBroker
from app.services.assessment_service import AssessmentService


def _driver(seqs):
    drv = AsyncMock()
    results = []
    for recs in seqs:
        r = MagicMock()
        r.records = recs
        results.append(r)
    drv.execute_query = AsyncMock(side_effect=results)
    return drv


def _events_for(broker: AssessmentEventBroker, graph_id: str, run_id: str):
    """Return the list of buffered events for (graph_id, run_id)."""
    return list(broker._history.get((graph_id, run_id), []))  # noqa: SLF001


# ── update_module_run ──────────────────────────────────────────────────────────


class TestModuleRunStatusEvents:
    async def test_status_change_publishes_status_changed_event(self):
        broker = AssessmentEventBroker()
        # First call: probe for prev_status; second: SET.
        drv = _driver([[{"status": "planned"}], [{"id": "mr-1"}]])
        svc = AssessmentService(drv, event_broker=broker)

        ok = await svc.update_module_run(
            "g1",
            "r1",
            "mr-1",
            UpdateModuleRunRequest(status="running"),
        )
        assert ok is True

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        ev = events[0]
        assert ev.type == "module_run.status_changed"
        assert ev.payload["prev_status"] == "planned"
        assert ev.payload["new_status"] == "running"
        assert ev.payload["module_run_id"] == "mr-1"

    async def test_no_status_change_emits_no_event(self):
        """status==prev_status should not emit a status_changed event."""
        broker = AssessmentEventBroker()
        drv = _driver([[{"status": "running"}], [{"id": "mr-1"}]])
        svc = AssessmentService(drv, event_broker=broker)

        await svc.update_module_run(
            "g1", "r1", "mr-1", UpdateModuleRunRequest(status="running")
        )

        assert _events_for(broker, "g1", "r1") == []

    async def test_heartbeat_only_update_emits_heartbeat(self):
        broker = AssessmentEventBroker()
        # No status field => no probe, just one execute_query for the SET.
        drv = _driver([[{"id": "mr-1"}]])
        svc = AssessmentService(drv, event_broker=broker)

        await svc.update_module_run(
            "g1",
            "r1",
            "mr-1",
            UpdateModuleRunRequest(last_heartbeat_at=datetime.now(UTC)),
        )

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        assert events[0].type == "module_run.heartbeat"

    async def test_failed_update_emits_no_event(self):
        """If the Cypher returns no records (row not found), no event fires."""
        broker = AssessmentEventBroker()
        drv = _driver([[{"status": "planned"}], []])  # probe ok, SET returns nothing
        svc = AssessmentService(drv, event_broker=broker)

        ok = await svc.update_module_run(
            "g1", "r1", "mr-1", UpdateModuleRunRequest(status="running")
        )
        assert ok is False
        assert _events_for(broker, "g1", "r1") == []


# ── record_finding_bulk ───────────────────────────────────────────────────────


class TestFindingBulkEvents:
    async def test_bulk_emits_single_delta_event(self):
        broker = AssessmentEventBroker()
        # Parent existence probe + 1 finding MERGE + refresh-evidence-count
        # The MERGE for a finding returns _created=true.
        drv = _driver(
            [
                [{"id": "mr-1"}],  # parent ModuleRun probe
                [{"created": True}],  # finding MERGE
                [],  # refresh_evidence_count
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)

        findings = [
            Finding(
                finding_id="f-1",
                graph_id="g1",
                run_id="r1",
                module_run_id="mr-1",
                claim="x",
            )
        ]
        await svc.record_finding_bulk("g1", "r1", "mr-1", findings)

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        ev = events[0]
        assert ev.type == "finding.recorded"
        assert ev.payload["finding_count_delta"] == 1
        assert ev.payload["module_run_id"] == "mr-1"

    async def test_bulk_with_zero_new_findings_emits_no_event(self):
        """Idempotent replay (every finding already_existed) emits no event."""
        broker = AssessmentEventBroker()
        drv = _driver(
            [
                [{"id": "mr-1"}],  # parent probe
                [{"created": False}],  # finding MERGE — already_existed
                [],  # refresh_evidence_count
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)
        findings = [
            Finding(
                finding_id="f-1",
                graph_id="g1",
                run_id="r1",
                module_run_id="mr-1",
                claim="x",
            )
        ]
        await svc.record_finding_bulk("g1", "r1", "mr-1", findings)

        assert _events_for(broker, "g1", "r1") == []


# ── record_conflict ───────────────────────────────────────────────────────────


class TestConflictEvents:
    async def test_record_conflict_emits_event(self):
        broker = AssessmentEventBroker()
        # probe (no prior), MERGE (created=True), and one [:INVOLVES] edge merge.
        drv = _driver(
            [
                [],  # existence_probe — no prior row
                [{"created": True}],  # conflict MERGE
                [],  # [:INVOLVES] edge
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)

        conf = Conflict(
            conflict_id="c-1",
            graph_id="g1",
            run_id="r1",
            topic="t",
            summary="s",
            status="open",
            involved_finding_ids=["f-1"],
        )
        await svc.record_conflict("g1", "r1", conf)

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        assert events[0].type == "conflict.recorded"
        assert events[0].payload["conflict_id"] == "c-1"
        assert events[0].payload["status"] == "open"


# ── record_unresolved_question ────────────────────────────────────────────────


class TestUnresolvedQuestionEvents:
    async def test_new_question_emits_event(self):
        broker = AssessmentEventBroker()
        # existence probe + MERGE
        drv = _driver(
            [
                [],  # existence_probe — no prior row
                [{"created": True}],  # MERGE result
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)

        q = UnresolvedQuestion(
            question_id="q-1",
            graph_id="g1",
            run_id="r1",
            module_run_id="mr-1",
            text="why?",
            status="open",
        )
        await svc.record_unresolved_question("g1", "r1", "mr-1", q)

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        assert events[0].type == "unresolved_question.raised"
        assert events[0].payload["question_id"] == "q-1"

    async def test_replay_emits_no_event(self):
        broker = AssessmentEventBroker()
        drv = _driver(
            [
                [],  # existence_probe — no prior row
                [{"created": False}],  # MERGE matched existing
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)

        q = UnresolvedQuestion(
            question_id="q-1",
            graph_id="g1",
            run_id="r1",
            module_run_id="mr-1",
            text="why?",
            status="open",
        )
        await svc.record_unresolved_question("g1", "r1", "mr-1", q)

        assert _events_for(broker, "g1", "r1") == []


# ── persist_deliverable ───────────────────────────────────────────────────────


class TestDeliverableEvents:
    async def test_persist_emits_event(self):
        broker = AssessmentEventBroker()
        drv = _driver(
            [
                [],  # existence_probe — no prior row
                [{"created": True}],  # MERGE result
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)

        d = Deliverable(
            deliverable_id="d-1",
            graph_id="g1",
            run_id="r1",
            module_run_id=None,
            kind="module-md",
            filename="x.md",
            ordinal=0,
        )
        await svc.persist_deliverable("g1", "r1", d)

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        ev = events[0]
        assert ev.type == "deliverable.persisted"
        assert ev.payload["deliverable_id"] == "d-1"
        assert ev.payload["filename"] == "x.md"


# ── finalize_run ──────────────────────────────────────────────────────────────


class TestFinalizeEvents:
    async def test_finalize_emits_run_finalized(self):
        broker = AssessmentEventBroker()
        drv = _driver(
            [
                [  # counts query
                    {
                        "direct_count": 5,
                        "inferred_count": 1,
                        "deliverable_count": 2,
                        "unresolved_conflict_count": 0,
                        "open_question_count": 0,
                    }
                ],
                [],  # SET status='finished'
            ]
        )
        svc = AssessmentService(drv, event_broker=broker)
        resp = await svc.finalize_run("g1", "r1")

        events = _events_for(broker, "g1", "r1")
        assert len(events) == 1
        ev = events[0]
        assert ev.type == "run.finalized"
        assert ev.payload["status"] == resp.status
        assert ev.payload["passed"] is True
        assert ev.payload["evidence_count_direct"] == 5


# ── Fire-and-forget contract ──────────────────────────────────────────────────


class TestFireAndForget:
    async def test_publish_failure_does_not_roll_back_write(self):
        """A broker exception during publish must NOT raise into the caller."""

        class BoomBroker(AssessmentEventBroker):
            def publish(self, *a, **kw):  # type: ignore[override]
                raise RuntimeError("simulated broker failure")

        broker = BoomBroker()
        drv = _driver([[{"status": "planned"}], [{"id": "mr-1"}]])
        svc = AssessmentService(drv, event_broker=broker)

        # Must NOT raise.
        ok = await svc.update_module_run(
            "g1", "r1", "mr-1", UpdateModuleRunRequest(status="running")
        )
        assert ok is True


pytestmark = pytest.mark.asyncio
